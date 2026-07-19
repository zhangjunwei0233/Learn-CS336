from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from os import PathLike, getpid
from time import perf_counter

import regex as re
import os
import heapq

"""
NOTE: Optimization 1: More efficient data structure design

We use the following data structures:

1. Pretoken_table: index for pretokens and its frequency:

pretoken_id     sequence                    frequency
0               (b"a", b"l", b"l")          3
1               (b"b", b"a", b"l", b"l")    2
2               (b"c", b"a", b"t")           5
...

2. pair_table: mappings from adjacent pairs to its counts and originating pretokens

pair            count       pretokens
(b"l", b"l")    100         {0, 1}
(b"a", b"l")    13          {0, 1, 4}
...

The working loop is as follows

1. identify the pairs to merge, say b"l" and b"l" to b"ll"
2. add b"ll" to vocab dict and update merge list
3. perform merging in the pretoken table (indicated by pair-pretokens mapping)

NOTE: Optimization 2: Heap for best pair selection

Instantiate a pair_heap alongside pair_table to sort pairs along counts.

The problem is how to synchronize the heap and the table, this is achieved by:
- Timely update: push a new record to heap whenever table updates
- Lazy deletion: keep the outdated record inside the heap, only make sure the selected entry is up-to-date

NOTE: Optimization 3: Multi-Process for pre-tokenization

Together, these optimizations reduced runtime on TinyStories validation set from over 1m to 0.7s.
"""

# Regex Pattern for Pretokenization
GPT2_PERTOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Define type aliases for clarity
Token = bytes
Pretoken = tuple[Token, ...]
Pair = tuple[Token, Token]

@dataclass(frozen=True, slots=True)
class Chunk:
    """A byte range that starts and ends at safe document boundaries."""
    input_path: str
    start: int
    end: int
    special_tokens: list[str]


@dataclass(slots=True)
class PretokenEntry:
    seq: Pretoken
    freq: int

@dataclass(slots=True)
class PairEntry:
    count: int
    pids: set[int]

@dataclass(frozen=True, slots=True)
class HeapItem:
    count: int
    pair: Pair

    def __lt__(self, other):
        return (self.count, self.pair) > (other.count, other.pair)

# Reusing these objects avoids constructing a new one-byte ``bytes`` object
# every time a distinct pre-token is converted to its initial token sequence.
BYTE_TOKENS = tuple(bytes([value]) for value in range(256))

def _find_chunk_boundaries(
    input_path: str | PathLike[str],
    desired_num_chunks: int,
    split_special_token: list[str],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """

    with open(input_path, "rb") as f:
        # Get total file size in bytes
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            f.seek(initial_position)  # Start at boundary guess
            while True:
                found = 0
                mini_chunk = f.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                for special_token in split_special_token:
                    found_at = mini_chunk.find(special_token.encode("utf-8"))
                    if found_at != -1:
                        chunk_boundaries[bi] = initial_position + found_at
                        found = 1
                        break

                if found == 1:
                    break
                initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Remove special tokens while preserving them as hard boundries."""
    if not special_tokens:
        return [text]
    
    # Reorder longer special tokens in the front to resolve overlap
    alternatives = sorted((re.escape(token) for token in special_tokens), key=len, reverse=True)

    # Split the text by constructing delimiters
    delimiter = re.compile("|".join(alternatives))
    return delimiter.split(text)


def _worker(chunk: Chunk) -> Counter[bytes]:
    print(f"Worker process {getpid()} recieved chunk [{chunk.start}, {chunk.end}]")
    counts: Counter[bytes] = Counter()

    with open(chunk.input_path, "rb") as f:
        f.seek(chunk.start)
        chunk_bytes = f.read(chunk.end - chunk.start)
    
    text = chunk_bytes.decode("utf-8")
    
    for ordinary_text in _split_on_special_tokens(text, chunk.special_tokens):
        # For each split, use PATTERN to separate to pretokens
        for match in GPT2_PERTOKEN_PATTERN.finditer(ordinary_text):
            # Convert pretokens to utf-8 encoded bytes
            counts[match.group().encode("utf-8")] += 1

    return counts

def _build_pretoken_table(
    input_path: str | PathLike[str],
    special_tokens: list[str],
    num_processes: int,
) -> list[PretokenEntry]:
    """Pre-tokenize text and return pretoken table. Uses Multi-process"""

    # Count through multi-process
    path = os.fspath(input_path)
    boundaries = _find_chunk_boundaries(input_path, num_processes, special_tokens)
    args = [
        Chunk(path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if start < end
    ]

    worker_count = min(num_processes, len(args))
    print(f"Launching {worker_count} workers...")
    with Pool(processes=worker_count) as pool:
        partial_counts = pool.map(_worker, args)
    
    # Aggregate the results
    total_counts: Counter[bytes] = Counter()
    for partial_count in partial_counts:
        total_counts.update(partial_count)

    print("Closed multi-process.")
    return [
        PretokenEntry(
            seq=tuple(BYTE_TOKENS[value] for value in p),
            freq=c
        )
        for p, c in total_counts.items()
    ]

def _build_pair_table(
    pretoken_table: list[PretokenEntry]
) -> dict[Pair, PairEntry]:
    """Initialize pair table on a given pretoken table."""
    pair_table: dict[Pair, PairEntry] = {}

    for pid, entry in enumerate(pretoken_table):
        pretoken = entry.seq
        freq = entry.freq

        for pair in zip(pretoken, pretoken[1:]):
            if pair in pair_table:
                pair_table[pair].count += freq
                pair_table[pair].pids.add(pid)
            else:
                pair_table[pair] = PairEntry(count=freq, pids={pid})

    return pair_table

def _merge_pair(pretoken: Pretoken, pair: Pair) -> Pretoken:
    """Replace non-overlapping occurences of pair from left to right."""
    left, right = pair
    output: list[Token] = []

    index = 0
    while index < len(pretoken):
        has_pair = (
            index + 1 < len(pretoken)
            and pretoken[index] == left
            and pretoken[index + 1] == right
        )
        if has_pair:
            output.append(left + right)
            index += 2
        else:
            output.append(pretoken[index])
            index += 1

    return tuple(output)

def _pop_best_pair(
    heap: list[HeapItem],
    reference: dict[Pair, PairEntry],
) -> tuple[Pair, PairEntry]:
    while heap: # Pop from heap until entry is up-to-date
        candidate = heapq.heappop(heap)
        entry = reference.get(candidate.pair)
        if entry is not None and entry.count == candidate.count:
            return candidate.pair, reference.pop(candidate.pair)
    raise RuntimeError("pair heap is empty while the pair table is not")

def _merge_and_find_changes(
    pretoken: Pretoken,
    selected_pair: Pair,
) -> tuple[Pretoken, dict[Pair, int], set[Pair]]:
    """
    Merge selected pairs inside pretoken
    Return the merged pretoken, old pair changes and pairs in new pretoken
    """
    output: list[Token] = []
    pair_changes: dict[Pair, int] = defaultdict(int)
    new_pairs: set[Pair] = set()
    
    position = 0
    while position < len(pretoken):
        is_selected = (
            position + 1 < len(pretoken)
            and (pretoken[position], pretoken[position + 1]) == selected_pair
        )

        # Consider old pretoken change
        if is_selected:
            # Remove the selected old pair
            pair_changes[selected_pair] -= 1

            # The second selected token may have an old right-hand pair
            if position + 2 < len(pretoken):
                old_right_pair = (pretoken[position + 1], pretoken[position + 2])
                pair_changes[old_right_pair] -= 1
            
            next_token = pretoken[position] + pretoken[position + 1]
            position += 2
        else:
            # Remove this token's old outgoing pair
            if position + 1 < len(pretoken):
                old_pair = (pretoken[position], pretoken[position + 1])
                pair_changes[old_pair] -= 1

            next_token = pretoken[position]
            position += 1
        
        # Consider new pretoken change
        if output:
            # Add the new pair created in the output pretoken
            new_pair = (output[-1], next_token)
            pair_changes[new_pair] += 1
            new_pairs.add(new_pair)

        output.append(next_token)

    return tuple(output), pair_changes, new_pairs

def _apply_merge(
    selected_pair: Pair,
    selected_pair_entry: PairEntry,
    pretoken_table: list[PretokenEntry],
    pair_table: dict[Pair, PairEntry],
) -> set[Pair]:
    """Apply one merge and return pairs whose heap priorities changed."""
    changed_pairs: set[Pair] = set()

    for pid in selected_pair_entry.pids:
        pretoken_entry = pretoken_table[pid]
        old_pretoken = pretoken_entry.seq
        new_pretoken, pair_changes, new_pairs = _merge_and_find_changes(
            old_pretoken,
            selected_pair
        )

        # Update pair table
        for p in pair_changes.keys():
            # Do not operate on selected pair
            if p == selected_pair:
                continue

            count_change = pair_changes[p] * pretoken_entry.freq

            # If count unchanged, keep it untouched
            if count_change == 0:
                continue

            # Otherwise count has changed, update it
            pentry = pair_table.get(p)
            if pentry is None:  # Insert a new pair
                if count_change <= 0:
                    raise RuntimeError("cannot remove a pair absent from the table")
                pair_table[p] = PairEntry(count_change, {pid})
            else:
                pentry.count += count_change
                if p in new_pairs:
                    pentry.pids.add(pid)
                else:
                    pentry.pids.discard(pid)
                
                if pentry.count < 0:
                    raise RuntimeError("pair count became negative")
                if pentry.count == 0:
                    if pentry.pids:
                        raise RuntimeError("zero-count pair still references pre-tokens")
                    del pair_table[p]
            
            # Then add it to changed pairs
            changed_pairs.add(p)
        
        # Update pretoken table
        pretoken_entry.seq = new_pretoken

    return changed_pairs

def _refresh_heap(
    pair_heap: list[HeapItem],
    pair_table: dict[Pair, PairEntry],
    changed_pairs: set[Pair],
) -> None:
    """Push current priorities and occasianlly discard accumulated garbage."""
    for p in changed_pairs:
        pentry = pair_table.get(p)
        if pentry is not None:
            heapq.heappush(pair_heap, HeapItem(pentry.count, p))

    # Lazy deletion is fast but leaves stale entries behind. Rebuilding keeps
    # memory bounded and prevents future pops from wading through too much
    # historical state.
    live_pair_count = len(pair_table)
    if live_pair_count and len(pair_heap) > 4 * live_pair_count:
        pair_heap[:] = [
            HeapItem(pentry.count, p)
            for p, pentry in pair_table.items()
        ]
        heapq.heapify(pair_heap)


def train_bpe_tokenizer(
    input_path: str | PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 4,
    profile: bool = False,
    log_step: int = 500,
) -> tuple[dict[int, Token], list[Pair]]:
    """Train a byte-level BPE tokenizer and return its vocabulary and merges."""

    # Sanity Check
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")
    if log_step is not None and log_step < 1:
        raise ValueError("log_step must be positive or None")
    if any(not token for token in special_tokens):
        raise ValueError("special tokens must not be empty strings")

    initial_vocab_size = 256 + len(special_tokens)
    if vocab_size < initial_vocab_size:
        raise ValueError(
            f"vocab_size must be at least {initial_vocab_size} "
            "to contain all bytes and special tokens"
        )

    total_start = perf_counter()

    # Create initial vocab and merges
    vocab = {token_id: BYTE_TOKENS[token_id] for token_id in range(256)}
    for token_id, special_token in enumerate(special_tokens, start=256):
        vocab[token_id] = special_token.encode("utf-8")

    merges: list[Pair] = []

    # Build pretoken_table
    print("Building Pretoken Table...")
    pretokenize_start = perf_counter()
    pretoken_table = _build_pretoken_table(input_path, special_tokens, num_processes)
    pretokenize_seconds = perf_counter() - pretokenize_start

    # Build pair_table
    print("Building pair Table...")
    build_pair_start = perf_counter()
    pair_table = _build_pair_table(pretoken_table)
    build_pair_seconds = perf_counter() - build_pair_start

    # Build pair_heap
    print("Building pair heap...")
    build_heap_start = perf_counter()
    pair_heap = [
        HeapItem(entry.count, pair)
        for pair, entry in pair_table.items()
    ]
    heapq.heapify(pair_heap)
    build_heap_seconds = perf_counter() - build_heap_start

    # Main loop: perform merge until target vocab size
    print("Running main loop...")
    pair_selection_seconds = 0.0
    table_update_seconds = 0.0
    heap_update_seconds = 0.0

    iter_count = 1
    round_start = perf_counter()  # log_step iters is a round
    for new_token_id in range(initial_vocab_size, vocab_size):
        if not pair_table: # Prevent crashing
            break

        # Get the pair with largest count
        section_start = perf_counter()
        best_pair, best_entry = _pop_best_pair(pair_heap, pair_table)
        pair_selection_seconds += perf_counter() - section_start

        # Update vocab and merge list
        vocab[new_token_id] = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        # Update pretoken & pair table
        section_start = perf_counter()
        changed_pairs = _apply_merge(
            selected_pair=best_pair,
            selected_pair_entry=best_entry,
            pretoken_table=pretoken_table,
            pair_table=pair_table,
        ) # Document changed pairs to update the heap
        table_update_seconds += perf_counter() - section_start

        # Update the heap
        section_start = perf_counter()
        _refresh_heap(pair_heap, pair_table, changed_pairs)
        heap_update_seconds += perf_counter() - section_start

        # Print logging message
        if iter_count % log_step == 0:
            average_iter_seconds = (perf_counter() - round_start) / log_step
            print(f"Iter {iter_count}: {average_iter_seconds} s/iter.")
            round_start = perf_counter()

        iter_count += 1
    
    # Print profile info
    if profile:
        total_seconds = perf_counter() - total_start
        sections = (
            ("pre-tokenization", pretokenize_seconds),
            ("build pair", build_pair_seconds),
            ("build heap", build_heap_seconds),
            ("pair selection", pair_selection_seconds),
            ("table update", table_update_seconds),
            ("heap update", heap_update_seconds),
        )

        print("\nBPE training profile")
        for section_name, seconds in sections:
            percentage = 100 * seconds / total_seconds if total_seconds else 0.0
            print(f"  {section_name:<20} {seconds:9.3f}s  ({percentage:5.1f}%)")
        print(f"  {'total':<20} {total_seconds:9.3f}s")


    return vocab, merges