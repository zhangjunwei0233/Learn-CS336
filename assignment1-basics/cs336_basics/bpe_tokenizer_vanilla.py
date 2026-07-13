from collections import Counter
from os import PathLike
from time import perf_counter

import regex as re

# Regex Pattern for Pretokenization
GPT2_PERTOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Define type aliases for clarity
Token = bytes
Pretoken = tuple[Token, ...]
Pair = tuple[Token, Token]

def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Remove special tokens while preserving them as hard boundries."""
    if not special_tokens:
        return [text]
    
    # Reorder longer special tokens in the front to resolve overlap
    alternatives = sorted((re.escape(token) for token in special_tokens), key=len, reverse=True)

    # Split the text by constructing delimiters
    delimiter = re.compile("|".join(alternatives))
    return delimiter.split(text)

def _count_pretokens(text: str, special_tokens: list[str]) -> Counter[Pretoken]:
    """Pre-tokenize text and return each distinct byte sequence's frequency."""
    counts: Counter[Pretoken] = Counter()

    # Split text according to special tokens
    for ordinary_text in _split_on_special_tokens(text, special_tokens):
        # For each split, use PATTERN to separate to pretokens
        for match in GPT2_PERTOKEN_PATTERN.finditer(ordinary_text):
            # Convert pretokens to utf-8 encoded bytes
            encoded = match.group().encode("utf-8")
            # Convert bytes into a tuple of byte (Pretoken Type)
            pretoken = tuple(bytes([byte_value]) for byte_value in encoded)
            # Add to counter
            counts[pretoken] += 1

    return counts

def _count_pairs(pretoken_counts: Counter[Pretoken]) -> Counter[Pair]:
    """Count adjacent pairs without allowing pairs across pre-token boundaries."""
    pair_counts: Counter[Pair] = Counter()

    # Iterate through all pretokens
    for pretoken, frequency in pretoken_counts.items():
        # For each pretoken, calculate it's token pairs
        for pair in zip(pretoken, pretoken[1:]):
            # And add to pair counts
            pair_counts[pair] += frequency
    
    return pair_counts

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


def train_bpe_tokenizer(
    input_path: str | PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    profile: bool = False,
    log_step: int = 100,
) -> tuple[dict[int, Token], list[Pair]]:
    """Train a byte-level BPE tokenizer and return its vocabulary and merges."""

    total_start = perf_counter()

    # Sanity Check
    initial_vocab_size = 256 + len(special_tokens)
    if vocab_size < initial_vocab_size:
        raise ValueError(
            f"vocab_size must be at least {initial_vocab_size} "
            "to contain all bytes and special tokens"
        )

    # Create initial vocab and merges
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    for token_id, special_token in enumerate(special_tokens, start=256):
        vocab[token_id] = special_token.encode("utf-8")

    merges: list[Pair] = []

    # Read, then pretokenize. Keeping separate timers reveals whether disk I/O
    # or regex/token conversion is responsible for the startup cost.
    read_start = perf_counter()
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    read_seconds = perf_counter() - read_start
    print(f"Read file {input_path} as corpus.")

    print("Staring pretokenization...")
    pretokenize_start = perf_counter()
    pretoken_counts = _count_pretokens(text, special_tokens)
    pretokenize_seconds = perf_counter() - pretokenize_start
    del text  # The corpus string is no longer needed after pre-tokenization.
    print("Finished pretokenization...")

    pair_count_seconds = 0.0
    pair_selection_seconds = 0.0
    merge_update_seconds = 0.0
    completed_merges = 0

    # Perform merge until target vocab size
    iter_count = 1
    log_start = perf_counter()
    for new_token_id in range(initial_vocab_size, vocab_size):
        # Calculate byte pair counts
        section_start = perf_counter()
        pair_counts = _count_pairs(pretoken_counts)
        pair_count_seconds += perf_counter() - section_start
        if not pair_counts:
            break

        # Get the best pair
        section_start = perf_counter()
        best_pair = max(pair_counts, key = lambda pair: (pair_counts[pair], pair))
        pair_selection_seconds += perf_counter() - section_start
        left, right = best_pair

        # Insert the best pair to vocab
        vocab[new_token_id] = left + right
        merges.append(best_pair)
        completed_merges += 1

        # Update Pre-tokenized cache by creating a new one
        section_start = perf_counter()
        updated_pretoken_counts: Counter[Pretoken] = Counter()
        for pretoken, frequency in pretoken_counts.items():
            updated_pretoken_counts[_merge_pair(pretoken, best_pair)] += frequency
        pretoken_counts = updated_pretoken_counts
        merge_update_seconds += perf_counter() - section_start

        # Print Logging message
        if iter_count % log_step == 0:
            average_iter_seconds = (perf_counter() - log_start) / log_step
            print(f"Iter {iter_count}: {average_iter_seconds} s/iter.")
            log_start = perf_counter()

        iter_count += 1

    if profile:
        total_seconds = perf_counter() - total_start
        sections = (
            ("file reading", read_seconds),
            ("pre-tokenization", pretokenize_seconds),
            ("pair counting", pair_count_seconds),
            ("pair selection", pair_selection_seconds),
            ("merge/cache update", merge_update_seconds),
        )

        print("\nBPE training profile")
        print(f"  completed merges: {completed_merges}")
        print(f"  distinct pre-tokens: {len(pretoken_counts):,}")
        for section_name, seconds in sections:
            percentage = 100 * seconds / total_seconds if total_seconds else 0.0
            print(f"  {section_name:<20} {seconds:9.3f}s  ({percentage:5.1f}%)")
        print(f"  {'total':<20} {total_seconds:9.3f}s")

    return vocab, merges
