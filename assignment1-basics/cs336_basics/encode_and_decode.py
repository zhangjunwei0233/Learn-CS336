import pickle
import regex as re

# Regex Pattern for Pretokenization
GPT2_PERTOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Reusing these objects avoids constructing a new one-byte ``bytes`` object
# every time a distinct pre-token is converted to its initial token sequence.
BYTE_TOKENS = tuple(bytes([value]) for value in range(256))

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ) -> None:
        """
        Construct a tokenizer from a given vocabulary, list of merges,
        and (optionally) a list of special tokens.
        """
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}
        self.vocab_to_id = {v: k for k, v in vocab.items()}

        # Append unseen special_tokens to vocab, as required in the handout
        if special_tokens:
            next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.vocab_to_id:
                    self.vocab[next_id] = token_bytes
                    self.vocab_to_id[token_bytes] = next_id
                    next_id += 1

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None =None
    ):
        """
        Construct a tokenizer from vocab and merges file path.
        """
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)

        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        if not isinstance(vocab, dict):
            raise TypeError(f"vocab must be dict, got {type(vocab)}")
        if vocab:
            sample_key = next(iter(vocab))
            sample_value = vocab[sample_key]
            if not isinstance(sample_key, int):
                raise TypeError(f"vocab key must be int, got {type(sample_key)}")
            if not isinstance(sample_value, bytes):
                raise TypeError(f"vocab value must be bytes, got {type(sample_value)}")

        if not isinstance(merges, list):
            raise TypeError(f"merges must be list, got {type(merges)}")
        if merges:
            sample_merge = merges[0]
            if not isinstance(sample_merge, tuple) or len(sample_merge) != 2:
                raise TypeError(f"merges elements must be tuple of length 2, got {type(sample_merge)}")
            if not isinstance(sample_merge[0], bytes) or not isinstance(sample_merge[1], bytes):
                raise TypeError(f"merges tuple elements must be bytes, got {type(sample_merge[0])} and {type(sample_merge[1])}")

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)


    @staticmethod
    def _split_on_special_tokens(
        text: str,
        special_tokens: list[str] | None
    ) -> list[tuple[str, str]]:
        """Remove special tokens while preserving them as hard boundries."""
        if not special_tokens:
            return [("text", text)]
    
        # Reorder longer special tokens in the front to resolve overlap
        alternatives = sorted((re.escape(token) for token in special_tokens), key=len, reverse=True)

        # Split the text by constructing delimiters
        delimiter = re.compile("|".join(alternatives))

        chunks = []
        last_end = 0
        for match in delimiter.finditer(text):
            if match.start() > last_end:
                chunks.append(("text", text[last_end:match.start()]))
            chunks.append(("special token", match.group()))
            last_end = match.end()
        if last_end < len(text):
            chunks.append(("text", text[last_end:]))

        return chunks


    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        encoded = []

        chunks = self._split_on_special_tokens(text, self.special_tokens)

        for kind, chunk in chunks:
            if not chunk:
                continue

            # Special token type
            if kind == "special token":
                encoded.append(self.vocab_to_id[chunk.encode("utf-8")])
                continue

            # text type
            assert kind == "text", f"chunk should only be 'text' or 'special token', got {kind}"
            for match in GPT2_PERTOKEN_PATTERN.finditer(chunk):
                # Convert to byte tuple
                pretoken_bytes = match.group().encode("utf-8")
                pretoken = tuple(BYTE_TOKENS[byte] for byte in pretoken_bytes)

                # Merge the bytes according to self.merges
                while True:
                    # Try to find a smallest merge
                    index = 0
                    prior = float('inf')
                    for i in range(len(pretoken) - 1):
                        p = (pretoken[i], pretoken[i + 1])
                        pprior = self.merges_rank[p] if p in self.merges_rank else float('inf')
                        if pprior < prior:
                            prior = pprior
                            index = i

                    # No more merges, finished
                    if prior == float('inf'):
                        break

                    # Apply the merge
                    temp = pretoken
                    pretoken = temp[:index] + (temp[index] + temp[index + 1],) + temp[index + 2:]

                # Store to encoded
                encoded.extend([self.vocab_to_id[token] for token in pretoken])

        return encoded

    def encode_iterable(self, iterable):
        """Encode an iterable of texts, yielding token IDs."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        all_bytes = b''.join(self.vocab[id] for id in ids)
        return all_bytes.decode("utf-8", errors="replace")
