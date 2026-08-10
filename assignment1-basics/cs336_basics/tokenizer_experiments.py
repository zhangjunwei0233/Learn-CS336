from pathlib import Path
from time import perf_counter

from cs336_basics.encode_and_decode import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TINY_STORIES_VALID_PATH = PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
OWT_VALID_PATH = PROJECT_ROOT / "data" / "owt_valid.txt"
TOKENIZER_DIRECTORY = PROJECT_ROOT / "artifacts"
TINY_STORIES_VOCAB = TOKENIZER_DIRECTORY / "tinystories_vocab.pkl"
TINY_STORIES_MERGES = TOKENIZER_DIRECTORY / "tinystories_merges.pkl"
OWT_VOCAB = TOKENIZER_DIRECTORY / "owt_vocab.pkl"
OWT_MERGES = TOKENIZER_DIRECTORY / "owt_merges.pkl"

def _read_documents(filepath: Path, count: int) -> list[str]:
    results = []
    buffer = ""

    if count == -1:
        with open(filepath, "r", encoding="utf-8") as f:
            return [f.read()]

    with open(filepath, "r", encoding="utf-8") as f:
        while len(results) < count:
            chunk = f.read(4096)
            if not chunk:
                break

            buffer += chunk

            while "<|endoftext|>" in buffer and len(results) < count:
                idx = buffer.index("<|endoftext|>")
                results.append(buffer[:idx])
                buffer = buffer[idx + len("<|endoftext|>"):]

    return results

def main() -> None:
    ts_tokenizer = Tokenizer.from_files(
        vocab_filepath=str(TINY_STORIES_VOCAB),
        merges_filepath=str(TINY_STORIES_MERGES),
        special_tokens=["<|endoftext|>"]
    )

    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath=str(OWT_VOCAB),
        merges_filepath=str(OWT_MERGES),
        special_tokens=["<|endoftext|>"]
    )

    print("================== Question (a) ==================")

    """Sample 10 documents from TinyStories and OpenWebText.
    Using your previously-trained TinyStories and OpenWebText tokenizers
    (10K and 32K vocabulary size, respectively),
    encode these sampled documents into integer IDs.
    What is each tokenizer's compression ratio (bytes/token)?
    """

    # Sample five documents from each datasets
    documents = _read_documents(TINY_STORIES_VALID_PATH, 5) + _read_documents(OWT_VALID_PATH, 5)
    print("Chose 5 documents from TinyStories, 5 from OWT.")

    # Call encode
    total_document_len = 0
    ts_tokenizer_len = 0
    owt_tokenizer_len = 0
    for document in documents:
        total_document_len += len(document)
        ts_tokenizer_len += len(ts_tokenizer.encode(document))
        owt_tokenizer_len += len(owt_tokenizer.encode(document))

    ts_compression_ratio = total_document_len / ts_tokenizer_len
    owt_compression_ratio = total_document_len / owt_tokenizer_len
    print(f"TS Tokenizer Compression Ratio: {ts_compression_ratio}")
    print(f"OWT Tokenizer Compression Ratio: {owt_compression_ratio}")

    print("================== Question (b) ==================")

    """
    What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer?
    Compare the compression ratio and/or qualitatively describe what happens
    """

    ts_documents = _read_documents(TINY_STORIES_VALID_PATH, 10)
    owt_documents = _read_documents(OWT_VALID_PATH, 10)

    print(f"On TS Dataset:")
    total_document_len = 0
    ts_tokenizer_len = 0
    owt_tokenizer_len = 0
    for document in ts_documents:
        total_document_len += len(document)
        ts_tokenizer_len += len(ts_tokenizer.encode(document))
        owt_tokenizer_len += len(owt_tokenizer.encode(document))
    ts_compression_ratio = total_document_len / ts_tokenizer_len
    owt_compression_ratio = total_document_len / owt_tokenizer_len
    print(f"    TS Tokenizer Compression Ratio: {ts_compression_ratio}")
    print(f"    OWT Tokenizer Compression Ratio: {owt_compression_ratio}")

    print(f"On OWT Dataset:")
    total_document_len = 0
    ts_tokenizer_len = 0
    owt_tokenizer_len = 0
    for document in owt_documents:
        total_document_len += len(document)
        ts_tokenizer_len += len(ts_tokenizer.encode(document))
        owt_tokenizer_len += len(owt_tokenizer.encode(document))
    ts_compression_ratio = total_document_len / ts_tokenizer_len
    owt_compression_ratio = total_document_len / owt_tokenizer_len
    print(f"    TS Tokenizer Compression Ratio: {ts_compression_ratio}")
    print(f"    OWT Tokenizer Compression Ratio: {owt_compression_ratio}")

    print("================== Question (c) ==================")

    """Estimate the throughput of your tokenizer (e.g., in bytes/second)"""

    ts_documents = _read_documents(TINY_STORIES_VALID_PATH, -1)[0]
    owt_documents = _read_documents(OWT_VALID_PATH, -1)[0]

    print(f"On TS Dataset:")
    ts_document_len = len(ts_documents)
    ts_start = perf_counter()
    ts_tokenized_len = len(ts_tokenizer.encode(ts_documents))
    ts_seconds = perf_counter() - ts_start
    print(f"    Encoded {ts_document_len} bytes, used {ts_seconds} seconds.")
    print(f"    That's {ts_document_len/ts_seconds} bps.")
    print(f"    Total Compression Rate: {ts_document_len/ts_tokenized_len}")

    print(f"On OWT Dataset:")
    owt_document_len = len(owt_documents)
    owt_start = perf_counter()
    owt_tokenized_len = len(owt_tokenizer.encode(owt_documents))
    owt_seconds = perf_counter() - owt_start
    print(f"    Encoded {owt_document_len} bytes, used {owt_seconds} seconds.")
    print(f"    That's {owt_document_len/owt_seconds} bps.")
    print(f"    Total Compression Rate: {owt_document_len/owt_tokenized_len}")


if __name__ == "__main__":
    main()