import pickle
from pathlib import Path

from cs336_basics.bpe_tokenizer_vanilla import train_bpe_tokenizer as vanilla_bpe_tokenizer
from cs336_basics.bpe_tokenizer_optimized import train_bpe_tokenizer as optimized_bpe_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA = PROJECT_ROOT / "data" / "owt_train.txt"
OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts"
# VOCAB_OUTPUT = OUTPUT_DIRECTORY / "openwebtext_vocab.pkl"
# MERGES_OUTPUT = OUTPUT_DIRECTORY / "openwebtext_merges.pkl"
VOCAB_OUTPUT = OUTPUT_DIRECTORY / "owt_vocab.pkl"
MERGES_OUTPUT = OUTPUT_DIRECTORY / "owt_merges.pkl"


def main() -> None:
    vocab, merges = optimized_bpe_tokenizer(
        input_path=TRAINING_DATA,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        num_processes=16,
        profile=True,
        log_step=1000
    )

    longest_token = max(vocab.values(), key=len)
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    with open(VOCAB_OUTPUT, "wb") as vocab_file:
        pickle.dump(vocab, vocab_file)
    with open(MERGES_OUTPUT, "wb") as merges_file:
        pickle.dump(merges, merges_file)

    print(f"Trained vocabulary entries: {len(vocab):,}")
    print(f"Learned merges: {len(merges):,}")
    print(f"Longest token ({len(longest_token)} bytes): {longest_token!r}")
    print(f"Saved vocabulary to: {VOCAB_OUTPUT}")
    print(f"Saved merges to: {MERGES_OUTPUT}")


if __name__ == "__main__":
    main()
