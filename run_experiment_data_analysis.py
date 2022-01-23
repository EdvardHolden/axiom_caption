import numpy as np
import os
import config
from dataset import get_tokenizer, load_ids, load_clean_descriptions

# TODO make argparser?
BASE_DIR = "data/deepmath/"
proof_data_path = "data/raw/deepmath.pkl"


def compute_set_stats(tokenizer, proof_data_path, dataset_dir, id_set):

    # Get hold of the correct proof data
    ids = load_ids(os.path.join(dataset_dir, id_set))
    data = load_clean_descriptions(proof_data_path, ids, order=None)
    # Only works for deepmath data right now
    proofs = list(data.values())
    assert len(proofs) == len(ids)

    # WARNING: This includes the start and end tokens!
    print(f"Number of proofs: {len(proofs)}")
    print(f"Max proof length: {max([len(p.split(config.TOKEN_DELIMITER)) for p in proofs])}")
    print(f"Average proof length: {np.mean([len(p.split(config.TOKEN_DELIMITER)) for p in proofs]):.2f}")
    print(f"Median proof length: {np.median([len(p.split(config.TOKEN_DELIMITER)) for p in proofs]):.2f}")

    # Tokenise the proofs
    seq = tokenizer.texts_to_sequences(proofs)

    oov = tokenizer.texts_to_sequences([config.TOKEN_OOV])[0][0]
    print(f"Ratio of problems with OOV token: {sum([1 for s in seq if int(oov) in s])/len(seq):.2f}")

    # If the set without start and end only consists of one element and that element is oov, it is rendered completely useless.
    problem_loss = sum(1 for s in seq if len(set(s[1:-1])) == 1 and s[2] == oov)
    print(f"Ratio of proofs reduced completely to oov: {problem_loss / len(seq):.2f}")


def main():

    # Load tokenizer
    tokenizer_path = os.path.join(BASE_DIR, "tokenizer.json")
    tokenizer, vocab_size = get_tokenizer(tokenizer_path)

    # Get path

    # For each val, test, train
    # Compute stats
    print("# Train")
    compute_set_stats(tokenizer, proof_data_path, BASE_DIR, "train.txt")
    print("")
    print("# Validation")
    compute_set_stats(tokenizer, proof_data_path, BASE_DIR, "val.txt")
    print("")
    print("# Test")
    compute_set_stats(tokenizer, proof_data_path, BASE_DIR, "test.txt")


if __name__ == "__main__":
    main()
