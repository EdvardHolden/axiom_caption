import os
from dataset import load_ids, load_clean_descriptions, load_tokenizer

proof_data_path = "data/raw/deepmath.pkl"

BASE_DIR = "data/deepmath/"


def main():
    total_axioms_count = 0
    unseen_axioms_count = 0

    unseen_in_problem = 0

    total_axioms = set()
    unseen_axioms = set()

    # Load the tokenizer
    tokenizer_path = os.path.join(BASE_DIR, "tokenizer_axioms_train_6000.json")
    tokenizer = load_tokenizer(tokenizer_path)
    print("Number of axioms in tokenizer:", len(tokenizer.word_index))

    # Load test problems and corresponding proof data
    ids = load_ids("data/deepmath/test.txt")
    print("Number of test problems: ", len(ids))
    data = load_clean_descriptions(proof_data_path, ids, order=None)

    for sequence in data.values():
        pos_axioms = sequence.split("\n")[1:-1]
        solvable = True

        for ax in pos_axioms:
            total_axioms_count += 1
            total_axioms.add(ax)

            if ax not in tokenizer.word_index:
                unseen_axioms_count += 1
                unseen_axioms.add(ax)
                solvable = False

        if not solvable:
            unseen_in_problem += 1

    print("Total", total_axioms_count)
    print("Unseen", unseen_axioms_count)
    print()
    print("Total axioms set", len(total_axioms))
    print("Unseen axioms set", len(unseen_axioms))
    print("Unseen in problem", unseen_in_problem)


if __name__ == "__main__":
    main()
