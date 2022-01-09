"""
This program computes train-test-val sets for a given axiom dictionary downloaded
from the CASC results. The IDs of each set are saved in a text file in an appropriate
directory in 'data/'.
"""
import argparse
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_file", default="data/raw/vampire_jjt.pkl", help="Data file used to compute splits"
)

parser.add_argument(
    "--min_axioms", default=1, type=int, help="Min axiom count for being included in the split"
)
parser.add_argument(
    "--max_axioms", default=100, type=int, help="Max axiom count for being included in the split"
)

parser.add_argument(
    "--fof_only", default=False, action="store_true", help="Uses only FOF proofs if specified"
)
parser.add_argument("--random_state", default=7, type=int, help="Set random state for the splitting")

parser.add_argument("--train_size", default=0.8)
parser.add_argument("--val_size", default=0.1)


def save_set(path, ids):
    with open(path, "w") as f:
        f.write("\n".join(ids))


def main():

    args = parser.parse_args()

    # Load pickle
    with open(args.data_file, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} proofs from {args.data_file}")

    # Remove min proofs
    for prob in list(data.keys()):
        if len(data[prob]["axioms"]) < args.min_axioms or len(data[prob]["axioms"]) > args.max_axioms:
            del data[prob]
    print(
        f"{len(data)} problems remaining after filtering on axiom lengths min:{args.min_axioms} max:{args.max_axioms}"
    )

    # Get the problem ids - optionally only restrict to FOF
    if args.fof_only:
        print("Restricting problems to FOF")
        ids = [d["version"] for prob, d in data.items() if d["version"] and "+" in d["version"]]
    elif "version" in data[list(data.keys())[0]]:
        ids = [d["version"] for prob, d in data.items() if d["version"]]
    else:
        ids = sorted(data.keys())

    ids = sorted(ids)  # Sort the ids for good measure
    print("Number of problem ids: ", len(ids))

    # Compute splits
    train_id, test_id = train_test_split(
        ids, shuffle=True, train_size=args.train_size, random_state=args.random_state
    )
    train_id, val_id = train_test_split(
        train_id, shuffle=True, test_size=args.val_size, random_state=args.random_state
    )

    # Report stats
    print(f"Resulting datasets train:{len(train_id)}, test:{len(test_id)}, val:{len(val_id)}")

    # Create directory
    dir_path = os.path.join("data", Path(args.data_file).stem)
    if args.fof_only:
        dir_path += "_fof"
    dir_path += "/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print(f"Saving sets to {dir_path}")

    # Save sets to path
    save_set(dir_path + "train.txt", train_id)
    save_set(dir_path + "test.txt", test_id)
    save_set(dir_path + "val.txt", val_id)


if __name__ == "__main__":
    main()
