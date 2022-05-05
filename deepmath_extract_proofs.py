import pickle
import glob
from pathlib import Path
import os

DEST = "data/raw/deepmath.pkl"
SOURCE = os.path.join(Path.home(), "deepmath/nndata")


def get_proof_axioms(problem_path):

    # Open the file
    with open(problem_path, "r") as f:
        data = f.readlines()

    # Check if used in the proof (+) and extract
    axioms = []
    for d in data:
        if "+" == d[0]:
            axioms += [d[2:].strip()]

    return {"axioms": axioms}


def main():

    # Get hold of the problem paths
    problems = glob.glob(SOURCE + "/*")
    print(f"Number of problems: {len(problems)}")

    # Extract the axioms of each problem
    result = {}
    for prob in problems:
        result[Path(prob).stem] = get_proof_axioms(prob)

    # Save proof dict
    with open(DEST, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
