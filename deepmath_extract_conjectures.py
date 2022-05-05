import pickle
import glob
from pathlib import Path
import os

DEST = "data/raw/deepmath_conjectures.pkl"
SOURCE = os.path.join(Path.home(), "deepmath/nndata")


def get_proof_conjecture(problem_path):

    # Open the file
    with open(problem_path, "r") as f:
        # Conjecture is always the first line in the file
        conj = f.readlines()[0]

    # Check if used in the proof (+) and extract
    assert conj[0] == "C"
    # Remove C tag and newline
    conj = conj[2:].strip()
    # Only keep the actual formula to shorten the string
    conj = conj.split("axiom,")[-1]
    # Remove ').'
    conj = conj.replace(").", "")

    return conj


def main():

    # Get hold of the problem paths
    problems = glob.glob(SOURCE + "/*")
    print(f"Number of problems: {len(problems)}")

    # Extract the axioms of each problem
    result = {}
    for prob in problems:
        result[Path(prob).stem] = get_proof_conjecture(prob)

    # Save proof dict
    with open(DEST, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
