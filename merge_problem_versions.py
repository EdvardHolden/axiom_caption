'''
Script for mergeing the clauses in nndata with deepmath.
This is to ensure the axioms are the same for as this occassinally varies lsightly in between versions.
'''
from pathlib import Path
from tqdm import tqdm
import os

from process_problem import save_problem, get_problems_from_path, load_and_process_problem

DEEPMATH_PATH = '/shareddata/home/holden/gnn-entailment-caption/nndata/'
MIZAR_PATH = '/shareddata/home/holden/mizar40_nndata/'

DEST = '/shareddata/home/holden/gnn-entailment-caption/merged_problems/'


def main():

    # Get problem names
    problem_paths = get_problems_from_path(DEEPMATH_PATH, limit=None)
    problem_names = [Path(p).stem for p in problem_paths]

    for prob in tqdm(problem_names):
        res1 = load_and_process_problem(os.path.join(MIZAR_PATH, prob), deepmath=False)
        res2 = load_and_process_problem(os.path.join(DEEPMATH_PATH, prob), deepmath=True)
        res = set(res1).union(set(res2))
        res = sorted(set(res))
        # Remove any commented lines
        res = [r for r in res if r[0] != '%']

        save_problem(DEST, prob, '\n'.join(res).encode())


if __name__ == "__main__":
    main()
