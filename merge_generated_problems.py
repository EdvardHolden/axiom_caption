'''
Script for merging two versions of generated problems. This is convenient
for reducing the computation time when experimenting with caption+SiNE combinations.
The script assumes that axioms are quoted and non clausified. The resulting problems
are clausified.
# Input: dir1 dir2
# Output: Merged version of dir dir2
'''
import argparse
import os
import sys
import re
from pathlib import Path
from multiprocessing import Pool

from process_problem import get_problems_from_path, load_and_process_problem, save_problem
from clausifier import clausify

parser = argparse.ArgumentParser()
parser.add_argument('dir1', help='Dir to merge')
parser.add_argument('dir2', help='Dir to merge')
parser.add_argument('--dest_dir', default=None, help='Distination dir. Inferred if not set')
parser.add_argument('--workers', type=int, default=max(os.cpu_count() - 2, 1))


def infer_dest_dir(dir1, dir2):
    # This works ok

    # Get the directory path of the first dir - need to choose one
    base_dir = str(Path(dir1).parent)

    f1 = Path(dir1).name.split('_')
    f2 = Path(dir2).name.split('_')

    res = []
    f1_i = 0
    f2_i = 0
    while f1_i < len(f1) or f2_i < len(f2):
        if not f1_i == len(f1) and not f2_i == len(f2) and f1[f1_i] == f2[f2_i]:
            res += [f1[f1_i]]
            f1_i += 1
            f2_i += 1
        else:
            if f1_i < len(f1):
                res += [f1[f1_i]]
                f1_i += 1
            elif f2_i < len(f2):
                res += [f2[f2_i]]
                f2_i += 1

    # Replace original with clausified
    res = "_".join(res).replace("original", "clausified")

    dest_dir = os.path.join(base_dir, res)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    return dest_dir

def merge_problems(dir1, dir2, prob_name, dest_dir):

    # Load problems and merge clauses
    prob1 = load_and_process_problem(os.path.join(dir1, prob_name), deepmath=False)
    prob2 = load_and_process_problem(os.path.join(dir2, prob_name), deepmath=False)
    prob = set(prob1).union(set(prob2))

    # Re-compute distinct TODO make function

    # Check for $distinct axioms and merge them - might be in both problems?
    distinct_ax = [ax for ax in prob if "fof(a1, axiom, $distinct(" in ax]
    prob = [ax for ax in prob if "fof(a1, axiom, $distinct(" not in ax] # Remove distincts

    # Extract all the axioms
    digits = set()
    if len(distinct_ax) > 0:
        for dist in distinct_ax:
            d = re.findall(r"\'\d+\'", dist)
            digits = digits.union(d)

        # All digits are extracted - get new axiom
        distinct_number_axiom = "fof(a1, axiom, $distinct({0})).".format(", ".join(sorted(digits)))

        # Add new axiom to the problem
        prob += [distinct_number_axiom]


    # Clausify the merged problem
    prob = clausify(prob, skolem_prefix=None, sine_st=None, sine_sd=None, prob_name=prob_name)

    # Save to folder
    save_problem(dest_dir, prob_name, prob)


def main():
    print("Warning: We assume that the digits in the problems are quoted")

    # Get parser
    args = parser.parse_args()

    # Check that both directories exists
    if not os.path.exists(args.dir1):
        print("Dir1 does not exist!")
        sys.exit(1)
    if not os.path.exists(args.dir2):
        print("Dir2 does not exist!")
        sys.exit(1)

    # Get file paths from both
    probs1 = get_problems_from_path(args.dir1, verbose=0)
    probs2 = get_problems_from_path(args.dir2, verbose=0)

    # Sanity check
    assert len(probs1) == len(probs2) and len(probs1) > 0

    # Create destination directory name
    dest_dir = args.dest_dir
    if args.dest_dir is None:
        dest_dir = infer_dest_dir(args.dir1, args.dir2)
    print("Merged problem dir: ", dest_dir)

    # Only need the one filename
    #for prob in probs1
    prob_names = [Path(p).stem for p in probs1]

    star_args = [(args.dir1, args.dir2, prob_name, dest_dir) for prob_name in prob_names]
    pool = Pool(args.workers)
    pool.starmap(merge_problems, star_args)
    pool.close()
    pool.join()

    print("Finished")


if __name__ == "__main__":
    main()
