import argparse
import re
import sys
import numpy as np
from pathlib import Path
from multiprocessing import Pool

import generate_problems
from generate_problems import get_problems_from_path, load_and_process_problem, sine_process
from evaluate import jaccard_score_np, coverage_score_np

parser = argparse.ArgumentParser()
parser.add_argument("--sine_sd", default=None)
parser.add_argument("--sine_st", default=None)

NO_WORKERS = 8

clause_file_re = b"file\('(\/|\w)*',(\w*)\)\)."
clause_name_problem_re = b"^\+ fof\((\w+), axiom"  # Only positive axioms

# Would be good with multiprocessing on the problems!
DATASET_PATH = "/home/eholden/gnn-entailment-caption/nndata/"


def get_sine_clause_names(prob):
    res = re.findall(clause_file_re, prob)
    res = [r[1] for r in res]
    return res


def get_original_clause_names(prob):
    res = re.findall(clause_name_problem_re, prob, flags=re.MULTILINE)
    return res


def sine_score_problem(prob_path, sine_st, sine_sd):
    prob = load_and_process_problem(prob_path)
    # Unclear whether this is the right call to do rn
    prob_processed = sine_process(prob, sine_st=sine_st, sine_sd=sine_sd)
    del prob

    # Extract the clause names from the sine processed problem
    sine_names = get_sine_clause_names(prob_processed)

    # Extract the clause names from the original problem
    with open(DATASET_PATH + Path(prob_path).stem, "rb") as f:  # Need to open the original dataset file
        prob_original = f.read()
    prob_names = get_original_clause_names(prob_original)
    del prob_original

    if len(prob_names) == 0:
        print(f"ERROR: No clause names for problem {prob_path}", file=sys.stderr)

    # Compute the scores - no avg as it is single entry anyways
    jaccard = jaccard_score_np([prob_names], [sine_names], avg=False)[0]
    coverage = coverage_score_np([prob_names], [sine_names], avg=False)[0]

    # Return result
    return Path(prob_path).stem, {"jaccard": jaccard, "coverage": coverage}


def sine_score_set(problem_paths, sine_st, sine_sd):

    # Iterate over each problem and compute their SiNe representation and their scores
    map_args = [(prob_path, sine_st, sine_sd) for prob_path in problem_paths]
    pool = Pool(NO_WORKERS)
    res = pool.starmap(sine_score_problem, map_args)
    pool.close()
    pool.join()

    # Set the socre dict
    scores = {prob: s for prob, s in res}

    avg_jaccard = np.average([v["jaccard"] for v in scores.values()])
    avg_coverage = np.average([v["coverage"] for v in scores.values()])

    return avg_jaccard, avg_coverage


def main():
    # Parse input arguments
    # args = parser.parse_args()

    # Get path to all problems
    problem_paths = get_problems_from_path(limit=None)

    st_values = [1, 2, 3]
    sd_values = [1, 2, 3, 4]

    results = [[-1] * len(st_values)] * len(sd_values)

    # Need to have an extra script calling this
    for nt, st in enumerate(st_values):
        for nd, sd in enumerate(sd_values):
            print(f"sd:{sd} st{st}")
            res = sine_score_set(problem_paths, st, sd)
            results[nd][nt] = res

    # Function for printing table line
    def print_line():
        print("-" * (4 + (14 * len(st_values))))

    # Pretty print our results
    print("(Jaccard , Coverage)")
    print("   |" + "|".join([f"    {st:>4}     " for st in st_values]) + "|")
    print_line()
    for sd, res in zip(sd_values, results):
        print(f" {sd} |" + "|".join([f" {r[0]:>5.2f} {r[1]:>5.2f} " for r in res]) + "|")
        print_line()


if __name__ == "__main__":
    main()
    # Extension: segment on train test?
