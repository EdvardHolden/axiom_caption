import os
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

from process_problem import load_and_process_problem, get_problems_from_path
from evaluate import jaccard_score_np, coverage_score_np

import sys
sys.path.insert(0,'online')
from online.get_scores import get_solved_problem_name_time, get_solved_problem_name


# The direcotry to the proof axioms
PROOF_AXIOMS = 'generated_problems/analysis/output_original_positive_axioms/'

CONFIGS = {115498: 'generated_problems/analysis/output_original_ideal', # Upper bound
           #115554: 'generated_problems/analysis/output_original_clean/', # Raw merged problem
           115507: 'generated_problems/analysis/output_original_clean/', # Raw merged problem
           115555: 'generated_problems/analysis/output_original_sine_1_1/',
           115556: 'generated_problems/analysis/output_original_sine_3_0/',
           115591: 'generated_problems/analysis/output_original_caption/',
           #115633: 'generated_problems/analysis/output_original_caption_sine_1_1/', TODO need to recompute these problems
           115634: 'generated_problems/analysis/output_original_caption_sine_3_0/'}


# Make sure the result are not lost due to being truncated
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)


parser = argparse.ArgumentParser()

parser.add_argument("--print_df", default=False, action="store_true",
                    help="Print detailed stat information for each problem in each analysis case")
parser.add_argument("--debug", "-d", default=False, action="store_true",
                    help="Limit number of problems for efficient debugging")
parser.add_argument("--remove_unsolved", default=False, action="store_true",
                    help="Only looks at the solved problems in the general analysis")
args = parser.parse_args()

# Set debugging mode
if args.debug:
    LIMIT = 100
else:
    LIMIT = None

def load_generated_problem(problem_path):

    # Load generated problem
    prob = load_and_process_problem(problem_path, deepmath=False)
    if len(prob) < 1:
        # Report and skip empty problems
        print(f'Warning: empty problem {problem_path}')
        return []
    # Skip the conjecture to make a jaccard score of 1 possible
    if 'conjecture' in prob[0]:
        prob = prob[1:]

    return prob


def load_proof_axioms(problem_name):

    # Load proof axioms
    proof = load_and_process_problem(os.path.join(PROOF_AXIOMS, problem_name), deepmath=False)
    # Hack: check if quoting / $distinct number axioms is added. if so, remove ut
    if "a1, axiom, $distinct" in proof[-1]:
        proof = proof[:-1]

    return proof


def get_metrics(problem_paths):

    results = {}

    for problem_path in problem_paths:
        # Get the problem name
        name = Path(problem_path).stem

        proof = load_proof_axioms(name)

        prob = load_generated_problem(problem_path)

        results[name] = {"jaccard": jaccard_score_np([proof], [prob]),
                         "coverage": coverage_score_np([proof], [prob]),
                         "length": len(prob)}

    return results





# TODO utterly stupid variabel naming!
def analyse_problem_performance(exp_id, problem_dir, solved_set):

    # Load the problems from path
    problems = get_problems_from_path(problem_dir, limit=LIMIT, verbose=0)

    # Get solved status and time
    performance = get_performance_stats(exp_id)

    # Get length, jaccard and coverage score of each problem
    metrics = get_metrics(problems)

    # Merge dictionaries
    data = {}
    for prob_name in metrics:
        # Skip problem if solved_set is provided and it is not included in the set
        if solved_set is not None and prob_name not in solved_set:
            continue
        # Performance only contains solved entries
        if prob_name in performance:
            data[prob_name] = {**metrics[prob_name], **performance[prob_name]}
        else:
            data[prob_name] = metrics[prob_name]

    # Convert to pandas - problem names used as index
    df = pd.DataFrame.from_dict(data).T

    # Report the statistics
    print('## Solved partition')
    print(df.loc[df["solved_time"].notnull()].describe())
    print()
    print('## Unsolved partition')
    print(df.loc[df["solved_time"].isna()].describe())


def get_performance_stats(exp_id):

    # Get problem ID and runtime of the solved problems in the experiment(LTB)
    res = get_solved_problem_name_time(exp_id)

    # Convert to dict
    res = {name: {"solved_time": time} for name, time in res}

    return res


def run_initial_analysis(solved_set):
    print("# # # # # # # # # # # # #")
    print("# # Initial Analysis  # #")
    print()
    for exp_id, problem_dir in CONFIGS.items():
        print(f"## ## {problem_dir}")
        analyse_problem_performance(exp_id, problem_dir, solved_set)
        print()
        print()


def get_performance_coverage_data(CONFIGS, common_substring=''):

    result = {}

    # Want config_coverage, config_solved
    for exp_id, problem_dir in CONFIGS.items():

        conf = Path(problem_dir).stem.replace(common_substring, '')

        # Load the problems from path
        problems = get_problems_from_path(problem_dir, limit=LIMIT, verbose=0)
        print('# Number of problems found: ', conf, len(problems))

        # Initialise - if needed
        if len(result) == 0:
            result = {Path(p).stem: {} for p in problems}

        # Get solved status and time
        performance = get_performance_stats(exp_id)

        # Get coverage scores and add performance
        for problem_path in problems:
            name = Path(problem_path).stem
            prob = load_generated_problem(problem_path)
            proof = load_proof_axioms(name)
            result[name].update({f"{conf}_coverage": coverage_score_np([proof], [prob])})
            result[name].update({f"{conf}_jaccard": jaccard_score_np([proof], [prob])})
            result[name].update({f"{conf}_length": len(prob)})

            if name in performance:
                result[name].update({f"{conf}_solved": True})
                result[name].update({f"{conf}_time": performance[name]["solved_time"]})
            else:
                result[name].update({f"{conf}_solved": False})

    df = pd.DataFrame.from_dict(result).T

    return df


def print_avg_stats(df, base, other):

    print(f"Avg coverage base  :", "{0:.2f}".format(np.average(df[f"{base}_coverage"])))
    print(f"Avg jaccard  base  :", "{0:.2f}".format(np.average(df[f"{base}_jaccard"])))
    print(f"Avg length   base  :", "{0:.2f}".format(np.average(df[f"{base}_length"])))
    print(f"Avg coverage other :", "{0:.2f}".format(np.average(df[f"{other}_coverage"])))
    print(f"Avg jaccard  other :", "{0:.2f}".format(np.average(df[f"{other}_jaccard"])))
    print(f"Avg length   other :", "{0:.2f}".format(np.average(df[f"{other}_length"])))


def print_detailed_overview(df, base, other):

    #print(df[[f"{base}_solved", f"{base}_coverage", f"{base}_jaccard", f"{base}_length",
    #          f"{other}_solved", f"{other}_coverage", f"{other}_jaccard", f"{other}_length"]])
    print(df[[f"{base}_coverage", f"{base}_jaccard", f"{base}_length",
              f"{other}_coverage", f"{other}_jaccard", f"{other}_length"]])


def print_similar_problems_overview(df, base, other):

    print(df[[f"{base}_coverage", f"{base}_jaccard", f"{base}_length", f"{base}_time"]])


def compare_versions(df, base, other):

    print(f"### Comparing {base} to {other}")
    print(f"{base} solved: {len(df.loc[df[base+'_solved']])}")
    print(f"{other} solved: {len(df.loc[df[other+'_solved']])}")

    # Compute solved by base but not by other
    diff = sorted(set(df.loc[df[f'{base}_solved']].index).difference(set(df.loc[df[f'{other}_solved']].index)))
    df_comp = df.loc[diff]

    print(f"Solved by {base} and not other: ", len(diff))
    # Need to have at least one problem to compute a meaningful average
    if len(diff) > 0:
        # Print average of main stats between the different approaches
        print_avg_stats(df_comp, base, other)

        if args.print_df:
            # Print more detailed overview and separate of dissimilar stats
            # Check whether there are any problems with similar stats
            similar = df.loc[np.isclose(list(df[f"{base}_length"]), list(df[f"{other}_length"])) & np.isclose(list(df[f"{base}_coverage"]), list(df[f"{other}_coverage"])) & np.isclose(list(df[f"{base}_jaccard"]), list(df[f"{other}_jaccard"]))].index

            # Split based on whether the values are the same
            print("## Detailed Overview:")
            if len(similar) == 0:
                print_detailed_overview(df_comp, base, other)
            else:
                print("# Similar Problems: ")
                #print_detailed_overview(df_comp.loc[sorted(set(diff).intersection(set(similar)))], base, other) FIXME
                print_similar_problems_overview(df_comp.loc[sorted(set(diff).intersection(set(similar)))], base, other)
                print()
                print("# Other Problems: ")
                print_detailed_overview(df_comp.loc[sorted(set(diff).difference(set(similar)))], base, other)
    print()


def get_solved_set():
    solved = set()
    for exp, prob_dir in CONFIGS.items():
        if "ideal" not in prob_dir:
            print(prob_dir)
            res = get_solved_problem_name(exp)
            solved.update([r[0] for r in res])
        else:
            print("!! Skipping \"ideal\" when computing solved set")

    if LIMIT is not None:
        # Get problems from last path to truncate
        problems = get_problems_from_path(prob_dir, limit=LIMIT, verbose=0)
        solved = solved.intersection(set([Path(p).stem for p in problems]))

    print("## Number of problems solved in union: ", len(solved))
    return solved

def main():

    print('# ', args)

    # Get the problems solved by all configurations for reference
    solved_set = get_solved_set()
    if not args.remove_unsolved:
        # Do not keep the reference if not used
        solved_set = None
    print()

    # Run some basic analysis on solved/unsolved for metrics of each configuration
    run_initial_analysis(solved_set)

    ## Specific analysis
    print('\n\n')
    print("# # # # # # # # # # # # #")
    print("# # Specific Analysis # #")

    # Get the data needed
    df = get_performance_coverage_data(CONFIGS, common_substring='output_original_')

    # Check that the best configuration (e.g. cap_sine_3_0) is a superset of its combination
    #diff_problems = set(df.loc[df['caption_sine_3_0_solved']].index).difference(set(df.loc[df['sine_3_0_solved']].index).union(set(df.loc[df['caption_solved']].index)))
    diff_problems = (set(df.loc[df['sine_3_0_solved']].index).union(set(df.loc[df['caption_solved']].index))).difference(set(df.loc[df['caption_sine_3_0_solved']].index))
    print("Number of problems lost by combining sine and captioning:", len(diff_problems))
    if len(diff_problems) > 0 and args.print_df:
        print(df[['caption_sine_3_0_coverage', 'caption_sine_3_0_jaccard',
                  'sine_3_0_solved', 'sine_3_0_coverage', 'sine_3_0_jaccard',
                  'caption_solved', 'caption_coverage', 'caption_jaccard']].loc[diff_problems])

    # Compare the experiments
    print()
    compare_versions(df, 'sine_1_1', 'sine_3_0')
    print()
    compare_versions(df, 'sine_3_0', 'sine_1_1')
    print()
    #'''
    compare_versions(df, 'caption', 'sine_3_0')
    print()
    compare_versions(df, 'sine_3_0', 'caption')
    print()
    compare_versions(df, 'caption_sine_3_0', 'sine_3_0')
    print()
    compare_versions(df, 'caption_sine_3_0', 'caption')
    #'''



if __name__ == "__main__":
    main()
