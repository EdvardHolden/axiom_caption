import os
from pathlib import Path
import pandas as pd

from process_problem import load_and_process_problem, get_problems_from_path
from evaluate import jaccard_score_np, coverage_score_np

import sys
sys.path.insert(0,'online')
from online.get_scores import get_problem_stats, get_solved_problem_name_time


# The direcotry to the proof axioms
PROOF_AXIOMS = 'generated_problems/analysis/output_original_positive_axioms/'

CONFIGS = {115498: 'generated_problems/analysis/output_original_ideal', # Upper bound
           115554: 'generated_problems/analysis/output_original_clean/', # Raw merged problem
           115555: 'generated_problems/analysis/output_original_sine_1_1/',
           115556: 'generated_problems/analysis/output_original_sine_3_0/'}

LIMIT = 100 # TODO


def get_metrics(problem_paths):

    results = {}

    for problem_path in problem_paths:
        # Get the problem name
        name = Path(problem_path).stem

        # Load proof axioms
        proof = load_and_process_problem(os.path.join(PROOF_AXIOMS, name), deepmath=False)
        # Hack: check if quoting / $distinct number axioms is added. if so, remove ut
        if "a1, axiom, $distinct" in proof[-1]:
            proof = proof[:-1]

        # Load generated problem
        prob = load_and_process_problem(problem_path, deepmath=False)
        if len(prob) < 1:
            # Report and skip empty problems
            print(f'Warning: empty problem {prob}')
            continue
        # Skip the conjecture to make a jaccard score of 1 possible
        if 'conjecture' in prob[0]:
            prob = prob[1:]

        results[name] = {"jaccard": jaccard_score_np([proof], [prob]),
                         "coverage": coverage_score_np([proof], [prob]),
                         "length": len(prob)}

    return results





def run_initial_analysis(exp_id, problem_dir):

    # Load the problems from path
    problems = get_problems_from_path(problem_dir, limit=LIMIT)

    # Get solved status and time
    performance = get_performance_stats(exp_id)

    # Get length, jaccard and coverage score of each problem
    metrics = get_metrics(problems)

    # Merge dictionaries
    data = {}
    for prob_name in metrics:
        # Perfromance only contains solved entries
        if prob_name in performance:
            data[prob_name] = {**metrics[prob_name], **performance[prob_name]}

    # Convert to pandas - problem names used as index
    df = pd.DataFrame.from_dict(data).T

    # Report the statistics
    print(df.describe())


def get_performance_stats(exp_id):

    # Get problem ID and runtime of the solved problems in the experiment(LTB)
    res = get_solved_problem_name_time(exp_id)

    # Convert to dict
    res = {name: {"solved_time": time} for name, time in res}

    return res





def main():
    print("## Initial Analysis ##")
    print()
    for exp_id, problem_dir in CONFIGS.items():
        print(f"## {problem_dir}")
        run_initial_analysis(exp_id, problem_dir)
        print()


if __name__ == "__main__":
    main()
