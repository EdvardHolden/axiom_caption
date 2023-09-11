import sys

sys.path.insert(0, "online")
from online.get_scores import get_solved_problem_name

# Set baseline as iProver basic
BASELINE = 115507

TIMEOUT = 10


# List of experiments
EXPERIMENTS = {
    "115507": "Original iProver",
    "115508": "SInE 1 1",
    "117251": "Caption",
    "117090": "Caption (Remapping)",
    "115512": "Caption + SInE 1 1",
    "115509": "SInE 3 0",
    "115510": "Caption + SInE 3 0",
    "117091": "Caption (Remapping) + SInE 3 0",
    "117026": "Binary Graph",
    "117028": "Conjecture Transformer",
    "117029": "Conjecture RNN",
}


def main():
    # Get problems sovled in the baseline
    base_solved = set(get_solved_problem_name(BASELINE, upper_time_bound=TIMEOUT))

    # Print the header
    print(f"{'Method':30} &  {'Solved':5}  &  {'Difference':5}  \\\\")

    # Get the data for each experiment and print
    for exp_id, tag in EXPERIMENTS.items():
        solved = get_solved_problem_name(exp_id, upper_time_bound=TIMEOUT)
        solved_diff = set(solved) - set(base_solved)
        print(f"{tag:30} &  {len(solved):5}  &  {len(solved_diff):5}  \\\\")


if __name__ == "__main__":
    main()
