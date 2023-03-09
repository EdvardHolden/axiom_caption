import os
from pathlib import Path

from dataset import load_tokenizer
from enum_types import OutputFormat
from clausifier import clausify
from generate_problems import extract_rare_axioms
from process_problem import get_problems_from_path, load_and_process_problem, order_formulae, save_problem

TOKENISER_PATH = "data/deepmath/tokenizer_axioms_train_6000.json"
# TODO change to base and filter dirs!!

# base_dir = "generated_problems/analysis/output_original_caption_greedy_no_samples_4_length_22_conjecture_position_first"
base_dir = "generated_problems/analysis/output_original_caption_sine_3_0"

# filter_dir = "generated_problems/analysis/output_original_caption_sine_3_0"
filter_dir = "generated_problems/analysis/output_original_sine_3_0_conjecture_position_first"
result_dir = "generated_problems/analysis/output_{0}_caption_sine_3_0_sine_3_0_rare_intersection"

"""
filter_dir = "generated_problems/analysis/output_original_sine_3_0_conjecture_position_first"
result_dir = "generated_problems/analysis/output_{0}_caption_sine_3_0_rare_intersection"
# """

"""
filter_dir = "generated_problems/analysis/output_original_sine_1_1_conjecture_position_first"
result_dir = "generated_problems/analysis/output_{0}_caption_sine_1_1_rare_intersection"
"""


post_process = True  # Used for debugging

# TODO HACKK
output_format = OutputFormat.CLAUSIFIED
# output_format = OutputFormat.ORIGINAL

import logging
import sys

# logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger()

result_dir = result_dir.format(output_format)
log.info(f"Result dir: {result_dir}")


def main():
    # Get hold of the problems
    # Get path to all problems
    problem_paths = get_problems_from_path(base_dir)
    if len(problem_paths) == 0:
        raise ValueError(f'Error please check problem dir path, found no problem at "{base_dir}"')

    # Extract the names
    problem_names = [Path(prob).name for prob in problem_paths]
    # problem_names = problem_names[:10]

    # Get the tokeniser
    tokeniser = load_tokenizer(TOKENISER_PATH)

    # For every problem
    for prob_name in problem_names:
        log.info(f"# {prob_name}")

        # Get the rare axioms selected by sine
        filter_prob = load_and_process_problem(os.path.join(filter_dir, prob_name))
        log.debug(f"Filter problem size {len(filter_prob)}")
        rare_axioms_filter = extract_rare_axioms(tokeniser, filter_prob)
        log.debug(f"Rare filter axioms {len(rare_axioms_filter)}")

        # Load the caption problem
        base_prob = load_and_process_problem(os.path.join(base_dir, prob_name))
        log.debug(f"Caption problem size {len(base_prob)}")
        conjecture = set([base_prob[0]])
        base_axioms = set(base_prob[1:])

        # Split the caption problem into rare and non-rare
        rare_axioms_base = extract_rare_axioms(tokeniser, base_axioms)
        log.debug(f"Rare caption axioms {len(rare_axioms_base)}")

        base_non_rare_axioms = base_axioms.difference(rare_axioms_base)
        log.debug(f"Base non rare axiom: {len(base_non_rare_axioms)}")

        # Intersect the rare
        rare_axioms_intersection = rare_axioms_base.intersection(rare_axioms_filter)
        log.debug(f"Rare axiom intersection {len(rare_axioms_intersection)}")

        # Re-join the axioms into the problem
        new_problem = conjecture.union(base_non_rare_axioms).union(rare_axioms_intersection)
        log.debug(f"New problem size {len(new_problem)}")

        # Order
        new_problem = order_formulae(new_problem, "first")
        log.debug(f"New problem size: {len(new_problem)}")

        if len(rare_axioms_base) != len(rare_axioms_filter):
            log.warning(f"Difference in number of axioms for {prob_name}")

        # Write to file
        # Only clausify the problem if set
        if post_process:
            if output_format is OutputFormat.CLAUSIFIED:
                # Clausify the problem - this is only done once and in the final step, hence no application of SInE
                prob = clausify(new_problem, skolem_prefix=None, sine_st=None, sine_sd=None)
            elif output_format is OutputFormat.ORIGINAL:
                prob = "\n".join(new_problem).encode()
            else:
                raise ValueError()

            # Save to folder
            save_problem(result_dir, prob_name, prob)

    log.info("# Finished")
    print("# Finished")


if __name__ == "__main__":
    main()
