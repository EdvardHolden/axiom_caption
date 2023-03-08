import os
from pathlib import Path

from dataset import load_tokenizer
from enum_types import OutputFormat
from clausifier import clausify
from generate_problems import extract_rare_axioms
from process_problem import get_problems_from_path, load_and_process_problem, order_formulae, save_problem

caption_dir = "generated_problems/analysis/output_original_caption_greedy_no_samples_4_length_22_conjecture_position_first"
tokeniser_path = "data/deepmath/tokenizer_axioms_train_6000.json"

"""
sine_dir = "generated_problems/analysis/output_original_caption_sine_3_0"
result_dir = "generated_problems/analysis/output_{0}_caption_sine_3_0_rare_intersection"
"""

sine_dir = "generated_problems/analysis/output_original_caption_sine_1_1"
result_dir = "generated_problems/analysis/output_{0}_caption_sine_1_1_rare_intersection"


# TODO HACKK
# output_format = OutputFormat.CLAUSIFIED
output_format = OutputFormat.ORIGINAL

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger()
log.info("Hello")
log.debug


def main():
    # Get hold of the problems
    # Get path to all problems
    problem_paths = get_problems_from_path(caption_dir)
    if len(problem_paths) == 0:
        raise ValueError(f'Error please check problem dir path, found no problem at "{caption_dir}"')
    # Extract the names
    problem_names = [Path(prob).name for prob in problem_paths]
    problem_names = problem_names[:10]

    # Get the tokeniser # TODO HACK
    tokeniser = load_tokenizer(tokeniser_path)

    # For every problem
    for prob_name in problem_names:
        log.info(f"# {prob_name}")

        # Get the rare axioms selected by sine
        sine_prob = load_and_process_problem(os.path.join(sine_dir, prob_name))
        log.debug(f"Sine problem size {len(sine_prob)}")
        rare_axioms_sine = extract_rare_axioms(tokeniser, sine_prob)
        log.debug(f"Rare sine axioms {len(rare_axioms_sine)}")

        # Load the caption problem
        caption_prob = load_and_process_problem(os.path.join(caption_dir, prob_name))
        log.debug(f"Caption problem size {len(caption_prob)}")
        conjecture = set([caption_prob[0]])
        caption_axioms = set(caption_prob[1:])

        # Split the caption problem into rare and non-rare
        rare_axioms_caption = extract_rare_axioms(tokeniser, caption_axioms)
        log.debug(f"Rare caption axioms {len(rare_axioms_caption)}")

        caption_predicted = caption_axioms.difference(rare_axioms_caption)
        log.debug(f"Caption predicted axioms: {len(caption_predicted)}")

        # Intersect the rare
        rare_axioms_intersection = rare_axioms_caption.intersection(rare_axioms_sine)
        log.debug(f"Rare axiom intersection {len(rare_axioms_intersection)}")

        # Re-join the axioms into the problem
        new_problem = conjecture.union(caption_predicted).union(rare_axioms_intersection)
        log.debug(f"New problem size {len(new_problem)}")

        # Order
        new_problem = order_formulae(new_problem, "first")
        log.debug(f"New problem size: {len(new_problem)}")

        if len(rare_axioms_caption) != len(rare_axioms_sine):
            log.warning(f"Difference in number of axioms for {prob_name}")

        # Write to file
        # Only clausify the problem if set
        if output_format is OutputFormat.CLAUSIFIED:
            # Clausify the problem - this is only done once and in the final step, hence no application of SInE
            prob = clausify(new_problem, skolem_prefix=None, sine_st=None, sine_sd=None)
        elif output_format is OutputFormat.ORIGINAL:
            prob = "\n".join(new_problem).encode()
        else:
            raise ValueError()

        # Save to folder
        global result_dir
        result_dir = result_dir.format(output_format)
        log.info(f"Result dir: {result_dir}")
        save_problem(result_dir, prob_name, prob)

    log.info("# Finished")


if __name__ == "__main__":
    main()
