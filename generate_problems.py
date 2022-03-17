import glob
import tempfile
import atexit
import shutil
import os
import re
import subprocess
from pathlib import Path
import argparse
from keras.preprocessing.text import text_to_word_sequence
from tqdm import tqdm
from multiprocessing import Pool
import random
from itertools import chain
import numpy as np
import socket

from dataset import get_tokenizer
from dataset import load_photo_features
from model import get_model_params
from model import load_model
from evaluate import generate_step


random.seed(7)

CLAUSIFIER = "~/bin/vclausify_rel"

# Create temporary folder for storing clausifier results
TMP_DIR = tempfile.mkdtemp(prefix="iprover_out_")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    # default="clean",
    default="caption",
    choices=["clean", "ideal", "sine", "caption", "caption_sine"],
    help="The mode used to generate the modified DeepMath problems",
)
parser.add_argument("--sine_sd", default=None)
parser.add_argument("--sine_st", default=None)
parser.add_argument(
    "--result_dir", default="generated_problems/", help="Base folder for writing generated problems"
)

if socket.gethostname() == "kontor":
    default_problem_dir = "/home/eholden/gnn-entailment-caption/nndata"
else:
    default_problem_dir = "/shareddata/home/holden/gnn-entailment-caption/nndata"
parser.add_argument(
    "--problem_dir",
    default=default_problem_dir,
    help="Directory containing the base problems",
)
parser.add_argument(
    "--add_extra_axioms",
    default=False,
    action="store_true",
    help="Add a set number of extra axioms to each problem. Same axioms added to each problem.",
)
parser.add_argument(
    "--number_of_extra_axioms",
    default=1000,
    type=int,
    help="Number of extra axioms to add to each generated problem. Only used if add_extra_axioms flag is set",
)

parser.add_argument(
    "--feature_path",
    default="data/embeddings/deepmath/graph_features_deepmath_all.pkl",
    help="Path to the problem embeddings",
)
parser.add_argument(
    "--model_dir",
    default="experiments/hyperparam/initial/attention_False_axiom_order_length_batch_norm_False_dropout_rate_0.1_embedding_size_200_learning_rate_0.001_model_type_merge_inject_no_dense_units_32_no_rnn_units_32_normalize_True_rnn_type_lstm/",
    help="Path to the model used in the captioning modes",
)
parser.add_argument(
    "--workers",
    type=int,
    default=max(os.cpu_count() - 2, 1),
    help="Number of workers for multiprocessing (used in some modes)",
)
parser.add_argument(
    "-d", "--debug", action="store_true", default=False, help="Limit generation to 100 instances"
)

# Re pattern for finding each element in a clause
ELEMENT_PATTERN = re.compile("([\(\),=&?<>|])")

# Find an quote all numbers appearing in a formula
def quote_number_in_formula(formula):
    # Split formula elements
    elements = ELEMENT_PATTERN.split(formula)
    # Quote all the digits FIXME cannot use shorthand
    # elements = ["'" + e.strip() + "'" if e.strip().isdigit() else e for e in elements]
    formula = []
    digits = set()
    for e in elements:
        # Quote all digits
        if e.strip().isdigit():
            digit = e.strip()
            # Add quoted digit
            formula += "'" + digit + "'"
            # Add to set of digits
            digits.add(digit)
        else:
            # Add non-digit
            formula += e

    # Join the formula back up and return
    return digits, "".join(formula)


@atexit.register
def clean_tmp_folder():
    # Clean tmp folder
    try:
        shutil.rmtree(TMP_DIR)
    except FileNotFoundError:
        pass


def get_tmp_out_file():
    # Create the tmp file in the current tmp directory and return the file name
    fd, filepath = tempfile.mkstemp(prefix=TMP_DIR + "/")
    os.close(fd)  # Close the open file descriptor
    return filepath


def load_and_process_problem(path):

    # Load lines
    with open(path, "r") as f:
        # Load the conjecture and replace axiom tag with the appropriate conjecture tag
        conjecture = next(f)[2:].replace("axiom", "conjecture", 1).strip()

        # Load the axioms
        axioms = []
        lines = f.readlines()
        for line in lines:
            axioms += [line[2:].strip()]

    # Return the problem as a list of formulas
    return [conjecture] + axioms


def run_clausifier(prob, cmd, sine_st, sine_sd):

    # Build sine string
    if sine_st is not None and sine_sd is not None:
        cmd += f" -ss axioms -sd {sine_sd} -st {sine_st} "

    # Put processed problem in a tmp file such that we can process it
    tmp_file = get_tmp_out_file()
    with open(tmp_file, "w") as f:
        f.write("\n".join(prob))
    # Add file path to cmd
    cmd += f" {tmp_file} "

    # Make subprocess
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        outs, errs = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()

    if "--print_clausifier_premises" in cmd:
        if proc.returncode != 0 and proc.returncode != 1:  # For some reason it returns 1 for this
            print(f"Clausifier finished with exitcode: {proc.returncode}")
    else:
        if proc.returncode != 0 or errs != b"":
            print(f"Clausifier finished with exitcode: {proc.returncode}")
            print("Error: ", errs)
            print(cmd)

    return outs


def clausify(prob, sine_st=None, sine_sd=None):

    # Set clausifier mode and call clausifier
    cmd = f"{CLAUSIFIER} --mode clausify "
    return run_clausifier(prob, cmd, sine_st, sine_sd)


def sine_process(prob, sine_st=None, sine_sd=None):
    cmd = f"{CLAUSIFIER} --proof tptp --print_clausifier_premises on --output_axiom_names on --time_limit 1 "
    return run_clausifier(prob, cmd, sine_st, sine_sd)


def save_problem(dir_name, prob_name, prob):
    with open(os.path.join(dir_name, prob_name), "wb") as f:
        f.write(prob)


def extract_rare_axioms(tokenizer, axioms):
    """Some axioms we know appear in proofs, but they occur too rarely to be trained on.
    This function identifies and returns such axioms.
    """

    rare = set()
    # For each axiom in the problem, if it occurs rarely, keep it
    for formula in axioms:

        # Process the clause
        formula = text_to_word_sequence(formula, tokenizer.filters, tokenizer.lower, tokenizer.split)[0]

        # If the clause is known to use, but not in the top words, it is positively rare
        i = tokenizer.word_index.get(formula)
        if i is not None and i >= tokenizer.num_words:
            rare.update([formula])

    return rare


def compute_caption(tokenizer, model, problem_feature):

    # TODO add sampling type?
    # Run the model to get the predicted tokens
    # axiom_caption = generate_step( tokenizer, model, max_len, img_tensor, sampler, no_samples, sampler_temperature, sampler_top_k)
    axiom_caption = generate_step(
        tokenizer,
        model,
        22,
        [problem_feature],
        "greedy",
        1,
        None,
        None,
    )
    # Remove non-axiom tokens
    axiom_caption = list(filter(lambda x: x != 0 and x != 1 and x != 2 and x != 3, axiom_caption))
    # If this is reduced to the empty list, set captions as the empty set
    if len(axiom_caption) > 0:
        # Feed input as a nested list of single tokens as this is cleaner for transforming into a set
        inp = np.array([axiom_caption]).T
        # Tokenize the output
        axiom_caption = set(tokenizer.sequences_to_texts(inp))
        if axiom_caption == [""]:
            axiom_caption = set()
    else:
        # No useful output, set to the empty set
        axiom_caption = set()

    # print(axiom_caption)
    # print(len(axiom_caption), type(axiom_caption))

    return axiom_caption


def get_result_dir(result_dir, mode, sine_st, sine_sd, add_extra_axioms, number_of_extra_axioms):
    if mode == "clean":
        result_dir = os.path.join(result_dir, "clean")
    elif mode == "ideal":
        result_dir = os.path.join(result_dir, "ideal")
    elif mode == "sine":
        result_dir = os.path.join(result_dir, f"sine_{sine_st}_{sine_sd}")
    elif mode == "caption":
        result_dir = os.path.join(result_dir, "caption")
    elif mode == "caption_sine":
        result_dir = os.path.join(result_dir, f"caption_sine_{sine_st}_{sine_sd}")
    else:
        raise ValueError(f'Generative mode "{mode}" not implemented')

    if add_extra_axioms:
        result_dir += f"_extra_axioms_{number_of_extra_axioms}"

    # Create direcotry if not exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    return result_dir


def get_problems_from_path(problem_dir, limit=None):

    # Get path to all problems
    problem_paths = glob.glob(os.path.join(problem_dir, "") + "*")
    if limit is not None:
        return_limit = min(limit, len(problem_paths))
        problem_paths = problem_paths[:return_limit]
    print(f"Number of problems {len(problem_paths)}")

    return problem_paths


def validate_input_arguments(args):
    if args.mode in ["sine", "caption_sine"]:
        if (
            (args.sine_sd is not None and args.sine_st is None)
            or (args.sine_sd is None and args.sine_st is not None)
            or (args.sine_sd is None and args.sine_st is None)
        ):
            raise ValueError("Both sd and st must be set for the SiNE mode")


def get_model(model_dir):
    model_params = get_model_params(model_dir)
    model_dir = os.path.join(model_dir, "ckpt_dir")
    model = load_model(model_dir)
    model.no_rnn_units = model_params.no_rnn_units
    return model


def quote_number_in_problem(prob):

    # Quote numbers in each formula
    numbers, prob = zip(*list(map(quote_number_in_formula, prob)))

    # Get the set of numbers in the problem from each formulae
    numbers = set(chain.from_iterable(numbers))

    # If there are more than one number, add distinct number axiom
    if len(numbers) > 1:
        distinct_number_axiom = "fof(a1, axiom, $distinct({0})).".format(
            ", ".join(["'" + n + "'" for n in sorted(numbers)])
        )
        # Add axiom to the tuple of formulae
        prob += tuple([distinct_number_axiom])

    return prob


def standard_process_problem(prob_path, mode, sine_st, sine_sd, result_dir, extra_axioms):
    # Load problem formulae as a list
    prob = load_and_process_problem(prob_path)

    # If the problem should be ideal, we just remove the last half of the axioms are they are false
    if mode == "ideal":
        prob = prob[: len(prob) // 2 + 1]

    # Add extra axioms if provided
    if len(extra_axioms) > 0:
        prob = set(prob).union(set(extra_axioms))

    # Ensure all numbers are quoted
    prob = quote_number_in_problem(list(prob))

    # Run clean/sine mode and clausify the problem
    clausified_problem = clausify(prob, sine_st=sine_st, sine_sd=sine_sd)

    # Save to folder
    save_problem(result_dir, Path(prob_path).stem, clausified_problem)


def get_extra_axioms(problem_paths, no_axioms):

    # Get all axioms
    axioms = set()
    for problem in problem_paths:
        prob_ax = load_and_process_problem(problem)[1:]
        # Add axioms
        if len(prob_ax) > 0:
            # prob_ax = prob_ax[: len(prob_ax) // 2] # If inlcuding only positive
            axioms.update(set(prob_ax))

    # Check if enough axioms, otherwise return them all
    if no_axioms > len(axioms):
        print(
            f"Warning: Cannot add {no_axioms} to benchmark as there is not that many axioms. Truncating to: {len(axioms)}"
        )
        return axioms

    # Sample the axioms
    extra_axioms = set(random.sample(list(axioms), k=no_axioms))
    assert len(extra_axioms) == no_axioms
    return extra_axioms


def main():

    # Parse input arguments
    args = parser.parse_args()
    # Check if SiNE is set correctly
    validate_input_arguments(args)

    # Set result dir based on the mode
    result_dir = get_result_dir(
        args.result_dir,
        args.mode,
        args.sine_st,
        args.sine_sd,
        args.add_extra_axioms,
        args.number_of_extra_axioms,
    )

    # Get path to all problems
    problem_paths = get_problems_from_path(args.problem_dir)
    if len(problem_paths) == 0:
        raise ValueError(f'Error please check problem dir path, found no problem at "{args.problem_dir}"')
    if args.debug:
        print("Debug mode: Limiting to 100 problems")
        problem_paths = random.sample(problem_paths, k=5)

    # If captioning, load all the required resources
    if args.mode in ["caption", "caption_sine"]:
        problem_features = load_photo_features(args.feature_path, [Path(p).stem for p in problem_paths])
        tokenizer, _ = get_tokenizer("data/deepmath/tokenizer.json")
        model = get_model(args.model_dir)

    # Add extra axioms if set
    if args.add_extra_axioms:
        extra_axioms = get_extra_axioms(problem_paths, args.number_of_extra_axioms)
    else:
        extra_axioms = set()

    # ## Compute the problems

    # Compute the problems
    # In clean/ideal/sine mode we use a process pool for speedup
    if args.mode in ["clean", "ideal", "sine"]:

        star_args = [
            (prob_path, args.mode, args.sine_st, args.sine_sd, result_dir, extra_axioms)
            for prob_path in problem_paths
        ]
        pool = Pool(args.workers)
        pool.starmap(standard_process_problem, star_args)
        pool.close()
        pool.join()

    # Run other problem modes
    elif args.mode in ["caption", "caption_sine"]:

        # Process each problem
        for prob_path in tqdm(problem_paths):
            prob = load_and_process_problem(prob_path)

            # Split the problem into initial axioms and conjecture
            conjecture = prob[0]
            initial_axioms = set(prob[1:])

            # Set the current problem to be the conjecture
            new_problem = set([conjecture])

            # Update initial axioms with the extra axioms if set
            if args.add_extra_axioms:
                initial_axioms.update(extra_axioms)

            # Extract axioms that are found in proof but cannot be predicted
            rare_axioms = extract_rare_axioms(tokenizer, initial_axioms)
            # Add the rare axioms to the problem
            new_problem.update(rare_axioms)

            # Use the model to generate the axioms required for the proof
            axiom_caption = compute_caption(tokenizer, model, problem_features[Path(prob_path).stem])
            # Add the caption to the problem
            new_problem.update(axiom_caption)

            # Ensure all numbers are quoted
            new_problem = quote_number_in_problem(new_problem)

            # Clausify the problem
            clausified_problem = clausify(new_problem, sine_st=None, sine_sd=None)

            # Check if we should also include clauses from sine
            if args.mode == "caption_sine":
                # Clausify with sine
                sine_problem = clausify(new_problem, sine_st=args.sine_st, sine_sd=args.sine_sd)
                # Combine the clausified axioms with the sine output
                clausified_problem += b"\n" + sine_problem
            # Save to folder
            save_problem(result_dir, Path(prob_path).stem, clausified_problem)

    else:
        raise ValueError(f"Unrecognised mode '{args.mode}'")


if __name__ == "__main__":

    main()
    print("# Finished")
