import glob
import tempfile
import atexit
import shutil
import os
import re
import subprocess
from pathlib import Path
from keras.preprocessing.text import text_to_word_sequence
from tqdm import tqdm
from multiprocessing import Pool
import random
from itertools import chain
import numpy as np

from dataset import get_tokenizer
from dataset import load_photo_features
from enum_types import GenerationMode
from model import get_model_params
from evaluate import generate_step, get_model
from parser import get_generate_parser
import config

import tensorflow as tf

random.seed(7)

CLAUSIFIER = "~/bin/vclausify_rel"

# Create temporary folder for storing clausifier results
TMP_DIR = tempfile.mkdtemp(prefix="iprover_out_")

# Top dir of the result directory
BASE_RES_DIR = "generated_problems/merged/"

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


def include_axiom_files(problem_path, axioms, deepmath):

    # By convention, inclusion happens in the first n lines only
    no_axiom_files = 0
    for n, ax in enumerate(axioms):
        # break when there is nothing more to include
        if ax[0] != "%" and not ax[:7] == "include":
            break
        no_axiom_files += 1

        # Get hold of file
        axiom_file_name = re.findall("\(.+?\)", ax)[0][2:-2]
        axiom_file_path = os.path.join(os.path.dirname(problem_path), axiom_file_name)

        # Extract the axioms
        file_axioms = load_and_process_problem(axiom_file_path, deepmath=deepmath)

        # Add axioms to the set
        axioms.extend(file_axioms)

    # Delete inclusion statements as the axioms are now included
    for n in range(no_axiom_files):
        del axioms[0]

    return axioms


def load_and_process_problem(path, deepmath=False):
    """
    Loads a problem from the text file into lists consiting of string of formulae.
    It currently assumes that each formulae is on a separate line. For deepmath
    problems, the prefix of each formula is removed. It handles axiom includes
    by calling itself on the axiom file.
    """

    # Refractor this whole function to make it much more readable

    # Load lines
    with open(path, "r") as f:
        # List to store all the fof formulae
        formulae = []

        # If deepmath the first formula is a conjecture, load it and replace the axiom tag
        if deepmath:
            conjecture = next(f)[2:].replace("axiom", "conjecture", 1).strip()
            formulae += [conjecture]

        # Load the axioms
        axioms = f.read().splitlines()

    # If deepmath remove the pos/neg label tag
    if deepmath:
        axioms = [ax[2:] for ax in axioms]

    # No inclusion of axioms files for the deepmath format
    if not deepmath:
        axioms = include_axiom_files(path, axioms, deepmath)

    # Add axioms to the set of problem formulae
    formulae.extend(axioms)

    # Remove any newlines for consistency
    formulae = [f.strip() for f in formulae]

    # Return the problem as a list of formulas
    return formulae


def run_clausifier(prob, cmd, sine_st, sine_sd, prob_name, skolem_prefix=b''):

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
        # For some reason, the clausified problem could be added to stderr in this version
        outs += b"\n" + errs
        if proc.returncode != 0 and proc.returncode != 1:  # For some reason it returns 1 for this
            print(f"Clausifier finished on {prob_name} with exitcode: {proc.returncode}")
            print(cmd)
    else:
        if proc.returncode != 0 or errs != b"":
            print(f"Clausifier finished on {prob_name} with exitcode: {proc.returncode}")
            print("Error: ", errs)
            print(cmd)

    # Set a prefix for the Skolem functions in case the clausified problem is merged with other clauses
    outs = re.sub(b'(sK\d+)', skolem_prefix + b'\\1', outs)

    # Try to delete the file to save memory
    try:
        os.remove(tmp_file)
        pass
    except Exception as err:
        print(f"Warning could not remove file {tmp_file} because: {err}")

    return outs


def clausify(prob, skolem_prefix, sine_st=None, sine_sd=None, prob_name=None):

    # Set clausifier mode and call clausifier
    cmd = f"{CLAUSIFIER} --mode clausify "
    return run_clausifier(prob, cmd, sine_st, sine_sd, prob_name, skolem_prefix)


def sine_process(prob, sine_st=None, sine_sd=None, prob_name=None):
    # --proof (-p) Specifies whether proof (or similar e.g. model/saturation) will be output
    # --print_clausifier_premises Output how the clausified problem was derived.
    # --output_axiom_names Preserve names of axioms from the problem file in the proof output

    # cmd = f"{CLAUSIFIER} --proof tptp --print_clausifier_premises on --output_axiom_names on --time_limit 1 "
    cmd = f"{CLAUSIFIER} --mode clausify --proof tptp --print_clausifier_premises on --output_axiom_names on --time_limit 4 " # TODO changed timelimit from 1 to 4
    return run_clausifier(prob, cmd, sine_st, sine_sd, prob_name)


def save_problem(dir_name, prob_name, prob):
    try:
        with open(os.path.join(dir_name, prob_name), "wb") as f:
            f.write(prob)
    except OSError as err:
        print("Error: ", err)
        print("Could not save generated problem for: ", prob_name)


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


def compute_caption(
    tokenizer, model, problem_feature, sampler, max_length, no_samples, sampler_temperature, sampler_top_k
):

    # Run the model to get the predicted tokens
    # axiom_caption = generate_step( tokenizer, model, max_len, img_tensor, sampler, no_samples, sampler_temperature, sampler_top_k)
    axiom_caption = generate_step(
        tokenizer,
        model,
        max_length,
        [problem_feature],
        tf.ones([1, 1], dtype=tf.dtypes.int32),  # Dummy the caption to get dimension
        sampler,
        no_samples,
        sampler_temperature,
        sampler_top_k,
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


def get_result_dir(
    result_dir,
    mode,
    sine_st,
    sine_sd,
    extra_axioms,
    sampler,
    sampler_temperature,
    sampler_top_k,
    no_samples,
    max_length,
    prefix=None,
    postfix=None,
):
    # Add sampler arguments

    if prefix is not None:
        result_dir += prefix

    # Set the base destionation
    result_dir = os.path.join(result_dir, str(mode))

    # Add sine parameters
    if mode in [GenerationMode.SINE, GenerationMode.CAPTION_SINE]:
        result_dir += f"_{sine_st}_{sine_sd}"

    # Add sampling method if caption model is in use
    if mode in [GenerationMode.CAPTION, GenerationMode.CAPTION_SINE]:
        # Add sampling method
        result_dir += f"_{sampler}"

        # Add method details if set
        if sampler == "temperature":
            sampler += f"_{sampler_temperature}"
        elif sampler == "top_k":
            sampler += f"_{sampler_top_k}"

        # Add sampling size arguments
        result_dir += f"_no_samples_{no_samples}_length_{max_length}"

    # Add the numbr of extra axioms if set
    if extra_axioms is not None:
        result_dir += f"_extra_axioms_{extra_axioms}"

    if postfix is not None:
        result_dir += postfix

    # Create direcotry if not exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    return result_dir


def get_problems_from_path(problem_dir, limit=None, deepmath=True):

    # Get path to all problems
    problem_paths = glob.glob(os.path.join(problem_dir, "") + "*")
    """
    if deepmath:
        problem_paths = glob.glob(os.path.join(problem_dir, "") + "*")
    else:
        problem_paths = glob.glob(os.path.join(problem_dir, "") + "*.p")
    """

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


def standard_process_problem(prob_path, mode, sine_st, sine_sd, result_dir, extra_axioms, deepmath):
    # Load problem formulae as a list
    prob = load_and_process_problem(prob_path, deepmath=deepmath)

    # If the problem should be ideal, we just remove the last half of the axioms are they are false
    if mode is GenerationMode.IDEAL:
        prob = prob[: len(prob) // 2 + 1]

    elif mode is GenerationMode.POSITIVE_AXIOMS:
        # Remove conjecture
        prob = prob[1:]
        prob = prob[: len(prob) // 2]
        # Only keep positive axioms (first half)

    # Add extra axioms if provided
    if len(extra_axioms) > 0:
        prob = set(prob).union(set(extra_axioms))

    # Ensure all numbers are quoted
    prob = quote_number_in_problem(list(prob))

    # Run clean/sine mode and clausify the problem
    clausified_problem = clausify(prob, skolem_prefix=b'ST_', sine_st=sine_st, sine_sd=sine_sd, prob_name=Path(prob_path).stem)

    # Save to folder
    save_problem(result_dir, Path(prob_path).name, clausified_problem)


def get_extra_axioms(problem_paths, no_axioms, deepmath):

    # Get all axioms
    axioms = set()
    for problem in problem_paths:
        # Get all clauses in the problem
        prob_ax = load_and_process_problem(problem, deepmath=deepmath)
        # Add axioms
        if len(prob_ax) > 1:
            # prob_ax = prob_ax[: len(prob_ax) // 2] # If inlcuding only positive
            axioms.update(set(prob_ax[1:]))

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
    parser = get_generate_parser()
    args = parser.parse_args()
    # Check if SiNE is set correctly
    validate_input_arguments(args)

    # Deduce the problem format
    deepmath = args.problem_format == "deepmath"

    if len(args.no_samples) > 1:
        raise ValueError("Error: Multiple sample values not supported for generating problems")
    else:
        no_samples = args.no_samples[0]

    # Set result dir based on the mode if the path is not provided
    if args.result_dir is not None:
        result_dir = os.path.join(args.result_dir, "")
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
    else:
        result_dir = get_result_dir(
            BASE_RES_DIR,
            args.mode,
            args.sine_st,
            args.sine_sd,
            args.extra_axioms,
            args.sampler,
            args.sampler_temperature,
            args.sampler_top_k,
            no_samples,
            args.max_length,
            args.result_prefix,
        )
    print("Writing results to: ", result_dir)

    # Get path to all problems
    problem_paths = get_problems_from_path(args.problem_dir, deepmath=deepmath)
    if len(problem_paths) == 0:
        raise ValueError(f'Error please check problem dir path, found no problem at "{args.problem_dir}"')
    if args.debug:
        print("Debug mode: Limiting to 100 problems")
        problem_paths = random.sample(problem_paths, k=5)

    # If captioning, load all the required resources
    if args.mode in [GenerationMode.CAPTION, GenerationMode.CAPTION_SINE]:
        problem_features = load_photo_features(args.feature_path, [Path(p).stem for p in problem_paths])
        # TODO need to modify this work with the new laoding of params

        tokenizer, vocab_size = get_tokenizer(
            config.train_id_file,
            str(args.context),
            config.proof_data,
            get_model_params(args.model_dir).axiom_vocab_size,
        )

        # Load the model
        model = get_model(args.model_dir, vocab_size)

    # Add extra axioms if set
    if args.extra_axioms is not None:
        extra_axioms = get_extra_axioms(problem_paths, args.extra_axioms, deepmath)
    else:
        extra_axioms = set()

    # ## Compute the problems

    # Compute the problems
    # In clean/ideal/sine mode we use a process pool for speedup
    if args.mode in [GenerationMode.CLEAN, GenerationMode.IDEAL, GenerationMode.SINE, GenerationMode.POSITIVE_AXIOMS]:

        star_args = [
            (prob_path, args.mode, args.sine_st, args.sine_sd, result_dir, extra_axioms, deepmath)
            for prob_path in problem_paths
        ]
        pool = Pool(args.workers)
        pool.starmap(standard_process_problem, star_args)
        pool.close()
        pool.join()

    # Run other problem modes
    elif args.mode in [GenerationMode.CAPTION, GenerationMode.CAPTION_SINE]:

        # Process each problem
        for prob_path in tqdm(problem_paths):
            prob = load_and_process_problem(prob_path, deepmath)

            # Split the problem into initial axioms and conjecture
            conjecture = prob[0]
            initial_axioms = set(prob[1:])

            # Set the current problem to be the conjecture
            new_problem = set([conjecture])

            # Update initial axioms with the extra axioms if set
            if args.extra_axioms is not None:
                # These only affect the extraction of rare axioms
                initial_axioms.update(extra_axioms)

            # Extract axioms that are found in proof but cannot be predicted
            rare_axioms = extract_rare_axioms(tokenizer, initial_axioms)
            # Add the rare axioms to the problem
            new_problem.update(rare_axioms)

            # Use the model to generate the axioms required for the proof
            axiom_caption = compute_caption(
                tokenizer,
                model,
                problem_features[Path(prob_path).stem],
                args.sampler,
                args.max_length,
                no_samples,
                args.sampler_temperature,
                args.sampler_top_k,
            )
            # Add the caption to the problem
            new_problem.update(axiom_caption)

            # Ensure all numbers are quoted
            new_problem = quote_number_in_problem(new_problem)

            # Clausify the problem
            clausified_problem = clausify(new_problem, skolem_prefix=b"CAPTION_", sine_st=None, sine_sd=None)

            # Check if we should also include clauses from sine
            if args.mode is GenerationMode.CAPTION_SINE:
                # Clausify the original problem with sine and add to set
                sine_problem = clausify(
                    quote_number_in_problem(prob), skolem_prefix=b"SINE_", sine_st=args.sine_st, sine_sd=args.sine_sd
                )
                # Combine the clausified axioms with the sine output
                clausified_problem += b"\n" + sine_problem
            # Save to folder
            save_problem(result_dir, Path(prob_path).name, clausified_problem)

    else:
        raise ValueError(f"Unrecognised mode '{args.mode}'")


if __name__ == "__main__":

    main()
    print("# Finished")
