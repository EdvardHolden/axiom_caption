import os
from pathlib import Path
from keras.preprocessing.text import text_to_word_sequence
from tqdm import tqdm
from multiprocessing import Pool
import random
import numpy as np

from clausifier import clausify, quote_number_in_problem, get_clauses_from_sine
from enum_types import GenerationMode, OutputFormat
from model import get_model_params
from evaluate import generate_step, get_model
from parser import get_generate_parser
import config

from process_problem import save_problem, get_problems_from_path, load_and_process_problem, push_conjecture_to_front, order_formulae

random.seed(7)


# Top dir of the result directory
# BASE_RES_DIR = "generated_problems/merged/"
# BASE_RES_DIR = "generated_problems/skolem/"
BASE_RES_DIR = "generated_problems/analysis/"
#BASE_RES_DIR = "generated_problems/fix/merged/"


def extract_rare_axioms(tokenizer, axioms):
    """Some axioms we know appear in proofs, but they occur too rarely to be trained on.
    This function identifies and returns such axioms.
    """

    rare = set()
    # For each axiom in the problem, if it occurs rarely, keep it
    for formula in axioms:

        # Process the clause
        formula = text_to_word_sequence(formula, tokenizer.filters, tokenizer.lower, tokenizer.split)[0]

        # If the clause is known to us, but not in the top words, it is positively rare
        i = tokenizer.word_index.get(formula)
        if i is not None and i >= tokenizer.num_words:
            rare.update([formula])

    return rare


def compute_caption(
    tokenizer, model, problem_feature, caption, sampler, max_length, no_samples, sampler_temperature, sampler_top_k, axiom_remapping, warmstart_input
):

    import tensorflow as tf
    # Run the model to get the predicted tokens
    axiom_caption = generate_step(
        tokenizer,
        model,
        max_length,
        [problem_feature],
        tf.expand_dims(caption, 0),
        sampler,
        no_samples,
        sampler_temperature,
        sampler_top_k,
        axiom_remapping,
        warmstart_input=warmstart_input
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
    output_format,
    unquote,
    axiom_remapping,
    conjecture_position,
    warmstart,
    prefix=None,
    postfix=None
):
    # Add sampler arguments

    if prefix is not None:
        result_dir += prefix

    # Set the base destionation - use empty path to get correct string format
    result_dir = os.path.join(result_dir, "")

    # Add output format
    result_dir += f"output_{output_format}"

    if unquote:
        result_dir += "_unquoted"

    # Add the mode
    result_dir += f"_{mode}"

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

    if axiom_remapping:
        result_dir += "_axiom_remapping"

    if conjecture_position != "standard":
        result_dir += f"_conjecture_position_{conjecture_position}"

    if warmstart is not None:
        result_dir += f"_warmstart_{Path(warmstart).name}"

    if postfix is not None:
        result_dir += postfix

    # Create direcotry if not exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    return result_dir


def validate_input_arguments(args):
    if args.mode in ["sine", "caption_sine"]:
        if (
            (args.sine_sd is not None and args.sine_st is None)
            or (args.sine_sd is None and args.sine_st is not None)
            or (args.sine_sd is None and args.sine_st is None)
        ):
            raise ValueError("Both sd and st must be set for the SiNE mode")


def standard_process_problem(
    prob_path, mode, sine_st, sine_sd, result_dir, extra_axioms, deepmath, output_format, unquote, conjecture_position
):
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

    elif mode is GenerationMode.SINE:
        # Keep the conjecture
        conj = prob[0]
        # Process with SInE
        prob = get_clauses_from_sine(prob, prob_path, sine_st, sine_sd, deepmath)
        # Add conjecture
        prob += [conj]

    # Add extra axioms if provided
    if len(extra_axioms) > 0:
        prob = set(prob).union(set(extra_axioms))

    # Ensure all numbers are quoted - if unquote is not set
    if not unquote:
        prob = quote_number_in_problem(list(prob))

    prob = order_formulae(prob, conjecture_position)

    if output_format is OutputFormat.CLAUSIFIED:
        prob = clausify(prob, skolem_prefix=None, sine_st=None, sine_sd=None, prob_name=Path(prob_path).stem)
    elif output_format is output_format.ORIGINAL:
        prob = "\n".join(prob).encode()

    # Save to folder
    save_problem(result_dir, Path(prob_path).name, prob)


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
            args.output_format,
            args.unquote,
            args.axiom_remapping,
            args.conjecture_position,
            args.warmstart,
            prefix=args.result_prefix,
            postfix=args.result_postfix,
        )
    print("Writing results to: ", result_dir)

    # Get path to all problems
    problem_paths = get_problems_from_path(args.problem_dir)
    if len(problem_paths) == 0:
        raise ValueError(f'Error please check problem dir path, found no problem at "{args.problem_dir}"')
    if args.debug:
        print("Debug mode: Limiting to K random problems")
        problem_paths = random.sample(problem_paths, k=5)
        problem_paths = ['/shareddata/home/holden/gnn-entailment-caption/merged_problems/t62_chord', '/shareddata/home/holden/gnn-entailment-caption/merged_problems/t24_laplace', '/shareddata/home/holden/gnn-entailment-caption/merged_problems/t36_tsep_1', '/shareddata/home/holden/gnn-entailment-caption/merged_problems/t6_jordan']
        print("Debug problems: ", problem_paths)

    # If captioning, load all the required resources
    if args.mode in [GenerationMode.CAPTION, GenerationMode.CAPTION_SINE]:

        # Load functions specific to captioning
        from dataset import load_entity_features
        from dataset import get_caption_conjecture_tokenizers
        from dataset import load_caption_dict

        model_params = get_model_params(args.model_dir)

        # Load the tokenizers for this training setting
        caption_tokenizer, _, conjecture_tokenizer = get_caption_conjecture_tokenizers(
            model_params, args.proof_data, str(args.context), config.train_id_file, args.feature_path
        )

        # Extract the ids for use in function calls below
        ids = [Path(p).name for p in problem_paths]

        problem_features, caching = load_entity_features(model_params.encoder_input, args.feature_path, ids, conjecture_tokenizer, model_params.conjecture_input_length)
        if caching:
            raise ValueError("Caching not yet supported for problem generation.")

        # Get the captions - this is needed for functionality such as axiom remapping
        caption_dict, _ = load_caption_dict(config.proof_data, ids, model_params.axiom_order, None, caption_tokenizer, None, model_params.remove_unknown)

        # Load the data used to warmstart the prediction model - if set
        if args.warmstart is not None:
            print("# Computing warmstart data")
            from dataset import load_warmstart_data
            from train import get_axiom_frequency
            axiom_frequency = get_axiom_frequency(model_params.axiom_order, config.train_id_file, config.proof_data)
            warmstart_input_dict = load_warmstart_data(ids, args.warmstart, caption_tokenizer, model_params.axiom_order, model_params.remove_unknown, axiom_frequency, args.workers)
        else:
            warmstart_input_dict = None

        # Load the model
        model = get_model(args.model_dir, max_caption_length=args.max_length)

    # Add extra axioms if set
    if args.extra_axioms is not None:
        extra_axioms = get_extra_axioms(problem_paths, args.extra_axioms, deepmath)
    else:
        extra_axioms = set()

    # ## Compute the problems

    # Compute the problems
    # In clean/ideal/sine mode we use a process pool for speedup
    if args.mode in [
        GenerationMode.CLEAN,
        GenerationMode.IDEAL,
        GenerationMode.SINE,
        GenerationMode.POSITIVE_AXIOMS,
    ]:

        star_args = [
            (
                prob_path,
                args.mode,
                args.sine_st,
                args.sine_sd,
                result_dir,
                extra_axioms,
                deepmath,
                args.output_format,
                args.unquote,
                args.conjecture_position
            )
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
            rare_axioms = extract_rare_axioms(caption_tokenizer, initial_axioms)
            # Add the rare axioms to the problem
            new_problem.update(rare_axioms)

            # Extract warmstart input if set
            if warmstart_input_dict is None:
                warmstart_input = None
            else:
                warmstart_input = warmstart_input_dict[Path(prob_path).name]

            # Use the model to generate the axioms required for the proof
            axiom_caption = compute_caption(
                caption_tokenizer,
                model,
                problem_features[Path(prob_path).name],
                caption_dict[Path(prob_path).name],
                args.sampler,
                args.max_length,
                no_samples,
                args.sampler_temperature,
                args.sampler_top_k,
                args.axiom_remapping,
                warmstart_input,
            )
            # Add the caption to the problem
            new_problem.update(axiom_caption)

            # Check if we should also include clauses from sine
            if args.mode is GenerationMode.CAPTION_SINE:

                # Get the formulae output from SInE
                sine_formulae = get_clauses_from_sine(prob, prob_path, args.sine_st, args.sine_sd, deepmath)

                # Combine the clausified axioms with the sine output
                new_problem.update(sine_formulae)

            # Ensure all numbers are quoted - if unquote is not set
            if not args.unquote:
                new_problem = quote_number_in_problem(new_problem)

            # Order the formulae in the problem
            new_problem = order_formulae(new_problem, args.conjecture_position)

            # Only clausify the problem if set
            if args.output_format is OutputFormat.CLAUSIFIED:
                # Clausify the problem - this is only done once and in the final step, hence no application of SInE
                prob = clausify(new_problem, skolem_prefix=None, sine_st=None, sine_sd=None)
            elif args.output_format is OutputFormat.ORIGINAL:
                prob = "\n".join(new_problem).encode()

            # Save to folder
            save_problem(result_dir, Path(prob_path).name, prob)

    else:
        raise ValueError(f"Unrecognised mode '{args.mode}'")


if __name__ == "__main__":

    main()
    print("# Finished")
