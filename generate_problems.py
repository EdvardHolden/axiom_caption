import glob
import tempfile
import atexit
import shutil
import os
import subprocess
from pathlib import Path
import argparse
from keras.preprocessing.text import text_to_word_sequence

from dataset import get_tokenizer
from dataset import load_photo_features
from model import get_model_params
from model import load_model
from evaluate import generate_step


PROBLEM_PATH = "/home/eholden/gnn-entailment-caption/nndata/"
CLAUSIFIER = "~/bin/vclausify_rel"

# Create temporary folder for storing clausifier results
TMP_DIR = tempfile.mkdtemp(prefix="iprover_out_")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    # default="clean",
    default="caption",
    choices=["clean", "sine", "caption", "caption_sine"],
    help="The mode used to generate the modified DeepMath problems",
)
parser.add_argument("--sine_sd", default=None)
parser.add_argument("--sine_st", default=None)
parser.add_argument(
    "--result_dir", default="generated_problems/", help="Base folder for writing generated problems"
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

    return outs


def clausify(prob, sine_st=None, sine_sd=None):

    # Set clausifier mode and call clausifier
    cmd = f"{CLAUSIFIER} --mode clausify "
    return run_clausifier(prob, cmd, sine_st, sine_sd)


def sine_process(prob, sine_st=None, sine_sd=None):
    cmd = f"{CLAUSIFIER} --proof tptp --print_clausifier_premises on --output_axiom_names on --time_limit 1 "
    return run_clausifier(prob, cmd, sine_st, sine_sd)


def save_problem(dir_name, prob_name, prob):
    # print(prob)
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
    axiom_caption = filter(lambda x: x != 0 and x != 1 and x != 2 and x != 3, axiom_caption)
    # If this is reduced to the empty list, set captions as the empty set
    if len(list(axiom_caption)) > 0:
        # Tokenize the output
        axiom_caption = tokenizer.sequences_to_texts([axiom_caption])
        if axiom_caption == [""]:
            axiom_caption = set()
    else:
        # No useful output, set to the empty set
        axiom_caption = set()

    return axiom_caption


def get_result_dir(result_dir, mode, sine_st, sine_sd):
    if mode == "clean":
        result_dir = os.path.join(result_dir, "clean")
    elif mode == "sine":
        result_dir = os.path.join(result_dir, f"sine_{sine_st}_{sine_sd}")
    elif mode == "caption":
        result_dir = os.path.join(result_dir, "caption")
    elif mode == "caption_sine":
        result_dir = os.path.join(result_dir, f"caption_sine_{sine_st}_{sine_sd}")
    else:
        raise ValueError(f'Generative mode "{mode}" not implemented')

    # Create direcotry if not exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    return result_dir


def get_problems_from_path(path=PROBLEM_PATH, limit=None):

    # Get path to all problems
    problem_paths = glob.glob(path + "*")
    if limit is not None:
        return_limit = min(limit, len(problem_paths))
        problem_paths = problem_paths[:return_limit]
    print(f"Number of problems {len(problem_paths)}")

    return problem_paths


def main():

    # Parse input arguments
    args = parser.parse_args()

    # Check if SiNE is set correctly
    if args.mode in ["sine", "caption_sine"]:
        if (
            (args.sine_sd is not None and args.sine_st is None)
            or (args.sine_sd is None and args.sine_st is not None)
            or (args.sine_sd is None and args.sine_st is None)
        ):
            raise ValueError("Both sd and st must be set for the SiNE mode")
    # Set result dir based on the mode
    result_dir = get_result_dir(args.result_dir, args.mode, args.sine_st, args.sine_sd)

    # Get path to all problems
    problem_paths = get_problems_from_path()

    # If captioning, load all the required resources
    if args.mode in ["caption", "caption_sine"]:
        problem_features = load_photo_features(args.feature_path, [Path(p).stem for p in problem_paths])

        # Load the tokenizer
        tokenizer, _ = get_tokenizer("data/deepmath/tokenizer.json")

        # Load model
        print("Loading the model")
        model_params = get_model_params(args.model_dir)
        model_dir = os.path.join(args.model_dir, "ckpt_dir")
        model = load_model(model_dir)
        model.no_rnn_units = model_params.no_rnn_units

    print(f"Writing problems to: {result_dir}")

    # For each problem
    for prob_path in problem_paths:
        prob = load_and_process_problem(prob_path)

        # Maybe use flag instead?
        if args.mode == "clean" or args.mode == "sine":
            # Run clean/sine mode and clausify the problem
            clausified_problem = clausify(prob, sine_st=args.sine_st, sine_sd=args.sine_sd)

        elif args.mode in ["caption", "caption_sine"]:

            # Split the problem into initial axioms and conjecture
            conjecture, initial_axioms = prob[0], prob[1:]
            # Set the current problem to be the conjecture
            new_problem = set([conjecture])

            # Extract axioms that are found in proof but cannot be predicted
            rare_axioms = extract_rare_axioms(tokenizer, initial_axioms)
            # Add the rare axioms to the problem
            new_problem.update(rare_axioms)

            # Use the model to generate the axioms required for the proof
            axiom_caption = compute_caption(tokenizer, model, problem_features[Path(prob_path).stem])
            # Add the caption to the problem
            new_problem.update(axiom_caption)

            # Clausify the problem
            clausified_problem = clausify(new_problem, sine_st=None, sine_sd=None)

            # Check if we should also include clauses from sine
            if args.mode == "caption_sine":
                # Clausify with sine
                sine_problem = clausify(new_problem, sine_st=args.sine_st, sine_sd=args.sine_sd)
                # Combine the clausified axioms with the sine output
                clausified_problem += b"\n" + sine_problem
        else:
            raise ValueError(f"Unrecognised mode '{args.mode}'")

        # Save to folder
        save_problem(result_dir, Path(prob_path).stem, clausified_problem)


if __name__ == "__main__":

    main()
    print("# Finished")
