import glob
import tempfile
import atexit
import shutil
import os
import subprocess
from pathlib import Path
import argparse


PROBLEM_PATH = "/home/eholden/gnn-entailment-caption/nndata/"
CLAUSIFIER = "~/bin/vclausify_rel"

# Create temporary folder for storing clausifier results
TMP_DIR = tempfile.mkdtemp(prefix="iprover_out_")


# TODO lets add some modes!
# Maybe set the model decoding step as a static option?
# mode = {clean, sine, caption, caption_sine}
# TODO need to include the rare axioms as well!!

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    default="clean",
    choices=["clean", "sine"],
    help="The mode used to generate the modified DeepMath problems",
)
parser.add_argument("--sine_sd", default=None)
parser.add_argument("--sine_st", default=None)
parser.add_argument(
    "--result_dir", default="generated_problems/", help="Base folder for writing generated problems"
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


def clausify(prob, sine_st=None, sine_sd=None):

    # Need to make file in tmp directory
    tmp_file = get_tmp_out_file()
    with open(tmp_file, "w") as f:
        f.write("\n".join(prob))

    cmd = f"{CLAUSIFIER} --mode clausify {tmp_file}"

    if sine_st is not None and sine_sd is not None:
        cmd += f" -ss axioms -sd {sine_sd} -st {sine_st} "
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        outs, errs = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()

    if errs != b"" or proc.returncode != 0:
        print(errs)
        print(f"Clausifier finished with exitcode: {proc.returncode}")

    return outs


def save_problem(dir_name, prob_name, prob):
    # print(prob)
    with open(os.path.join(dir_name, prob_name), "wb") as f:
        f.write(prob)


def main():

    # Parse input arguments
    args = parser.parse_args()

    # Set result dir based on the mode
    if args.mode == "clean":
        result_dir = os.path.join(args.result_dir, "clean")
    elif args.mode == "sine":
        if (args.sine_sd is not None and args.sine_st is None) or (
            args.sine_sd is None and args.sine_st is not None
        ):
            raise ValueError("Both sd and st must be set for the SiNE mode")
        result_dir = os.path.join(args.result_dir, f"sine_{args.sine_st}_{args.sine_sd}")
    else:
        raise ValueError(f'Generative mode "{args.mode}" not implemented')

    # Create direcotry if not exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Get path to all problems
    problem_paths = glob.glob(PROBLEM_PATH + "*")
    problem_paths = [problem_paths[0]]  # FIXME
    print(f"Number of problems {len(problem_paths)}")

    print(f"Writing problems to: {result_dir}")

    # For each problem
    for prob_path in problem_paths:
        prob = load_and_process_problem(prob_path)

        # Optionally modify TODO

        # Clausify the problem
        prob = clausify(
            prob, sine_st=args.sine_st, sine_sd=args.sine_sd
        )  # TODO need to rename this variable!

        # TODO could append the caption stuff here?

        # Save to folder
        save_problem(result_dir, Path(prob_path).stem, prob)


if __name__ == "__main__":
    main()
