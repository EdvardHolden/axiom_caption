import atexit
import os
import re
import shutil
import subprocess
import tempfile

# Path to the clausifier - might put in config
from itertools import chain
from pathlib import Path

CLAUSIFIER = "~/bin/vclausify_rel"

# Create temporary folder for storing clausifier results
TMP_DIR = tempfile.mkdtemp(prefix="iprover_out_")


# Re pattern for finding each element in a clause
ELEMENT_PATTERN = re.compile("([\(\),=&?<>|])")

RE_CLAUSE_FILE = b"file\('(\/|\w)*',(\w*)\)\)."
RE_CLAUSE_NAME_PROBLEM = "^fof\((\w+), axiom"


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


def run_clausifier(prob, cmd, sine_st, sine_sd, prob_name, skolem_prefix=None):

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
    if skolem_prefix is not None:
        outs = re.sub(b"(sK\d+)", skolem_prefix + b"\\1", outs)

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


def clausify_output_axiom_names(prob, sine_st=None, sine_sd=None, prob_name=None):
    # --proof (-p) Specifies whether proof (or similar e.g. model/saturation) will be output
    # --print_clausifier_premises Output how the clausified problem was derived.
    # --output_axiom_names Preserve names of axioms from the problem file in the proof output
    # --mode clausify To clauify the problem and not attempt to solve it

    # cmd = f"{CLAUSIFIER} --proof tptp --print_clausifier_premises on --output_axiom_names on --time_limit 1 "
    cmd = f"{CLAUSIFIER} --mode clausify --proof tptp --print_clausifier_premises on --output_axiom_names on --time_limit 4 "  # FIXME changed timelimit from 1 to 4
    return run_clausifier(prob, cmd, sine_st, sine_sd, prob_name)


def get_sine_clause_names(prob):
    res = re.findall(RE_CLAUSE_FILE, prob)
    # res = re.findall(RE_CLAUSE_NAME_PROBLEM, prob)
    res = [r[1] for r in res]
    return res


def get_clause_names(prob):
    res = re.findall(RE_CLAUSE_NAME_PROBLEM, prob, flags=re.MULTILINE)
    return res


def quote_number_in_formula(formula):
    # Split formula elements
    elements = ELEMENT_PATTERN.split(formula)
    # Quote all the digits
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


def get_clausified_names(prob, sine_st, sine_sd, prob_path):

    # Clausify the problem - optionally with SInE
    prob_processed = clausify_output_axiom_names(
        prob, sine_st=sine_st, sine_sd=sine_sd, prob_name=Path(prob_path).stem
    )

    # Extract the clause names from the sine processed problem
    processed_names = get_sine_clause_names(prob_processed)
    return processed_names


def get_clauses_from_sine(prob, prob_path, sine_st, sine_sd, deepmath):

    processed_names = get_clausified_names(prob, sine_st, sine_sd, prob_path)

    # Quadtratic matching - not good
    # Every clause starts with 'fof({NAME}' so we can just check for that
    # Check for every selected clause
    result = []
    for clause_name in processed_names:
        for formula in prob[1:]:  # Skip the conjecture
            if f"fof({clause_name.decode()}," in formula:  # Check for name - needs to be byte
                result += [formula]

    return result
