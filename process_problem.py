import glob
import os
import re


def order_formulae(prob, conjecture_position):
    # Standard jsut return all lexicographically
    if conjecture_position == "standard":
        return sorted(prob)

    # Split on the conjecture
    conj, *ax = push_conjecture_to_front(prob)
    ax = sorted(ax)  # Ensure that the axioms are sorted

    if conjecture_position == "first":
        return [conj] + ax
    elif conjecture_position == "last":
        return ax + [conj]
    else:
        raise ValueError(f"Incorrect value given for conjecture position: '{conjecture_position}'")


def save_problem(dir_name, prob_name, problem_string):
    try:
        with open(os.path.join(dir_name, prob_name), "wb") as f:
            f.write(problem_string)
    except OSError as err:
        print("Error: ", err)
        print("Could not save generated problem for: ", prob_name)


def get_problems_from_path(problem_dir, limit=None, verbose=1):
    # Get path to all problems
    problem_paths = glob.glob(os.path.join(problem_dir, "") + "*")

    if limit is not None:
        return_limit = min(limit, len(problem_paths))
        problem_paths = problem_paths[:return_limit]

    if verbose > 0:
        print(f"Number of problems {len(problem_paths)}")

    return problem_paths


def _include_axiom_files(problem_path, axioms, deepmath):
    # By convention, inclusion happens in the first n lines only
    no_axiom_files = 0
    for n, ax in enumerate(axioms):
        # Skip commented lines
        if ax[0] == "%":
            continue

        # Break when there is nothing more to include
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


def push_conjecture_to_front(formulae):
    if isinstance(formulae, tuple) or isinstance(formulae, set):
        formulae = list(formulae)

    for n, f in enumerate(formulae):
        # Positive axioms setting does not neccessarily have a connjecture
        conjecture = None

        if "conjecture" in f:
            conjecture = formulae.pop(n)
            break
    if conjecture is not None:
        formulae = [conjecture] + formulae

    return formulae


def load_and_process_problem(path, deepmath=False):
    """
    Loads a problem from the text file into lists consiting of string of formulae.
    It currently assumes that each formulae is on a separate line. For deepmath
    problems, the prefix of each formula is removed. It handles axiom includes
    by calling itself on the axiom file.
    """

    # Refractor this whole function to make it much more readable

    # Load lines
    with open(os.path.expanduser(path), "r") as f:
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
        axioms = _include_axiom_files(path, axioms, deepmath)

    # Add axioms to the set of problem formulae
    formulae.extend(axioms)

    # Remove any newlines for consistency
    formulae = [f.strip() for f in formulae]

    # Need to ensure that the first entry is the conjecture
    if not deepmath and len(formulae) > 0 and "conjecture" not in formulae[0]:
        # Find the conjecture and push it to the front - Assumes only one FOF conjecture
        formulae = push_conjecture_to_front(formulae)

    # Return the problem as a list of formulas
    return formulae
