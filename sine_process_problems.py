"""
Given a problem folder and an meta file
this program preprocess problems using the sine algorithm.
The result is outputed to the approrpiate folder
"""

import argparse
import os
import sys
import pickle
import multiprocessing
import subprocess

CLAUSIFIER = "~/bin/vclausify_rel"
TIMELIMIT = 30

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('problem_dir', help='Directory containing the problems to process')
parser.add_argument('meta_dict', help='Path to dict containing the meta info')
parser.add_argument('--results_dir', help='Directory for storing the result', default='data/processed/jjt_sine_1_0/')
parser.add_argument('--sine_tolerance', default=1)
parser.add_argument('--sine_depth', default=0)

args = parser.parse_args()


def preprocess_sine(path, problem, res_dir):

    # Assume that if the problem contains underscore it is in tff
    if '_' in problem:
        mode = 'tclausify'
    else:
        mode = 'clausify'

    # Build SiNE command
    cmd = f'{CLAUSIFIER} -t {TIMELIMIT} --mode {mode} -ss axioms -sd {args.sine_depth} -st {args.sine_tolerance}'
    # Append problem
    cmd += f' {os.path.join(path, problem)}.p'
    # Append output direction
    cmd += f'  >{os.path.join(res_dir, problem)}.p'

    # Run process
    proc = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, preexec_fn=os.setsid)
    try:
        _, errs = proc.communicate(timeout=TIMELIMIT + 5)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, errs = proc.communicate()

    # Check for errors
    if proc.returncode != 0 or errs != b'':
        print(f'Clausification error for {problem} exitcode: {proc.returncode} error: {errs}')


# Get the problems used in proofs from the meta (annoying twist here that E 'attempts' nearly all)
def get_problems_versions(meta):
    problem_versions = set()
    for prover in meta.keys():
        for problem in meta[prover].values():
            version = problem['version']
            if problem['version'] is not None:
                problem_versions.add(version)
    return problem_versions


def main():

    # Check problem dir existence
    if not os.path.exists(args.problem_dir):
        print("Error - problem dir does not exist")
        sys.exit(0)

    # Extract all the problem names from the meta
    with open(args.meta_dict, 'rb') as f:
        meta = pickle.load(f)

    # Find the problem versions to process
    problem_versions = get_problems_versions(meta)

    # Check existence of results path
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    # Process the problems
    proc_args = [(args.problem_dir, prob, args.results_dir) for prob in problem_versions]

    # Multiprocess
    pool = multiprocessing.Pool(processes=5)
    pool.starmap(preprocess_sine, proc_args)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
    print("Finito")
