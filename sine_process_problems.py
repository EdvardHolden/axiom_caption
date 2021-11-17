"""
This program preprocesses the fof and tff problems in a directory using the SiNE algorithm.
The result is outputed to the approrpiate folder
"""

import argparse
import os
import sys
import multiprocessing
import subprocess
import glob

CLAUSIFIER = "~/bin/vclausify_rel"
TIMELIMIT = 30

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('problem_dir', help='Directory containing the problems to process')
parser.add_argument('--results_dir', help='Directory for storing the result', default='data/processed/jjt_sine_1_0/')
parser.add_argument('--sine_tolerance', default=1)
parser.add_argument('--sine_depth', default=0)
parser.add_argument('--no_processes', default=5, type=int, help='Number of processes for running SiNE')

args = parser.parse_args()


def preprocess_sine(path, problem, res_dir):

    # Assume that if the problem contains underscore it is in tff - this doesn't seem to matter anymore
    if '_' in problem:
        mode = 'tclausify'
    else:
        mode = 'clausify'

    # Build SiNE command
    cmd = f'{CLAUSIFIER} -t {TIMELIMIT} --mode {mode} -ss axioms -sd {args.sine_depth} -st {args.sine_tolerance}'
    # Append problem
    cmd += f' {os.path.join(path, problem)}'
    # Append output direction
    cmd += f'  >{os.path.join(res_dir, problem)}'

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


def main():

    # Check problem dir existence
    if not os.path.exists(args.problem_dir):
        print("Error - problem dir does not exist")
        sys.exit(0)

    # Get list of all problems in the directory
    problems = [prob.split('/')[-1] for prob in glob.glob(args.problem_dir + '/*')]
    # Keep only tff and fof problems
    problems = [prob for prob in problems if '_1' in prob or '+' in prob]

    # Check existence of results path
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    # Process the problems
    proc_args = [(args.problem_dir, prob, args.results_dir) for prob in problems]

    # Multiprocess
    pool = multiprocessing.Pool(processes=args.no_processes)
    pool.starmap(preprocess_sine, proc_args)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
    print("Finito")
