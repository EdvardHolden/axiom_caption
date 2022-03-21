"""Peform hyperparemeters search"""

import os
import itertools
import json
import subprocess
from subprocess import check_call
import config
from tqdm import tqdm
import utils

from utils import launch_training_job, get_train_parser


def get_hyperparam_parser():
    """
    This function extends the training parser with the parameters
    required for tuning the hyperparameters.
    """

    # Get the parser for the train script
    parser = get_train_parser()

    # Extend the argument parser
    parser.add_argument(
        "--experiment_dir",
        default="experiments/learning_rate",
        help="Directory for reporting the model experiments",
    )
    parser.add_argument(
        "--parameter_space",
        type=str,
        default="hyperparameter_space/example.json",
        help="Path to json file describing the parameter space",
    )
    parser.add_argument(
        "--rerun",
        default=False,
        action="store_true",
        help="Force rerunning of a config even if the job dir already exists",
    )
    return parser


def main():

    parser, training_parameters = get_hyperparam_parser()
    args = parser.parse_args()

    # Define the hyper-parameter space
    with open(args.parameter_space, "r") as f:
        hp_space = json.load(f)

    # Cartesian product of the parameter space
    hp_parameters = [dict(zip(hp_space, v)) for v in itertools.product(*hp_space.values())]

    # Iterate over each param config
    no_skipped_runs = 0
    for param_config in tqdm(hp_parameters):

        # Make config description
        job_name = "_".join([p + "_" + str(v) for p, v in sorted(param_config.items())])
        print(f"\n### Processing job: {job_name}")

        # If we are not forcing reruns and the jobdir already exists, we skip this configuration
        if not args.rerun and os.path.exists(os.path.join(args.experiment_dir, job_name)):
            no_skipped_runs += 1
            print(f"Skipping rerun of: {job_name}")
            continue

        # Create job dir
        job_dir = utils.create_job_dir(args.experiment_dir, job_name, params=param_config)

        # Update the model directory
        args.model_dir = job_dir

        # Launch job
        launch_training_job(job_dir, args)

    # Report the number of skipped runs if any
    if no_skipped_runs > 0:
        print(f"Skipped a total of {no_skipped_runs} job runs")
    print("Finished")


if __name__ == "__main__":

    main()
