"""
Script for running experiment which evaluates the different axiom orderings over a single model.
"""
import glob
import os
from pathlib import Path
import json
from tqdm import tqdm
import subprocess
from subprocess import check_call
import utils
import config

from train import get_train_parser

AXIOM_ORDERS = ["original", "lexicographic", "length", "random", "frequency"]


def get_order_exp_parser():
    """
    This function extends the training parser with the parameters
    required for the order experiments. This is to make it easier
    to change parameters such as dataset IDs and proof data between
    different experiments.

    It also return the set of parameters for the training script as
    these are useful when building the cmd for running the training job.
    """

    # Get the parser for the train script
    parser = get_train_parser()

    # Extract and store its original options for later use
    train_parameters = sorted(parser.parse_args([]).__dict__.keys())

    # Extend the argument parser
    parser.add_argument(
        "--experiment_dir",
        default="experiments/axiom_order",
        help="Directory for reporting the embedding results",
    )

    parser.add_argument(
        "--rerun",
        default=False,
        action="store_true",
        help="Force rerunning of a config even if the job dir already exists",
    )

    return parser, train_parameters


def launch_training_job_embedding(job_dir, args, training_parameters):
    """
    Launch training of a model configuration on the params file containing the order
    """

    # The initial training cmd
    cmd = f"{config.PYTHON} train.py --model_dir {job_dir}"

    # Add all other remaining training parameters
    for param in training_parameters:
        cmd += f" --{param} {args.__dict__[param]} "

    check_call(cmd, shell=True, stdout=subprocess.DEVNULL)


def main():

    parser, training_parameters = get_order_exp_parser()
    args = parser.parse_args()

    # Read the model config
    with open(os.path.join(args.model_dir, "params.json"), "r") as f:
        model_params = json.load(f)

    # For each embedding
    no_skipped_runs = 0
    for order in tqdm(AXIOM_ORDERS):
        print(f"Running experiment for order: {order}")

        # Update the model parameters with the model
        model_params["axiom_order"] = order

        # If we are not forcing reruns and the jobdir already exists, we skip this configuration
        if not args.rerun and os.path.exists(os.path.join(args.experiment_dir, order)):
            no_skipped_runs += 1
            print(f"Skipping rerun of: {order}")
            continue

        # Create directory for placing the results and storing the parameters of the model
        job_dir = utils.create_job_dir(args.experiment_dir, order, params=model_params)

        # Run the job on the directory and embedding
        launch_training_job_embedding(job_dir, args, training_parameters)

    if no_skipped_runs > 0:
        print(f"Skipped a total of {no_skipped_runs} job runs")

    print("# Finito")
    # """


if __name__ == "__main__":
    main()
