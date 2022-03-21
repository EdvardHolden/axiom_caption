"""
Script for running experiment which evaluates the different axiom orderings over a single model.
"""
import os
import json
from tqdm import tqdm
import utils

from utils import get_train_parser, launch_training_job

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

    return parser


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

        # Update model directory
        args.model_dir = job_dir

        # Run the job on the directory and embedding
        launch_training_job(job_dir, args)

    if no_skipped_runs > 0:
        print(f"Skipped a total of {no_skipped_runs} job runs")

    print("# Finito")
    # """


if __name__ == "__main__":
    main()
