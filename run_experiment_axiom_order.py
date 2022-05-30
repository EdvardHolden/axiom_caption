"""
Script for running experiment which evaluates the different axiom orderings over a single model.
"""
import os
import json
from tqdm import tqdm
import utils

from utils import launch_training_job
from parser import get_order_exp_parser

AXIOM_ORDERS = ["original", "lexicographic", "length", "random", "frequency", "random_global"]


def main():

    parser = get_order_exp_parser()
    args = parser.parse_args()

    # Read the model config
    with open(os.path.join(args.experiment_dir, "params.json"), "r") as f:
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


if __name__ == "__main__":
    main()
