"""
Script for running experiment which evaluates different graph embeddings over a single model.
"""
import glob
import os
from pathlib import Path
import json
from tqdm import tqdm
import utils

from utils import launch_training_job
from parser import get_embedding_exp_parser


def main():
    parser = get_embedding_exp_parser()
    args = parser.parse_args()

    # Get all embeddings in the embedding folder
    embeddings = sorted(glob.glob(os.path.join(args.embedding_dir, "*.pkl")), reverse=True)
    if len(embeddings) == 0:
        print("Warning: Could not find any embeddings in the provided directory")

    # Read the model config
    with open(os.path.join(args.experiment_dir, "params.json"), "r") as f:
        model_params = json.load(f)

    # For each embedding
    no_skipped_runs = 0
    for emb in tqdm(embeddings):
        print(f"Running experiment for: {emb}")

        # If we are not forcing reruns and the jobdir already exists, we skip this configuration
        if not args.rerun and os.path.exists(os.path.join(args.experiment_dir, Path(emb).stem)):
            no_skipped_runs += 1
            print(f"Skipping rerun of: {Path(emb).stem}")
            continue

        # Create directory for placing the results and storing the parameters of the model
        job_dir = utils.create_job_dir(args.experiment_dir, Path(emb).stem, params=model_params)

        # Update model dir
        args.model_dir = job_dir

        # Update embedding file
        args.problem_features = emb

        # Run the job on the directory and embedding
        launch_training_job(job_dir, args)

    # Report the number of skipped runs if any
    if no_skipped_runs > 0:
        print(f"Skipped a total of {no_skipped_runs} job runs")

    print("# Finito")


if __name__ == "__main__":
    main()
