"""
Script for running experiment which evaluates different graph embeddings over a single model.
"""
import argparse
import glob
import os
from pathlib import Path
import json
from tqdm import tqdm

import subprocess
from subprocess import check_call
import utils
import config


parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_dir",
    default="experiments/graph_embeddings",
    help="Directory for reporting the embedding results",
)
parser.add_argument(
    "--model_config",
    default="experiments/base_model",
    help="The model configuration used in the training experiments",
)
parser.add_argument(
    "--embedding_dir",
    default="data/embeddings",
    help="The directory containing the embeddings to use for this experiment",
)

# TODO need to set dataset split

# TODO need to get more informative results from the training loop!
# Create a model folder


def launch_training_job_embedding(job_dir, embedding):
    """
    Launch training of a model configuration on the given embedding
    """
    cmd = f"{config.PYTHON} train.py --model_dir {job_dir} --problem_features {embedding}"
    print(cmd)
    check_call(cmd, shell=True, stdout=subprocess.DEVNULL)


def main():
    args = parser.parse_args()

    # Get all embeddings in the embedding folder
    embeddings = glob.glob(os.path.join(args.embedding_dir, "*.pkl"))

    # Read the model config
    with open(os.path.join(args.model_config, "params.json"), "r") as f:
        model_params = json.load(f)

    # For each embedding
    for emb in tqdm(embeddings):
        # Create directory for placing the results and storing the parameters of the model
        job_dir = utils.create_job_dir(args.experiment_dir, Path(emb).stem, params=model_params)

        # Run the job on the directory and embedding
        launch_training_job_embedding(job_dir, emb)

    print("# Finito")


if __name__ == "__main__":
    main()
