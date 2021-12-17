"""Peform hyperparemeters search"""

import argparse
import itertools
import json
import os
import subprocess
from subprocess import check_call
import sys
from tqdm import tqdm


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument(
    "--parent_dir", default="experiments/learning_rate", help="Directory for reporting the model experiments"
)
parser.add_argument(
    "--parameter_space",
    type=str,
    default="hyperparameter_space/example.json",
    help="Path to json file describing the parameter space",
)


def launch_training_job(parent_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        parent_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, "params.json")
    with open(json_path, "w") as f:
        json.dump(params, f)

    # Launch training with this config
    cmd = "{python} train.py --model_dir {model_dir}".format(python=PYTHON, model_dir=model_dir)
    print(cmd)
    check_call(cmd, shell=True, stdout=subprocess.DEVNULL)


def main():
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()

    # Define the hyper-parameter space
    with open(args.parameter_space, "r") as f:
        hp_space = json.load(f)

    # Cartesian product of the parameter space
    hp_parameters = [dict(zip(hp_space, v)) for v in itertools.product(*hp_space.values())]

    # Iterate over each param config
    for param_config in tqdm(hp_parameters):

        # Make config description
        r = sorted(param_config.items())
        job_name = "_".join([p + "_" + str(v) for p, v in r])
        print(job_name)

        # Launch job
        launch_training_job(args.parent_dir, job_name, param_config)


if __name__ == "__main__":

    main()
