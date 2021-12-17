"""Peform hyperparemeters search"""

import argparse
import itertools
import json
import subprocess
from subprocess import check_call
import config
from tqdm import tqdm
import utils


# TODO should include options for including stuff about which embedding/dataset split to use here
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


# FIXME maybe this could be refactored?
def launch_training_job(job_dir):
    """
    Launch training of the model with a set of hyperparameters in parent_dir/job_name
    """
    cmd = f"{config.PYTHON} train.py --model_dir {job_dir}"
    print(cmd)
    check_call(cmd, shell=True, stdout=subprocess.DEVNULL)


def main():

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

        # Create job dir
        job_dir = utils.create_job_dir(args.parent_dir, job_name, params=param_config)

        # Launch job
        launch_training_job(job_dir)


if __name__ == "__main__":

    main()
