import os
import json
from subprocess import check_call
from argparse import Namespace

import config
from parser import get_train_parser


def create_job_dir(root_dir, job_name, params=None):

    # Create a new folder in parent_dir with unique_name "job_name"
    job_dir = os.path.join(root_dir, job_name)
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    # Write parameters in json file
    if params is not None:
        json_path = os.path.join(job_dir, "params.json")
        with open(json_path, "w") as f:
            json.dump(params, f)

    return job_dir


def launch_training_job(job_dir: str, args: Namespace) -> None:
    """
    Launch training of a model given the direcotry path containing configration file
    and a namesapce containing the training parameters.
    """

    # The base command
    cmd = f"{config.PYTHON} train.py "

    # Get the default parameters from the training script
    default_parameters = sorted(get_train_parser().parse_args([]).__dict__.keys())

    # Add all other remaining training parameters
    for param in default_parameters:
        if param == "save_model":
            if args.__dict__[param]:  # if set to true
                cmd += " --save_model "
        # Cannot handle None values so only set if not None
        elif param == "working_dir":
            if args.__dict__[param] is not None:
                cmd += f" --{param} {args.__dict__[param]} "
        else:
            # Include option and value
            cmd += f" --{param} {args.__dict__[param]} "

    print(cmd)
    check_call(cmd, shell=True, stdout=None)


# Use as a decorator to debug functions
def debug(func):
    def _debug(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"{func.__name__}(args: {args}, kwargs: {kwargs}) -> {result}")
        return result

    return _debug
