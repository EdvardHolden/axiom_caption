import os
import json
import config
from subprocess import check_call
import argparse
from argparse import Namespace


def get_train_parser(add_help=True):

    # Get the parser, need to remove 'help' if being used as a parent parser
    parser = argparse.ArgumentParser(add_help=add_help)

    # Dataset ID options
    parser.add_argument(
        "--train_id_file", default=config.train_id_file, help="File containing the training ids"
    )
    parser.add_argument(
        "--val_id_file", default=config.val_id_file, help="File containing the validation ids"
    )

    # Feature options
    parser.add_argument("--proof_data", default=config.proof_data, help="File containing the image features")
    parser.add_argument(
        "--problem_features", default=config.problem_features, help="File containing the image descriptions"
    )
    parser.add_argument(
        "--remove_unknown",
        action="store_true",
        default=False,
        help="Remove tokens mapped to oov. Can reduce the number of samples.",
    )

    # Model options
    parser.add_argument("--model_dir", default=config.base_model, help="Directory containing params.json")

    # FIXME this might not remove that much memory load due to the checkpoints
    parser.add_argument(
        "--save_model", default=False, action="store_true", help="Set if final model should be saved"
    )

    return parser


def get_sampler_parser(add_help=True):

    # Get the parser, need to remove 'help' if being used as a parent parser
    parser = argparse.ArgumentParser(add_help=add_help)

    # Sampling options
    parser.add_argument(
        "--sampler",
        default="greedy",
        choices=["greedy", "temperature", "top_k"],
        help="The method used to sample the next word in the prediction",
    )
    parser.add_argument(
        "--no_samples",
        default=[1],
        type=int,
        nargs="+",
        help="The number of samples to draw at each iteration (only one is passed to the model)",
    )
    parser.add_argument(
        "--sampler_temperature",
        default=1.0,
        type=float,
        help="The temperature when using the temperature sampler (0, 1]",
    )
    parser.add_argument(
        "--sampler_top_k",
        default=10,
        type=int,
        help="The top k predictions to use when recomputing the prediction distributions",
    )

    parser.add_argument("--max_length", default=22, type=int, help="The maximum length of the predictions")

    return parser


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
        # Cannot handle flags so only set them if true
        if param == "remove_unknown":
            if args.__dict__[param]:  # if set to true
                cmd += " --remove_unknown "
        elif param == "save_model":
            if args.__dict__[param]:  # if set to true
                cmd += " --save_model "
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
