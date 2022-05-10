import os
import json
import config
from subprocess import check_call
import argparse
from argparse import Namespace
from enum import Enum


class Context(Enum):
    PROOF = "proof"
    FLICKR = "flickr"

    def __str__(self):
        return self.value


class EncoderInput(Enum):
    SEQUENCE = "sequence"
    FLAT = "flat"

    def __str__(self):
        return self.value


# Set axiom order type
class AxiomOrder(Enum):
    ORIGINAL = "original"
    LEXICOGRAPHIC = "lexicographic"
    LENGTH = "length"
    FREQUENCY = "frequency"
    RANDOM = "random"
    RANDOM_GLOBAL = "random_global"

    def __str__(self):
        return self.value


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
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help="Path to the tokenizer. Uses tokenizer.json in directory of the train id if not specified",
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

    parser.add_argument(
        "--encoder_input",
        default="flat",
        choices=["flat", "sequence"],
        help="Changes between flat (normal) entity inputs and sequence (conjecture) input to the encoder",
    )
    parser.add_argument(
        "--conjecture_tokenizer",
        default=None,
        help="The path to the conjecture tokenizer. Only used if encoder_input is set to sequence",
    )

    # Model options
    parser.add_argument("--model_dir", default=config.base_model, help="Directory containing params.json")

    parser.add_argument(
        "--working_dir",
        default=None,
        help="Directory for saving ckp, model and history. Same as model_dir if not set.",
    )

    # FIXME this might not remove that much memory load due to the checkpoints
    parser.add_argument(
        "--save_model", default=False, action="store_true", help="Set if final model should be saved"
    )

    parser.add_argument(
        "--context",
        choices=list(Context),
        type=Context,
        default="proof",
        help="Alternate context between Flickr8 and Proof datasets",
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

    # FIXME could make this more compact
    # Add all other remaining training parameters
    for param in default_parameters:
        # Cannot handle flags so only set them if true
        if param == "remove_unknown":
            if args.__dict__[param]:  # if set to true
                cmd += " --remove_unknown "
        elif param == "save_model":
            if args.__dict__[param]:  # if set to true
                cmd += " --save_model "
        # Cannot handle None values so only set if not None
        elif param == "tokenizer_path":
            if args.__dict__[param] is not None:
                cmd += f" --{param} {args.__dict__[param]} "
        elif param == "conjecture_tokenizer":
            if args.__dict__[param] is not None:
                cmd += f" --{param} {args.__dict__[param]} "
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
