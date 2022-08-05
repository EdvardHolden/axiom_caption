import os
import json
from subprocess import check_call
from argparse import Namespace
import tensorflow as tf

import config
from parser import get_train_parser


class NameSpace:
    # TODO a bit dangerous with such similar names to argparse.Namespace
    # but this is mostly used to testing when implementing.
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@tf.function
def get_initial_decoder_input(tokenizer, target, sequence=False):
    """
    Returns the decoder input consisting of the start token.
    Sequence is set to true if using e.g. TransformerDecoder where
    we need to supply the full sequence predicted.
    """

    # Make list of start tokens
    dec_input = [tokenizer.word_index[config.TOKEN_START]] * target.shape[0]

    if sequence:
        # TODO why does this work in the guide but not for me?
        input_array = tf.TensorArray(dtype=tf.int32, size=target.shape[1])
        # input_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        return input_array.write(0, dec_input)
    else:
        # Need to expand the dimensions when feeding single tokens
        return tf.expand_dims(dec_input, 1)


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
