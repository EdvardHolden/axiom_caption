import argparse
import socket
import os

from utils import Context, Mode
import config


def get_train_parser():

    # Get the parser, need to remove 'help' if being used as a parent parser
    parser = argparse.ArgumentParser()

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
        default="axioms",
        help="Axioms for axiom tokenizer mode, and words for natural language (for the tokenizer).",
    )

    return parser


def get_sampler_parser():

    # Get the parser, need to remove 'help' if being used as a parent parser
    parser = argparse.ArgumentParser()

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


def get_evaluate_parser():

    # Get the parser, need to remove 'help' if being used as a parent parser
    parser = get_sampler_parser()

    parser.add_argument(
        "--model_dir", default="experiments/base_model", help="Directory containing params.json"
    )
    parser.add_argument(
        "--problem_ids",
        default=[config.test_id_file],
        nargs="+",
        type=str,
        help="List of files containing IDs for evaluation",
    )

    parser.add_argument(
        "--problem_features", default=config.problem_features, help="File containing the image features"
    )
    parser.add_argument(
        "--proof_data", default=config.proof_data, help="File containing the image descriptions"
    )

    parser.add_argument("-v", "--verbose", action="count", default=0)

    parser.add_argument(
        "--context",
        choices=list(Context),
        type=Context,
        default="axioms",
        help="Axioms for axiom tokenizer mode, and words for natural language (for the tokenizer).",
    )

    parser.add_argument(
        "--tokenizer_id_file",
        default=config.train_id_file,
        help="The ID file used to compute the tokenizer is the training setting",
    )

    return parser


def get_generate_parser():

    parser = get_sampler_parser()
    parser.add_argument(
        "--mode",
        default="caption",
        type=Mode,
        choices=list(Mode),
        help="The mode used to generate the modified DeepMath problems",
    )
    parser.add_argument("--sine_sd", default=None)
    parser.add_argument("--sine_st", default=None)
    parser.add_argument("--result_dir", default=None, help="Root folder for writing generated problems")

    parser.add_argument(
        "--result_prefix", default=None, help="File name prefix of the result dir (if result_dir is not set)"
    )

    if socket.gethostname() == "kontor":
        default_problem_dir = "/home/eholden/gnn-entailment-caption/nndata"
    else:
        default_problem_dir = "/shareddata/home/holden/gnn-entailment-caption/nndata"
    parser.add_argument(
        "--problem_dir",
        default=default_problem_dir,
        help="Directory containing the base problems",
    )
    parser.add_argument(
        "--extra_axioms",
        default=None,
        type=int,
        help="Number of extra axioms to add to each generated problem is set.",
    )

    parser.add_argument(
        "--feature_path",
        default="data/embeddings/deepmath/graph_features_deepmath_all.pkl",
        help="Path to the problem embeddings",
    )
    parser.add_argument(
        "--model_dir",
        default="experiments/hyperparam/initial/attention_False_axiom_order_length_batch_norm_False_dropout_rate_0.1_embedding_size_200_learning_rate_0.001_model_type_merge_inject_no_dense_units_32_no_rnn_units_32_normalize_True_rnn_type_lstm/",
        help="Path to the model used in the captioning modes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(os.cpu_count() - 2, 1),
        help="Number of workers for multiprocessing (used in some modes)",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="Limit generation to 100 instances"
    )
    parser.add_argument(
        "--problem_format",
        default="deepmath",
        choices=["deepmath", "mptp"],
        help="The problem format of the benchmark",
    )

    parser.add_argument(
        "--context",
        choices=list(Context),
        type=Context,
        default="axioms",
        help="Axioms for axiom tokenizer mode, and words for natural language (for the tokenizer).",
    )

    return parser


def get_embedding_exp_parser():
    """
    This function extends the training parser with the parameters
    required for the embedding experiments. This is to make it easier
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
        default="experiments/graph_embeddings",
        help="Directory for reporting the embedding results",
    )
    parser.add_argument(
        "--embedding_dir",
        default="data/embeddings",
        help="The directory containing the embeddings to use for this experiment",
    )
    parser.add_argument(
        "--rerun",
        default=False,
        action="store_true",
        help="Force rerunning of a config even if the job dir already exists",
    )

    return parser


def get_hyperparam_parser():
    """
    This function extends the training parser with the parameters
    required for tuning the hyperparameters.
    """

    # Get the parser for the train script
    parser = get_train_parser()

    # Extend the argument parser
    parser.add_argument(
        "--experiment_dir",
        default="experiments/learning_rate",
        help="Directory for reporting the model experiments",
    )
    parser.add_argument(
        "--parameter_space",
        type=str,
        default="hyperparameter_space/example.json",
        help="Path to json file describing the parameter space",
    )
    parser.add_argument(
        "--rerun",
        default=False,
        action="store_true",
        help="Force rerunning of a config even if the job dir already exists",
    )
    return parser


def get_score_parser():

    parser = get_sampler_parser()

    parser.add_argument("--sine_sd", type=int, nargs="+", default=None)
    parser.add_argument("--sine_st", type=float, nargs="+", default=None)

    parser.add_argument(
        "--model_dir", default=None, help="Path to sequence model, used if including predictions"
    )
    parser.add_argument(
        "--include_rare_axioms", default=False, action="store_true", help="Include rare axioms"
    )
    parser.add_argument(
        "--feature_path",
        default=config.problem_features,
        help="Path to the problem embeddings used with the captioning model",
    )
    parser.add_argument(
        "--number_of_samples",
        type=int,
        default=None,
        help="Number of samples to use for computing the score (None for all)",
    )
    parser.add_argument(
        "--problem_format",
        default="deepmath",
        choices=["deepmath", "mptp"],
        help="The problem format of the benchmark",
    )

    parser.add_argument(
        "--problem_dir",
        default=None,
        help="The directory of the problems processed (typically by SInE)",
    )

    return parser


def get_compute_tokenizer_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id_file",
        default=config.train_id_file,
        help="File containing the ids used to construct the tokenizer",
    )
    parser.add_argument(
        "--tokenizer_data_path",
        default=config.proof_data,
        help="File containing the proof|conjecture|text data",
    )
    parser.add_argument(
        "--vocab_word_limit",
        default=None,
        type=int,
        help="Number of top K words to include in the vocabulary. None for all words",
    )
    parser.add_argument(
        "--tokenizer_mode",
        default="axioms",
        choices=["axioms", "words", "conj_char", "conj_word"],
        help="Set preprocessing based on natural language, conjecture or axioms",
    )
    return parser


def get_plot_result_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Directory containing the history.pkl of interest")
    parser.add_argument(
        "--metric",
        default="loss",
        choices=["loss", "l", "coverage", "c", "jaccard", "j"],
        help="The metric to plot",
    )
    return parser


def main():
    print("Calling all parser methods for testing")
    get_train_parser()
    get_sampler_parser()
    get_order_exp_parser()
    get_evaluate_parser()
    get_generate_parser()
    get_embedding_exp_parser()
    get_hyperparam_parser()
    get_score_parser()
    get_compute_tokenizer_parser()


if __name__ == "__main__":
    main()
