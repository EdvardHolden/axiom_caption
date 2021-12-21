import config
import argparse
from nltk.translate.bleu_score import corpus_bleu
import os
from itertools import starmap
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from model import get_model_params

from dataset import get_dataset, get_tokenizer, compute_max_length
from model import load_model


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="experiments/base_model", help="Directory containing params.json")
parser.add_argument("--problem_ids", default=config.test_id_file, help="File containing IDs for evaluation")

parser.add_argument(
    "--problem_features", default=config.problem_features, help="File containing the image features"
)
parser.add_argument("--proof_data", default=config.proof_data, help="File containing the image descriptions")

# Sampling options
parser.add_argument(
    "--sampler",
    default="greedy",
    choices=["greedy", "temperature", "top_k"],
    help="The method used to sample the next word in the prediction",
)
parser.add_argument(
    "--no_samples",
    default=1,
    type=int,
    help="The number of samples to draw at each iteration (only on is passed to the model)",
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

parser.add_argument("--max_length", default=None, type=int, help="The maximum length of the predictions")
parser.add_argument("-v", "--verbose", action="count", default=0)


# TODO maybe add some BeamSearch in here?


def greedy_sampler(pred, no_samples):

    # Get the top K predictions (negate values as argsort is ascending)
    return np.argsort(-pred)[0][0:no_samples]


def temperature_sampler(pred, no_samples, temp=1.0):

    # Scale numbers by the temperature
    pred = np.asarray(softmax(pred[0])).astype("float64")
    # pred = np.asarray(pred).astype("float64")
    conditional_probability = np.log(pred) / temp
    # Compute the softmax of the new distribution
    exp_preds = np.exp(conditional_probability)
    conditional_probability = exp_preds / np.sum(exp_preds)
    # Perform no_samples picking experiments from the new distribution
    prob = np.random.multinomial(no_samples, conditional_probability, 1)

    # Greedily get the top distributed samples
    return greedy_sampler(prob, no_samples)


def softmax(z):
    return np.exp(z) / sum(np.exp(z))


def top_k_sampler(pred, no_samples, k=10):

    # Get the k-top words
    top_k_probabilities, top_k_indices = tf.math.top_k(softmax(pred[0]), k=k, sorted=True)
    # Compute the softmax of the top indices
    top_k_indices = np.asarray(top_k_indices).astype("int32")
    top_k_redistributed_probability = softmax(np.log(top_k_probabilities))
    top_k_redistributed_probability = np.asarray(top_k_redistributed_probability).astype("float32")

    # Sample no_samples without replacement
    sampled_token = np.random.choice(
        top_k_indices, size=no_samples, replace=False, p=top_k_redistributed_probability
    )

    return sampled_token


def coverage_score(actual, predicted, avg=True):

    scores = [*starmap(lambda a, p: len(set(a).intersection(set(p))) / len(set(a)), zip(actual, predicted))]
    if avg:
        return np.average(scores)
    else:
        return scores


def jaccard_score(actual, predicted, avg=True):
    """
    Jaccard(A,B) = |A/\B|/|A\/B|
    """
    scores = [
        *starmap(
            lambda a, p: len(set(a).intersection(set(p))) / len(set(a).union(set(p))), zip(actual, predicted)
        )
    ]
    if avg:
        return np.average(scores)
    else:
        return scores


def generate_step(
    tokenizer, model, max_len, img_tensor, sampler, no_samples, sampler_temperature, sampler_top_k
):

    # List for storing predicted sequence
    result = []

    # Reset LSTM states between each batch
    model.reset_states()

    # Get start token
    dec_input = tf.expand_dims([tokenizer.word_index[config.TOKEN_START]], 0)

    # Run the model until we reach the max length or the end token
    for i in range(max_len):
        # Predict probabilities
        pred = model([img_tensor, dec_input])

        # Sample the next token(s)
        if sampler == "greedy":
            predictions = greedy_sampler(pred, no_samples)
        elif sampler == "temperature":
            predictions = temperature_sampler(pred, no_samples, temp=sampler_temperature)
        elif sampler == "top_k":
            predictions = top_k_sampler(pred, no_samples, k=sampler_top_k)
        else:
            raise ValueError(f"Unrecognised sampler '{sampler}'")

        # Return sequence if we predicted the end token
        if tokenizer.index_word[predictions[0]] == config.TOKEN_END:
            return result

        # Add predicted IDs to the result
        for pred_id in predictions:
            result.append(tokenizer.index_word[pred_id])

        # Set the top predicted word as the next model input
        dec_input = tf.expand_dims([predictions[0]], 0)

    # Reached max length, returning the sequence
    return result


def evaluate_model(
    tokenizer, model, test_data, max_len, sampler, no_samples, sampler_temperature, sampler_top_k, verbose=0
):

    # Initialise results
    actual, predicted = list(), list()

    # Predict for each problem in the data
    for (_, (img_tensor, caption)) in tqdm(enumerate(test_data)):
        # Generate caption based on the image tensor
        yhat = generate_step(
            tokenizer, model, max_len, img_tensor, sampler, no_samples, sampler_temperature, sampler_top_k
        )

        # Extract the string value from the tensor, remove start/end tokens,
        # convert to utf-8 and make into array
        caption = caption.numpy()[0].split(bytes(config.TOKEN_DELIMITER, "utf-8"))[1:-1]

        # Store the actual token
        actual.append(caption)
        predicted.append(yhat)

        if verbose:
            print("Actual:    %s" % " ".join(caption))
            print("Predicted: %s" % " ".join(yhat))

    # Calculate BLEU score
    bleu = corpus_bleu(actual, predicted)
    # Compute the set coverage
    coverage = coverage_score(actual, predicted)

    return {"bleu": bleu, "coverage": coverage}


def main():

    # Get the arguments
    args = parser.parse_args()

    # Get pre-trained tokenizer
    tokenizer_path = os.path.join(os.path.dirname(args.problem_ids), "tokenizer.json")
    tokenizer, _ = get_tokenizer(tokenizer_path)

    # If maximum length is not provided, we compute it based on the training set in config
    if args.max_length is None:
        max_len = compute_max_length(config.train_id_file, args.proof_data)
    else:
        max_len = args.max_length
    print("Max caption length: ", max_len)

    # Get the axiom ordering from the model parameter file
    axiom_order = get_model_params("experiments/base_model").axiom_order
    print("Using axiom order: ", axiom_order)

    # Get the test dataset with batch 1 as we need to treat each caption separately
    # Also, we want the raw text so not providing a tokenizer
    test_data, _ = get_dataset(
        args.problem_ids, args.proof_data, args.problem_features, batch_size=1, order=axiom_order
    )

    # Load model
    model_dir = os.path.join(args.model_dir, "ckpt_dir")
    loaded_model = load_model(model_dir)

    # Run evaluation
    scores = evaluate_model(
        tokenizer,
        loaded_model,
        test_data,
        max_len,
        args.sampler,
        args.no_samples,
        args.sampler_temperature,
        args.sampler_top_k,
        verbose=args.verbose,
    )
    print("# Scores ")
    for score in sorted(scores):
        print(f"{score:<8}: {scores[score]:.2f}")


if __name__ == "__main__":
    main()
