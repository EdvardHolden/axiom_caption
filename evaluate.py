import config
import os
from itertools import starmap
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
from pathlib import Path
from tensorflow.sets import size, intersection, union

from dataset import get_dataset, compute_max_caption_length, get_caption_conjecture_tokenizers
from model import (
    get_model_params,
    load_model,
    reset_model_decoder_state,
    initialise_model,
    call_encoder,
    call_model_decoder,
    get_hidden_state,
    decoder_sequence_input,
)
from parser import get_evaluate_parser
from utils import get_initial_decoder_input


os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


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


@tf.function
def coverage_score(actual, predicted, avg=True):

    actual = tf.sparse.from_dense(actual)
    predicted = tf.sparse.from_dense(predicted)

    scores = size(intersection(actual, predicted)) / size(actual)
    if avg:
        return np.average(scores)
    else:
        return scores


@tf.function
def jaccard_score(actual, predicted, avg=True):
    """
    Jaccard(A,B) = |A/\B|/|A\/B|
    """

    actual = tf.sparse.from_dense(actual)
    predicted = tf.sparse.from_dense(predicted)

    scores = size(intersection(actual, predicted)) / size(union(actual, predicted))
    if avg:
        return np.average(scores)
    else:
        return scores


def coverage_score_np(actual, predicted, avg=True):

    scores = [*starmap(lambda a, p: len(set(a).intersection(set(p))) / len(set(a)), zip(actual, predicted))]
    if avg:
        return np.average(scores)
    else:
        return scores


def jaccard_score_np(actual, predicted, avg=True):
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


# @tf.function
def generate_step(
    tokenizer, model, max_len, img_tensor, caption, sampler, no_samples, sampler_temperature, sampler_top_k
):

    # List for storing predicted sequence
    result = set()

    # Reset LSTM states between each batch
    reset_model_decoder_state(model)

    # Check whether decoder input should be a sequence
    sequence = decoder_sequence_input(model)

    # Initialise the hidden shape of the model - makes the above lines redundant
    hidden = get_hidden_state(model, caption.shape[0])

    # Get start token - supply dummy for computing target shapes
    dec_input = get_initial_decoder_input(tokenizer, tf.random.uniform([1, max_len]), sequence=sequence)

    # Placeholder for mask if not used
    input_mask = None

    # Call the encoder to pre-compute the entity features for use in the decoder call
    img_tensor, input_mask, hidden = call_encoder(model, img_tensor, False, input_mask, hidden)

    # Run the model until we reach the max length or the end token
    for i in range(max_len):

        # Call the decoder/model to produce the final predictions
        pred, hidden = call_model_decoder(model, img_tensor, dec_input, input_mask, hidden, training=False)

        # Sample the next token(s)
        if sampler == "greedy":
            predictions = greedy_sampler(pred, no_samples)
        elif sampler == "temperature":
            predictions = temperature_sampler(pred, no_samples, temp=sampler_temperature)
        elif sampler == "top_k":
            predictions = top_k_sampler(pred, no_samples, k=sampler_top_k)
        else:
            raise ValueError(f"Unrecognised sampler '{sampler}'")

        # Add predicted IDs to the result
        result.update(predictions)

        # Return sequence if we predicted the end token
        if tokenizer.index_word[predictions[0]] == tokenizer.word_index[config.TOKEN_END]:
            return result

        # Set the top predicted word as the next model input
        next_token = predictions[0]
        if sequence:
            dec_input = dec_input.write(i + 1, next_token)
        else:
            dec_input = tf.expand_dims([next_token], 0)

    if isinstance(dec_input, tf.TensorArray):
        dec_input.close()

    # Reached max length, returning the sequence
    return result


def evaluate_model(
    tokenizer, model, test_data, max_len, sampler, no_samples, sampler_temperature, sampler_top_k, verbose=0
):

    # Create lambda expression for filtering start, end, and pad tokens
    token_filter = lambda x: x != 0 and x != 2 and x != 3
    # Initialise results
    actual, predicted = list(), list()

    # Predict for each problem in the data
    for (_, (img_tensor, caption)) in tqdm(enumerate(test_data)):
        # Generate caption based on the image tensor
        yhat = generate_step(
            tokenizer,
            model,
            max_len,
            img_tensor,
            caption,
            sampler,
            no_samples,
            sampler_temperature,
            sampler_top_k,
        )

        # Extract the string value from the tensor, remove start, end and pad tokens,
        actual.append(list(filter(token_filter, caption[0].numpy())))
        predicted.append(list(filter(token_filter, yhat)))

        if verbose:
            print("Actual:    %s" % " ".join(caption))
            print("Predicted: %s" % " ".join(yhat))

    # Compute decoding metrics
    coverage = coverage_score_np(actual, predicted)
    jaccard = jaccard_score_np(actual, predicted)
    avg_size = np.mean([len(set(p)) for p in predicted])

    return {"coverage": coverage, "jaccard": jaccard, "avg_size": avg_size}


def get_model(model_dir, vocab_size):

    # Load the model parameters
    model_params = get_model_params(model_dir)

    # Load the checkpointed model
    ckpt_dir = os.path.join(model_dir, "ckpt_dir")
    if model_params.model_type == "inject_decoder":
        encoder = load_model(os.path.join(ckpt_dir, "encoder"))
        decoder = load_model(os.path.join(ckpt_dir, "decoder"))
        loaded_model = (encoder, decoder)
    else:
        loaded_model = load_model(ckpt_dir)

    # As stateful=True might be set, we need to get a new fresh model with the weights of the loaded
    # model. This is to use a different batch size in the evaluation instead of the training batch size.
    model = get_new_trained_model(loaded_model, model_params, vocab_size)
    return model


def get_new_trained_model(trained_model, model_params, vocab_size):

    # Loading the dataset (without tokenizer) as dummy data stopped working
    normalisation_data, _ = get_dataset(
        config.val_id_file, config.proof_data, config.problem_features, batch_size=1
    )

    model = initialise_model(
        model_params.model_type, vocab_size, model_params, training_data=normalisation_data
    )  # Loaded weights will override this

    # Run the model call once to infer the main input shapes? - fails otherwise
    inp1 = tf.random.uniform([1, 400])
    inp2 = tf.ones([1, 1], dtype=tf.dtypes.int32)
    hidden = tf.zeros((1, model_params.no_rnn_units))

    # Run model to infer input shapes
    if isinstance(model, tuple):
        inp1 = model[0](inp1)
        model[1]([inp1, inp2, hidden])
    else:
        model([inp1, inp2, hidden])

    # Set the weights of the new model with the weights of the old model
    if isinstance(model, tuple):
        model[0].set_weights(trained_model[0].get_weights())
        model[1].set_weights(trained_model[1].get_weights())

    else:
        model.set_weights(trained_model.get_weights())

    return model


def main(
    model_dir,
    proof_data,
    problem_features,
    problem_ids,
    max_length,
    sampler,
    no_samples,
    sampler_temperature,
    sampler_top_k,
    context,
    tokenizer_id_file,
    verbose,
):

    # Get the arguments
    print(
        f"Sampler configuration: (sampler: {sampler}) (no_samples: {no_samples}) (sampler_temperature: {sampler_temperature}) (sampler_top_k: {sampler_top_k})"
    )

    # Get the axiom ordering from the model parameter file
    model_params = get_model_params(model_dir)
    axiom_order = None  # We do not care about the order in this case as all metrics are based on the set
    print("Using axiom order: ", axiom_order)

    # Load the tokenizers for this evaluation setting
    caption_tokenizer, vocab_size, conjecture_tokenizer = get_caption_conjecture_tokenizers(
        model_params, proof_data, context, tokenizer_id_file, problem_features
    )

    # If maximum length is not provided, we compute it based on the training set in config
    if max_length is None:
        max_len = compute_max_caption_length(config.train_id_file, proof_data)
    else:
        max_len = max_length
    print("Max caption length: ", max_len)

    model = get_model(model_dir, vocab_size)
    print("Evaluating on model: ", model)

    # Get the test dataset with batch 1 as we need to treat each caption separately
    # Also, we want the raw text so not providing a tokenizer
    for n in no_samples:
        print(f"### No samples: {n}")
        for ids in problem_ids:

            print(f"## Evaluating on: {Path(ids).stem}")

            # Get ground truth
            test_data, _ = get_dataset(
                ids,
                proof_data,
                problem_features,
                batch_size=1,
                caption_tokenizer=caption_tokenizer,
                # order=model_params.axiom_order,
                order=None,  # We do not need an order for this as we treating it as a set for evaluation
                # axiom_frequency=axiom_frequency,
                remove_unknown=model_params.remove_unknown,
                encoder_input=model_params.encoder_input,
                conjecture_tokenizer=conjecture_tokenizer,
            )

            # Run evaluation
            scores = evaluate_model(
                caption_tokenizer,
                model,
                test_data,
                max_len,
                sampler,
                n,
                sampler_temperature,
                sampler_top_k,
                verbose=verbose,
            )
            print("# Scores ")
            for score in sorted(scores):
                print(f"{score:<8}: {scores[score]:.2f}")
            print()

    return scores


def run_main():

    # Get the parser and parse the arguments
    parser = get_evaluate_parser()
    args = parser.parse_args()

    # Run main with the arguments
    main(**vars(args))


if __name__ == "__main__":
    run_main()
    # main()
    print("# Finished")
