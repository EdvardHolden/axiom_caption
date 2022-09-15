import config
import os
from itertools import starmap
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
from pathlib import Path
from tensorflow.sets import size, intersection, union
from sklearn.neighbors import NearestNeighbors

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
from enum_types import ModelType, EncoderType, DecoderType
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


def remap_predictions_to_problem_axioms(model, predictions, caption, tokenizer):
    """
    Remap predicted axioms to the axioms conisisting in the original problem.

    In some cases we do not want to introduce 'new' axioms into the problem.
    Instead, we select the most similar axioms contained within the problem
    based on their encoding by the axiom decoding layer. This means that we
    are limited to the axiom that consists on both the problem and vocabulary.

    We use the given axiom to avoid re-reading the problem.
    """

    # Extract the axiom embedding layer of the decoder part
    if isinstance(model, tuple):
        emb_layer = model[1].layers[0]
    else:
        raise ValueError("Remapping not yet supported for non-split models")


    # Extract values and ensure there are no pad/start/unk tokens - map back to ndarray
    caption = [c for c in caption.numpy()[0] if not (c == tokenizer.word_index[config.TOKEN_PAD]
                                                     or c == tokenizer.word_index[config.TOKEN_START]
                                                     or c == tokenizer.word_index[config.TOKEN_OOV]
                                                     or c == tokenizer.word_index[config.TOKEN_END])]
    caption = np.array(caption)

    # Check if there are any selectable axioms, otherwise, return the original predictions
    if len(caption) == 0:
        return predictions

    cap_embedding = emb_layer(caption)
    pred_embedding = emb_layer(predictions)

    # Create nearest neighbour model and predict
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(cap_embedding)
    distances, indices = nbrs.kneighbors(pred_embedding)

    # Re-assign predictions to mapped indices
    for n, i in enumerate(indices):
        predictions[n] = caption[i]

    return predictions


# @tf.function
def generate_step(
    tokenizer, model, max_len, img_tensor, caption, sampler, no_samples, sampler_temperature, sampler_top_k, axiom_remapping
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
    dec_input = get_initial_decoder_input(tokenizer, tf.random.uniform([1, max_len + 1]), sequence=sequence)

    # Placeholder for mask if not used
    input_mask = None

    # Call the encoder to pre-compute the entity features for use in the decoder call
    img_tensor, input_mask, hidden = call_encoder(model, img_tensor, False, hidden)

    # Run the model until we reach the max length or the end token
    for i in range(1, max_len + 1):

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

        # Map predicted axioms back to the problem axioms if set
        if axiom_remapping:
            # Quick hack to keep the end token if predicted
            if tokenizer.index_word[predictions[0]] == config.TOKEN_END:
                pred_prefix = predictions[0]
            else:
                pred_prefix = []

            # Perform axiom remapping
            predictions = remap_predictions_to_problem_axioms(model, predictions, caption, tokenizer)

            # Prepend pred_prefix
            predictions = np.insert(predictions,0, pred_prefix)

        # Add predicted IDs to the result
        result.update(predictions)

        # Return sequence if we predicted the end token as the first token
        if tokenizer.index_word[predictions[0]] == config.TOKEN_END:
            return result

        # Set the top predicted word as the next model input
        next_token = predictions[0]
        if sequence:
            new_dec_input = tf.TensorArray(
                dtype=tf.int32, size=dec_input.shape[0], name="decoder_sequence_input"
            ).unstack(dec_input)
            dec_input = new_dec_input.write(i, [next_token]).stack()
        else:
            dec_input = tf.expand_dims([next_token], 0)

    if isinstance(dec_input, tf.TensorArray):
        dec_input.close()

    # Reached max length, returning the sequence
    return result


def evaluate_model(
    tokenizer, model, test_data, max_len, sampler, no_samples, sampler_temperature, sampler_top_k, axiom_remapping, verbose=0
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
            axiom_remapping
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


def get_model(model_dir, max_caption_length=None):

    # Load the model parameters
    model_params = get_model_params(model_dir)
    model_params.max_caption_length = max_caption_length  # Required for transformer decoder

    # Load the checkpointed model
    ckpt_dir = os.path.join(model_dir, "ckpt_dir")
    if model_params.model_type is ModelType.SPLIT:
        encoder = load_model(os.path.join(ckpt_dir, "encoder"))
        decoder = load_model(os.path.join(ckpt_dir, "decoder"))
        loaded_model = (encoder, decoder)
    else:
        loaded_model = load_model(ckpt_dir)

    # As stateful=True might be set, we need to get a new fresh model with the weights of the loaded
    # model. This is to use a different batch size in the evaluation instead of the training batch size.
    model = get_new_trained_model(loaded_model, model_params)
    return model


def get_new_trained_model(trained_model, model_params):

    # Loading the dataset (without tokenizer) as dummy data stopped working
    normalisation_data, _ = get_dataset(
        config.val_id_file, config.proof_data, config.problem_features, batch_size=1
    )

    # Initialise the mdoel - loading of weights will override this
    model = initialise_model(model_params, training_data=normalisation_data)

    # Run the model call once to infer the main input shapes? - fails otherwise
    if model_params.encoder_type is EncoderType.TRANSFORMER:
        inp1 = tf.random.uniform([1, model_params.conjecture_input_length])
    else:
        inp1 = tf.random.uniform([1, 400])

    if model_params.decoder_type is DecoderType.TRANSFORMER:
        print(
            "Warning: Max length needs to be same as during training, otherwise, model loading will not work."
        )
        inp2 = tf.ones([model_params.max_caption_length, 1], dtype=tf.dtypes.int32)
    else:
        inp2 = tf.ones([1, 1], dtype=tf.dtypes.int32)
    hidden = tf.zeros((1, model_params.no_rnn_units))

    # Run model to infer input shapes
    if isinstance(model, tuple):  # Call decoder if split model
        inp1, input_mask, _ = call_encoder(model, inp1, False, hidden)
    else:
        input_mask = None

    # Initialise shapes of the model/decoder
    call_model_decoder(model, inp1, inp2, input_mask, hidden, False)

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
    axiom_remapping,
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

    model = get_model(model_dir, max_caption_length=max_len)
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
                max_cap_len=max_len,
                # axiom_frequency=axiom_frequency,
                remove_unknown=model_params.remove_unknown,
                encoder_input=model_params.encoder_input,
                conjecture_tokenizer=conjecture_tokenizer,
                conjecture_input_length=model_params.conjecture_input_length,
            )
            print(f"Size of dataset: {len(test_data)}")

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
                axiom_remapping,
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
