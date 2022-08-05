import os
import time
import tensorflow as tf
from pickle import dump
import numpy as np
import random

import config
from dataset import (
    get_dataset,
    compute_axiom_frequency,
    compute_random_global_axiom_frequency,
    get_caption_conjecture_tokenizers,
)
from model import (
    get_model_params,
    initialise_model,
    get_hidden_state,
    reset_model_decoder_state,
    RNNEncoder,
    ImageEncoder,
    TransformerEncoder,
)
from evaluate import jaccard_score, coverage_score
from enum_types import AxiomOrder
from parser import get_train_parser
from model_transformer import create_padding_mask, TransformerDecoder

# Make script deterministic to see if we can avoid the gpu issue
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

# Make path for tensorboard
LOG_DIR = os.path.join("logdir", time.strftime("%Y%m%d-%H%M%S"))


@tf.function
def loss_function(real, pred):

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


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


@tf.function
def get_next_decoder_input_token(
    dec_input, pred, target, training, teacher_forcing_rate, iteration, sequence=False
):

    # Check if applying teacher-forcing in training mode
    """
    if training and tf.random.uniform(()) <= teacher_forcing_rate:
        # Teacher forcing - using the correct input target
        dec_input = tf.expand_dims(target[:, i], 1)
    else:
        # Using the predicted tokens
        dec_input = tf.expand_dims(tf.cast(pred, tf.int32), 1)
    """

    if not training or tf.random.uniform(()) <= teacher_forcing_rate:
        next_tokens = target[:, iteration]
    else:
        next_tokens = tf.cast(pred, tf.int32)

    # If we are using sequences - add in-place, otherwise return expanded tokens
    if sequence:
        return dec_input.write(iteration, next_tokens)
    else:
        return tf.expand_dims(next_tokens, 1)


@tf.function
def call_encoder(model, img_tensor, training, input_mask, hidden):
    """
    Function which makes an encoder call upon the input. If the model
    is not split into encoder-decoder, there is no effect from the call.
    Raises ValueError if the model is a tuple but call for the encoder
    class is not implemented.
    """
    if isinstance(model, tuple):
        # Call and update variables according to the type of encoder being used
        if isinstance(model[0], ImageEncoder):
            img_tensor = model[0](img_tensor, training=training)

        elif isinstance(model[0], TransformerEncoder):
            input_mask = create_padding_mask(img_tensor)
            img_tensor = model[0](img_tensor, mask=input_mask, training=training)

        elif isinstance(model[0], RNNEncoder):
            img_tensor, hidden, input_mask = model[0](img_tensor, training=training)
        else:
            raise ValueError(f"Encoder call not implemented for {model[0]}")

    return img_tensor, input_mask, hidden


@tf.function
def call_model_decoder(model, img_tensor, dec_input, input_mask, hidden, training):
    # Predict the next token - either by using the full model or just the decoder
    # encodes the image each time
    if isinstance(model, tuple) and isinstance(model[1], TransformerDecoder):
        # Reshape the sequence fed to the decoder
        transformer_dec_input = tf.transpose(dec_input.stack())

        # Call transformer decoder
        decoder_mask = create_padding_mask(transformer_dec_input)
        y_hat, _ = model[1](transformer_dec_input, img_tensor, decoder_mask, input_mask, training=training)

    elif isinstance(model, tuple):
        # Call decoder
        y_hat, hidden = model[1]([img_tensor, dec_input, hidden], training=training, mask=input_mask)
    else:
        # Call whole model on all the input data
        y_hat, hidden = model([img_tensor, dec_input, hidden], training=training)

    return y_hat, hidden


@tf.function
def train_step(tokenizer, model, optimizer, img_tensor, target, teacher_forcing_rate, training):

    # Initial loss on the batch is zero
    loss = 0

    # Reset the LSTM states of the decoder between each batch
    reset_model_decoder_state(model)

    # Initialise the hidden shape of the model - used for attention mainly
    hidden = get_hidden_state(model, target.shape[0])

    # Determine whether the decoding stage is operating over a sequence (relevant for transformer)
    sequence = isinstance(model, tuple) and isinstance(model[1], TransformerDecoder)

    # Get the initial decoder input consisting of the start token
    dec_input = get_initial_decoder_input(
        tokenizer,
        target,
        sequence=sequence,
    )

    # List for holding the predictions
    predictions = []

    # Placeholder for mask - not used in all model types
    input_mask = None

    with tf.GradientTape() as tape:

        # Call the encoder to pre-compute the entity features for use in the decoder call
        img_tensor, input_mask, hidden = call_encoder(model, img_tensor, training, input_mask, hidden)

        for i in range(1, target.shape[1]):

            # Call the decoder/model to produce the final predictions
            y_hat, hidden = call_model_decoder(model, img_tensor, dec_input, input_mask, hidden, training)

            # Append predictions
            pred = tf.math.argmax(y_hat, axis=1)
            predictions.append(pred)

            # Compute loss of predictions
            loss += loss_function(target[:, i], y_hat)

            # Get the next decoder input tokens
            dec_input = get_next_decoder_input_token(
                dec_input, pred, target, training, teacher_forcing_rate, i, sequence=sequence
            )

    # Close dec_input if TensorArray as it does not like being written to without being used (for the last iteration)
    if isinstance(dec_input, tf.TensorArray):
        dec_input.close()

    # Compute the total loss for the sequence
    sequence_loss = loss / int(target.shape[1])

    # Backprop if in training mode
    if training:
        if isinstance(model, tuple):
            trainable_variables = model[0].trainable_variables + model[1].trainable_variables
        else:
            trainable_variables = model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, sequence_loss, tf.transpose(predictions)


@tf.function
def compute_pred_stats(captions, predictions):
    # First need to remove pad and start tokens, and end tokens
    # <pad>: 0, <start>: 2, <end>: 3
    # This will be converted to a sparse tensor where 0 is unrepresented
    FILTER = lambda x: x if x != 2 and x != 3 else tf.constant(0, dtype=tf.int64)

    # Need to cast the captions from int32 to int64
    captions = tf.cast(captions, dtype=tf.int64)

    # Filter some tokens - if this starts failing we should include fn_output_signature=tf.int64 in function call
    captions = tf.map_fn(
        lambda x: tf.map_fn(FILTER, x),
        captions,
    )
    predictions = tf.map_fn(
        lambda x: tf.map_fn(FILTER, x),
        predictions,
    )

    cov_score = coverage_score(captions, predictions, avg=False)
    jac_score = jaccard_score(captions, predictions, avg=False)

    return cov_score, jac_score


epoch_jac_score = tf.keras.metrics.Mean()
epoch_cov_score = tf.keras.metrics.Mean()
epoch_loss = tf.keras.metrics.Mean()
num_steps = tf.Variable(0)


def epoch_step(model, tokenizer, optimizer, data, teacher_forcing_rate=0.0, training=False, epoch=None):
    """
    Runs one epoch of data over the model. Used to train/validate the model.

    If the epoch is not None we report the loss of each batch.
    """

    # Keep track of the number of steps/batches for correct reportin
    num_steps.assign(0)

    # Re-initialise the metrics for this epoch
    epoch_loss.reset_state()
    epoch_jac_score.reset_state()
    epoch_cov_score.reset_state()

    # Train on each batch in the data
    for img_tensor, captions in data:
        # Increment the number of steps
        num_steps.assign_add(1)

        batch_loss, s_loss, predictions = train_step(
            tokenizer,
            model,
            optimizer,
            img_tensor,
            captions,
            teacher_forcing_rate=teacher_forcing_rate,
            training=training,
        )

        # Check if reporting batch
        # if epoch is not None and batch % 10 == 0:
        if epoch is not None and num_steps % 10 == 0:
            average_batch_loss = batch_loss.numpy() / int(captions.shape[1])
            tf.print(f"Epoch {epoch + 1} Batch {num_steps.numpy()} Train Loss {average_batch_loss:.4f}")

        # Compute statistics
        cov_batch, jac_batch = compute_pred_stats(captions, predictions)

        # Update metrics
        epoch_loss.update_state(s_loss)
        epoch_jac_score.update_state(jac_batch)
        epoch_cov_score.update_state(cov_batch)

    # Return metrics as a dict
    return {
        "loss": epoch_loss.result(),
        "coverage": epoch_cov_score.result(),
        "jaccard": epoch_jac_score.result(),
    }


def add_new_metrics(history, new_stats, prefix=""):

    for stat in new_stats:
        index = prefix + stat
        if index in history:
            history[index] += [new_stats[stat]]
        else:
            history[index] = [new_stats[stat]]

    return history


def train_loop(
    tokenizer, model, ckpt_managers, optimizer, train_data, val_data, teacher_forcing_rate, es_patience=None
):

    # Dictionary to store all the metrics
    metrics = {}

    # Initialise EarlyStopping wait variable
    if es_patience is not None:
        tf.print("EarlyStopping patience set to: ", es_patience)
        es_wait = 0
        es_best_loss = np.inf

    # If in developing mode, compute the initial loss of the model as a sanity check
    # This only works in developing mode as we need to call training=True on the train_step
    # first in order to build the graph correctly.
    if config.DEVELOPING:
        initial_stats = epoch_step(model, tokenizer, optimizer, train_data, training=False)
        tf.print(f"Initial model training loss: {initial_stats['loss']:.2f}")

    # Loop through each epoch
    for epoch in tf.range(0, config.EPOCHS):
        start = time.time()

        # Train the model for one epoch
        train_epoch_metrics = epoch_step(
            model, tokenizer, optimizer, train_data, teacher_forcing_rate, training=True, epoch=epoch
        )
        metrics = add_new_metrics(metrics, train_epoch_metrics, prefix="train_")

        # Run model over the validation data
        val_epoch_metrics = epoch_step(model, tokenizer, optimizer, val_data, training=False)
        metrics = add_new_metrics(metrics, val_epoch_metrics, prefix="val_")

        # Save the model after every epoch
        if epoch % 1 == 0:
            for ckpt in ckpt_managers:
                ckpt.save()

        tf.print(f"Epoch {epoch + 1} Total Train Loss {metrics['train_loss'][-1]:.6f}")
        tf.print(f"Epoch {epoch + 1} Total Val   Loss {metrics['val_loss'][-1]:.6f}")
        tf.print(f"Time spent on epoch {time.time() - start:.2f} sec\n")

        # The early stopping strategy: stop the training if `val_loss` does not
        # decrease over a certain number of epochs.
        if es_patience is not None:
            es_wait += 1
            if metrics["val_loss"][-1] < es_best_loss:
                es_best_loss = metrics["val_loss"][-1]
                es_wait = 0
            elif es_wait >= es_patience:
                tf.print(
                    "Terminated training with early stopping after {0} epochs of no improvement".format(
                        es_wait
                    )
                )
                break

    # Add a final metric evaluation that ensures no drop out is used (with training off)
    tf.print("Computing metrics on the training set with the final model parameters, without dropout")
    train_epoch_metrics = epoch_step(model, tokenizer, optimizer, train_data, training=False, epoch=epoch)
    metrics = add_new_metrics(metrics, train_epoch_metrics, prefix="train_")

    # Return training history
    return metrics


# Compute the axiom frequencies if required
def get_axiom_frequency(axiom_order, train_id_file, proof_data):
    """
    Gets the axiom frequencies. Either comptued from data, or randomly imposed
    on the global set of axioms. Returns None if an order which does not
    require this is set.
    """

    if axiom_order is AxiomOrder.FREQUENCY:
        axiom_frequency = compute_axiom_frequency(proof_data, train_id_file)
    elif axiom_order is AxiomOrder.RANDOM_GLOBAL:
        axiom_frequency = compute_random_global_axiom_frequency(proof_data)
    else:
        axiom_frequency = None
    return axiom_frequency


def main(
    model_dir,
    problem_features,
    proof_data,
    train_id_file,
    val_id_file,
    save_model,
    working_dir,
    context,
):

    # Instantiate Tensorflow environment
    """
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    """
    tf.config.run_functions_eagerly(config.DEVELOPING)

    # Check if working_dir is set, otherwise, default to model_dir
    if working_dir is None:
        working_dir = model_dir

    # Load model params from model file - and set the encoder input
    model_params = get_model_params(model_dir)

    # Load the tokenizers for this training setting
    caption_tokenizer, _, conjecture_tokenizer = get_caption_conjecture_tokenizers(
        model_params, proof_data, context, train_id_file, problem_features
    )

    # Get the axiom frequencies from this dataset
    axiom_frequency = get_axiom_frequency(model_params.axiom_order, train_id_file, proof_data)

    # Get the training dataset
    tf.print("Loading training dataset")
    train_data, max_len = get_dataset(
        train_id_file,
        proof_data,
        problem_features,
        caption_tokenizer=caption_tokenizer,
        order=model_params.axiom_order,
        axiom_frequency=axiom_frequency,
        remove_unknown=model_params.remove_unknown,
        encoder_input=model_params.encoder_input,
        conjecture_tokenizer=conjecture_tokenizer,
        conjecture_input_length=model_params.conjecture_input_length,
    )
    tf.print("Max caption length: ", max_len)
    model_params.max_caption_length = max_len  # Set variable in case we are using the transformer
    # Compute validation dataset based on the max length of the training data
    tf.print("Loading validation dataset")
    val_data, _ = get_dataset(
        val_id_file,
        proof_data,
        problem_features,
        caption_tokenizer=caption_tokenizer,
        max_cap_len=max_len,
        order=model_params.axiom_order,
        axiom_frequency=axiom_frequency,
        remove_unknown=model_params.remove_unknown,
        encoder_input=model_params.encoder_input,
        conjecture_tokenizer=conjecture_tokenizer,
        conjecture_input_length=model_params.conjecture_input_length,
    )

    # Remove axiom frequencies as it is no longer needed and can be very large
    del axiom_frequency

    # Not longer needed
    del conjecture_tokenizer

    # Initialise the model
    model = initialise_model(model_params, training_data=train_data)
    tf.print("Training on: ", model)

    # Initialise the optimiser
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_params.learning_rate)

    # Initialise the checkpoint manager
    checkpoint_path = os.path.join(working_dir, "ckpt_dir")
    if isinstance(model, tuple):
        # USe separate checkpoints for encoder and decoder
        encoder_ckpt = tf.train.Checkpoint(model[0])
        decoder_ckpt = tf.train.Checkpoint(model[1])

        encoder_checkpoint_path = os.path.join(checkpoint_path, "encoder")
        decoder_checkpoint_path = os.path.join(checkpoint_path, "decoder")

        ckpt_managers = [
            tf.train.CheckpointManager(encoder_ckpt, encoder_checkpoint_path, max_to_keep=5),
            tf.train.CheckpointManager(decoder_ckpt, decoder_checkpoint_path, max_to_keep=5),
        ]
    else:
        ckpt = tf.train.Checkpoint(model)
        ckpt_managers = [tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)]

    # Call training loop
    history = train_loop(
        caption_tokenizer,
        model,
        ckpt_managers,
        optimizer,
        train_data,
        val_data,
        model_params.teacher_forcing_rate,
        es_patience=config.ES_PATIENCE,
    )

    # Save the training history
    with open(os.path.join(working_dir, "history.pkl"), "wb") as f:
        dump(history, f)

    # Save the model
    if save_model:
        if isinstance(model, tuple):
            model[0].save(encoder_checkpoint_path)
            model[1].save(decoder_checkpoint_path)
        else:
            model.save(checkpoint_path)


def run_main():
    """
    Helper function for running the main as a main script.
    This helper function is needed to avoid shadowing any
    of the names in the outer-scope.
    """

    # Get the parser and parse the arguments
    parser = get_train_parser()
    args = parser.parse_args()

    # Run main with the arguments
    main(**vars(args))


if __name__ == "__main__":

    # Call the helper function for running the main
    run_main()

    print("# Finito")
