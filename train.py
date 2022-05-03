import os
import time
import tensorflow as tf
from pickle import dump
import numpy as np
import random

import config
from dataset import (
    get_dataset,
    get_tokenizer,
    compute_axiom_frequency,
    compute_random_global_axiom_frequency,
)
from model import get_model_params, initialise_model, DenseModel, reset_model_decoder_state
from evaluate import jaccard_score, coverage_score
from utils import get_train_parser, AxiomOrder

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
def train_step(tokenizer, model, optimizer, img_tensor, target, training=True):
    # Initial loss on the batch is zero
    loss = 0

    # Reset the LSTM states of the decoder between each batch
    reset_model_decoder_state(model)

    # Initialise the hidden shape of the model - makes the above lines redundant
    if isinstance(model, DenseModel):
        hidden = None  # no hidden state in the dense model
    else:
        hidden = model.word_decoder.reset_state(batch_size=target.shape[0])
        # hidden = tf.zeros((1, model.no_rnn_units))

    # Initialise input vector with the start token
    dec_input = tf.expand_dims([tokenizer.word_index[config.TOKEN_START]] * target.shape[0], 1)

    predictions = []

    with tf.GradientTape() as tape:
        # for i in tf.range(1, target.shape[1]):
        for i in range(1, target.shape[1]):
            # Predict the next token - slightly expensive way of doing it as it
            # encodes the image each time
            y_hat, hidden = model([img_tensor, dec_input, hidden], training=training)

            pred = tf.math.argmax(y_hat, axis=1)
            predictions.append(pred)

            # Compute loss of predictions
            loss += loss_function(target[:, i], y_hat)

            # Use teacher forcing to decide next model input
            dec_input = tf.expand_dims(target[:, i], 1)

    # Compute the total loss for the sequence
    sequence_loss = loss / int(target.shape[1])

    # Backprop if in training mode
    if training:
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

# TODO put tf function here?
# @tf.function
def epoch_step(model, tokenizer, optimizer, data, training=False, epoch=None):
    """
    Runs one epoch of data over the model. Used to train/validate the model.

    If the epoch is not None we report the loss of each batch.
    """

    # Need to keep track of the number of steps to compute the loss correctly.
    num_steps.assign(0)

    # Re-initialise the metrics for this epoch
    epoch_loss.reset_state()
    epoch_jac_score.reset_state()
    epoch_cov_score.reset_state()

    # Train on each batch in the data
    for (batch, (img_tensor, captions)) in enumerate(data):
        # num_steps += 1
        num_steps.assign_add(1)

        batch_loss, s_loss, predictions = train_step(
            tokenizer, model, optimizer, img_tensor, captions, training=training
        )

        # Check if reporting batch
        if epoch is not None and batch % 10 == 0:
            average_batch_loss = batch_loss.numpy() / int(captions.shape[1])
            tf.print(f"Epoch {epoch + 1} Batch {batch} Train Loss {average_batch_loss:.4f}")

        # Compute statistics
        cov_batch, jac_batch = compute_pred_stats(captions, predictions)
        # Update metric
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


# TODO tf.function?
def train_loop(tokenizer, model, ckpt_manager, optimizer, train_data, val_data, es_patience=None):

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
    # for epoch in tf.range(0, config.EPOCHS):
    for epoch in range(0, config.EPOCHS):
        start = time.time()

        # Train the model for one epoch
        train_epoch_metrics = epoch_step(model, tokenizer, optimizer, train_data, training=True, epoch=epoch)
        metrics = add_new_metrics(metrics, train_epoch_metrics, prefix="train_")

        # Run model over the validation data
        val_epoch_metrics = epoch_step(model, tokenizer, optimizer, val_data, training=False)
        metrics = add_new_metrics(metrics, val_epoch_metrics, prefix="val_")

        # Save the model after every epoch
        if epoch % 1 == 0:
            ckpt_manager.save()

        tf.print(f"Epoch {epoch + 1} Total Train Loss {metrics['train_loss'][-1]:.6f}")
        tf.print(f"Epoch {epoch + 1} Total Val   Loss {metrics['val_loss'][-1]:.6f}")
        tf.print(f"Time spent on epoch {time.time() - start:.2f} sec\n")

        # The early stopping strategy: stop the training if `val_loss` does not
        # decrease over a certain number of epochs.
        if es_patience is not None:
            es_wait += 1
            # FIXME use tf operators?
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

    # Add a final metric evaluation that ensures no drop out is ised (with training off)
    tf.print("Computing metrics on the training set with the final model parameters, without dropout")
    train_epoch_metrics = epoch_step(model, tokenizer, optimizer, train_data, training=False, epoch=epoch)
    metrics = add_new_metrics(metrics, train_epoch_metrics, prefix="train_")

    # Return training history
    return metrics


def main(
    model_dir, problem_features, proof_data, train_id_file, val_id_file, save_model, remove_unknown, context
):

    # Instantiate Tensorflow environment
    # TODO
    """
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    """
    tf.config.run_functions_eagerly(config.DEVELOPING)  # TODO

    # Get pre-trained tokenizer
    tokenizer_path = os.path.join(os.path.dirname(train_id_file), "tokenizer.json")  # FIXME
    tokenizer, vocab_size = get_tokenizer(tokenizer_path)
    print("Number of words: ", vocab_size)

    # Load model params from model file
    model_params = get_model_params(model_dir, context=context)

    # Compute the axiom frequencies if required # FIXME make function
    if model_params.axiom_order is AxiomOrder.FREQUENCY:
        axiom_frequency = compute_axiom_frequency(proof_data, train_id_file)
    elif model_params.axiom_order is AxiomOrder.RANDOM_GLOBAL:
        axiom_frequency = compute_random_global_axiom_frequency(proof_data)
    else:
        axiom_frequency = None

    # Get the training dataset
    train_data, max_len = get_dataset(
        train_id_file,
        proof_data,
        problem_features,
        tokenizer=tokenizer,
        order=model_params.axiom_order,
        axiom_frequency=axiom_frequency,
        remove_unknown=remove_unknown,
    )
    print("Max len: ", max_len)
    # Compute validation dataset based on the max length of the training data
    val_data, _ = get_dataset(
        val_id_file,
        proof_data,
        problem_features,
        tokenizer=tokenizer,
        max_cap_len=max_len,
        order=model_params.axiom_order,
        axiom_frequency=axiom_frequency,
        remove_unknown=remove_unknown,
    )

    # Remove axiom frequencies as it is no longer needed and can be very large
    del axiom_frequency

    # Initialise the model
    model = initialise_model(model_params.model_type, vocab_size, model_params, training_data=train_data)
    print("Training on: ", model)

    # TODO adding tensorboard stuff here - need to add more log points for this to apply
    # tb_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)
    # tb_callback.set_model(model)

    # Initialise the optimiser
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_params.learning_rate)

    # Initialise the checkpoint manager
    checkpoint_path = os.path.join(model_dir, "ckpt_dir")
    ckpt = tf.train.Checkpoint(model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # Call training loop
    history = train_loop(
        tokenizer, model, ckpt_manager, optimizer, train_data, val_data, es_patience=config.ES_PATIENCE
    )

    # Save the model
    if save_model:
        model.save(checkpoint_path)

    # Save the training history
    with open(os.path.join(model_dir, "history.pkl"), "wb") as f:
        dump(history, f)


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
