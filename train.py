import os
import time
import tensorflow as tf
import argparse
from pickle import dump
import numpy as np

import config
from dataset import get_dataset, get_tokenizer
from model import get_model_params, initialise_model
from evaluate import jaccard_score, coverage_score

# TODO only give folder to ids?
# Would be nice to have some compositive Argument parser so scripts that uses this function doesn't have to duplicate this
parser = argparse.ArgumentParser()
parser.add_argument("--train_id_file", default=config.train_id_file, help="File containing the training ids")
parser.add_argument("--val_id_file", default=config.val_id_file, help="File containing the validation ids")

parser.add_argument("--proof_data", default=config.proof_data, help="File containing the image features")
parser.add_argument(
    "--problem_features", default=config.problem_features, help="File containing the image descriptions"
)

parser.add_argument("--model_dir", default="experiments/base_model", help="Directory containing params.json")

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")


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

    # Reset the LSTM states between each batch
    model.reset_states()

    # Initialise input vector with the start token
    dec_input = tf.expand_dims([tokenizer.word_index[config.TOKEN_START]] * target.shape[0], 1)

    predictions = []

    with tf.GradientTape() as tape:
        for i in range(1, target.shape[1]):
            # Predict the next token - slightly expensive way of doing it as it
            # encodes the image each time
            y_hat = model([img_tensor, dec_input], training=training)

            # TODO: Could argmax the predictions and return them?
            # What about padding token and start/stop?
            predictions.append(np.argmax(y_hat, axis=1))

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

    return loss, sequence_loss, np.array(predictions).T


def compute_pred_stats(captions, predictions):
    # First need to remove pad and start tokens
    FILTER = lambda x: x != 0 and x != 2
    filtered_pred = []
    filtered_cap = []
    for pred, cap in zip(predictions, captions):
        filtered_pred.append([*filter(FILTER, pred)])
        filtered_cap.append([*filter(FILTER, cap.numpy())])

    # Compute and return the stats for each prediction
    cov_score = coverage_score(filtered_cap, filtered_pred, avg=False)
    jac_score = jaccard_score(filtered_cap, filtered_pred, avg=False)
    return cov_score, jac_score


def epoch_step(model, tokenizer, optimizer, data, training=False, epoch=None):
    """
    Runs one epoch of data over the model. Used to train/validate the model.

    If the epoch is not None we report the loss of each batch.
    """

    # Need to keep track of the number of steps to compute the loss correctly.
    num_steps = 0
    total_loss = 0
    jac_scores = []
    cov_scores = []

    # Train on each batch in the data
    for (batch, (img_tensor, captions)) in enumerate(data):
        num_steps += 1
        batch_loss, s_loss, predictions = train_step(
            tokenizer, model, optimizer, img_tensor, captions, training=training
        )
        total_loss += s_loss

        # Check if reporting batch
        if epoch is not None and batch % 10 == 0:
            average_batch_loss = batch_loss.numpy() / int(captions.shape[1])
            print(f"Epoch {epoch + 1} Batch {batch} Train Loss {average_batch_loss:.4f}")

        # Compute statistics
        cov_batch, jac_batch = compute_pred_stats(captions, predictions)
        jac_scores += jac_batch
        cov_scores += cov_batch

    # Store the training loss for plotting
    loss_epoch = total_loss.numpy() / num_steps
    jac_epoch = np.average(jac_scores)
    cov_epoch = np.average(cov_scores)

    # Return metrics as a dict
    return {"loss": loss_epoch, "coverage": cov_epoch, "jaccard": jac_epoch}


def add_new_metrics(history, new_stats, prefix=""):

    for stat in new_stats:
        index = prefix + stat
        if index in history:
            history[index] += [new_stats[stat]]
        else:
            history[index] = [new_stats[stat]]

    return history


def train_loop(tokenizer, model, ckpt_manager, optimizer, train_data, val_data, es_patience=None):

    # Dictionary to store all the metrics
    metrics = {}

    # Initialise EarlyStopping wait variable
    if es_patience is not None:
        print("EarlyStopping patience set to: ", es_patience)
        es_wait = 0
        es_best_loss = np.inf

    # If in developing mode, compute the initial loss of the model as a sanity check
    # This only works in developing mode as we need to call training=True on the train_step
    # first in order to build the graph correctly.
    if config.DEVELOPING:
        initial_stats = epoch_step(model, tokenizer, optimizer, train_data, training=False)
        print(f"Initial model training loss: {initial_stats['loss']:.2f}")

    # Loop through each epoch
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

        print(f"Epoch {epoch + 1} Total Train Loss {metrics['train_loss'][-1]:.6f}")
        print(f"Epoch {epoch + 1} Total Val   Loss {metrics['val_loss'][-1]:.6f}")
        print(f"Time spent on epoch {time.time() - start:.2f} sec\n")

        # The early stopping strategy: stop the training if `val_loss` does not
        # decrease over a certain number of epochs.
        if es_patience is not None:
            es_wait += 1
            if metrics["val_loss"][-1] < es_best_loss:
                es_best_loss = metrics["val_loss"][-1]
                es_wait = 0
            elif es_wait >= es_patience:
                print(
                    "Terminated training with early stopping after {0} epochs of no improvement".format(
                        es_wait
                    )
                )
                break

    # Return training history
    return metrics


def main():

    # TODO add caption order argument!

    # Parse input arguments
    args = parser.parse_args()

    # Get pre-trained tokenizer
    tokenizer_path = os.path.join(os.path.dirname(args.train_id_file), "tokenizer.json")  # FIXME
    tokenizer, vocab_size = get_tokenizer(tokenizer_path)
    print("Number of words: ", vocab_size)

    # Load model params from model file
    model_params = get_model_params(args.model_dir)

    # Get the training dataset
    train_data, max_len = get_dataset(
        args.train_id_file,
        args.proof_data,
        args.problem_features,
        tokenizer=tokenizer,
        order=model_params.axiom_order,
    )
    print("Max len: ", max_len)
    # Compute validation dataset based on the max length of the training data
    val_data, _ = get_dataset(
        args.val_id_file,
        args.proof_data,
        args.problem_features,
        tokenizer=tokenizer,
        max_cap_len=max_len,
        order=model_params.axiom_order,
    )

    # Initialise the model
    print("Train type ", type(train_data))  # TODO remvoe
    model = initialise_model(
        model_params.model_type, max_len, vocab_size, model_params, training_data=train_data
    )
    print("Training on: ", model)

    # Initialise the optimiser
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_params.learning_rate)

    # Initialise the checkpoint manager
    checkpoint_path = os.path.join(args.model_dir, "ckpt_dir")
    ckpt = tf.train.Checkpoint(model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # Call training loop
    history = train_loop(
        tokenizer, model, ckpt_manager, optimizer, train_data, val_data, es_patience=config.ES_PATIENCE
    )

    # Save the model
    model.save(checkpoint_path)

    # Save the training history
    with open(os.path.join(args.model_dir, "history.pkl"), "wb") as f:
        dump(history, f)


if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    tf.config.run_functions_eagerly(config.DEVELOPING)
    main()
    print("# Finito")
