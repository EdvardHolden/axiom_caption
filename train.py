import os
import time
import tensorflow as tf
import argparse
from pickle import dump
import sys
import numpy as np

import config
from dataset import get_dataset, get_tokenizer
from model import MergeInjectModel, InjectModel, get_model_params


# TODO only give folder to ids?
parser = argparse.ArgumentParser()
parser.add_argument('--train_id_file', default=config.train_id_file,
                    help="File containing the training ids")
parser.add_argument('--val_id_file', default=config.val_id_file,
                    help="File containing the validation ids")

parser.add_argument('--proof_data', default=config.proof_data,
                    help="File containing the image features")
parser.add_argument('--problem_features', default=config.problem_features,
                    help="File containing the image descriptions")
parser.add_argument('--axiom_order', default=None, choices=[None, 'default', 'length', 'lexicographic'],
                    help='The order of the axioms in caption')

parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


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

    with tf.GradientTape() as tape:
        for i in range(1, target.shape[1]):
            # Predict the next token - slightly expensive way of doing it as it
            # encodes the image each time
            predictions = model([img_tensor, dec_input], training=training)

            # Compute loss of predictions
            loss += loss_function(target[:, i], predictions)

            # Use teacher forcing to decide next model input
            dec_input = tf.expand_dims(target[:, i], 1)

    # Compute the total loss for the sequence
    sequence_loss = (loss / int(target.shape[1]))

    # Backprop if in training mode
    if training:
        trainable_variables = model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, sequence_loss


def train_loop(tokenizer, model, ckpt_manager, optimizer, train_data, val_data, es_patience=None):

    loss_plot_train = []
    loss_plot_val = []

    # Initialise EarlyStopping wait variable
    if es_patience is not None:
        print("EarlyStopping patience set to: ", es_patience)
        es_wait = 0
        es_best_loss = np.inf

    # Loop through each epoch
    for epoch in range(0, config.EPOCHS):
        start = time.time()
        num_train_steps = 0  # Compute this on the fly for each epoch
        num_val_steps = 0  # Compute this on the fly for each epoch
        total_train_loss = 0
        total_val_loss = 0

        # Train on each batch in the data
        for (batch, (img_tensor, caption)) in enumerate(train_data):
            num_train_steps += 1
            batch_loss, s_loss = train_step(tokenizer, model, optimizer, img_tensor, caption, training=True)
            total_train_loss += s_loss

            # Check if reporting batch
            if batch % 10 == 0:
                average_batch_loss = batch_loss.numpy() / int(caption.shape[1])
                print(f'Epoch {epoch + 1} Batch {batch} Train Loss {average_batch_loss:.4f}')

        # Store the training loss for plotting
        train_loss_epoch = total_train_loss.numpy() / num_train_steps
        loss_plot_train.append(train_loss_epoch)

        # Validate model after each epoch and validation data is provided
        for (batch, (img_tensor, caption)) in enumerate(val_data):
            num_val_steps += 1
            batch_loss, s_loss = train_step(
                tokenizer, model, optimizer, img_tensor, caption, training=False)
            total_val_loss += s_loss

        # Store the training loss for plotting
        val_loss_epoch = total_val_loss.numpy() / num_val_steps
        loss_plot_val.append(val_loss_epoch)

        # Save the model after every epoch
        if epoch % 1 == 0:
            ckpt_manager.save()

        print(f'Epoch {epoch + 1} Total Train Loss {train_loss_epoch:.6f}')
        print(f'Epoch {epoch + 1} Total Val   Loss {val_loss_epoch:.6f}')
        print(f'Time spent on epoch {time.time() - start:.2f} sec\n')

        # The early stopping strategy: stop the training if `val_loss` does not
        # decrease over a certain number of epochs.
        if es_patience is not None:
            es_wait += 1
            if val_loss_epoch < es_best_loss:
                es_best_loss = val_loss_epoch
                es_wait = 0
            elif es_wait >= es_patience:
                print("Terminated training with early stopping after {0} epochs of no improvement".format(es_wait))
                break

    # Return training history
    return {"train_loss": loss_plot_train, "val_loss": loss_plot_val}


def initialise_model(model_type, max_len, vocab_size, model_params):
    if model_type == "merge_inject":
        model = MergeInjectModel(max_len, vocab_size, model_params)
    elif model_type == "inject":
        model = InjectModel(max_len, vocab_size, model_params)
    else:
        print("Unrecognised model type: ", model_type, file=sys.stderr)
        sys.exit(1)
    return model


def main():

    # TODO add caption order argument!

    # Parse input arguments
    args = parser.parse_args()

    # Get pre-trained tokenizer
    tokenizer_path = os.path.join(os.path.dirname(args.train_id_file), 'tokenizer.json') # FIXME
    tokenizer, vocab_size = get_tokenizer(tokenizer_path)
    print("Number of words: ", vocab_size)
    import sys

    # Get the training dataset
    train_data, max_len = get_dataset(args.train_id_file, args.proof_data,
                                      args.problem_features, tokenizer=tokenizer,
                                      order=args.axiom_order)
    print("Max len: ", max_len)
    # Compute validation dataset based on the max length of the training data
    val_data, _ = get_dataset(args.val_id_file, args.proof_data,
                              args.problem_features, tokenizer=tokenizer,
                              max_cap_len=max_len, order=args.axiom_order)

    # Load model params from model file
    model_params = get_model_params(args.model_dir)

    # Initialise the model
    model = initialise_model(model_params.model_type, max_len, vocab_size, model_params)
    print("Training on: ", model)

    # Initialise the optimiser
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_params.learning_rate)

    # Initialise the checkpoint manager
    checkpoint_path = os.path.join(args.model_dir, 'ckpt_dir')
    ckpt = tf.train.Checkpoint(model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # Call training loop
    history = train_loop(tokenizer, model, ckpt_manager, optimizer, train_data, val_data, es_patience=config.ES_PATIENCE)

    # Save the model
    model.save(checkpoint_path)

    # Save the training history
    with open(os.path.join(args.model_dir, 'history.pkl'), 'wb') as f:
        dump(history, f)


if __name__ == "__main__":
    tf.config.run_functions_eagerly(config.DEVELOPING)
    main()
