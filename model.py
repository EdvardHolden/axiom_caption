import sys

import tensorflow as tf
from keras.models import Model
from tensorflow.keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Normalization
from keras.layers import BatchNormalization
from keras.layers.merge import concatenate
from argparse import Namespace
import json
import os

from dataset import AxiomOrder


def adapt_normalization_layer(model, embedding_vectors):
    if embedding_vectors is None:
        raise ValueError("Cannot initialize model with normalization layer without supplying training data")
    # Adapt the normalisation layer to the embedding vector
    model.image_encoder.normalize.adapt(embedding_vectors)
    return model


def initialise_model(model_type, max_len, vocab_size, model_params, training_data=None):

    model = get_model(model_type, max_len, vocab_size, model_params)

    # If normalisation on the embedding graph is set, we have to adapt the
    # layer before compiling (or re-compile) the model
    if model_params.normalize:
        # Only supply the embedding vectors
        model = adapt_normalization_layer(model, training_data.map(lambda x1, x2: x1))

    return model


class ImageEncoder(layers.Layer):
    def __init__(self, no_dense_units, dropout_rate, normalize, batch_norm, name="image_encoder", **kwargs):
        super(ImageEncoder, self).__init__(name=name, **kwargs)

        if normalize:
            self.normalize = Normalization()
        else:
            self.normalize = None

        if batch_norm:
            self.batch_norm = BatchNormalization()
        else:
            self.batch_norm = None

        self.fe2 = Dense(no_dense_units, activation="relu")
        self.d2 = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = inputs

        # Apply normalization if set
        if self.normalize:
            assert self.normalize.is_adapted, "Need to adapt the normalisation layer before using"
            x = self.normalize(x)

        # Apply batch norm if set
        if self.batch_norm:
            x = self.batch_norm(x, training=training)

        x = self.fe2(x)
        x = self.d2(x, training=training)
        return x

    def build_graph(self):
        # Input shape of a single word
        x = Input(shape=(400,))
        return Model(inputs=x, outputs=self.call(x))


class WordEncoder(layers.Layer):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        rnn_type,
        no_rnn_units,
        no_dense_units,
        dropout_rate,
        name="word_encoder",
        **kwargs,
    ):
        super(WordEncoder, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.emb1 = Embedding(vocab_size, embedding_size, mask_zero=True)
        self.d1 = Dropout(dropout_rate)

        rnn = get_rnn(rnn_type)

        self.emb2 = rnn(no_rnn_units, return_sequences=True, dropout=dropout_rate)

        # TODO need to understand the use of this?
        self.emb3 = TimeDistributed(Dense(no_dense_units, activation="relu"))
        self.d2 = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = inputs
        x = self.emb1(x)
        x = self.d1(x, training=training)
        x = self.emb2(x, training=training)
        x = self.emb3(x)
        x = self.d2(x, training=training)
        return x

    def build_graph(self):
        # Input shape of a single word
        x = Input(shape=(self.vocab_size,))
        return Model(inputs=x, outputs=self.call(x))


class WordDecoder(layers.Layer):
    def __init__(
        self, vocab_size, rnn_type, no_rnn_units, no_dense_units, dropout_rate, name="word_decoder", **kwargs
    ):
        super(WordDecoder, self).__init__(name=name, **kwargs)

        rnn = get_rnn(rnn_type)
        self.rnn = rnn(
            no_rnn_units,
            dropout=dropout_rate,
            return_state=True,
            return_sequences=True,
            recurrent_initializer="glorot_uniform",
        )

        self.no_rnn_units = no_rnn_units

        self.d1 = Dropout(dropout_rate)
        self.fc = Dense(no_dense_units, activation="relu")
        self.d2 = Dropout(dropout_rate)
        self.out = Dense(vocab_size)

    def call(self, inputs, training=None):
        x = inputs

        if isinstance(self.rnn, LSTM):
            # LSTM also returns the cell state, which we do not use
            x, hidden, _ = self.rnn(x, training=training)
        else:
            # GRU does not return the cell state
            x, hidden = self.rnn(x, training=training)

        x = self.d1(x, training=training)
        x = self.fc(x)
        x = self.d2(x, training=training)
        x = self.out(x)

        x = tf.reshape(x, (-1, x.shape[2]))
        return x, hidden

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.no_rnn_units))

    def build_graph(self):
        # Input shape of a single word
        x = Input(
            shape=(
                22,
                256,
            )
        )
        return Model(inputs=x, outputs=self.call(x))


class DenseModel(tf.keras.Model):
    def __init__(self, max_length, vocab_size, model_params, name="dense", **kwargs):
        super(DenseModel, self).__init__(name=name, **kwargs)
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.no_rnn_units = model_params.no_rnn_units

        self.axiom_order = model_params.axiom_order

        self.image_encoder = ImageEncoder(
            model_params.no_dense_units,
            model_params.dropout_rate,
            model_params.normalize,
            model_params.batch_norm,
        )

        self.word_embedder = tf.keras.layers.Embedding(vocab_size, model_params.embedding_size)

        # Define the dense model
        self.fc = Dense(model_params.no_dense_units, activation="relu")
        self.dropout = Dropout(model_params.dropout_rate)
        self.out = Dense(vocab_size)

        self.flatten = Flatten()

    def call(self, inputs, training=None):
        input_image, input_word, hidden_state = inputs

        # Compute image embedding
        image_emb = self.image_encoder(input_image, training=training)
        # Pass word through embedding layer
        # word_emb = tf.squeeze( self.word_embedder(input_word, training=training))  # TODO maybe use flatten instead?
        word_emb = self.word_embedder(input_word, training=training)
        # Flatten the embedding as we are not using LSTM
        word_emb = self.flatten(word_emb)

        x = concatenate([image_emb, word_emb])

        # Pass through the dense layer
        x = self.fc(x, training=training)
        x = self.dropout(x, training=training)
        output = self.out(x)

        return output, output

    def get_config(self):
        config = super(DenseModel, self).get_config()
        config.update({"max_length": self.max_length, "vocab_size": self.vocab_size, "name": self.name})
        return config

    def build_graph(self):
        x = [Input(shape=(400,)), Input(shape=(1,)), Input(shape=(2,))]
        return Model(inputs=x, outputs=self.call(x))


class InjectModel(tf.keras.Model):
    def __init__(self, max_length, vocab_size, model_params, name="inject", **kwargs):
        super(InjectModel, self).__init__(name=name, **kwargs)
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.no_rnn_units = model_params.no_rnn_units

        self.axiom_order = model_params.axiom_order

        self.word_embedder = tf.keras.layers.Embedding(vocab_size, model_params.embedding_size)
        self.image_encoder = ImageEncoder(
            model_params.no_dense_units,
            model_params.dropout_rate,
            model_params.normalize,
            model_params.batch_norm,
        )

        if model_params.attention:
            self.attention = BahdanauAttention(model_params.no_rnn_units)
        else:
            self.attention = None

        self.word_decoder = WordDecoder(
            vocab_size,
            model_params.rnn_type,
            model_params.no_rnn_units,
            model_params.no_dense_units,
            model_params.dropout_rate,
        )

        self.repeat = RepeatVector(1)

    def call(self, inputs, training=None):
        input_image, input_word, hidden_state = inputs

        # Compute image embedding
        image_emb = self.image_encoder(input_image, training=training)
        # Pass word through embedding layer
        word_emb = self.word_embedder(input_word, training=training)

        # Add attention embedding
        if self.attention is not None:
            image_emb, _ = self.attention(image_emb, hidden_state)

        # Concatenate the features - original paper passes attention vector instead
        image_emb = self.repeat(image_emb)  # Quickfix for expanding the dimension
        merged_emb = concatenate([image_emb, word_emb])

        # Give to decoder
        output, hidden_state = self.word_decoder(merged_emb, training=training)
        return output, hidden_state

    def get_config(self):
        config = super(InjectModel, self).get_config()
        config.update({"max_length": self.max_length, "vocab_size": self.vocab_size, "name": self.name})
        return config

    def build_graph(self):
        x = [Input(shape=(400,)), Input(shape=(1,)), Input(shape=(self.no_rnn_units,))]
        return Model(inputs=x, outputs=self.call(x))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class MergeInjectModel(tf.keras.Model):
    def __init__(self, max_length, vocab_size, model_params, name="merge_inject", **kwargs):
        super(MergeInjectModel, self).__init__(name=name, **kwargs)
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.no_rnn_units = model_params.no_rnn_units

        self.axiom_order = model_params.axiom_order

        self.image_encoder = ImageEncoder(
            model_params.no_dense_units,
            model_params.dropout_rate,
            model_params.normalize,
            model_params.batch_norm,
        )
        self.word_encoder = WordEncoder(
            vocab_size,
            model_params.embedding_size,
            model_params.rnn_type,
            model_params.no_rnn_units,
            model_params.no_dense_units,
            model_params.dropout_rate,
        )

        # Add attention
        if model_params.attention:
            print(
                "Warning: Attention functionality not fully implemented for the merge architecture",
                file=sys.stderr,
            )
            self.attention = BahdanauAttention(model_params.no_rnn_units)
        else:
            self.attention = None

        self.word_decoder = WordDecoder(
            vocab_size,
            model_params.rnn_type,
            model_params.no_rnn_units,
            model_params.no_dense_units,
            model_params.dropout_rate,
        )

        # Add repeat vector for avoid calling the image encoder all the time
        # QUICKFIX - setting length to 1 to expand the dimension of the output for concat
        self.repeat = RepeatVector(1)

    def call(self, inputs, training=None):
        input_image, input_word, hidden_state = inputs

        image_emb = self.image_encoder(input_image, training=training)
        word_emb = self.word_encoder(
            input_word, training=training
        )  # TODO maybe this should return the state as well?

        # Perform attention on the image embedding
        if self.attention is not None:
            image_emb, _ = self.attention(image_emb, hidden_state)
        image_emb = self.repeat(image_emb)

        # Concatenate both inputs
        merged_emb = concatenate([image_emb, word_emb])

        # Decode the embedding
        output, hidden = self.word_decoder(merged_emb, training=training)
        return output, hidden_state

    def get_config(self):
        config = super(MergeInjectModel, self).get_config()
        config.update({"max_length": self.max_length, "vocab_size": self.vocab_size, "name": self.name})
        return config

    def build_graph(self):
        # TODO this does not show the submodels - needs more work
        # https://stackoverflow.com/questions/61427583/how-do-i-plot-a-keras-tensorflow-subclassing-api-model
        x = [
            Input(shape=(400,)),
            Input(shape=(1,)),
            Input(shape=(self.no_rnn_units,)),
        ]
        return Model(inputs=x, outputs=self.call(x))


def get_model(model_type, max_length, vocab_size, params):
    if model_type == "merge_inject":
        model = MergeInjectModel(max_length, vocab_size, params)
    elif model_type == "inject":
        model = InjectModel(max_length, vocab_size, params)
    elif model_type == "dense":
        model = DenseModel(max_length, vocab_size, params)
    else:
        print("Unrecognised model type: ", model_type, file=sys.stderr)
        sys.exit(1)

    return model


def load_model(ckpt_dir):
    loaded_model = tf.keras.models.load_model(ckpt_dir)
    latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    load_status = loaded_model.load_weights(latest_checkpoint)
    print(f"Restored from {latest_checkpoint}.")

    return loaded_model


def get_rnn(rnn_type):
    if rnn_type == "lstm":
        rnn = LSTM
    elif rnn_type == "gru":
        rnn = GRU
    else:
        raise ValueError(f'RNN type "{rnn_type}" not supported')

    return rnn


def get_model_params(model_dir):

    # Load parameters from model directory and create namespace
    with open(os.path.join(model_dir, "params.json"), "r") as f:
        params = json.load(f)
        params = Namespace(**params)

    if params.axiom_order:
        params.axiom_order = _axiom_order_string_to_type(params.axiom_order)

    return params


def _axiom_order_string_to_type(string_value):

    if string_value == "original":
        return AxiomOrder.ORIGINAL
    elif string_value == "lexicographic":
        return AxiomOrder.LEXICOGRAPHIC
    elif string_value == "length":
        return AxiomOrder.LENGTH
    elif string_value == "random":
        return AxiomOrder.RANDOM
    elif string_value == "frequency":
        return AxiomOrder.FREQUENCY
    else:
        raise ValueError(f"No string mapping between '{string_value}' and enum in AxiomOrder")


if __name__ == "__main__":
    # Function for testing the script
    # Load base model parameters
    params = get_model_params("experiments/base_model")
    params.normalize = False  # Quick hack
    print("Model params: ", params)
    print("# # # MergeInject # # #")
    m = get_model("merge_inject", 123, 20, params)
    m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(m)
    print(m.build_graph().summary())
    print()

    print("# # # Inject # # #")
    m = get_model("inject", 123, 20, params)
    print(m)
    print(m.build_graph().summary())

    print("# # # Dense # # #")
    m = get_model("dense", 123, 20, params)
    print(m)
    print(m.build_graph().summary())

"""
    print("# # # Inject # # #")
    m = get_model("inject", 123, 20, params)
    m = get_model("attention_inject", 123, 20, params)
    print(m)
    print(m.build_graph().summary())
"""
