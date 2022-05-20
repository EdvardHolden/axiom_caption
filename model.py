import sys

import tensorflow as tf
from tensorflow.keras.models import Model
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
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers.merge import concatenate
from argparse import Namespace
import json
import os

from dataset import AxiomOrder
from utils import Context, EncoderInput, AttentionMechanism


def adapt_normalization_layer(model, embedding_vectors):

    # If we use a cached dataset it is a MapDataset, and we need to unpack it and access all the elements as a new dataset.
    # I cannot figure out how to check for MapDataset, so we apply this to all cases.
    embedding_vectors = list(embedding_vectors.unbatch().as_numpy_iterator())
    embedding_vectors = tf.data.Dataset.from_tensor_slices(embedding_vectors)

    if embedding_vectors is None:
        raise ValueError("Cannot initialize model with normalization layer without supplying training data")
    # Adapt the normalisation layer to the embedding vector
    if isinstance(model, tuple):  # Check if decoder and encoder is separated
        model[0].normalize.adapt(embedding_vectors)
    else:
        model.encoder.normalize.adapt(embedding_vectors)
    return model


def reset_model_decoder_state(model):
    """Function used to reset rnn state of the word decoder when it is stateful.
    If it is the first call to the model (e.g. input sizes are not set), the
    model returns ValueError. At the moment we handle try and ask for forgiveness
    but a proper check might be better.
    Warning, function can not be wrapped in tf.function as it alters the behaviour"""
    # Check if we are given the full model
    if isinstance(model, tuple) and hasattr(model[1], "word_decoder"):
        # Check that rnn is stateful
        if model[1].word_decoder.rnn.stateful:
            try:
                model[1].word_decoder.rnn.reset_states()
            except ValueError:
                pass
    elif isinstance(model, InjectModel):
        # Check that rnn is stateful
        if model.inject_decoder.word_decoder.rnn.stateful:
            try:
                model.inject_decoder.word_decoder.rnn.reset_states()
            except ValueError:
                pass
    elif hasattr(model, "word_decoder"):
        # Check that rnn is stateful
        if model.word_decoder.rnn.stateful:
            try:
                model.word_decoder.rnn.reset_states()
            except ValueError:
                pass
    # Check if we are given the word decoder
    elif hasattr(model, "rnn"):
        # Check that rnn is stateful
        if model.rnn.stateful:
            try:
                model.rnn.reset_states()
            except ValueError:
                pass


def _get_sequence_mask(sequence, dtype=tf.float32):
    """
    Return the mask of the input sequence for use with the attention mechanism
    """
    mask = tf.math.logical_not(tf.math.equal(sequence, 0))
    mask = tf.cast(mask, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    return mask


@tf.function
def get_hidden_state(model, batch_size):
    """
    Get the initial hidden state of the model - based on the number of rnn parameters
    """
    if isinstance(model, DenseModel):
        hidden = None  # no hidden state in the dense model
    elif isinstance(model, tuple) and isinstance(model[1], InjectDecoder):
        hidden = model[1].word_decoder.reset_state(batch_size=batch_size)
    elif isinstance(model, InjectModel):
        hidden = model.inject_decoder.word_decoder.reset_state(batch_size=batch_size)
    else:
        # Assume model structure with a word_decoder
        hidden = model.word_decoder.reset_state(batch_size=batch_size)

    return hidden


def initialise_model(model_type, vocab_size, model_params, training_data=None):

    model = get_model(model_type, vocab_size, model_params)

    # If normalisation on the embedding graph is set, we have to adapt the
    # layer before compiling (or re-compile) the model. This is done over
    # the embedding vectors only.
    if model_params.normalize:
        # Only supply the embedding vectors
        model = adapt_normalization_layer(model, training_data.map(lambda x1, x2: x1))

    return model


# class ImageEncoder(layers.Layer):
class ImageEncoder(tf.keras.Model):
    def __init__(
        self,
        params,
        name="image_encoder",
        **kwargs,
    ):
        super(ImageEncoder, self).__init__(name=name, **kwargs)

        # Initialise with parameters
        if params.normalize:
            self.normalize = Normalization()
        else:
            self.normalize = None

        if params.global_max_pool:  # FIXME unclear whether this should go before normalisation
            self.max_pool = GlobalMaxPooling2D()
        else:
            self.max_pool = None

        if params.batch_norm:
            self.batch_norm = BatchNormalization()
        else:
            self.batch_norm = None

        self.fe2 = Dense(params.no_dense_units, activation="relu")
        self.d2 = Dropout(params.dropout_rate)

    def call(self, inputs, training=None):
        x = inputs

        # Apply normalization if set
        if self.normalize:
            assert self.normalize.is_adapted, "Need to adapt the normalisation layer before using"
            x = self.normalize(x)

        # Apply max pooling if set
        if self.max_pool:
            x = self.max_pool(x)

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


# TODO add bidirectional encoder
class Encoder(tf.keras.layers.Layer):
    """
    Encoder function for processing input sequences
    """

    def __init__(self, params, name="conjecture_encoder", **kwargs):

        super(Encoder, self).__init__(name=name, **kwargs)

        # Set the vocabulary size
        self.vocab_size = params.conjecture_vocab_size

        # The embedding layer converts tokens to vectors
        self.embedding = tf.keras.layers.Embedding(params.conjecture_vocab_size, params.embedding_size)

        # Get the RNN type
        rnn = get_rnn(params.rnn_type)

        self.no_rnn_units = params.no_rnn_units

        # Initialise RNN model - always set stateful to False as we are processing the full sequence at once
        self.rnn = rnn(
            params.no_rnn_units,
            stateful=False,
            dropout=params.dropout_rate,
            return_state=True,
            return_sequences=True,
            recurrent_initializer="glorot_uniform",
        )

    def call(self, sequence_input, training=None):

        # Embed the sequences into tokens
        x = self.embedding(sequence_input)

        # Process the input sequence
        if isinstance(self.rnn, LSTM):
            # LSTM also returns the cell state, which we do not use
            x, hidden, _ = self.rnn(x, training=training)
        else:
            # GRU does not return the cell state
            x, hidden = self.rnn(x, training=training)

        # Compute the mask for the input sequence
        mask = _get_sequence_mask(sequence_input)

        # 4. Returns the new sequence, its state and a mask for the input sequence
        return x, hidden, mask

    def build_graph(self):
        # Input shape of a single word
        x = Input(shape=(self.vocab_size,))
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
        print("Warning: Deprecated")

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
        self,
        vocab_size,
        params,
        name="word_decoder",
        **kwargs,
    ):
        super(WordDecoder, self).__init__(name=name, **kwargs)
        self.params = params

        # Get the RNN type
        rnn = get_rnn(params.rnn_type)
        # Initialise RNN model
        self.rnn = rnn(
            params.no_rnn_units,
            stateful=params.stateful,
            dropout=params.dropout_rate,
            return_state=True,
            return_sequences=True,
            recurrent_initializer="glorot_uniform",
        )

        self.no_rnn_units = params.no_rnn_units

        self.d1 = Dropout(params.dropout_rate)
        self.fc = Dense(params.no_dense_units, activation="relu")
        self.d2 = Dropout(params.dropout_rate)
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
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.d2(x, training=training)
        x = self.out(x)

        return x, hidden

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.no_rnn_units))

    def build_graph(self):
        # Input shape of a single word
        x = Input(
            shape=(
                1,
                self.params.embedding_size + self.params.no_dense_units,
            )
        )
        return Model(inputs=x, outputs=self.call(x))


class DenseModel(tf.keras.Model):
    def __init__(self, vocab_size, model_params, name="dense", **kwargs):
        super(DenseModel, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.no_rnn_units = model_params.no_rnn_units

        self.axiom_order = model_params.axiom_order

        self.encoder = ImageEncoder(model_params)

        self.word_embedder = tf.keras.layers.Embedding(
            vocab_size, model_params.embedding_size, name="layer_word_embedding"
        )

        # Define the dense model
        self.fc = Dense(model_params.no_dense_units, activation="relu", name="layer_dense_1")
        self.dropout = Dropout(model_params.dropout_rate, name="layer_dropout")
        self.out = Dense(vocab_size, name="layer_output")

        self.flatten = Flatten(name="layer_flatten")

    def call(self, inputs, training=None):
        input_image, input_word, hidden_state = inputs

        # Compute image embedding
        image_emb = self.encoder(input_image, training=training)

        # Pass word through embedding layer
        # word_emb = tf.squeeze( self.word_embedder(input_word, training=training))  # TODO maybe use flatten instead?
        word_emb = self.word_embedder(input_word, training=training)
        # Flatten the embedding as we are not using LSTM
        word_emb = self.flatten(word_emb)

        x = concatenate([image_emb, word_emb], name="layer_concatenate")

        # Pass through the dense layer
        x = self.fc(x, training=training)
        x = self.dropout(x, training=training)
        output = self.out(x)

        return output, output

    def get_config(self):
        config = super(DenseModel, self).get_config()
        config.update({"vocab_size": self.vocab_size, "name": self.name})
        return config

    def build_graph(self):
        x = [Input(shape=(400,)), Input(shape=(1,)), Input(shape=(2,))]
        return Model(inputs=x, outputs=self.call(x))


class InjectDecoder(tf.keras.Model):
    def __init__(self, vocab_size, model_params, name="inject_decoder", **kwargs):

        super(InjectDecoder, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.no_rnn_units = model_params.no_rnn_units

        self.axiom_order = model_params.axiom_order

        self.word_embedder = tf.keras.layers.Embedding(vocab_size, model_params.embedding_size)

        self.attention = get_attention_mechanism(model_params)

        self.word_decoder = WordDecoder(vocab_size, model_params)

        self.repeat = RepeatVector(1)

    def call(self, inputs, training=None, mask=None):

        # Extract the input elements
        image_emb, input_word, hidden_state = inputs

        # Compute image embedding
        # Pass word through embedding layer
        word_emb = self.word_embedder(input_word, training=training)

        # Add attention embedding
        if self.attention is not None:
            image_emb, _ = self.attention(image_emb, hidden_state, mask=mask)

        # Concatenate the features - original paper passes attention vector instead
        image_emb = self.repeat(image_emb)  # Quickfix for expanding the dimension
        merged_emb = concatenate([image_emb, word_emb])

        # Give to decoder
        output, hidden_state = self.word_decoder(merged_emb, training=training)
        return output, hidden_state

    def get_config(self):
        config = super(InjectDecoder, self).get_config()
        config.update({"vocab_size": self.vocab_size, "name": self.name})
        return config

    def build_graph(self):
        x = [Input(shape=(400,)), Input(shape=(1,)), Input(shape=(self.no_rnn_units,))]
        return Model(inputs=x, outputs=self.call(x))


class InjectModel(tf.keras.Model):
    def __init__(self, vocab_size, model_params, name="inject_model", **kwargs):
        super(InjectModel, self).__init__(name=name, **kwargs)

        self.encoder = _get_encoder(model_params)
        self.inject_decoder = InjectDecoder(vocab_size, model_params)

        self.vocab_size = vocab_size
        self.no_rnn_units = model_params.no_rnn_units
        self.axiom_order = model_params.axiom_order

    def call(self, inputs, training=None, mask=None):

        # Extract the input elements
        input_entity, input_word, hidden_state = inputs

        # Compute image embedding
        if isinstance(self.encoder, Encoder):
            entity_embedding, hidden_state, mask = self.encoder(input_entity, training=training)
        elif isinstance(self.encoder, ImageEncoder):
            entity_embedding = self.encoder(input_entity, training=training)
        else:
            raise ValueError(f'ERROR: Calling of encoder "{self.encoder}" not implemented for InjectModel')

        # Give to decoder
        output, hidden_state = self.inject_decoder(
            [entity_embedding, input_word, hidden_state], training=training, mask=mask
        )
        return output, hidden_state

    def get_config(self):
        config = super(InjectModel, self).get_config()
        config.update({"vocab_size": self.vocab_size, "name": self.name})
        return config

    def build_graph(self):
        x = [Input(shape=(400,)), Input(shape=(1,)), Input(shape=(self.no_rnn_units,))]
        return Model(inputs=x, outputs=self.call(x))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, params):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(params.no_rnn_units)
        self.W2 = tf.keras.layers.Dense(params.no_rnn_units)
        self.V = tf.keras.layers.Dense(1)
        self.softmax = tf.keras.layers.Softmax(axis=1)

    def call(self, features, hidden, mask=None):

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = self.softmax(score, mask=mask)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def build_graph(self):
        # Input shape of a single word
        x = Input(shape=(64,))
        hidden = Input(shape=(50,))
        return Model(inputs=[x, hidden], outputs=self.call(x, hidden))


class AdditiveFlatAttention(tf.keras.Model):
    def __init__(self, params):
        super(AdditiveFlatAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(params.no_rnn_units)
        self.W2 = tf.keras.layers.Dense(params.no_rnn_units)
        self.V = tf.keras.layers.Dense(params.no_dense_units)

        # self.softmax = tf.keras.layers.Softmax(axis=1) # Dimensions?
        self.softmax = tf.keras.layers.Softmax()

    def call(self, features, hidden, mask=None):

        attention_hidden_layer = tf.nn.tanh(self.W1(features) + self.W2(hidden))

        # Give score for each embedding value
        score = self.V(attention_hidden_layer)

        attention_weights = self.softmax(score, mask=mask)

        # Weigh the features
        context_vector = attention_weights * features
        # context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def build_graph(self):
        # Input shape of a single word
        x = Input(shape=(64,))
        hidden = Input(shape=(50,))
        return Model(inputs=[x, hidden], outputs=self.call(x, hidden))


class MergeInjectModel(tf.keras.Model):
    def __init__(self, vocab_size, model_params, global_max_pool=False, name="merge_inject", **kwargs):
        super(MergeInjectModel, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.no_rnn_units = model_params.no_rnn_units

        self.axiom_order = model_params.axiom_order

        self.image_encoder = ImageEncoder(model_params)

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
            self.attention = BahdanauAttention(model_params)
        else:
            self.attention = None

        self.word_decoder = WordDecoder(vocab_size, model_params)

        # Add repeat vector for avoid calling the image encoder all the time
        # QUICKFIX - setting length to 1 to expand the dimension of the output for concat
        self.repeat = RepeatVector(1)
        print("Warning: Model DEPRECATED (need updates)")

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
        config.update({"vocab_size": self.vocab_size, "name": self.name})
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


def _get_encoder(params):

    if params.encoder_input is EncoderInput.FLAT:
        encoder = ImageEncoder(params)
    elif params.encoder_input is EncoderInput.SEQUENCE:
        encoder = Encoder(params)
    else:
        raise ValueError(f'Cannot get model with the supplied "params.encoder_input": {params.encoder_input}')
    return encoder


def get_model(model_type, vocab_size, params):

    if model_type == "merge_inject":
        print("Merge Inject model is deprecated", file=sys.stderr)
        model = MergeInjectModel(vocab_size, params)
    elif model_type == "inject":
        model = InjectModel(vocab_size, params)
    elif model_type == "inject_decoder":
        # Model where encoder and decoder is separate for more efficient training
        encoder = _get_encoder(params)
        # Initialise the decoder
        decoder = InjectDecoder(vocab_size, params)
        # Wrap the model as a tuple
        model = (encoder, decoder)
    elif model_type == "dense":
        model = DenseModel(vocab_size, params)
    else:
        print("Unrecognised model type: ", model_type, file=sys.stderr)
        sys.exit(1)

    return model


def load_model(ckpt_dir):

    loaded_model = tf.keras.models.load_model(ckpt_dir)
    latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    load_status = loaded_model.load_weights(latest_checkpoint)
    print(f"Restored model from {latest_checkpoint}.")

    return loaded_model


def get_rnn(rnn_type):
    if rnn_type == "lstm":
        rnn = LSTM
    elif rnn_type == "gru":
        rnn = GRU
    else:
        raise ValueError(f'RNN type "{rnn_type}" not supported')

    return rnn


def get_attention_mechanism(model_params):
    if model_params.attention is AttentionMechanism.BAHDANAU:
        return BahdanauAttention(model_params)
    elif model_params.attention is AttentionMechanism.FLAT:
        return AdditiveFlatAttention(model_params)
    elif model_params.attention is AttentionMechanism.NONE:
        return None
    else:
        raise ValueError(f"Cannot retrieve attention mechanism: {model_params.attention}")


def get_model_params(model_dir):

    # Load parameters from model directory and create namespace
    with open(os.path.join(model_dir, "params.json"), "r") as f:
        params = json.load(f)
        params = Namespace(**params)

    # Set the axiom order
    if params.axiom_order:
        params.axiom_order = AxiomOrder(params.axiom_order)

    if params.attention:
        params.attention = AttentionMechanism(params.attention)

    if params.encoder_input:
        params.encoder_input = EncoderInput(params.encoder_input)

    if params.conjecture_vocab_size == "all":
        # We use None for all in the code
        params.conjecture_vocab_size = None

    return params


def check_inject():
    # Load base model parameters
    params = get_model_params("experiments/base_model")
    params.normalize = False  # quick hack
    params.attention = True  # quick hack
    params.stateful = False  # FIXME is this right?
    print("model params: ", params)

    # TODO add  the other components
    # Get inject
    vocab_size = 1000

    # Attention
    print("\n# # # Attention # # #")
    m = BahdanauAttention(params)
    m.build_graph().summary()

    print("\n# # # Inject # # #")
    m = InjectModel(vocab_size, params)
    m.build_graph().summary()
    # print("### No trainable variables: ", m.trainable_variables)
    import numpy as np

    tran = np.sum([np.prod(v.get_shape().as_list()) for v in m.trainable_variables])
    print("### Number of trainable variables: ", tran)

    # ImageEncoder
    print("\n# # # Image Encoder # # #")
    m = ImageEncoder(params)
    m.build_graph().summary()

    # WordDecoder
    print("\n# # # WordDecoder # # #")
    m = WordDecoder(vocab_size, params)
    m.build_graph().summary()


def check_models():
    # Function for testing the script
    # Load base model parameters
    params = get_model_params("experiments/base_model")
    params.normalize = False  # quick hack
    print("model params: ", params)
    """
    # Deprecated model
    print("# # # MergeInject # # #")
    m = get_model("merge_inject", 123, 20, params)
    m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(m)
    print(m.build_graph().summary())
    print()
    """

    print("# # # Inject # # #")
    params.stateful = False  # Need to turn off the state as do not want to specify batch size
    m = get_model("inject", 123, params)
    print(m)
    print(m.build_graph().summary())

    print("# # # Dense # # #")
    dense = get_model("dense", 123, params)
    print(dense)
    print(dense.summary())
    print(dense.build_graph().summary())

    """
    print("# # # Inject # # #")
    m = get_model("inject", 123, 20, params)
    m = get_model("attention_inject", 123, 20, params)
    print(m)
    print(m.build_graph().summary())
    """


def check_encoder():
    params = get_model_params("experiments/base_model")
    # This needs to be supplied from data
    params.sequence_vocab_size = 80
    enc = Encoder(params)
    print(enc)
    enc.build_graph().summary()


if __name__ == "__main__":
    # check_models()
    # check_inject()
    check_encoder()
