import sys

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Normalization
from keras.layers import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import concatenate
from argparse import Namespace
import json
import os

from enum_types import AxiomOrder, EncoderInput, AttentionMechanism, ModelType, EncoderType, DecoderType
from model_transformer import TransformerEncoder, TransformerDecoder, create_padding_mask


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
    """
    Function which makes a decoder/model call upon the input. Creates a unified
    interface for calling both complete model and models with separate encoder
    and decoders.
    """
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
        return None  # no hidden state in the dense model
    elif isinstance(model, tuple) and isinstance(model[1], InjectDecoder):
        return model[1].word_decoder.reset_state(batch_size=batch_size)
    elif isinstance(model, tuple) and isinstance(model[1], TransformerDecoder):
        return None  # No hidden state for this decoder
    elif isinstance(model, InjectModel):
        return model.inject_decoder.word_decoder.reset_state(batch_size=batch_size)

    # Default - Assume model structure with a word_decoder
    return model.word_decoder.reset_state(batch_size=batch_size)


def initialise_model(model_params, training_data=None):

    model = get_model(model_params)

    # If normalisation on the embedding graph is set, we have to adapt the
    # layer before compiling (or re-compile) the model. This is done over
    # the embedding vectors only.
    if model_params.normalize:
        # Only supply the embedding vectors
        model = adapt_normalization_layer(model, training_data.map(lambda x1, x2: x1))

    return model


# class ImageEncoder(layers.Layer):
class ImageEncoder(tf.keras.Model):  # TODO possible issue here?
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

        if params.global_max_pool:
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


class RNNEncoder(tf.keras.layers.Layer):
    """
    Encoder function for processing input sequences
    """

    def __init__(self, params, name="recurrent_encoder", **kwargs):

        super(RNNEncoder, self).__init__(name=name, **kwargs)

        # Set the vocabulary size
        self.input_vocab_size = params.input_vocab_size

        # The embedding layer converts tokens to vectors
        self.embedding = tf.keras.layers.Embedding(params.input_vocab_size, params.embedding_size)

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
        x = Input(shape=(self.input_vocab_size,))
        return Model(inputs=x, outputs=self.call(x))


class WordDecoder(layers.Layer):
    def __init__(
        self,
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
    def __init__(self, model_params, name="dense_model", **kwargs):
        super(DenseModel, self).__init__(name=name, **kwargs)
        self.target_vocab_size = model_params.target_vocab_size
        self.no_rnn_units = model_params.no_rnn_units

        self.axiom_order = model_params.axiom_order

        self.encoder = ImageEncoder(model_params)

        self.word_embedder = tf.keras.layers.Embedding(
            self.target_vocab_size, model_params.embedding_size, name="layer_word_embedding"
        )

        # Define the dense model
        self.fc = Dense(model_params.no_dense_units, activation="relu", name="layer_dense_1")
        self.dropout = Dropout(model_params.dropout_rate, name="layer_dropout")
        self.out = Dense(self.target_vocab_size, name="layer_output")

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
        config.update({"target_vocab_size": self.target_vocab_size, "name": self.name})
        return config

    def build_graph(self):
        x = [Input(shape=(400,)), Input(shape=(1,)), Input(shape=(2,))]
        return Model(inputs=x, outputs=self.call(x))


class InjectDecoder(tf.keras.Model):
    def __init__(self, model_params, name="inject_decoder", **kwargs):

        super(InjectDecoder, self).__init__(name=name, **kwargs)
        self.target_vocab_size = model_params.target_vocab_size
        self.no_rnn_units = model_params.no_rnn_units
        self.no_dense_units = model_params.no_dense_units

        self.axiom_order = model_params.axiom_order

        self.word_embedder = tf.keras.layers.Embedding(self.target_vocab_size, model_params.embedding_size)

        self.attention = get_attention_mechanism(model_params)

        self.word_decoder = WordDecoder(model_params)

        self.out_layer = Dense(self.target_vocab_size)

        self.repeat = RepeatVector(1)
        self.flatten = Flatten()

    def call(self, inputs, training=None, mask=None):

        # Extract the input elements
        image_emb, input_word, hidden_state = inputs

        # If input is more than 2 dimension, we flatten it to fit the rest of the setup
        if len(image_emb.shape) > 2:
            image_emb = self.flatten(image_emb)

        # Pass word through embedding layer
        word_emb = self.word_embedder(input_word, training=training)

        # Run Bahdanau attention if set
        if self.attention is not None and (
            isinstance(self.attention, BahdanauAttention) or isinstance(self.attention, AdditiveFlatAttention)
        ):
            image_emb, _ = self.attention(image_emb, hidden_state, mask=mask)

        # Expand dims of image_emb and concatenate with the word embedding
        merged_emb = concatenate([self.repeat(image_emb), word_emb])

        # Give to decoder
        decoder_out, hidden_state = self.word_decoder(merged_emb, training=training)

        # Run Loung attention if set
        if self.attention is not None and isinstance(self.attention, tf.keras.layers.Attention):
            context = self.attention([decoder_out, image_emb], training=training)
            decoder_out = concatenate([context, decoder_out])

        # Run output layer to get the predictions
        output = self.out_layer(decoder_out)

        return output, hidden_state

    def get_config(self):
        config = super(InjectDecoder, self).get_config()
        config.update({"target_vocab_size": self.target_vocab_size, "name": self.name})
        return config

    def build_graph(self):
        x = [Input(shape=(self.no_dense_units,)), Input(shape=(1,)), Input(shape=(self.no_rnn_units,))]
        return Model(inputs=x, outputs=self.call(x))


class InjectModel(tf.keras.Model):
    def __init__(self, model_params, name="inject_model", **kwargs):
        super(InjectModel, self).__init__(name=name, **kwargs)

        self.encoder = _get_encoder(model_params)
        self.inject_decoder = InjectDecoder(model_params)

        self.target_vocab_size = model_params.target_vocab_size
        self.no_rnn_units = model_params.no_rnn_units
        self.axiom_order = model_params.axiom_order

    def call(self, inputs, training=None, mask=None):

        # Extract the input elements
        input_entity, input_word, hidden_state = inputs

        # Compute image embedding
        if isinstance(self.encoder, RNNEncoder):
            entity_embedding, hidden_state, mask = self.encoder(input_entity, training=training)
        elif isinstance(self.encoder, ImageEncoder):
            entity_embedding = self.encoder(input_entity, training=training)
        elif isinstance(self.encoder, TransformerEncoder):
            entity_embedding = self.encoder(
                input_entity,
                mask=None,  # We pass mask as None as it will be derived in the function call of the encoder
                training=training,
            )
        else:
            raise ValueError(f'ERROR: Calling of encoder "{self.encoder}" not implemented for InjectModel')

        # Give to decoder
        output, hidden_state = self.inject_decoder(
            [entity_embedding, input_word, hidden_state], training=training, mask=mask
        )
        return output, hidden_state

    def get_config(self):
        config = super(InjectModel, self).get_config()
        config.update({"target_vocab_size": self.target_vocab_size, "name": self.name})
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


def _get_encoder(params):

    if params.encoder_type is EncoderType.IMAGE:
        return ImageEncoder(params)
    elif params.encoder_type is EncoderType.RECURRENT:
        return RNNEncoder(params)
    elif params.encoder_type is EncoderType.TRANSFORMER:
        return TransformerEncoder(params)

    raise ValueError(f'Cannot get encoder with the supplied "params.encoder_type": {params.encoder_type}')


def _get_decoder(params):

    if params.decoder_type is DecoderType.INJECT:
        return InjectDecoder(params)
    elif params.decoder_type is DecoderType.TRANSFORMER:
        return TransformerDecoder(params)

    raise ValueError(f'Cannot get decoder with the supplied "params.encoder_type": {params.encoder_type}')


def get_model(params):

    if params.model_type is ModelType.INJECT:
        return InjectModel(params)
    elif params.model_type is ModelType.SPLIT:
        # Return separate the encoder and decoder is separate for more efficient training
        encoder = _get_encoder(params)
        decoder = _get_decoder(params)

        # Wrap the model as a tuple
        return (encoder, decoder)
    elif params.model_type is ModelType.DENSE:
        return DenseModel(params)

    raise ValueError(f"Unrecognised model type: {params.model_type}")


def load_model(ckpt_dir):

    loaded_model = tf.keras.models.load_model(ckpt_dir)
    latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    load_status = loaded_model.load_weights(latest_checkpoint)
    print(f"Restored model from {latest_checkpoint}.")

    return loaded_model


def get_rnn(rnn_type):
    if rnn_type == "lstm":
        return LSTM
    elif rnn_type == "gru":
        return GRU

    raise ValueError(f'RNN type "{rnn_type}" not supported')


def get_attention_mechanism(model_params):
    if model_params.attention is AttentionMechanism.BAHDANAU:
        return BahdanauAttention(model_params)
    elif model_params.attention is AttentionMechanism.FLAT:
        return AdditiveFlatAttention(model_params)
    elif model_params.attention is AttentionMechanism.LOUNG_CONCAT:
        return tf.keras.layers.Attention(score_mode="concat", dropout=model_params.dropout_rate)
    elif model_params.attention is AttentionMechanism.LOUNG_DOT:
        return tf.keras.layers.Attention(score_mode="dot", dropout=model_params.dropout_rate)
    elif model_params.attention is AttentionMechanism.NONE:
        return None
    else:
        raise ValueError(f"Cannot retrieve attention mechanism: {model_params.attention}")


def get_model_params(model_dir):

    # Load parameters from model directory and create namespace
    with open(os.path.join(model_dir, "params.json"), "r") as f:
        params = json.load(f)
        params = Namespace(**params)

    if params.model_type:
        params.model_type = ModelType(params.model_type)

    if params.axiom_order:
        params.axiom_order = AxiomOrder(params.axiom_order)

    if params.attention:
        params.attention = AttentionMechanism(params.attention)

    if params.encoder_type:
        params.encoder_type = EncoderType(params.encoder_type)

    # Infer input type from the encoder type - sort of need both variables
    if params.encoder_type is EncoderType.TRANSFORMER or params.encoder_type is EncoderType.RECURRENT:
        params.encoder_input = EncoderInput.SEQUENCE
    elif params.encoder_type is EncoderType.IMAGE:
        params.encoder_input = EncoderInput.FLAT
    else:
        ValueError(f"Could not determine encoder_input type from the encoder type {params.encoder_type}")

    if params.decoder_type:
        params.decoder_type = DecoderType(params.decoder_type)

    if params.input_vocab_size == "all":
        # We use None for all in the code
        params.input_vocab_size = None

    if params.model_type is ModelType.INJECT and params.decoder_type is DecoderType.TRANSFORMER:
        raise ValueError("Incompatible parameters with inject model and transformer decoder")

    return params


def check_inject():

    # Load base model parameters
    params = get_model_params("experiments/base_model")
    params.normalize = False  # quick hack
    params.attention = AttentionMechanism.NONE  # quick hack
    params.stateful = False
    params.target_vocab_size = int(params.target_vocab_size)
    print("model params: ", params)

    # Attention
    print("\n# # # Attention # # #")
    m = BahdanauAttention(params)
    m.build_graph().summary()

    print("\n# # # Inject # # #")
    m = InjectModel(params)
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
    m = WordDecoder(params)
    m.build_graph().summary()

    # InjectDecoder
    print("\n# # # InjectDecoder # # #")
    m = InjectDecoder(params)
    m.build_graph().summary()


def check_models():
    # Function for testing the script
    # Load base model parameters
    params = get_model_params("experiments/base_model")
    params.normalize = False  # quick hack
    params.target_vocab_size = int(params.target_vocab_size)
    print("model params: ", params)

    print("# # # Inject # # #")
    params.stateful = False  # Need to turn off the state as do not want to specify batch size
    params.model_type = ModelType.INJECT
    m = get_model((params))
    print(m)
    print(m.build_graph().summary())

    print("# # # Dense # # #")
    params.model_type = ModelType.DENSE
    dense = get_model(params)
    print(dense)
    print(dense.build_graph().summary())


def check_encoder():
    params = get_model_params("experiments/base_model")
    # This needs to be supplied from data
    params.sequence_vocab_size = 80
    params.input_vocab_size = 200  # HACK
    params.target_vocab_size = int(params.target_vocab_size)
    enc = RNNEncoder(params)
    enc.build_graph().summary()


if __name__ == "__main__":
    check_models()
    check_inject()
    check_encoder()
