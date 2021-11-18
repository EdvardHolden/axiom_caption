import sys

import tensorflow as tf
from keras.models import Model
from tensorflow.keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.layers.merge import concatenate
from keras.layers.pooling import GlobalMaxPooling2D
from tensorflow.keras.utils import plot_model
from argparse import Namespace
import json
import os


class ImageEncoder(layers.Layer):

    def __init__(self, no_dense_units, dropout_rate, name="image_encoder", **kwargs):
        super(ImageEncoder, self).__init__(name=name, **kwargs)
        #self.fe1 = GlobalMaxPooling2D()  # TODO - change to flatten or other pooling?
        #self.d1 = Dropout(dropout_rate)
        self.fe2 = Dense(no_dense_units, activation='relu')
        self.d2 = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = inputs
        #x = self.fe1(x)
        #x = self.d1(x, training=training)
        x = self.fe2(x)
        x = self.d2(x, training=training)
        return x

    def build_graph(self):
        # Input shape of a single word
        x = Input(shape=(400,))
        return Model(inputs=x, outputs=self.call(x))


class WordEncoder(layers.Layer):

    def __init__(self, vocab_size, embedding_size, no_lstm_units,
                 no_dense_units, dropout_rate, name="word_encoder", **kwargs):
        super(WordEncoder, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.emb1 = Embedding(vocab_size, embedding_size, mask_zero=True)
        self.d1 = Dropout(dropout_rate)
        self.emb2 = LSTM(no_lstm_units, return_sequences=True, dropout=dropout_rate)
        # TODO need to understand the use of this?
        self.emb3 = TimeDistributed(Dense(no_dense_units, activation='relu'))
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

    def __init__(self, vocab_size, no_lstm_units, no_dense_units,
                 dropout_rate, name="word_decoder", **kwargs):
        super(WordDecoder, self).__init__(name=name, **kwargs)
        self.lm = LSTM(no_lstm_units, dropout=dropout_rate)
        self.d1 = Dropout(dropout_rate)
        self.fc = Dense(no_dense_units, activation='relu')
        self.d2 = Dropout(dropout_rate)
        self.out = Dense(vocab_size)

    def call(self, inputs, training=None):
        x = inputs
        x = self.lm(x, training=training)
        x = self.d1(x, training=training)
        x = self.fc(x)
        x = self.d2(x, training=training)
        x = self.out(x)
        return x

    def build_graph(self):
        # Input shape of a single word
        x = Input(shape=(22, 256,))
        return Model(inputs=x, outputs=self.call(x))


class InjectModel(tf.keras.Model):
    def __init__(self, max_length, vocab_size, model_params, name="inject", **kwargs):
        super(InjectModel, self).__init__(name=name, **kwargs)
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.word_embedder = tf.keras.layers.Embedding(vocab_size, model_params.embedding_size)
        self.image_encoder = ImageEncoder(model_params.no_dense_units, model_params.dropout_rate)
        self.word_decoder = WordDecoder(vocab_size, model_params.no_lstm_units,
                                        model_params.no_dense_units, model_params.dropout_rate)

        self.repeat = RepeatVector(1)

    def call(self, inputs, training=None):
        input_image, input_word = inputs

        # Compute image embedding
        image_emb = self.image_encoder(input_image, training=training)
        # Pass word through embedding layer
        word_emb = self.word_embedder(input_word, training=training)

        # Concatenate the features - original paper passes attention vector instead
        image_emb = self.repeat(image_emb)  # Quickfix for expanding the dimension
        merged_emb = concatenate([image_emb, word_emb])

        # Give to decoder
        output = self.word_decoder(merged_emb, training=training)
        return output

    def get_config(self):
        config = super(InjectModel, self).get_config()
        config.update({"max_length": self.max_length,
                      "vocab_size": self.vocab_size, "name": self.name})
        return config

    def build_graph(self):
        x = [Input(shape=(400,)), Input(shape=(1,))]
        return Model(inputs=x, outputs=self.call(x))


class MergeInjectModel(tf.keras.Model):

    def __init__(self, max_length, vocab_size, model_params, name="merge_inject", **kwargs):
        super(MergeInjectModel, self).__init__(name=name, **kwargs)
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.image_encoder = ImageEncoder(model_params.no_dense_units, model_params.dropout_rate)
        self.word_encoder = WordEncoder(vocab_size, model_params.embedding_size, model_params.no_lstm_units,
                                        model_params.no_dense_units, model_params.dropout_rate)
        self.word_decoder = WordDecoder(vocab_size, model_params.no_lstm_units,
                                        model_params.no_dense_units, model_params.dropout_rate)

        # Add repeat vector for avoid calling the image encoder all the time
        # QUICKFIX - setting length to 1 to expand the dimension of the output for concat
        self.repeat = RepeatVector(1)

    def call(self, inputs, training=None):
        input_image, input_word = inputs
        image_emb = self.image_encoder(input_image, training=training)
        image_emb = self.repeat(image_emb)
        word_emb = self.word_encoder(input_word, training=training)

        # Concatenate both inputs
        merged_emb = concatenate([image_emb, word_emb])

        # Decode the embedding
        output = self.word_decoder(merged_emb, training=training)
        return output

    def get_config(self):
        config = super(MergeInjectModel, self).get_config()
        config.update({"max_length": self.max_length,
                      "vocab_size": self.vocab_size, "name": self.name})
        return config

    def build_graph(self):
        # TODO this does not show the submodels - needs more work
        # https://stackoverflow.com/questions/61427583/how-do-i-plot-a-keras-tensorflow-subclassing-api-model
        x = [Input(shape=(400,)), Input(shape=(1,))]
        return Model(inputs=x, outputs=self.call(x))


def get_model(model_type, max_length, vocab_size, params):
    if model_type == "merge_inject":
        model = MergeInjectModel(max_length, vocab_size, params)
    elif model_type == "inject":
        model = InjectModel(max_length, vocab_size, params)
    else:
        print("Unrecognised model type: ", model_type, file=sys.stderr)
        sys.exit(1)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def load_model(ckpt_dir):
    loaded_model = tf.keras.models.load_model(ckpt_dir)
    latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    load_status = loaded_model.load_weights(latest_checkpoint)
    print(f'Restored from {latest_checkpoint}.')

    return loaded_model


def get_model_params(model_dir):

    # Load parameters from model directory and create namespace
    with open(os.path.join(model_dir, 'params.json'), 'r') as f:
        params = json.load(f)
        params = Namespace(**params)
    return params


if __name__ == "__main__":
    # Function for testing the script
    # Load base model parameters
    params = get_model_params('experiments/base_model')
    print("Model params: ", params)
    print("# # # MergeInject # # #")
    m = get_model("merge_inject", 123, 20, params)
    print(m)
    print(m.build_graph().summary())
    print()
    print("# # # Inject # # #")
    m = get_model("inject", 123, 20, params)
    print(m)
    print(m.build_graph().summary())