import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras import Model
from enum_types import TransformerInputOrder


def _get_angles(pos, i, transformer_dense_units):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(transformer_dense_units))
    return pos * angle_rates


class PosEncodingEmbedding(tf.keras.layers.Layer):
    """
    Custom layer to apply the positional embedding to a sequence of embeddings.
    """

    def __init__(self, transformer_dense_units, **kwargs):
        super().__init__(**kwargs)
        self.transformer_dense_units = transformer_dense_units

    def build(self, inputs_shape):
        angle_rads = _get_angles(
            np.arange(inputs_shape[1])[:, np.newaxis],
            np.arange(self.transformer_dense_units)[np.newaxis, :],
            self.transformer_dense_units,
        )

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        self.pos_encoding_matrix = tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):

        seq_len = tf.shape(x)[1]

        # Apply encoding to input features
        x += self.pos_encoding_matrix[:, :seq_len, :]
        return x


class PosLinearEmbedding(tf.keras.layers.Layer):
    """
    Custom layer to learn positional embeddings to the inputs."""

    def __init__(self, posemb_init=None, **kwargs):
        super().__init__(**kwargs)
        if posemb_init is None:
            self.posemb_init = tf.keras.initializers.RandomNormal(stddev=0.02)
        else:
            self.posemb_init = posemb_init

    def build(self, inputs_shape):
        pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
        self.pos_embedding = self.add_weight("pos_embedding", pos_emb_shape, initializer=self.posemb_init)

    def call(self, x):
        # inputs.shape is (batch_size, seq_len, emb_dim).
        pos_embedding = tf.cast(self.pos_embedding, x.dtype)

        return x + pos_embedding


class IdentityLayer(tf.keras.layers.Layer):
    """
    Return the same ouput as input.
    """

    def __init__(self, posemb_init=None, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x


def _get_positional_encoding_function(transformer_input_order, transformer_dense_units):
    """
    Function for getting the positional embedding matrix of a specified embedding type.
    """

    if transformer_input_order is TransformerInputOrder.SEQUENTIAL:
        return PosEncodingEmbedding(transformer_dense_units)
    elif transformer_input_order is TransformerInputOrder.ORIGINAL:
        return IdentityLayer()
    elif transformer_input_order is TransformerInputOrder.LINEAR:
        return PosLinearEmbedding()

    raise ValueError(f'Transformer input order "{transformer_input_order}" not implemented.')


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class TransformerMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_params, name="transformer_attention_head"):
        super(TransformerMultiHeadAttention, self).__init__()
        self.num_attention_heads = model_params.num_attention_heads
        self.transformer_dense_units = model_params.transformer_dense_units

        assert model_params.transformer_dense_units % self.num_attention_heads == 0

        self.depth = model_params.transformer_dense_units // self.num_attention_heads

        self.wq = tf.keras.layers.Dense(model_params.transformer_dense_units)
        self.wk = tf.keras.layers.Dense(model_params.transformer_dense_units)
        self.wv = tf.keras.layers.Dense(model_params.transformer_dense_units)

        self.dense = tf.keras.layers.Dense(model_params.transformer_dense_units)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_attention_heads, depth).
        Transpose the result such that the shape is (batch_size, num_attention_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, transformer_dense_units)
        k = self.wk(k)  # (batch_size, seq_len, transformer_dense_units)
        v = self.wv(v)  # (batch_size, seq_len, transformer_dense_units)

        q = self.split_heads(q, batch_size)  # (batch_size, num_attention_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_attention_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_attention_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_attention_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_attention_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_attention_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.transformer_dense_units)
        )  # (batch_size, seq_len_q, transformer_dense_units)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, transformer_dense_units)

        return output, attention_weights

    def build_graph(self):
        v = Input(shape=(60, 512))
        k = Input(shape=(60, 512))
        q = Input(shape=(60, 512))
        return Model(inputs=[v, k, q], outputs=self.call(v, k, q, None))


def point_wise_feed_forward_network(model_params):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                model_params.transformer_dense_units, activation="relu"
            ),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(
                model_params.transformer_dense_units
            ),  # (batch_size, seq_len, transformer_dense_units)
        ]
    )


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_params, name="transformer_encoder_layer"):
        super(TransformerEncoderLayer, self).__init__()

        self.transformer_dense_units = model_params.transformer_dense_units

        self.mha = TransformerMultiHeadAttention(model_params)
        self.ffn = point_wise_feed_forward_network(model_params)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(model_params.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(model_params.dropout_rate)

    def call(self, x, mask, training=None):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, transformer_dense_units)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, transformer_dense_units)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, transformer_dense_units)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, transformer_dense_units)

        return out2

    def build_graph(self):
        x = Input(shape=(43, self.transformer_dense_units))
        return Model(inputs=x, outputs=self.call(x, None))


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_params, name="transformer_deocder_layer"):
        super(TransformerDecoderLayer, self).__init__()

        self.transformer_dense_units = model_params.transformer_dense_units
        self.mha1 = TransformerMultiHeadAttention(model_params)
        self.mha2 = TransformerMultiHeadAttention(model_params)

        self.ffn = point_wise_feed_forward_network(model_params)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(model_params.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(model_params.dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(model_params.dropout_rate)

    def call(self, x, enc_output, look_ahead_mask, padding_mask, training=None):
        # enc_output.shape == (batch_size, input_seq_len, transformer_dense_units)

        attn1, attn_weights_block1 = self.mha1(
            x, x, x, look_ahead_mask
        )  # (batch_size, target_seq_len, transformer_dense_units)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask
        )  # (batch_size, target_seq_len, transformer_dense_units)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, transformer_dense_units)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, transformer_dense_units)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, transformer_dense_units)

        return out3, attn_weights_block1, attn_weights_block2

    def build_graph(self):
        x = Input(shape=(50, self.transformer_dense_units))
        enc_output = Input(shape=(62, self.transformer_dense_units))
        return Model(inputs=[x, enc_output], outputs=self.call(x, enc_output, None, None, None))


class TransformerEncoder(tf.keras.Model):
    def __init__(self, model_params, name="tranformer_encoder"):
        super(TransformerEncoder, self).__init__()

        self.transformer_dense_units = model_params.transformer_dense_units
        self.transformer_num_layers = model_params.transformer_num_layers
        self.conjecture_input_length = model_params.conjecture_input_length

        self.embedding = tf.keras.layers.Embedding(
            model_params.input_vocab_size, model_params.transformer_dense_units
        )

        self.pos_encoding = _get_positional_encoding_function(
            model_params.transformer_input_order, model_params.transformer_dense_units
        )

        self.enc_layers = [
            TransformerEncoderLayer(model_params) for _ in range(model_params.transformer_num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(model_params.dropout_rate)

    def call(self, x, mask=None, training=None):

        # Compute the input mask padding mask if it is not provided
        if mask is None:
            mask = create_padding_mask(x)

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, transformer_dense_units)
        x *= tf.math.sqrt(tf.cast(self.transformer_dense_units, tf.float32))
        x = self.pos_encoding(x)

        x = self.dropout(x, training=training)

        for i in range(self.transformer_num_layers):
            x = self.enc_layers[i](x, mask, training=training)

        return x  # (batch_size, input_seq_len, transformer_dense_units)

    def build_graph(self):
        x = Input(shape=(62))
        return Model(inputs=x, outputs=self.call(x, None))


class TransformerDecoder(tf.keras.Model):
    def __init__(self, model_params, name="transformer_decoder"):
        super(TransformerDecoder, self).__init__()

        self.max_caption_length = model_params.max_caption_length

        self.transformer_dense_units = model_params.transformer_dense_units
        self.transformer_num_layers = model_params.transformer_num_layers

        self.embedding = tf.keras.layers.Embedding(
            model_params.target_vocab_size, model_params.transformer_dense_units
        )

        self.pos_encoding = _get_positional_encoding_function(
            model_params.transformer_input_order, model_params.transformer_dense_units
        )

        self.dec_layers = [
            TransformerDecoderLayer(model_params) for _ in range(model_params.transformer_num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(model_params.dropout_rate)
        self.output_layer = tf.keras.layers.Dense(model_params.target_vocab_size)
        self.flatten_ouput = Flatten()

    def call(self, x, enc_output, look_ahead_mask, padding_mask, training=None):

        if look_ahead_mask is None:
            look_ahead_mask = create_padding_mask(x)

        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, transformer_dense_units)
        x *= tf.math.sqrt(tf.cast(self.transformer_dense_units, tf.float32))

        x = self.pos_encoding(x)

        x = self.dropout(x, training=training)

        for i in range(self.transformer_num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, look_ahead_mask, padding_mask, training=training
            )

            attention_weights[f"decoder_layer{i+1}_block1"] = block1
            attention_weights[f"decoder_layer{i+1}_block2"] = block2

        # x.shape == (batch_size, target_seq_len, transformer_dense_units)

        # This might be a bit unexpected if a full sequence might be given
        x = self.flatten_ouput(x)

        x = self.output_layer(x)  # (batch_size, tar_seq_len*target_vocab_size)

        return x, attention_weights

    def build_graph(self):
        x = Input(shape=(self.max_caption_length))
        enc_output = Input(shape=(62, self.transformer_dense_units))
        return Model(inputs=[x, enc_output], outputs=self.call(x, enc_output, None, None))


class TransformerModel(tf.keras.Model):
    def __init__(self, model_params, name="transformer_model"):
        super().__init__()
        self.max_caption_length = model_params.max_caption_length
        self.encoder = TransformerEncoder(model_params)
        self.decoder = TransformerDecoder(model_params)
        self.conjecture_input_length = model_params.conjecture_input_length

    def call(self, inputs, training=None):
        # FIXME this is not adapted to be used with the training loop - and provbably should not be
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        padding_mask, look_ahead_mask = self.create_masks(inp, tar)

        enc_output = self.encoder(
            inp, padding_mask, training=training
        )  # (batch_size, inp_seq_len, transformer_dense_units)

        # dec_output.shape == (batch_size, tar_seq_len, transformer_dense_units)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, look_ahead_mask, padding_mask, training=training
        )

        return dec_output, attention_weights

    def create_masks(self, inp, tar):
        # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
        padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.

        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return padding_mask, look_ahead_mask

    def build_graph(self):
        inp = Input(shape=(self.conjecture_input_length))
        target = Input(shape=(self.max_caption_length))
        return Model(inputs=[inp, target], outputs=self.call([inp, target]))


def main():
    from utils import Namespace

    model_params = Namespace(
        transformer_num_layers=2,
        transformer_dense_units=512,
        num_attention_heads=8,
        dropout_rate=0.1,
        input_vocab_size=200,
        target_vocab_size=400,
        max_caption_length=22,
        conjecture_input_length=200,
    )

    # Initialise basic components - and test
    sample_ffn = point_wise_feed_forward_network(model_params)
    sample_ffn(tf.random.uniform((64, 50, 512))).shape

    print("\n", "# " * 16)
    print(" # # TransformerMultiHeadAttention")
    temp_mha = TransformerMultiHeadAttention(model_params)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, transformer_dense_units)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    out.shape, attn.shape
    temp_mha.build_graph().summary()

    print("\n", "# " * 16)
    print(" # # EncoderLayer")
    sample_encoder_layer = TransformerEncoderLayer(model_params)
    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((64, 43, model_params.transformer_dense_units)), None, False
    )
    sample_encoder_layer_output.shape  # (batch_size, input_seq_len, transformer_dense_units)
    sample_encoder_layer.build_graph().summary()

    print("\n", "# " * 16)
    print(" # # DecoderLayer")
    sample_decoder_layer = TransformerDecoderLayer(model_params)
    sample_decoder_layer_output, _, _ = sample_decoder_layer(
        tf.random.uniform((64, 50, model_params.transformer_dense_units)),
        sample_encoder_layer_output,
        None,
        None,
        False,
    )
    sample_decoder_layer_output.shape  # (batch_size, target_seq_len, transformer_dense_units)
    sample_decoder_layer.build_graph().summary()

    print("\n", "# " * 16)
    print(" # # Encoder")
    sample_encoder = TransformerEncoder(model_params)
    temp_input = tf.random.uniform(
        (
            64,
            min(63, model_params.conjecture_input_length),
        ),
        dtype=tf.int64,
        minval=0,
        maxval=200,
    )
    sample_encoder_output = sample_encoder(temp_input, mask=None, training=None)
    sample_encoder_output.shape
    sample_encoder.build_graph().summary()

    print("\n", "# " * 16)
    print(" # # Decoder")
    sample_decoder = TransformerDecoder(model_params)
    temp_input = tf.random.uniform(
        (64, model_params.max_caption_length), dtype=tf.int64, minval=0, maxval=200
    )
    output, attn = sample_decoder(
        temp_input, enc_output=sample_encoder_output, look_ahead_mask=None, padding_mask=None, training=None
    )
    output.shape, attn["decoder_layer2_block2"].shape
    sample_decoder.build_graph().summary()

    print(" # # Transformer")
    sample_transformer = TransformerModel(model_params)
    temp_input = tf.random.uniform(
        (64, model_params.conjecture_input_length), dtype=tf.int64, minval=0, maxval=200
    )
    temp_target = tf.random.uniform(
        (64, model_params.max_caption_length), dtype=tf.int64, minval=0, maxval=200
    )
    fn_out, _ = sample_transformer([temp_input, temp_target], training=False)
    fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)
    sample_transformer.build_graph().summary()


if __name__ == "__main__":

    main()
