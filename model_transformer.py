import tensorflow as tf
import numpy as np

import config


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


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


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_params, name="transformer_attention_head"):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = model_params.num_heads
        self.d_model = model_params.d_model

        assert model_params.d_model % self.num_heads == 0

        self.depth = model_params.d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(model_params.d_model)
        self.wk = tf.keras.layers.Dense(model_params.d_model)
        self.wv = tf.keras.layers.Dense(model_params.d_model)

        self.dense = tf.keras.layers.Dense(model_params.d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(model_params):
    # TODO need to change this to tf.layer - fro saving and loading purposes
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(model_params.dff, activation="relu"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(model_params.d_model),  # (batch_size, seq_len, d_model)
        ]
    )


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_params, name="transformer_model_layer"):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(model_params)
        self.ffn = point_wise_feed_forward_network(model_params)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(model_params.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(model_params.dropout_rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_params, name="transformer_model_layer"):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(model_params)
        self.mha2 = MultiHeadAttention(model_params)

        self.ffn = point_wise_feed_forward_network(model_params)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(model_params.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(model_params.dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(model_params.dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(
            x, x, x, look_ahead_mask
        )  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask
        )  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, model_params, name="tranformer_encoder"):
        super(Encoder, self).__init__()

        self.d_model = model_params.d_model
        self.num_layers = model_params.num_layers

        self.embedding = tf.keras.layers.Embedding(model_params.conjecture_vocab_size, model_params.d_model)
        # TODO make optional
        self.pos_encoding = positional_encoding(config.CONJECTURE_INPUT_MAX_LENGTH, self.d_model)

        self.enc_layers = [EncoderLayer(model_params) for _ in range(model_params.num_layers)]

        self.dropout = tf.keras.layers.Dropout(model_params.dropout_rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, model_params, name="transformer_decoder"):
        super(Decoder, self).__init__()

        self.d_model = model_params.d_model
        self.num_layers = model_params.num_layers

        self.embedding = tf.keras.layers.Embedding(model_params.axiom_vocab_size, model_params.d_model)
        self.pos_encoding = positional_encoding(model_params.max_caption_length, self.d_model)

        self.dec_layers = [DecoderLayer(model_params) for _ in range(model_params.num_layers)]
        self.dropout = tf.keras.layers.Dropout(model_params.dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights[f"decoder_layer{i+1}_block1"] = block1
            attention_weights[f"decoder_layer{i+1}_block2"] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, model_params, name="transformer_model"):
        super().__init__()
        self.encoder = Encoder(model_params)
        self.decoder = Decoder(model_params)
        self.final_layer = tf.keras.layers.Dense(model_params.axiom_vocab_size)

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        padding_mask, look_ahead_mask = self.create_masks(inp, tar)

        enc_output = self.encoder(inp, training, padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

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


def main():
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    model_params = Namespace(
        num_layers=2,
        d_model=512,
        dff=2048,
        num_heads=8,
        dropout_rate=0.1,
        conjecture_vocab_size=200,
        axiom_vocab_size=400,
        max_caption_length=22,
    )

    # Initialise basic components - and test
    temp_mha = MultiHeadAttention(model_params)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    out.shape, attn.shape

    sample_ffn = point_wise_feed_forward_network(model_params)
    sample_ffn(tf.random.uniform((64, 50, 512))).shape

    sample_encoder_layer = EncoderLayer(model_params)
    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((64, 43, model_params.d_model)), False, None
    )
    sample_encoder_layer_output.shape  # (batch_size, input_seq_len, d_model)

    sample_decoder_layer = DecoderLayer(model_params)
    sample_decoder_layer_output, _, _ = sample_decoder_layer(
        tf.random.uniform((64, 50, model_params.d_model)), sample_encoder_layer_output, False, None, None
    )
    sample_decoder_layer_output.shape  # (batch_size, target_seq_len, d_model)

    sample_encoder = Encoder(model_params)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)
    sample_encoder_output.shape

    sample_decoder = Decoder(model_params)
    temp_input = tf.random.uniform(
        (64, model_params.max_caption_length), dtype=tf.int64, minval=0, maxval=200
    )
    output, attn = sample_decoder(
        temp_input, enc_output=sample_encoder_output, training=False, look_ahead_mask=None, padding_mask=None
    )
    output.shape, attn["decoder_layer2_block2"].shape

    sample_transformer = Transformer(model_params)
    temp_input = tf.random.uniform(
        (64, config.CONJECTURE_INPUT_MAX_LENGTH), dtype=tf.int64, minval=0, maxval=200
    )
    temp_target = tf.random.uniform(
        (64, model_params.max_caption_length), dtype=tf.int64, minval=0, maxval=200
    )
    fn_out, _ = sample_transformer([temp_input, temp_target], training=False)
    fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)


if __name__ == "__main__":

    main()
