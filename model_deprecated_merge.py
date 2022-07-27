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

        self.word_decoder = WordDecoder(model_params)

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
