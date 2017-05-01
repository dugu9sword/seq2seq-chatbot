import tensorflow as tf


class Model:
    def __init__(self, config, is_train=True, embedding_init_value=None, num_of_layer=1):
        if embedding_init_value is None:
            embedding = tf.get_variable("embedding", [config.VOCAB_SIZE, config.EMBED_SIZE], dtype=tf.float32)
        else:
            embedding = tf.get_variable("embedding", [config.VOCAB_SIZE, config.EMBED_SIZE], dtype=tf.float32,
                                        initializer=tf.constant_initializer(embedding_init_value))

        self.utter_indices = tf.placeholder(shape=[config.BATCH_SIZE, 2, config.SEQ_SIZE], name="utter_indices",
                                            dtype=tf.int32)
        self.utter_lengths = tf.placeholder(shape=[config.BATCH_SIZE, 2], name="utter_lengths", dtype=tf.int32)
        self.utter_weights = tf.placeholder(shape=[config.BATCH_SIZE, config.SEQ_SIZE], name="utter_weights",
                                            dtype=tf.int32)
        utter_embs = tf.nn.embedding_lookup(embedding, self.utter_indices)
        encoder = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(config.UNIT_SIZE, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
             for _ in range(num_of_layer)], state_is_tuple=True)
        decoder = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(config.UNIT_SIZE, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
             for _ in range(num_of_layer)], state_is_tuple=True)

        self.initial_enc_state = encoder.zero_state(config.BATCH_SIZE, tf.float32)
        enc_state = self.initial_enc_state
        with tf.variable_scope("encoder"):
            enc_outputs, _ = tf.nn.dynamic_rnn(encoder, utter_embs[:, 0, :, :], self.utter_lengths[:, 0],
                                               initial_state=enc_state)
            utter_lens = self.utter_lengths[:, 0]
            mask = tf.logical_and(tf.sequence_mask(utter_lens, config.SEQ_SIZE),
                                  tf.logical_not(tf.sequence_mask(utter_lens - 1, config.SEQ_SIZE)))
            enc_output = tf.boolean_mask(enc_outputs, mask)
            # for time_step in range(config.SEQ_SIZE):
            #     if time_step > 0:
            #         tf.get_variable_scope().reuse_variables()
            #     enc_output, enc_state = encoder(utter_embs[:, 0, time_step, :], enc_state)

        self.initial_dec_state = decoder.zero_state(config.BATCH_SIZE, tf.float32)
        dec_state = self.initial_dec_state
        dec_outputs = []
        softmax_w = tf.get_variable("softmax_w", [config.EMBED_SIZE, config.VOCAB_SIZE], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [config.VOCAB_SIZE], dtype=tf.float32)

        with tf.variable_scope("decoder"):
            for time_step in range(config.SEQ_SIZE):
                if time_step == 0:
                    dec_output, dec_state = decoder(enc_output, dec_state)
                else:
                    tf.get_variable_scope().reuse_variables()
                    if is_train:
                        dec_output, dec_state = decoder(utter_embs[:, 1, time_step - 1, :], dec_state)
                    else:
                        dec_output_index = tf.argmax(tf.matmul(dec_output, softmax_w) + softmax_b, axis=1)
                        previous_embedding = tf.nn.embedding_lookup(embedding, dec_output_index)
                        dec_output, dec_state = decoder(previous_embedding, dec_state)
                dec_outputs.append(dec_output)  # outputs: SEQ_SIZE * BATCH * EMB_SIZE

        outputs = tf.reshape(tf.concat(dec_outputs, axis=1), [-1, config.EMBED_SIZE])

        logits = tf.matmul(outputs, softmax_w) + softmax_b
        self.pred = tf.argmax(logits, 1)
        self.fuck_logits = logits
        targets = tf.reshape(self.utter_indices[:, 1], [-1])
        # targets = tf.reshape(utter_indices[:, 1],[config.BATCH_SIZE, config.SEQ_SIZE])
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [targets],
            [tf.to_float(tf.reshape(self.utter_weights, [config.BATCH_SIZE * config.SEQ_SIZE]))])
        self.cost = tf.reduce_sum(loss) / config.BATCH_SIZE
        opt = tf.train.AdamOptimizer()
        self.grad_and_vars = opt.compute_gradients(loss)
        self.train_op = opt.apply_gradients(self.grad_and_vars)

        tf.summary.scalar('loss', self.cost)
        self.merged = tf.summary.merge_all()
