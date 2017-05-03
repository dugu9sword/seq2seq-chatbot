import tensorflow as tf


class Model:
    def __init__(self, config, is_train=True, embedding_init_value=None, num_of_layer=1):
        if embedding_init_value is None:
            embedding = tf.get_variable("embedding", [config.VOCAB_SIZE, config.EMBED_SIZE], dtype=tf.float32)
        else:
            embedding = tf.get_variable("embedding", [config.VOCAB_SIZE, config.EMBED_SIZE], dtype=tf.float32,
                                        initializer=tf.constant_initializer(embedding_init_value))

        self.utter_indices = tf.placeholder(shape=[None, 2, config.SEQ_SIZE], name="utter_indices",
                                            dtype=tf.int32)
        self.utter_lengths = tf.placeholder(shape=[None, 2], name="utter_lengths", dtype=tf.int32)
        self.utter_weights = tf.placeholder(shape=[None, config.SEQ_SIZE], name="utter_weights",
                                            dtype=tf.int32)
        batch_size = tf.shape(self.utter_indices)[0]
        utter_embs = tf.nn.embedding_lookup(embedding, self.utter_indices)
        encoder = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(config.UNIT_SIZE, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
             for _ in range(num_of_layer)], state_is_tuple=True)
        decoder = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(config.UNIT_SIZE, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
             for _ in range(num_of_layer)], state_is_tuple=True)

        self.initial_enc_state = encoder.zero_state(batch_size, tf.float32)
        enc_state = self.initial_enc_state
        with tf.variable_scope("encoder"):
            enc_outputs, enc_states = tf.nn.dynamic_rnn(encoder, utter_embs[:, 0, :, :], self.utter_lengths[:, 0],
                                                        initial_state=enc_state)
            # utter_lens = self.utter_lengths[:, 0]
            # mask = tf.logical_and(tf.sequence_mask(utter_lens, config.SEQ_SIZE),
            #                       tf.logical_not(tf.sequence_mask(utter_lens - 1, config.SEQ_SIZE)))
            # enc_output = tf.boolean_mask(enc_outputs, mask)
            # enc_state = tf.boolean_mask(enc_states, mask)
            # for time_step in range(config.SEQ_SIZE):
            #     if time_step > 0:
            #         tf.get_variable_scope().reuse_variables()
            #     enc_output, enc_state = encoder(utter_embs[:, 0, time_step, :], enc_state)

        # self.initial_dec_state = decoder.zero_state(config.BATCH_SIZE, tf.float32)
        # dec_state = self.initial_dec_state
        dec_state = enc_states
        dec_first_input = tf.zeros([batch_size, config.EMBED_SIZE])
        dec_outputs = []
        softmax_w = tf.get_variable("softmax_w", [config.EMBED_SIZE, config.VOCAB_SIZE], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [config.VOCAB_SIZE], dtype=tf.float32)

        if is_train:
            with tf.variable_scope("decoder"):
                for time_step in range(config.SEQ_SIZE):
                    if time_step == 0:
                        dec_output, dec_state = decoder(dec_first_input, dec_state)
                    else:
                        tf.get_variable_scope().reuse_variables()
                        dec_output, dec_state = decoder(utter_embs[:, 1, time_step - 1, :], dec_state)
                    dec_outputs.append(dec_output)  # outputs: SEQ_SIZE * BATCH * EMB_SIZE
        else:
            # suppose batch_size is 1 when decoding
            # then we employ a batch as a beam
            beam_size = 10
            beam_previous_inputs = tf.zeros(shape=[beam_size, config.EMBED_SIZE], dtype=tf.float32)
            beam_last_indices = tf.zeros(shape=[beam_size], dtype=tf.int32)
            beam_kept_indices = tf.zeros(shape=[beam_size, config.SEQ_SIZE], dtype=tf.int32)
            beam_last_probs = tf.zeros(shape=[beam_size], dtype=tf.float32)
            beam_states = tf.reshape(
                tf.stack([enc_states for _ in range(beam_size)]), shape=[beam_size, -1])

            with tf.variable_scope("decoder"):
                for time_step in range(config.SEQ_SIZE):
                    if time_step != 0:
                        tf.get_variable_scope().reuse_variables()
                    beam_outputs, beam_states = decoder(beam_previous_inputs, beam_states)

                    beam_logits = tf.matmul(beam_outputs, softmax_w) + softmax_b  # [beam, vocab]
                    if time_step == 0:
                        top_values, top_indices = tf.nn.top_k(beam_logits[0, :], k=beam_size)
                        beam_last_indices = tf.reshape(top_indices, shape=[beam_size])
                        beam_last_probs = tf.reshape(top_values, shape=[beam_size])
                    else:
                        beam_last_probs = tf.stack([beam_last_probs for _ in range(config.VOCAB_SIZE)], axis=1)
                        beam_probs = beam_last_probs * beam_logits
                        beam_probs = tf.reshape(beam_probs, shape=[beam_size * config.VOCAB_SIZE])
                        top_values, top_indices = tf.nn.top_k(beam_probs, k=beam_size)
                        beam_last_probs = tf.reshape(top_values, shape=[beam_size])
                        beam_tmp_kept_indices = tf.zeros_like(beam_kept_indices)
                        for i, index in enumerate(top_indices):
                            this_index = tf.mod(index, config.VOCAB_SIZE)
                            prev_index = tf.div(index, config.VOCAB_SIZE)
                            beam_last_indices[i] = this_index
                            beam_tmp_kept_indices[i, 0:time_step] = beam_kept_indices[prev_index, 0:time_step]
                            beam_tmp_kept_indices[i, time_step] = this_index
                        beam_kept_indices = beam_tmp_kept_indices
                    beam_previous_inputs = tf.nn.embedding_lookup(embedding, beam_kept_indices)

        outputs = tf.reshape(tf.concat(dec_outputs, axis=1), [-1, config.EMBED_SIZE])

        logits = tf.matmul(outputs, softmax_w) + softmax_b
        self.pred = tf.argmax(logits, 1)
        targets = tf.reshape(self.utter_indices[:, 1], [-1])
        # targets = tf.reshape(utter_indices[:, 1],[config.BATCH_SIZE, config.SEQ_SIZE])
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [targets],
            [tf.to_float(tf.reshape(self.utter_weights, [batch_size * config.SEQ_SIZE]))])
        self.cost = tf.reduce_sum(loss) / tf.to_float(batch_size)
        opt = tf.train.AdamOptimizer()
        self.grad_and_vars = opt.compute_gradients(loss)
        self.train_op = opt.apply_gradients(self.grad_and_vars)

        tf.summary.scalar('loss', self.cost)
        self.merged = tf.summary.merge_all()
