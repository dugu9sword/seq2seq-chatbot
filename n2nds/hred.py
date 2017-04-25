import tensorflow as tf


class Config:
    SEQ_SIZE = 5
    BATCH_SIZE = 2
    EMBED_SIZE = UNIT_SIZE = 20
    VOCAB_SIZE = 10


class Data:
    def __init__(self):
        self.utterances = [[[2, 3, 4, 5, 0],
                            [4, 5, 6, 7, 0],
                            [6, 7, 8, 9, 0]],
                           [[1, 2, 3, 4, 0],
                            [3, 4, 5, 6, 0],
                            [5, 6, 7, 8, 0]]]


class Model:
    def __init__(self, config, data):
        embedding = tf.get_variable("embedding", [config.VOCAB_SIZE, config.EMBED_SIZE], dtype=tf.float32)
        utterances = tf.nn.embedding_lookup(embedding, data.utterances)
        encoder = tf.contrib.rnn.BasicLSTMCell(config.UNIT_SIZE, reuse=tf.get_variable_scope().reuse)
        context = tf.contrib.rnn.BasicLSTMCell(config.UNIT_SIZE, reuse=tf.get_variable_scope().reuse)
        decoder = tf.contrib.rnn.BasicLSTMCell(config.UNIT_SIZE, reuse=tf.get_variable_scope().reuse)

        enc_state = encoder.zero_state(config.BATCH_SIZE, tf.float32)
        with tf.variable_scope("encoder_1"):
            for time_step in config.SEQ_SIZE:
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                enc_output, enc_state = encoder(utterances[:, 0, time_step, :], enc_state)

        con_state = context.zero_state(config.BATCH_SIZE, tf.float32)
        with tf.variable_scope("context"):
            con_output, con_state = context(enc_output, con_state)

        dec_state = decoder.zero_state(config.BATCH_SIZE, tf.float32)
        dec_outputs_1 = []
        with tf.variable_scope("decoder_1"):
            for time_step in config.SEQ_SIZE:
                if time_step == 0:
                    dec_output, dec_state = decoder(con_output, dec_state)
                else:
                    tf.get_variable_scope().reuse_variables()
                    dec_output, dec_state = decoder(utterances[:, 1, time_step, :], dec_state)
                dec_outputs_1.append(dec_output)

                enc_state = encoder.zero_state(config.BATCH_SIZE, tf.float32)
        with tf.variable_scope("encoder_2"):
            for time_step in config.SEQ_SIZE:
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                enc_output, enc_state = encoder(utterances[:, 1, time_step, :], enc_state)

        con_state = context.zero_state(config.BATCH_SIZE, tf.float32)
        with tf.variable_scope("context"):
            con_output, con_state = context(enc_output, con_state)

        dec_state = decoder.zero_state(config.BATCH_SIZE, tf.float32)
        dec_outputs_2 = []
        with tf.variable_scope("decoder_2"):
            for time_step in config.SEQ_SIZE:
                if time_step == 0:
                    dec_output, dec_state = decoder(con_output, dec_state)
                else:
                    tf.get_variable_scope().reuse_variables()
                    dec_output, dec_state = decoder(utterances[:, 2, time_step, :], dec_state)
                dec_outputs_2.append(dec_output)

        outputs = tf.concat([dec_outputs_1, dec_outputs_2], axis=0)
        outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, config.EMBED_SIZE])

        softmax_w = tf.get_variable("softmax_w", [config.EMBED_SIZE, config.VOCAB_SIZE], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [config.VOCAB_SIZE], dtype=tf.float32)
        logits = tf.matmul(outputs, softmax_w) + softmax_b
        self.pred = tf.argmax(logits, 1)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(data.y, [-1])],
            [tf.ones([config.BATCH_SIZE * config.SEQ_SIZE], dtype=tf.float32)])
        self.cost = tf.reduce_sum(loss) / config.BATCH_SIZE
        opt = tf.train.AdamOptimizer()
        self.minimizier = opt.minimize(loss)


def main():
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", initializer=tf.random_uniform_initializer(-0.01, 0.01)) as scope:
            model = Model(Config(), Data())


if __name__ == '__main__':
    main()
