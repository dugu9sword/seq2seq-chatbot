import tensorflow as tf


class Config:
    SEQ_SIZE = 5
    BATCH_SIZE = 2
    EMBED_SIZE = UNIT_SIZE = 20
    VOCAB_SIZE = 10


class Data:
    def __init__(self):
        # Should be [batch_size * 2 * sequence_size]
        self.utterances = [[[2, 3, 4, 5, 0],
                            [4, 5, 0, 0, 0]],
                           [[1, 2, 3, 0, 0],
                            [3, 4, 5, 0, 0]]]
        # Should be [batch_size * 2]
        self.length = [[5, 4],
                       [3, 4]]
        # Should be [batch_size * sequence_size]
        self.weights = [[1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 0]]


class Model:
    def __init__(self, config, data):
        embedding = tf.get_variable("embedding", [config.VOCAB_SIZE, config.EMBED_SIZE], dtype=tf.float32)
        utter_embs = tf.nn.embedding_lookup(embedding, data.utterances)
        utter_indices = tf.Variable(data.utterances)
        utter_length = tf.Variable(data.length)
        utter_weights = tf.Variable(data.weights)
        encoder = tf.contrib.rnn.BasicLSTMCell(config.UNIT_SIZE, reuse=tf.get_variable_scope().reuse)
        decoder = tf.contrib.rnn.BasicLSTMCell(config.UNIT_SIZE, reuse=tf.get_variable_scope().reuse)

        # enc_state = encoder.zero_state(config.BATCH_SIZE, tf.float32)
        with tf.variable_scope("encoder"):
            enc_outputs, _ = tf.nn.dynamic_rnn(encoder, utter_embs[:, 0, :, :], utter_length[:, 0],
                                               initial_state=encoder.zero_state(config.BATCH_SIZE, tf.float32))
            utter_lens = tf.Variable(utter_length[:, 0])
            mask = tf.logical_and(tf.sequence_mask(utter_lens, config.SEQ_SIZE),
                                  tf.logical_not(tf.sequence_mask(utter_lens - 1, config.SEQ_SIZE)))
            enc_output = tf.boolean_mask(enc_outputs, mask)
            # for time_step in range(config.SEQ_SIZE):
            #     if time_step > 0:
            #         tf.get_variable_scope().reuse_variables()
            #     enc_output, enc_state = encoder(utter_embs[:, 0, time_step, :], enc_state)

        dec_state = decoder.zero_state(config.BATCH_SIZE, tf.float32)
        dec_outputs = []
        with tf.variable_scope("decoder"):
            for time_step in range(config.SEQ_SIZE):
                if time_step == 0:
                    dec_output, dec_state = decoder(enc_output, dec_state)
                else:
                    tf.get_variable_scope().reuse_variables()
                    dec_output, dec_state = decoder(utter_embs[:, 1, time_step, :], dec_state)
                dec_outputs.append(dec_output)  # outputs: BATCH * SEQ_SIZE * EMB_SIZE

        outputs = tf.reshape(dec_outputs, [-1, config.EMBED_SIZE])

        softmax_w = tf.get_variable("softmax_w", [config.EMBED_SIZE, config.VOCAB_SIZE], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [config.VOCAB_SIZE], dtype=tf.float32)
        logits = tf.matmul(outputs, softmax_w) + softmax_b
        self.pred = tf.argmax(logits, 1)
        _, targets = tf.split(data.utterances, [1], axis=1)
        targets = tf.reshape(targets, [-1])
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [targets],
            [tf.to_float(tf.reshape(utter_weights, [-1]))])
        self.cost = tf.reduce_sum(loss) / config.BATCH_SIZE
        opt = tf.train.AdamOptimizer()
        self.minimizier = opt.minimize(loss)

        tf.summary.scalar('loss', self.cost)
        self.merged = tf.summary.merge_all()


def main():
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", initializer=tf.random_uniform_initializer(-0.01, 0.01)) as scope:
            model = Model(Config(), Data())

    with tf.Session() as sess:
        _ = sess.run(tf.global_variables_initializer())
        sum_writer = tf.summary.FileWriter('summary/sum', sess.graph)
        while True:
            _, cost, merged = sess.run([model.minimizier, model.cost, model.merged])
            sum_writer.add_summary(merged)
            print(cost)
            if cost < 0.1:
                break
        pred = sess.run([model.pred])
        print(pred)


if __name__ == '__main__':
    main()
