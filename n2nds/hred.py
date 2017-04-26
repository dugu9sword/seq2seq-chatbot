import tensorflow as tf


class Config:
    SEQ_SIZE = 5
    BATCH_SIZE = 2
    EMBED_SIZE = UNIT_SIZE = 20
    VOCAB_SIZE = 10


class Data:
    def __init__(self):
        self.utterances = [[[2, 3, 4, 5, 0],
                            [4, 5, 0, 0, 0],
                            [6, 7, 8, 0, 0]],
                           [[1, 2, 3, 0, 0],
                            [3, 4, 5, 0, 0],
                            [5, 6, 7, 0, 0]]]
        self.length = [[5, 4],
                       [3, 4],
                       [4, 4]]
        self.weights = [[[1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 0]],
                        [[1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 0]]]


class Model:
    def __init__(self, config, data):
        embedding = tf.get_variable("embedding", [config.VOCAB_SIZE, config.EMBED_SIZE], dtype=tf.float32)
        utter_embs = tf.nn.embedding_lookup(embedding, data.utterances)
        encoder = tf.contrib.rnn.BasicLSTMCell(config.UNIT_SIZE, reuse=tf.get_variable_scope().reuse)
        context = tf.contrib.rnn.BasicLSTMCell(config.UNIT_SIZE, reuse=tf.get_variable_scope().reuse)
        decoder = tf.contrib.rnn.BasicLSTMCell(config.UNIT_SIZE, reuse=tf.get_variable_scope().reuse)

        # enc_state = encoder.zero_state(config.BATCH_SIZE, tf.float32)
        with tf.variable_scope("encoder"):
            enc_outputs, _ = tf.nn.dynamic_rnn(encoder, utter_embs[:, 0, :, :], data.length[0],
                                               initial_state=encoder.zero_state(config.BATCH_SIZE, tf.float32))
            utter_lens = tf.Variable(data.length[0])
            mask = tf.logical_and(tf.sequence_mask(utter_lens, config.SEQ_SIZE),
                                  tf.logical_not(tf.sequence_mask(utter_lens - 1, config.SEQ_SIZE)))
            enc_output = tf.boolean_mask(enc_outputs, mask)
            # for time_step in range(config.SEQ_SIZE):
            #     if time_step > 0:
            #         tf.get_variable_scope().reuse_variables()
            #     enc_output, enc_state = encoder(utter_embs[:, 0, time_step, :], enc_state)

        con_state = context.zero_state(config.BATCH_SIZE, tf.float32)
        with tf.variable_scope("context"):
            con_output, con_state = context(enc_output, con_state)

        dec_state = decoder.zero_state(config.BATCH_SIZE, tf.float32)
        dec_outputs_1 = []
        with tf.variable_scope("decoder"):
            for time_step in range(config.SEQ_SIZE):
                if time_step == 0:
                    dec_output, dec_state = decoder(con_output, dec_state)
                else:
                    tf.get_variable_scope().reuse_variables()
                    dec_output, dec_state = decoder(utter_embs[:, 1, time_step, :], dec_state)
                dec_outputs_1.append(dec_output)  # outputs: BATCH * SEQ_SIZE * EMB_SIZE

        # enc_state = encoder.zero_state(config.BATCH_SIZE, tf.float32)
        with tf.variable_scope("encoder"):
            tf.get_variable_scope().reuse_variables()
            enc_output, _ = tf.nn.dynamic_rnn(encoder, utter_embs[:, 1, :, :], data.length[1],
                                              initial_state=encoder.zero_state(config.BATCH_SIZE, tf.float32))
            utter_lens = tf.Variable(data.length[1])
            mask = tf.logical_and(tf.sequence_mask(utter_lens, config.SEQ_SIZE),
                                  tf.logical_not(tf.sequence_mask(utter_lens - 1, config.SEQ_SIZE)))
            enc_output = tf.boolean_mask(enc_outputs, mask)
            # print(tf.get_variable_scope().reuse)
            # for time_step in range(config.SEQ_SIZE):
            #     enc_output, enc_state = encoder(utter_embs[:, 1, time_step, :], enc_state)

        with tf.variable_scope("context"):
            tf.get_variable_scope().reuse_variables()
            con_output, con_state = context(enc_output, con_state)

        dec_state = decoder.zero_state(config.BATCH_SIZE, tf.float32)
        dec_outputs_2 = []
        with tf.variable_scope("decoder"):
            tf.get_variable_scope().reuse_variables()
            for time_step in range(config.SEQ_SIZE):
                if time_step == 0:
                    dec_output, dec_state = decoder(con_output, dec_state)
                else:
                    dec_output, dec_state = decoder(utter_embs[:, 2, time_step, :], dec_state)
                dec_outputs_2.append(dec_output)

        outputs = tf.concat([dec_outputs_1, dec_outputs_2], axis=0)
        outputs = tf.reshape(outputs, [-1, config.EMBED_SIZE])

        softmax_w = tf.get_variable("softmax_w", [config.EMBED_SIZE, config.VOCAB_SIZE], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [config.VOCAB_SIZE], dtype=tf.float32)
        logits = tf.matmul(outputs, softmax_w) + softmax_b
        self.pred = tf.argmax(logits, 1)
        _, targets = tf.split(data.utterances, [1, 2], axis=1)
        targets = tf.reshape(targets, [-1])
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [targets],
            [tf.to_float(tf.reshape(data.weights, [-1]))])
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
