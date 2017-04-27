import tensorflow as tf

from n2nds.config import Config
from n2nds.data import Data
from n2nds.reader import WeiboReader


class Model:
    def __init__(self, config, data, is_train=True):
        embedding = tf.get_variable("embedding", [config.VOCAB_SIZE, config.EMBED_SIZE], dtype=tf.float32)
        utter_indices = tf.Variable(data.indices, name="utter_indices")
        utter_lengths = tf.Variable(data.lengths, name="utter_lengths")
        utter_weights = tf.Variable(data.weights, name="utter_weights")
        utter_embs = tf.nn.embedding_lookup(embedding, utter_indices)
        encoder = tf.contrib.rnn.BasicLSTMCell(config.UNIT_SIZE, reuse=tf.get_variable_scope().reuse)
        decoder = tf.contrib.rnn.BasicLSTMCell(config.UNIT_SIZE, reuse=tf.get_variable_scope().reuse)

        self.initial_enc_state = enc_state = encoder.zero_state(config.BATCH_SIZE, tf.float32)
        with tf.variable_scope("encoder"):
            enc_outputs, _ = tf.nn.dynamic_rnn(encoder, utter_embs[:, 0, :, :], utter_lengths[:, 0],
                                               initial_state=enc_state)
            utter_lens = utter_lengths[:, 0]
            mask = tf.logical_and(tf.sequence_mask(utter_lens, config.SEQ_SIZE),
                                  tf.logical_not(tf.sequence_mask(utter_lens - 1, config.SEQ_SIZE)))
            enc_output = tf.boolean_mask(enc_outputs, mask)
            # for time_step in range(config.SEQ_SIZE):
            #     if time_step > 0:
            #         tf.get_variable_scope().reuse_variables()
            #     enc_output, enc_state = encoder(utter_embs[:, 0, time_step, :], enc_state)

        dec_state = decoder.zero_state(config.BATCH_SIZE, tf.float32)
        dec_outputs = []
        softmax_w = tf.get_variable("softmax_w", [config.EMBED_SIZE, config.VOCAB_SIZE], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [config.VOCAB_SIZE], dtype=tf.float32)

        dec_indices = tf.get_variable(shape=[config.BATCH_SIZE, 1], dtype=tf.float32, name="dec_indices")
        dec_input = enc_output
        with tf.variable_scope("decoder"):
            if is_train:
                for time_step in range(config.SEQ_SIZE):
                    if time_step == 0:
                        dec_output, dec_state = decoder(enc_output, dec_state)
                    else:
                        tf.get_variable_scope().reuse_variables()
                        dec_output, dec_state = decoder(utter_embs[:, 1, time_step, :], dec_state)
                    dec_outputs.append(dec_output)  # outputs: BATCH * SEQ_SIZE * EMB_SIZE
            else:
                dec_output, dec_state = decoder(dec_input, dec_state)
                self.dec_output_index = tf.argmax(tf.matmul(dec_output, softmax_w) + softmax_b, 1)

        if not is_train:
            return

        outputs = tf.reshape(dec_outputs, [-1, config.EMBED_SIZE])

        logits = tf.matmul(outputs, softmax_w) + softmax_b
        self.pred = tf.argmax(logits, 1)
        targets = tf.reshape(utter_indices[:, 1], [-1])
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
    data = Data()
    config = Config()
    weibo = WeiboReader("../dataset/stc_weibo_train_post_generated_10",
                        "../dataset/stc_weibo_train_response_generated_10")
    data, config = weibo.gen_data_and_config_from_dataset()

    with tf.name_scope("Train"):
        with tf.variable_scope("Model", initializer=tf.random_uniform_initializer(-0.01, 0.01)) as scope:
            train_model = Model(config, data, is_train=True)

    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True):
            valid_model = Model(config, data, is_train=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sum_writer = tf.summary.FileWriter('summary/sum', sess.graph)
        iter = 0
        while True:
            iter += 1
            sess.run(train_model.minimizier)
            if iter % 50 == 0:
                cost, merged = sess.run([train_model.cost, train_model.merged])
                sum_writer.add_summary(merged, global_step=iter / 50)
                print("epoch %d : cost %f" % (iter // 50, cost))
                if cost < 1:
                    break

        print("Predict model")
        pred = sess.run([train_model.pred])
        pred = pred[0].tolist()
        i = 0
        while True:
            print(weibo.gen_words_from_indices(pred[i:i + config.SEQ_SIZE]))
            i += config.SEQ_SIZE
            if i >= len(pred):
                break

        print("Valid model")

        pred = sess.run([valid_model.dec_output_index])
        pred = pred[0].tolist()
        print(weibo.gen_words_from_indices(pred))
        # i = 0
        # while True:
        #     print(weibo.gen_words_from_indices(pred[i:i + config.SEQ_SIZE]))
        #     i += config.SEQ_SIZE
        #     if i >= len(pred):
        #         break


if __name__ == '__main__':
    main()
