from n2nds.data import Data
from n2nds.config import Config
from n2nds.reader import WeiboReader
import tensorflow as tf
from n2nds.seq2seq import Model


def main():
    train_weibo = WeiboReader("dataset/stc_weibo_train_post_generated_10",
                              "dataset/stc_weibo_train_response_generated_10",
                              batch_size=10)

    with tf.name_scope("Train"):
        with tf.variable_scope("Model", initializer=tf.random_uniform_initializer(-0.01, 0.01)) as scope:
            train_model = Model(train_weibo.config, is_train=True)

    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True):
            valid_model = Model(train_weibo.config, is_train=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sum_writer = tf.summary.FileWriter('summary/sum', sess.graph)
        iter = 0
        while True:
            iter += 1

            data = train_weibo.next_batch()
            # init_enc = sess.run(train_model.initial_enc_state)
            # init_dec = sess.run(train_model.initial_dec_state)
            feed_dict = {}
            # feed_dict[train_model.initial_enc_state.c] = init_enc.c
            # feed_dict[train_model.initial_enc_state.h] = init_enc.h
            # feed_dict[train_model.initial_dec_state.c] = init_dec.c
            # feed_dict[train_model.initial_dec_state.h] = init_dec.h
            feed_dict[train_model.utter_indices] = data.indices
            feed_dict[train_model.utter_lengths] = data.lengths
            feed_dict[train_model.utter_weights] = data.weights

            sess.run(train_model.minimizier, feed_dict)
            if iter % 50 == 0:
                cost, merged = sess.run([train_model.cost, train_model.merged], feed_dict)
                # sum_writer.add_summary(merged, global_step=iter / 50)
                print("iter %d : cost %f" % (iter // 50, cost))
                if cost < 0.9:
                    break

        print("Train model")
        data = train_weibo.next_batch()
        # init_enc = sess.run(train_model.initial_enc_state)
        # init_dec = sess.run(train_model.initial_dec_state)
        feed_dict = {}
        # feed_dict[train_model.initial_enc_state.c] = init_enc.c
        # feed_dict[train_model.initial_enc_state.h] = init_enc.h
        # feed_dict[train_model.initial_dec_state.c] = init_dec.c
        # feed_dict[train_model.initial_dec_state.h] = init_dec.h
        feed_dict[train_model.utter_indices] = data.indices
        feed_dict[train_model.utter_lengths] = data.lengths
        feed_dict[train_model.utter_weights] = data.weights
        pred = sess.run([train_model.pred], feed_dict)
        pred = pred[0].tolist()
        print(pred)
        i = 0
        while True:
            print(train_weibo.gen_words_from_indices(pred[i:i + train_weibo.config.SEQ_SIZE]))
            i += train_weibo.config.SEQ_SIZE
            if i >= len(pred):
                break

                # print("Valid model")
                #
                # init_enc = sess.run(valid_model.initial_enc_state)
                # init_dec = sess.run(valid_model.initial_dec_state)
                # feed_dict = {}
                # feed_dict[valid_model.initial_enc_state.c] = init_enc.c
                # feed_dict[valid_model.initial_enc_state.h] = init_enc.h
                # feed_dict[valid_model.initial_dec_state.c] = init_dec.c
                # feed_dict[valid_model.initial_dec_state.h] = init_dec.h
                #
                # pred= sess.run(valid_model.pred, feed_dict)
                # pred = pred.tolist()
                # print(pred)
                #
                # i = 0
                # while True:
                #     print(train_weibo.gen_words_from_indices(pred[i:i + config.SEQ_SIZE]))
                #     i += config.SEQ_SIZE
                #     if i >= len(pred):
                #         break


if __name__ == '__main__':
    main()
