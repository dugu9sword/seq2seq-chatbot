from n2nds.data import Data
from n2nds.config import Config
from n2nds.reader import WeiboReader
import tensorflow as tf
from n2nds.seq2seq import Model


def main():
    data = Data()
    config = Config()

    train_weibo = WeiboReader("dataset/stc_weibo_train_post_generated_100",
                        "dataset/stc_weibo_train_response_generated_100")
    data, config = train_weibo.gen_data_and_config_from_dataset()

    # valid_data, valid_config = train_weibo.gen_data_and_config_from_dataset()
    # valid_data = Data()
    # valid_config = Config()
    # valid_data.indices=[[[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]]
    # valid_data.lengths=[[13,11]]
    # valid_data.weights=[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


    # valid_data.indices = [[[52, 53, 38, 54, 55, 56, 57, 58, 32, 59, 60, 61, 62, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [63, 64, 65, 32, 66, 67, 36, 44, 68, 69, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]]
    # valid_data.lengths = [[14, 11]]
    # valid_data.weights = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # valid_config.BATCH_SIZE = 1
    # valid_config.SEQ_SIZE = config.SEQ_SIZE
    # valid_config.VOCAB_SIZE = config.VOCAB_SIZE

    with tf.name_scope("Train"):
        with tf.variable_scope("Model", initializer=tf.random_uniform_initializer(-0.01, 0.01)) as scope:
            train_model = Model(config, data, is_train=True)

    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True):
            valid_model = Model(config, data, is_train=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sum_writer = tf.summary.FileWriter('summary/sum', sess.graph)
        iter = 0
        while True:
            iter += 1
            init_enc = sess.run(train_model.initial_enc_state)
            init_dec = sess.run(train_model.initial_dec_state)
            feed_dict = {}
            feed_dict[train_model.initial_enc_state.c] = init_enc.c
            feed_dict[train_model.initial_enc_state.h] = init_enc.h
            feed_dict[train_model.initial_dec_state.c] = init_dec.c
            feed_dict[train_model.initial_dec_state.h] = init_dec.h

            sess.run(train_model.minimizier, feed_dict)
            if iter % 50 == 0:
                cost, merged = sess.run([train_model.cost, train_model.merged])
                # sum_writer.add_summary(merged, global_step=iter / 50)
                print("iter %d : cost %f" % (iter // 50, cost))
                if cost < 1:
                    break

        print("Train model")
        pred = sess.run([train_model.pred])
        pred = pred[0].tolist()
        print(pred)
        i = 0
        while True:
            print(train_weibo.gen_words_from_indices(pred[i:i + config.SEQ_SIZE]))
            i += config.SEQ_SIZE
            if i >= len(pred):
                break

        print("Valid model")

        init_enc = sess.run(valid_model.initial_enc_state)
        init_dec = sess.run(valid_model.initial_dec_state)
        feed_dict = {}
        feed_dict[valid_model.initial_enc_state.c] = init_enc.c
        feed_dict[valid_model.initial_enc_state.h] = init_enc.h
        feed_dict[valid_model.initial_dec_state.c] = init_dec.c
        feed_dict[valid_model.initial_dec_state.h] = init_dec.h

        pred, fuck_inputs, fuck_outputs, fuck_logits, fuck_previous_embeddings = sess.run(
            [valid_model.pred, valid_model.fuck_inputs, valid_model.fuck_outputs,
             valid_model.fuck_logits, valid_model.fuck_previous_embeddings], feed_dict)
        pred = pred.tolist()
        print(pred)

        # fuck_inputs = fuck_inputs.tolist()
        # print("fuck previous embeddings")
        # print(fuck_previous_embeddings)
        # print("fuck inputs")
        # print(fuck_inputs)
        # print("fuck outputs")
        # print(fuck_outputs)
        # print("fuck logits")
        # print(fuck_logits)
        i = 0
        while True:
            print(train_weibo.gen_words_from_indices(pred[i:i + config.SEQ_SIZE]))
            i += config.SEQ_SIZE
            if i >= len(pred):
                break

        # print("Word fed to the lstm")
        # i = 0
        # while True:
        #     print(weibo.gen_words_from_indices(fuck_inputs[i:i + config.SEQ_SIZE - 1]))
        #     i += config.SEQ_SIZE -1
        #     if i >= len(fuck_inputs):
        #         break

        # print(weibo.gen_words_from_indices(fuck_inputs))
        # print(weibo.gen_words_from_indices(pred))


if __name__ == '__main__':
    main()
