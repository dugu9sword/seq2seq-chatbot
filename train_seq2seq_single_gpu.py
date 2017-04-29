import os

import tensorflow as tf

from n2nds.reader import WeiboReader
from n2nds.seq2seq import Model


def main():
    data_set_used = 100
    batch_size = 10
    post_path = "dataset/stc_weibo_train_post_generated_%d" % data_set_used
    response_path = "dataset/stc_weibo_train_response_generated_%d" % data_set_used

    # Load the data set
    train_weibo = WeiboReader(post_path, response_path, batch_size=batch_size)
    train_weibo.config.EMBED_SIZE = 100
    train_weibo.config.UNIT_SIZE = 100

    # Set the log path for storing model and summary
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    model_id = "a_%d_b_%d" % (data_set_used, batch_size)
    if not os.path.exists("tmp/output_%s" % model_id):
        os.mkdir("tmp/output_%s" % model_id)
    log_dir_path = "tmp/logdir_%s" % model_id
    train_output_path = "tmp/output_%s/train_output.txt" % model_id
    valid_output_path = "tmp/output_%s/valid_output.txt" % model_id

    # Create the model
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", initializer=tf.random_uniform_initializer(-0.01, 0.01)):
            train_model = Model(train_weibo.config, is_train=True)

    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True):
            valid_model = Model(train_weibo.config, is_train=False)

    # Start the supervised session
    sv = tf.train.Supervisor(logdir=log_dir_path,
                             save_model_secs=300,
                             save_summaries_secs=60,
                             summary_op=None)
    with sv.managed_session() as sess:
        # sum_writer = tf.summary.FileWriter('summary/sum', sess.graph)

        iter = 0
        while True:
            iter += 1

            data = train_weibo.next_batch()
            feed_dict = {}
            feed_dict[train_model.utter_indices] = data.indices
            feed_dict[train_model.utter_lengths] = data.lengths
            feed_dict[train_model.utter_weights] = data.weights

            sess.run(train_model.train_op, feed_dict)
            if iter % 50 == 0:
                cost, merged = sess.run([train_model.cost, train_model.merged], feed_dict)
                sv.summary_computed(sess, summary=merged, global_step=iter)
                # sum_writer.add_summary(merged, global_step=iter / 50)
                print("iter %d : cost %f" % (iter // 50, cost))
                if cost < 0.001:
                    break

        train_output = open(train_output_path, "w")
        valid_output = open(valid_output_path, "w")
        for model, output in list(zip([train_model, valid_model], [train_output, valid_output])):
            data = train_weibo.next_batch()
            feed_dict = {}
            feed_dict[model.utter_indices] = data.indices
            feed_dict[model.utter_lengths] = data.lengths
            feed_dict[model.utter_weights] = data.weights
            pred = sess.run(model.pred, feed_dict)
            pred = pred.tolist()
            i = 0
            while True:
                pred_res = train_weibo.gen_words_from_indices(pred[i:i + train_weibo.config.SEQ_SIZE])
                print(pred_res)
                output.write(pred_res)
                output.write("\n")
                i += train_weibo.config.SEQ_SIZE
                if i >= len(pred):
                    break


if __name__ == '__main__':
    main()