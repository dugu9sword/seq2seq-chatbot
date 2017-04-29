import tensorflow as tf
import os
from n2nds.reader import WeiboReader
from n2nds.seq2seq import Model


def main():
    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    # Load the data set
    train_weibo = WeiboReader("dataset/stc_weibo_train_post_generated_10",
                              "dataset/stc_weibo_train_response_generated_10",
                              batch_size=10)

    # Create the model
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", initializer=tf.random_uniform_initializer(-0.01, 0.01)):
            train_model = Model(train_weibo.config, is_train=True)

    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True):
            valid_model = Model(train_weibo.config, is_train=False)

    # Start the supervised session
    sv = tf.train.Supervisor(logdir="tmp/logdir/",
                             save_model_secs=2,
                             save_summaries_secs=1,
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

            sess.run(train_model.minimizier, feed_dict)
            if iter % 50 == 0:
                cost, merged = sess.run([train_model.cost, train_model.merged], feed_dict)
                sv.summary_computed(sess, summary=merged, global_step=iter)
                # sum_writer.add_summary(merged, global_step=iter / 50)
                print("iter %d : cost %f" % (iter // 50, cost))
                if cost < 0.1:
                    break


        train_output = open("tmp/train_output.txt", "w")
        valid_output = open("tmp/valid_output.txt", "w")
        for model, output in list(zip([train_model, valid_model],[train_output, valid_output])):
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
