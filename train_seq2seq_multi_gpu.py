import os

import tensorflow as tf

from n2nds.reader import WeiboReader, SpToken
from n2nds.seq2seq import Model

data_set_used = 1000000
batch_size = 500
gpu_nums = 4
layer_nums = 4
info = "normal"
post_path = "dataset/stc_weibo_train_post_generated_%d" % data_set_used
response_path = "dataset/stc_weibo_train_response_generated_%d" % data_set_used

# Load the data set
train_weibo = WeiboReader(post_path, response_path, batch_size=batch_size)
train_weibo.config.EMBED_SIZE = 200
train_weibo.config.UNIT_SIZE = 200


def multi_gpu_model(num_gpus=1):
    grads = []
    reuse_flag = False
    for i in range(num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope("tower_%d" % i):
                with tf.variable_scope("Model", reuse=reuse_flag):
                    model = Model(train_weibo.config, is_train=True,
                                  embedding_init_value=train_weibo.embedding)
                    reuse_flag = True
                tf.add_to_collection("train_model", model)
                tf.add_to_collection("train_cost", model.cost)
                tf.add_to_collection("train_merged", model.merged)
                grads.append(model.grad_and_vars)
    with tf.device('/cpu:0'):
        average_grads = average_gradients(grads)
        opt = tf.train.AdamOptimizer()
        train_op = opt.apply_gradients(average_grads)
    return train_op


def main():
    # Set the log path for storing model and summary
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    model_id = "a_%d_b_%d_l_%d_g_%d_i_%s" % (data_set_used, batch_size, layer_nums, gpu_nums, info)
    if not os.path.exists("tmp/output_%s" % model_id):
        os.mkdir("tmp/output_%s" % model_id)
    log_dir_path = "tmp/logdir_%s" % model_id
    train_output_path = "tmp/output_%s/train_output.txt" % model_id
    valid_output_path = "tmp/output_%s/valid_output.txt" % model_id

    # Create the model
    with tf.name_scope("Train"):
        train_op = multi_gpu_model(gpu_nums)

    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True):
            valid_model = Model(train_weibo.config, is_train=False)

    # Start the supervised session
    sv = tf.train.Supervisor(logdir=log_dir_path,
                             save_model_secs=300,
                             save_summaries_secs=60,
                             summary_op=None)
    with sv.managed_session(config=tf.ConfigProto(
            log_device_placement=True,
            allow_soft_placement=True, )) as sess:
        # sum_writer = tf.summary.FileWriter('summary/sum', sess.graph)

        iter = 0
        while True:
            iter += 1

            models = tf.get_collection("train_model")
            feed_dict = dict()
            for model in models:
                data = train_weibo.next_batch()
                feed_dict[model.utter_indices] = data.indices
                feed_dict[model.utter_lengths] = data.lengths
                feed_dict[model.utter_weights] = data.weights
            sess.run(train_op, feed_dict)

            if iter % 50 == 0:
                cost = tf.get_collection("train_cost")
                merged = tf.get_collection("train_merged")
                merged, cost = sess.run([merged, cost], feed_dict)

                sv.summary_computed(sess, summary=merged[0], global_step=iter)
                # sum_writer.add_summary(merged, global_step=iter / 50)
                print("iter %d : cost %s" % (iter // 50, cost))
                if cost[0] < 65:
                    break

        models = tf.get_collection("train_model")
        train_output = open(train_output_path, "w")
        valid_output = open(valid_output_path, "w")
        data = train_weibo.next_batch()
        for model, output in list(zip([models[0], valid_model], [train_output, valid_output])):
            feed_dict = {}
            feed_dict[model.utter_indices] = data.indices
            feed_dict[model.utter_lengths] = data.lengths
            feed_dict[model.utter_weights] = data.weights

            pred = sess.run(model.pred, feed_dict)
            chunks = lambda arr, n: [arr[i:i + n] for i in range(0, len(arr), n)]
            pred = chunks(pred.tolist(), train_weibo.config.SEQ_SIZE)

            for p_r_pair, r in zip(data.indices, pred):
                def _index_of_or_len(lst, ele):
                    return lst.index(ele) if ele in lst else len(lst)

                post = p_r_pair[0][0:_index_of_or_len(p_r_pair[0], train_weibo.vocabulary[SpToken.EOS])]
                response = p_r_pair[1][0:_index_of_or_len(p_r_pair[1], train_weibo.vocabulary[SpToken.EOS])]
                pred_response = r[0:_index_of_or_len(r, train_weibo.vocabulary[SpToken.EOS])]
                output.write("post: %s \n"
                             "response: %s \n"
                             "predict: %s \n"
                             "\n" %
                             (train_weibo.gen_words_from_indices(post),
                              train_weibo.gen_words_from_indices(response),
                              train_weibo.gen_words_from_indices(pred_response)))


def average_gradients(tower_grads):
    """
    See https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


if __name__ == '__main__':
    main()
