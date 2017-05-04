import os

import tensorflow as tf
from tensorflow.python.client import device_lib

from n2nds.reader import WeiboReader, SpToken
from n2nds.beam_seq2seq import Model

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('dataset', 100, '')
flags.DEFINE_integer('batch_size', 100, '')
flags.DEFINE_integer('layer_num', 1, '')
flags.DEFINE_integer('gpu_num', 0, 'The gpu_num is the number of gpu used on the machine where'
                                   'the model is trained, instead of the machine where the model'
                                   'is running on. If 0, trained on a cpu, else on gpu(s)')
flags.DEFINE_string('info', 'normal', '')
flags.DEFINE_boolean('train_mode', False, '')

# Check gpu available
gpu_available = False
for x in device_lib.list_local_devices():
    if x.device_type == 'GPU':
        gpu_available = True

if not gpu_available and FLAGS.train_mode and FLAGS.gpu_num != 0:
    print("GPU not available on this device, please set gpu_num te be 0")
    exit(0)

if gpu_available and FLAGS.gpu_num == 0:
    print("Please set the gpu number")
    exit(0)

# Generate the model id
model_id = "a_%d_b_%d_l_%d_g_%d_i_%s" % (FLAGS.dataset,
                                         FLAGS.batch_size,
                                         FLAGS.layer_num,
                                         FLAGS.gpu_num,
                                         FLAGS.info)

# Set the log path for storing model and summary
post_path = "dataset/stc_weibo_train_post_generated_%d" % FLAGS.dataset
response_path = "dataset/stc_weibo_train_response_generated_%d" % FLAGS.dataset
if not os.path.exists("tmp"):
    os.mkdir("tmp")
if not os.path.exists("tmp/output_%s" % model_id):
    os.mkdir("tmp/output_%s" % model_id)
log_dir_path = "tmp/logdir_%s" % model_id
train_output_path = "tmp/output_%s/train_output.txt" % model_id
valid_output_path = "tmp/output_%s/valid_output.txt" % model_id

# Load the data set
train_weibo = WeiboReader(post_path, response_path,
                          batch_size=FLAGS.batch_size, pre_trained_path="pre_trained/wiki_char_200.txt")
train_weibo.config.EMBED_SIZE = 200
train_weibo.config.UNIT_SIZE = 200


def multi_concurrent_model(num=1, device='gpu'):
    grads = []
    reuse_flag = False
    for i in range(num):
        device_used = '/gpu:%d' % i if device == 'gpu' else '/cpu:0'
        with tf.device(device_used):
            with tf.name_scope("tower_%d" % i):
                with tf.variable_scope("Model", reuse=reuse_flag):
                    model = Model(train_weibo.config, is_train=True,
                                  embedding_init_value=train_weibo.embedding,
                                  num_of_layer=FLAGS.layer_num)
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
    # Create the train model
    with tf.name_scope("Train"):
        if gpu_available:
            # Train on GPUs and Test on GPUs
            train_op = multi_concurrent_model(num=FLAGS.gpu_num, device='gpu')
        else:
            if FLAGS.gpu_num > 0:
                # Train on GPUs and test on CPU
                train_op = multi_concurrent_model(FLAGS.gpu_num, device='cpu')
            else:
                # Train on CPU and test on CPU
                train_op = multi_concurrent_model(1, device='cpu')

    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True):
            valid_model = Model(train_weibo.config, is_train=False,
                                embedding_init_value=train_weibo.embedding,
                                num_of_layer=FLAGS.layer_num)

    # Training code
    if FLAGS.train_mode:
        sv = tf.train.Supervisor(logdir=log_dir_path,
                                 save_model_secs=60,
                                 save_summaries_secs=60,
                                 summary_op=None)
        with sv.managed_session(config=tf.ConfigProto(log_device_placement=True,
                                                      allow_soft_placement=True, )) as sess:
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
                    print("iter %d : cost %s" % (iter, cost))
                    if cost[0] < 1:
                        break

            batch_test(sess, tf.get_collection("train_model")[0], valid_model)

    # Testing code
    if not FLAGS.train_mode:
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "%s/model.ckpt" % log_dir_path)
            print("Model loaded successfully")
            # batch_test(sess, tf.get_collection("train_model")[0], valid_model)

            # sentence = "你好"
            # data_indices, data_lengths = train_weibo.gen_indices_and_lengths(sentence)
            feed_dict = dict()
            # feed_dict[valid_model.utter_indices] = data_indices
            # feed_dict[valid_model.utter_lengths] = data_lengths
            _b_o_p, _c_i, _i_o_i = sess.run([valid_model.beam_output_probs,
                                           valid_model.chosen_indices,
                                           valid_model.indices_of_input], {
                valid_model.beam_input_indices : [9999, 201, 399],
                valid_model.beam_input_probs: [0.5, 0.7, 0.7]
            })
            print(_b_o_p)
            print(_c_i)
            print(_i_o_i)
            # pred = pred[0: pred.index(train_weibo.vocabulary[SpToken.EOS])]
            # resp = train_weibo.gen_words_from_indices(pred)
            # print(resp)


def batch_test(sess, train_model, valid_model):
    """
    Run a batch of data using the train model and valid model
    """
    train_output = open(train_output_path, "w")
    valid_output = open(valid_output_path, "w")
    data = train_weibo.next_batch()
    for model, output in list(zip([train_model, valid_model], [train_output, valid_output])):
        feed_dict = dict()
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
