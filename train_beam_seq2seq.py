import os

import tensorflow as tf
import numpy as np
from copy import deepcopy
from tensorflow.python.client import device_lib

from n2nds.reader import WeiboReader, SpToken
from n2nds.beam_seq2seq import Model

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('dataset', 100, '')
flags.DEFINE_integer('batch_size', 100, '')
flags.DEFINE_integer('layer_num', 1, '')
flags.DEFINE_integer('gpu_num', 4, 'The gpu_num is the number of gpu used on the machine where'
                                   'the model is trained, instead of the machine where the model'
                                   'is running on. If 0, trained on a cpu, else on gpu(s)')
flags.DEFINE_string('info', 'normal', '')
flags.DEFINE_boolean('train_mode', True, '')

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
        with tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                              allow_soft_placement=True, )) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "%s/model.ckpt" % log_dir_path)
            print("Model loaded successfully")
            # batch_test(sess, tf.get_collection("train_model")[0], valid_model)


            sentence = "中国移动营销行来发展报告上"

            # Greedy
            data_indices, data_lengths = train_weibo.gen_indices_and_lengths(sentence)
            feed_dict = dict()
            feed_dict[valid_model.utter_indices] = data_indices
            feed_dict[valid_model.utter_lengths] = data_lengths
            # feed_dict[valid_model.utter_weights] = data.weights
            pred = sess.run(valid_model.pred, feed_dict).tolist()
            pred = pred[0: pred.index(train_weibo.vocabulary[SpToken.EOS])]
            resp = train_weibo.gen_words_from_indices(pred).replace(SpToken.UNK, "")
            print("Greedy Result:")
            print(resp)

            # Beam
            while True:
                beam = int(input("beam: (If 0, exit)"))
                if beam==0:
                    break
                sentence = input("Please input the sentence:")
                # beam = 10

                data_indices, data_lengths = train_weibo.gen_indices_and_lengths(sentence)
                enc_state = sess.run(
                    [
                        valid_model.encoder_state
                    ],
                    feed_dict={
                        valid_model.utter_indices: data_indices,
                        valid_model.utter_lengths: data_lengths
                    }
                )
                # print(enc_state)
                k = beam
                topk_nodes = []
                is_eos=lambda node: node.index==train_weibo.vocabulary[SpToken.EOS]
                for time_step in range(30):
                    if time_step == 0:
                        bs, bp = sess.run(
                            [
                                valid_model.b_output_state,
                                valid_model.b_probs
                            ],
                            {
                                valid_model.b_input_index: 0,
                                valid_model.b_input_state: enc_state
                            }
                        )
                        bp = bp[0]  # bp is [[]], we fetch batch 0
                        indices = list(reversed(np.argpartition(bp, -k)[-k:]))
                        probs = list(reversed(np.partition(bp, -k)[-k:]))
                        for i in range(k):
                            topk_nodes.append(BeamSearchNode(index=indices[i],
                                                             prob=probs[i],
                                                             state=bs))

                    else:
                        topk_prob_table = []
                        topk_state_list=[]
                        for topk_node in topk_nodes:
                            bs, bp = sess.run(
                                [
                                    valid_model.b_output_state,
                                    valid_model.b_probs
                                ],
                                {
                                    valid_model.b_input_index: topk_node.index,
                                    valid_model.b_input_state: topk_node.state
                                }
                            )
                            if is_eos(topk_node):
                                topk_prob_table.append(np.zeros(train_weibo.config.VOCAB_SIZE))
                                topk_state_list.append(bs)
                            else:
                                bp = np.array(bp[0])
                                topk_prob_table.append(bp * topk_node.prob)
                                topk_state_list.append(bs)
                        flatten_probs = np.reshape(topk_prob_table, -1)
                        indices = np.argpartition(flatten_probs, -k)[-k:]
                        chosen_indices = indices % train_weibo.config.VOCAB_SIZE
                        prev_indices_in_topk = indices // train_weibo.config.VOCAB_SIZE
                        probs = np.partition(flatten_probs, -k)[-k:]

                        tmp_topk_nodes=[]
                        for i in range(k):
                            node = BeamSearchNode(index=chosen_indices[i],
                                                  state=topk_state_list[prev_indices_in_topk[i]],
                                                  prob=probs[i],
                                                  prev_node=topk_nodes[prev_indices_in_topk[i]])
                            tmp_topk_nodes.append(node)
                        for topk_node in topk_nodes:
                            if is_eos(topk_node):
                                tmp_topk_nodes.append(topk_node)
                        topk_nodes=sorted(tmp_topk_nodes, key=lambda node:node.prob, reverse=True)[0:k]

                    print("beam: ")
                    for node in topk_nodes:
                        print("prob: %f, node: %d %s, prev node: %d %s,"
                              %(node.prob,
                                node.index,
                                train_weibo.gen_words_from_indices([node.index]),
                                -1 if node.prev_node is None else node.prev_node.index,
                                "<BOS>" if node.prev_node is None else
                                    train_weibo.gen_words_from_indices([node.prev_node.index])))

                print("Beam result: ")
                for node in topk_nodes:
                    print("prob %f with %s"%(
                        node.prob,
                        train_weibo.gen_words_from_indices(node.get_sentence_indices())))




class BeamSearchNode:
    def __init__(self, index, prob, state, prev_node=None):
        self.index = index
        self.prob = prob
        self.state = state
        self.prev_node = prev_node

    def get_sentence_indices(self):
        if self.prev_node is None:
            return [self.index]
        else:
            lst = self.prev_node.get_sentence_indices()
            lst.append(self.index)
            return lst


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
