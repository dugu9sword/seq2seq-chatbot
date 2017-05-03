import os

import tensorflow as tf

from flask import Flask
from n2nds.reader import WeiboReader, SpToken
from n2nds.seq2seq import Model

app = Flask(__name__)



from tensorflow.python.client import device_lib

from n2nds.reader import WeiboReader, SpToken
from n2nds.seq2seq import Model

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('dataset', 100, '')
flags.DEFINE_integer('batch_size', 100, '')
flags.DEFINE_integer('layer_num', 4, '')
flags.DEFINE_integer('gpu_num', 4, 'The gpu_num is the number of gpu used on the machine where'
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
train_weibo = WeiboReader(post_path, response_path, batch_size=FLAGS.dataset)
train_weibo.config.EMBED_SIZE = 200
train_weibo.config.UNIT_SIZE = 200

sess = None
valid_model = None

def multi_concurrent_model(num=1, device='gpu'):
    grads = []
    reuse_flag = False
    for i in range(num):
        device_used = '/gpu:%d' % i if device == 'gpu' else '/cpu:0'
        with tf.device(device_used):
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


def build_model():
    # Create the train model
    with tf.name_scope("Train"):
        if gpu_available:
            train_op = multi_concurrent_model(num=FLAGS.gpu_num, device='gpu')
        else:
            if not FLAGS.train_mode and FLAGS.gpu_num > 0:
                # Train on multi gpu and test on single cpu machine
                train_op = multi_concurrent_model(FLAGS.gpu_num, device='cpu')
            else:
                train_op = multi_concurrent_model(1, device='cpu')

    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True):
            valid_model = Model(train_weibo.config, is_train=False)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "%s/model.ckpt" % log_dir_path)

    return sess, valid_model


@app.route("/<sentence>")
def response(sentence):
    data_indices, data_lengths = train_weibo.gen_indices_and_lengths(sentence)
    feed_dict = dict()
    feed_dict[valid_model.utter_indices] = data_indices
    feed_dict[valid_model.utter_lengths] = data_lengths
    # feed_dict[valid_model.utter_weights] = data.weights
    pred = sess.run(valid_model.pred, feed_dict).tolist()
    pred = pred[0: train_weibo.config.SEQ_SIZE]
    pred = pred[0: pred.index(train_weibo.vocabulary[SpToken.EOS])]
    resp = train_weibo.gen_words_from_indices(pred)
    return resp


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
    sess, valid_model = build_model()
    app.run('0.0.0.0')
