import os

import tensorflow as tf

from flask import Flask
from n2nds.reader import WeiboReader, SpToken
from n2nds.seq2seq import Model

app = Flask(__name__)


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
    return train_weibo.gen_words_from_indices(pred)


data_set_used = 1000000
batch_size = 500
num_of_layer = 4
info = "normal"
post_path = "dataset/stc_weibo_train_post_generated_%d" % data_set_used
response_path = "dataset/stc_weibo_train_response_generated_%d" % data_set_used

# Load the data set
train_weibo = WeiboReader(post_path, response_path, batch_size=batch_size)
train_weibo.config.EMBED_SIZE = 200
train_weibo.config.UNIT_SIZE = 200

# Set the log path for storing model and summary
model_id = "a_%d_b_%d_l_%d_i_%s" % (data_set_used, batch_size, num_of_layer, info)
log_dir_path = "tmp/logdir_%s" % model_id

# Create the model
with tf.name_scope("Valid"):
    with tf.variable_scope("Model"):
        valid_model = Model(train_weibo.config, is_train=False, num_of_layer=num_of_layer)

# Restore
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "%s/model.ckpt" % log_dir_path)

if __name__ == '__main__':
    app.run()
