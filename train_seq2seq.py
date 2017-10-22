import os
import pickle

import tensorflow as tf
from tensorflow.python.client import device_lib

from n2nds.reader import DataSetReader, SpToken
from n2nds.seq2seq import Model
from n2nds.config import Config

from flask import Flask

app = Flask(__name__)

class FLAGS:
    dataset=1000000
    batch_size=200
    layer_num=2
    info='normal'
    mode = 'train'  # train/deploy
    embedding_size=200
    hidden_size=200
    save_model_secs=60
    save_summaries_secs=60
    show_loss_iters=50
    batch_test_iters=1000

# Check gpu available
gpu_available = False
for x in device_lib.list_local_devices():
    if x.device_type == 'GPU':
        gpu_available = True
if not gpu_available:
    print("Without GPU, the program will be very very very slow...")
    exit()

# Generate the model id
model_id = "%d_%s" % (FLAGS.dataset, FLAGS.info)

# Set the log path for storing model and summary
post_path = "dataset/post_%d.txt" % FLAGS.dataset
response_path = "dataset/response_%d.txt" % FLAGS.dataset
if not os.path.exists("tmp"):
    os.mkdir("tmp")
if not os.path.exists("tmp/output_%s" % model_id):
    os.mkdir("tmp/output_%s" % model_id)
log_dir_path = "tmp/logdir_%s" % model_id
train_output_path = "tmp/output_%s/train_output.txt" % model_id
valid_output_path = "tmp/output_%s/valid_output.txt" % model_id

# Load the data set
if not os.path.exists("tmp/dataset_%s" % model_id):
    pre_trained_path = "pre_trained/wiki_char_%d.txt" % FLAGS.embedding_size
    if not os.path.exists(pre_trained_path):
        pre_trained_path = None
    train_dataset = DataSetReader(post_path, response_path,
                                  pre_trained_path=pre_trained_path)
    pickle.dump(train_dataset, open("tmp/dataset_%s" % model_id, "wb"))
else:
    train_dataset = pickle.load(open("tmp/dataset_%s" % model_id, "rb"))

train_config=Config()
train_config.EMBED_SIZE = FLAGS.embedding_size
train_config.UNIT_SIZE = FLAGS.hidden_size
train_config.BATCH_SIZE = FLAGS.batch_size
train_config.SEQ_SIZE = train_dataset.SEQ_SIZE
train_config.VOCAB_SIZE = len(train_dataset.vocabulary)

g_sess = None
g_model = None

def build_model():
    # Create the train model
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=False):
            model = Model(train_config, is_train=True,
                          embedding_init_value=train_dataset.embedding)
                          # embedding_init_value=None)

    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True):
            valid_model = Model(train_config, is_train=False)
    
    return model, valid_model

def main():
    model, valid_model = build_model()
    
    # Training code
    if FLAGS.mode=='train':
        sv = tf.train.Supervisor(logdir=log_dir_path,
                                 save_model_secs=FLAGS.save_model_secs,
                                 save_summaries_secs=FLAGS.save_summaries_secs,
                                 summary_op=None)
        with sv.managed_session(config=tf.ConfigProto(log_device_placement=True,
                                                      allow_soft_placement=True, 
                                                      gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333))) as sess:
            iter = 0
            while True:
                iter += 1

                feed_dict = dict()
                data = train_dataset.next_batch(FLAGS.batch_size)
                feed_dict[model.utter_indices] = data.indices
                feed_dict[model.utter_lengths] = data.lengths
                feed_dict[model.utter_weights] = data.weights
                sess.run(model.train_op, feed_dict)

                if iter % FLAGS.show_loss_iters == 0:
                    merged, cost = sess.run([model.merged, model.cost], feed_dict)

                    sv.summary_computed(sess, summary=merged, global_step=iter)
                    print("iter %d : cost %s" % (iter, cost))
                    if cost < 1:
                        break
                
                if iter % FLAGS.batch_test_iters == 0:
                    batch_test(sess, model, valid_model)
        
    # Deploying mode
    if FLAGS.mode=='deploy':
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                                allow_soft_placement=True, ))
        saver = tf.train.Saver()
        saver.restore(sess, "%s/model.ckpt" % log_dir_path)
        global g_sess
        g_sess = sess
        global g_model
        g_model = valid_model
        app.run('0.0.0.0')
    
    
@app.route("/<sentence>")
def response(sentence):
    print(sentence)
    data_indices, data_lengths = train_dataset.gen_indices_and_lengths(sentence)
    feed_dict = dict()
    print("--------------------------------------------")
    print(g_model)
    feed_dict[g_model.utter_indices] = data_indices
    feed_dict[g_model.utter_lengths] = data_lengths
    # feed_dict[valid_model.utter_weights] = data.weights
    pred = g_sess.run(g_model.pred, feed_dict).tolist()
    
    if train_dataset.vocabulary[SpToken.EOS] in pred:
        pred = pred[0: pred.index(train_dataset.vocabulary[SpToken.EOS])]
        resp = train_dataset.gen_words_from_indices(pred).replace(SpToken.UNK, "")
    else:
        resp = u"嘻嘻~"
    return resp

def batch_test(sess, train_model, valid_model):
    """
    Run a batch of data using the train model and valid model
    """
    train_output = open(train_output_path, "w", encoding="utf-8")
    valid_output = open(valid_output_path, "w", encoding="utf-8")
    data = train_dataset.next_batch(FLAGS.batch_size)
    for model, output in list(zip([train_model, valid_model], [train_output, valid_output])):
        feed_dict = dict()
        feed_dict[model.utter_indices] = data.indices
        feed_dict[model.utter_lengths] = data.lengths
        feed_dict[model.utter_weights] = data.weights

        pred = sess.run(model.pred, feed_dict)
        chunks = lambda arr, n: [arr[i:i + n] for i in range(0, len(arr), n)]
        pred = chunks(pred.tolist(), train_dataset.SEQ_SIZE)

        for p_r_pair, r in zip(data.indices, pred):
            def _index_of_or_len(lst, ele):
                return lst.index(ele) if ele in lst else len(lst)

            post = p_r_pair[0][0:_index_of_or_len(p_r_pair[0], train_dataset.vocabulary[SpToken.EOS])]
            # print(post)
            response = p_r_pair[1][0:_index_of_or_len(p_r_pair[1], train_dataset.vocabulary[SpToken.EOS])]
            pred_response = r[0:_index_of_or_len(r, train_dataset.vocabulary[SpToken.EOS])]
            verbose = "post: %s \nresponse: %s \npredict: %s \n\n" %(train_dataset.gen_words_from_indices(post),
                          train_dataset.gen_words_from_indices(response),
                          train_dataset.gen_words_from_indices(pred_response))
            print(verbose)
            output.write(verbose)

if __name__ == '__main__':
    main()
