"""
Seq2seq
"""

import tensorflow as tf
import numpy as np


SEQ_SIZE = 8
BATCH_SIZE = 5
SIZE = 20
VOCAB_SIZE = 10


class Model:
    def __init__(self, data):
        uni_initer = tf.random_uniform_initializer(-0.01, 0.01)
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, SIZE],
                                    dtype=tf.float32, initializer=uni_initer)
        _encoder_inputs = tf.nn.embedding_lookup(embedding, data.x)
        _decoder_inputs = tf.nn.embedding_lookup(embedding, data.y)
        encoder_inputs=[]
        decoder_inputs=[]
        for i in range(SEQ_SIZE):
            encoder_inputs.append(_encoder_inputs[:,i,:])
            decoder_inputs.append(_decoder_inputs[:,i,:])

        lstm = tf.contrib.rnn.BasicLSTMCell(SIZE,
                                            reuse=tf.get_variable_scope().reuse)
        init_state = lstm.zero_state(BATCH_SIZE, tf.float32)
        state=init_state

        outputs, state=tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, lstm, dtype=tf.float32)
        outputs = tf.concat(outputs,axis=1)
        outputs = tf.reshape(tf.concat(axis=1, values=outputs),[-1,SIZE])

        softmax_w = tf.get_variable("softmax_w", [SIZE, VOCAB_SIZE],
                                    dtype=tf.float32, initializer=uni_initer)
        softmax_b = tf.get_variable("softmax_b", [VOCAB_SIZE],
                                    dtype=tf.float32, initializer=uni_initer)
        logits = tf.matmul(outputs, softmax_w) + softmax_b
        self.pred = tf.argmax(logits, 1)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(data.y, [-1])],
            [tf.ones([BATCH_SIZE * SEQ_SIZE], dtype=tf.float32)])
        self.cost = tf.reduce_sum(loss) / BATCH_SIZE
        opt = tf.train.AdamOptimizer()
        self.minimizier = opt.minimize(loss)


class Data:
    def __init__(self):
        self.x = [[0, 1, 2, 3, 4, 5, 6, 7],
                  [1, 1, 7, 1, 6, 1, 4, 1],
                  [2, 6, 7, 2, 3, 3, 2, 1],
                  [3, 8, 6, 9, 9, 1, 2, 5],
                  [4, 1, 1, 8, 1, 1, 3, 4]]
        self.y = [[3, 1, 4, 1, 5, 9, 2, 6],
                  [2, 7, 1, 8, 2, 8, 1, 8],
                  [1, 4, 1, 4, 2, 1, 3, 5],
                  [1, 7, 3, 2, 1, 1, 1, 1],
                  [2, 2, 3, 6, 1, 1, 1, 4]]


def main():
    data = Data()
    with tf.name_scope("Train"):
        with tf.variable_scope("Model"):
            m1 = Model(data=data)
    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True) as scope:
            m2 = Model(data=Data())

    with tf.Session() as sess:
        _ = sess.run(tf.global_variables_initializer())
        no = 0
        while True:
            # no += 1
            # if no > 3:
            #     break
            _, cost = sess.run([m1.minimizier, m1.cost], feed_dict={

            })
            print(cost)
            if cost < 0.5:
                break

        for model in [m1]:
            pred = sess.run([model.pred])
            print(np.reshape(pred, [BATCH_SIZE, SEQ_SIZE]))
            # print(ss)

if __name__ == '__main__':
    main()
