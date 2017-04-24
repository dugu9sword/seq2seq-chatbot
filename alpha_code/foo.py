import tensorflow as tf
import numpy as np

SEQ_SIZE = 5
BATCH_SIZE = 5
SIZE = 10
VOCAB_SIZE = 13


class Model:
    def __init__(self, data):
        uni_initer = tf.random_uniform_initializer(-0.01, 0.01)
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, SIZE],
                                    dtype=tf.float32, initializer=uni_initer)
        inputs = tf.nn.embedding_lookup(embedding, data.x)

        lstm = tf.contrib.rnn.BasicLSTMCell(SIZE,
                                            reuse=tf.get_variable_scope().reuse)
        init_state = lstm.zero_state(BATCH_SIZE, tf.float32)
        state=init_state

        cell_outputs = []
        self.states=[]

        with tf.variable_scope("lstm", initializer=uni_initer):
            for time_step in range(SEQ_SIZE):
                self.states.append(state)
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = lstm(inputs[:, time_step, :], state)
                cell_outputs.append(cell_output)

        outputs = tf.reshape(tf.concat(axis=1, values=cell_outputs), [-1, SIZE])

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
        self.x = [[0, 1, 2, 3, 4],
                  [1, 2, 3, 4, 5],
                  [2, 3, 4, 5, 6],
                  [3, 4, 5, 6, 7],
                  [4, 5, 6, 7, 8]]
        self.y = [[1, 3, 5, 7, 9],
                  [2, 4, 6, 8, 10],
                  [3, 4, 5, 6, 7],
                  [4, 6, 8, 10, 12],
                  [5, 6, 7, 8, 9]]
        # self.y = self.x

class ValidData:
    def __init__(self):
        self.x = [[2, 3, 4, 5, 6],
                  [3, 4, 5, 6, 7],
                  [4, 5, 6, 7, 8],
                  [5, 6, 7, 8, 9],
                  [6, 7, 8, 9, 10]]
        self.y = [[3, 4, 5, 6, 7],
                  [4, 5, 6, 7, 8],
                  [5, 6, 7, 8, 9],
                  [6, 7, 8, 9, 10],
                  [7, 8, 9, 10, 11]]
        # self.y = self.x


def main():
    data = Data()
    with tf.name_scope("Train"):
        with tf.variable_scope("Model"):
            m1 = Model(data=data)
    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True) as scope:
            m2 = Model(data=ValidData())

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
            pred, ss = sess.run([model.pred, model.states])
            print(np.reshape(pred, [5, 5]))
            # print(ss)

if __name__ == '__main__':
    main()
