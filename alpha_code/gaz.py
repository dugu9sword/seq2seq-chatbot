"""
Multi GPU
"""

import tensorflow as tf

gpu_mode = True
gpu_nums = 4


def build_model():
    a = tf.Variable([[1.0]])
    b = tf.Variable([[2.0]])
    c = tf.matmul(a, b)
    init = tf.global_variables_initializer()
    return a, b, c, init

cs=[]
if gpu_mode is True:
    for i in range(gpu_nums):
        with tf.device('/gpu:%d' % i):
            a, b, c, init = build_model()
            c = c + i
            cs.append(c)
else:
    a, b, c, init = build_model()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    print(sess.run(cs))
