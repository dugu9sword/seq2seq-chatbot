import tensorflow as tf

x = tf.placeholder(shape=[None, 2], name="x", dtype=tf.float32)
lstm = tf.contrib.rnn.BasicLSTMCell(10,
                                    state_is_tuple=True,
                                    reuse=tf.get_variable_scope().reuse)
state = lstm.zero_state(tf.shape(x)[0], tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    _state = sess.run(state, feed_dict={
        x: [[1.0, 2.0],
            [3.0, 4.0]]
    })
    print(_state)
