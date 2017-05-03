import tensorflow as tf

x = tf.placeholder(name="x", shape=[None], dtype=tf.float32)
a = tf.get_variable("a", shape=[1])
a = a + 1
y = x * a

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(y, feed_dict={
        x: [2.0]
    })
    print(res)
    res = sess.run(a)
    print(res)
    res = sess.run(a)
    print(res)
    res = sess.run(y, feed_dict={
        x: [2.0]
    })
    print(res)