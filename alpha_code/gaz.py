import tensorflow as  tf

c = tf.Variable(1.0)
h = tf.Variable(2.0)

d = c

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([c, h, d], feed_dict={
        c:4
    }))
