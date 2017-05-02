"""
Integrated with flask
"""

import tensorflow as tf
from flask import Flask

app = Flask(__name__)


# model = None
#
#
# def build_model():
#     model = Model()


class Model:
    def __init__(self):
        self.init_op = tf.global_variables_initializer()
        self.x = tf.placeholder(name="x", dtype=tf.float32)
        self.res = self.x * self.x

model = Model()
sess = tf.Session()

@app.route("/<sentence>")
def response(sentence):
    sess.run(model.init_op)
    response = sess.run(model.res, feed_dict={model.x: float(sentence)})
    print(response)
    return str(response)


if __name__ == '__main__':
    app.run()
