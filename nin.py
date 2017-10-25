import tensorflow as tf
import numpy as np

def linear(x, unit, layer_name='linear'):
    with tf.name_scope(layer_name):
        x = tf.layers.dense(inputs=x, units=unit)
        return x


def nin(x, unit, layer_name='nin'):
    with tf.name_scope(layer_name):
        s = list(map(int, x.get_shape()))
        x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
        x = linear(x, unit, layer_name)
        x = tf.reshape(x, s[:-1] + [unit])
        return x