"""
Single Test model
"""

import tensorflow as tf


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def test_layer(input):
    # input = tf.nn.softmax(input)
    # 128, 768
    output = layer_norm(input)
    return output


class TestModel(object):
    def __init__(self,
                 input):
        with tf.variable_scope("layer"):
            x = test_layer(input=input)
            self.output = tf.layers.dense(x, 2)

    def get_output(self):
        return self.output
