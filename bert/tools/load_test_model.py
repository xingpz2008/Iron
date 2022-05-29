import tensorflow as tf
import sys

sys.path.append('..')
from single_test import TestModel

input = tf.placeholder(dtype=tf.float32, shape=(128, 768), name='input')
output = '../output/single_test.ckpt'

with tf.Session() as sess:
    model = TestModel(input=input)
    sess.run(tf.initialize_all_variables())
    sess.run(model.get_output(), feed_dict={input: sess.run(tf.random_uniform([128, 768],
                                                                              dtype=tf.float32))})
    saver = tf.train.Saver()
    saver.save(sess, output)