import tensorflow as tf
import sys

sys.path.append('..')
from single_attention import SingleAttention

def construct_input():
    '''
    input_ids -> (batch_size, 128), int32
    input_mask -> (batch_size, 128), int 32
    token_type_ids -> (batch_size, 128), int 32
    :return:
    '''
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    # segment_emebdding. 表示第一个样本前两个词属于句子1，后一个词属于句子2.
    # 第二个样本的第一个词属于句子1， 第二次词属于句子2，第三个元素0表示padding
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
    return input_ids, input_mask, token_type_ids


def optimistic_restore(session, save_file):
    """
  restore only those variable that exists in the model
  :param session:
  :param save_file:
  :return:
  """
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for
                        var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                print("going to restore.var_name:", var_name, ";saved_var_name:", saved_var_name)
                restore_vars.append(curr_var)
            else:
                print("variable not trained.var_name:", var_name)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


batch_size = 1
bert_config_file = '../uncased_L-12_H-768_A-12/bert_config.json'
ckpt = '../uncased_L-12_H-768_A-12/bert_model.ckpt'
output = '../output/single_attention.ckpt'
# config = BertConfig.from_json_file(bert_config_file)
# input_ids, input_mask, token_type_ids = construct_input()

from_tensor = tf.placeholder(dtype=tf.float32, shape=(128, 768), name='from_tensor')
to_tensor = tf.placeholder(dtype=tf.float32, shape=(128, 768), name='to_tensor')
attention_mask = tf.placeholder(dtype=tf.float32, shape=(batch_size, 128, 128), name='attention_mask')
# x = embedding_processor.output_layer()


with tf.Session() as sess:
    model = SingleAttention(from_tensor=from_tensor,
                            to_tensor=to_tensor,
                            attention_mask=attention_mask,
                            num_attention_heads=12,
                            size_per_head=64,
                            query_act=None,
                            key_act=None,
                            value_act=None,
                            attention_probs_dropout_prob=0.0,
                            initializer_range=0.02,
                            do_return_2d_tensor=True,
                            batch_size=1,
                            from_seq_length=128,
                            to_seq_length=128)
    # input_tensor=embedding_processor.output_layer())
    sess.run(tf.initialize_all_variables())
    sess.run(model.get_output(), feed_dict={from_tensor: sess.run(tf.random_uniform([128, 768],
                                                                                    dtype=tf.float32)),
                                            to_tensor: sess.run(tf.random_uniform([128, 768],
                                                                                  dtype=tf.float32)),
                                            attention_mask: sess.run(tf.random_uniform([1, 128, 128],
                                                                                       dtype=tf.float32))})
    # optimistic_restore(sess, ckpt)
    saver = tf.train.Saver()
    saver.save(sess, output)

    # TODO: load weight

    # TODO: Operations unsupported
