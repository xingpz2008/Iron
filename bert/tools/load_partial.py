import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

sys.path.append('..')
import single_model
from single_model import BertModel
from modeling import BertConfig
from separated_embedding import embedding_layer

batch_size = 1
bert_config_file = '../uncased_L-2_H-128_A-2/bert_config.json'
ckpt = '/home/gfg/xingpengzhi/Transformer_EzPC/bert_pure/tmp/qnli_output/model.ckpt-9819'
output = '../output/tiny_qnli.ckpt'
config = BertConfig.from_json_file(bert_config_file)


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


# input_ids, input_mask, token_type_ids = construct_input()

x = tf.placeholder(dtype=tf.float32, shape=(batch_size, 128, config.hidden_size), name='input')
input_ids = tf.placeholder(dtype=tf.int32, shape=(batch_size, 128), name='input_ids')
input_mask = tf.placeholder(dtype=tf.float32, shape=(batch_size, 128), name='input_mask')
token_type_ids = tf.placeholder(dtype=tf.int32, shape=(batch_size, 128), name='token_type_ids')
# x = embedding_processor.output_layer()


with tf.Session() as sess1:
    embedding_processor = embedding_layer(config=config,
                                          input_ids=input_ids,
                                          token_type_ids=token_type_ids)
    # optimistic_restore(sess1, ckpt)
    init = tf.global_variables_initializer()
    sess1.run(init)
    embedding_tensor = embedding_processor.output_layer()
    embedding_output = sess1.run(embedding_tensor,
                                 feed_dict={input_ids: sess1.run(tf.random_uniform([batch_size, 128],
                                                                                   maxval=35,
                                                                                   minval=0,
                                                                                   dtype=tf.int32)),
                                            token_type_ids: sess1.run(tf.random_uniform([batch_size, 128],
                                                                                        dtype=tf.int32,
                                                                                        minval=0,
                                                                                        maxval=2))})

with tf.Session() as sess:
    model = BertModel(config=config,
                      is_training=False,
                      input_ids=input_ids,
                      input_mask=input_mask,
                      token_type_ids=token_type_ids,
                      use_one_hot_embeddings=False,
                      scope=None,
                      input_tensor=x)

    # input_tensor=embedding_processor.output_layer())
    init = tf.global_variables_initializer()
    sess.run(init)
    value0 = sess.run(tf.get_default_graph().get_tensor_by_name('output_bias:0'))
    sess.run(model.get_nsp_output(), feed_dict={x: sess.run(tf.ones([batch_size, 128, config.hidden_size],
                                                                    dtype=tf.float32)),
                                                input_mask: sess.run(tf.ones([batch_size, 128],
                                                                             dtype=tf.float32)),
                                                input_ids: sess.run(tf.ones([batch_size, 128],
                                                                            dtype=tf.int32))})
    optimistic_restore(sess, ckpt)
    value1 = sess.run(tf.get_default_graph().get_tensor_by_name('output_bias:0'))
    print(value0)
    print(value1)
    saver = tf.train.Saver()
    saver.save(sess, output)

    # TODO: load weight

    # TODO: Operations unsupported
