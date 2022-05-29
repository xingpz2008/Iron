from single_model import BertModel
from modeling import BertConfig
import tensorflow as tf
import numpy as np


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


input_file = "/home/gfg/xingpengzhi/Transformer_EzPC/bert/mrpc_npy_file/tiny/mrpcFile000_dev.npy"
input_mask_file = "/home/gfg/xingpengzhi/Transformer_EzPC/bert/mrpc_npy_file/tiny/mrpcFile000_AM_dev.npy"
ckpt = '/home/gfg/xingpengzhi/Transformer_EzPC/bert/output/tiny_mrpc.ckpt'
bert_config_file = './uncased_L-2_H-128_A-2/bert_config.json'
config = BertConfig.from_json_file(bert_config_file)

batch_size = 1

input = np.load(input_file)
mask = np.load(input_mask_file)

x = tf.placeholder(dtype=tf.float32, shape=(batch_size, 128, config.hidden_size), name='input')
input_mask = tf.placeholder(dtype=tf.float32, shape=(batch_size, 128), name='input_mask')
token_type_ids = tf.placeholder(dtype=tf.int32, shape=(batch_size, 128), name='token_type_ids')

with tf.Session() as sess:
    model = BertModel(config=config,
                      is_training=False,
                      input_ids=tf.ones([batch_size, 128],
                                        dtype=tf.int32),
                      input_mask=input_mask,
                      token_type_ids=token_type_ids,
                      use_one_hot_embeddings=False,
                      scope=None,
                      input_tensor=x)
    # input_tensor=embedding_processor.output_layer())
    init = tf.global_variables_initializer()
    sess.run(init)
    optimistic_restore(sess, ckpt)
    final = sess.run(model.get_nsp_output(), feed_dict={x: input,
                                                input_mask: mask})
    layer_wise_output = sess.run(model.get_attention_list()[0], feed_dict={x: input,
                                                                           input_mask: mask})
    pass
