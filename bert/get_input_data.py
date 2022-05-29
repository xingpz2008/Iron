from run_classifier import convert_examples_to_features, MrpcProcessor, Sst2Processor, QnliProcessor
from tokenization import FullTokenizer
from separated_embedding import embedding_layer
from modeling import BertConfig
import tensorflow as tf
import numpy as np
import time

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

# data_processor = MrpcProcessor()
# data_processor = Sst2Processor()
data_processor = QnliProcessor()
bert_config_file = './uncased_L-2_H-128_A-2/bert_config.json'
original_file_path = './glue_data/QNLI'
vocab_file = './uncased_L-2_H-128_A-2/vocab.txt'
ckpt = "/home/gfg/xingpengzhi/Transformer_EzPC/bert_pure/tmp/qnli_output/model.ckpt-9819"
output_npy_file_path = './qnli_npy_file/tiny/'
saved_label_file_name = output_npy_file_path + 'qnliLabel.npy'
batch_size = 1
max_seq_length = 128
flag = 1

# x = np.load(saved_label_file_name)

config = BertConfig.from_json_file(bert_config_file)
tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)


input_ids = tf.placeholder(dtype=tf.int32, shape=(batch_size, max_seq_length), name='input_ids')
token_type_ids = tf.placeholder(dtype=tf.int32, shape=(batch_size, max_seq_length), name='token_type_ids')

examples = data_processor.get_dev_examples(original_file_path)
features = convert_examples_to_features(examples, data_processor.get_labels(), max_seq_length, tokenizer)

'''
Here, in the features class:
input_ids -> input_ids
input_mask -> attention_mask
segment_ids -> token_type_ids
'''

with tf.Session() as sess:
    embedding_layer = embedding_layer(config=config, input_ids=input_ids, token_type_ids=token_type_ids).output_layer()
    # sess.run(tf.initialize_all_variables())
    optimistic_restore(sess, ckpt)
    label_list = []
    for i in range(len(features)):
        time1 = time.time()
        if flag:
            attention_mask_tensor = sess.run(tf.expand_dims(features[i].input_mask, 0))
            saved_file_name = output_npy_file_path + 'qnliFile' + f'{i:03}' + '_AM_dev' + '.npy'
            np.save(saved_file_name, attention_mask_tensor)
        else:
            input_ids_tensor = tf.expand_dims(features[i].input_ids, 0)
            token_type_ids_tensor = tf.expand_dims(features[i].segment_ids, 0)
            label_list.append(features[i].label_id)
            embedding_output = sess.run(embedding_layer,
                                        feed_dict={input_ids: sess.run(input_ids_tensor),
                                                   token_type_ids: sess.run(token_type_ids_tensor)})
            saved_file_name = output_npy_file_path + 'qnliFile' + f'{i:03}' + '_dev' + '.npy'
            saved_file_name_input_ids = output_npy_file_path + 'qnliFile' + f'{i:03}' + 'ID_dev' + '.npy'
            np.save(saved_file_name, embedding_output)
            # np.save(saved_file_name_input_ids, sess.run(input_ids_tensor))
        time2 = time.time()
        remaining_time = (time2 - time1) * (len(features)-1-i)
        print(f'File {saved_file_name} saved. Remaining time estimation: {remaining_time}s')
    if not flag:
        np.save(saved_label_file_name, label_list)
pass
