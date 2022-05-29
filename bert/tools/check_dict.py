import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

model_dir = "../uncased_L-12_H-768_A-12/bert_model"

ckpt = tf.train.get_checkpoint_state(model_dir)
ckpt_path = ckpt.model_checkpoint_path

reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
param_dict = reader.get_variable_to_shape_map()

for key, val in param_dict.items():
    try:
        print(key, val)
    except:
        raise