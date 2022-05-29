from tensorflow.python import pywrap_tensorflow
checkpoint_path = '../uncased_L-12_H-768_A-12/bert_model.ckpt'
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
f = open('./output.txt', 'w')
for key in var_to_shape_map:
    print("tensor_name: ", key, file=f)