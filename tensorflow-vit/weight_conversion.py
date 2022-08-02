import torch
import numpy as np


def name_mapping(tf_name):
    """
    This function maps a tf node name to pytorch node name
    :param tf_name:
    :return: str
    """
    class_header = 'classifier.'

    if tf_name == 'bert/pooler/norm':
        return class_header + 'norm.'
    if tf_name == 'cls/fc':
        return class_header + 'fc.'
    if "attention_pool" in tf_name:
        return class_header + 'attention_pool.'

    block_header = 'blocks.'
    block_index = -1
    segmentation = tf_name.split('/')

    for idx, ctx in enumerate(segmentation):
        if 'layer_' in ctx:
            block_index = segmentation[idx].replace('layer_', '')
            break

    assert block_index != -1
    block_header = block_header + block_index + '.'

    if 'pre_norm' in tf_name:
        return class_header + block_header + 'pre_norm.'
    # classifier.blocks.0.pre_norm.
    if '/linear1' in tf_name:
        return class_header + block_header + 'linear1.'
    if 'norm1' in tf_name:
        return class_header + block_header + 'norm1.'
    if 'linear2' in tf_name:
        return class_header + block_header + 'linear2.'
    if 'qkv' in tf_name:
        return class_header + block_header + 'self_attn.qkv.'
    if 'proj' in tf_name:
        return class_header + block_header + 'self_attn.proj.'

    raise NotImplementedError(f"No mapping established @ {tf_name}")


def get_specific_weight(torch_name, filepath='./output/_cct_7_3x1_32_sine_cifar100_5000epochs.pth'):
    """
    This function returns a numpy array that matches the precise torch_name
    :param torch_name:
    :param filepath:
    :return:
    """
    state_dict = torch.load(filepath)
    return np.array(state_dict[torch_name])
