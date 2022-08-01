import os
import sys

import numpy as np
import threading
import subprocess


"""
Supported model:
tiny -> for nlp Sentence Classification Task
cct -> for Image Classification task
"""
model = 'cct'
task = 'ImageNet'
thread_num = 10
scale = 12

test_num_dict = {
    "mrpc": 408,
    "qnli": 2000,
    "mnli": 2000,
    "sst2": 872,
    "ImageNet": 1500
}

cv_task = ['ImageNet', 'cifar100']
nlp_task = ['mrpc', 'qnli', 'mnli', 'sst2']

input_restriction = {
    "mnli": [0, 1, 2],
    "qnli": [0, 1],
    "sst2": [0, 1],
    "mrpc": [0, 1],
}

if scale == 12:
    binary_file = f'./{model}_{task}'
else:
    binary_file = f'./{model}_{task}_{scale}'
file_dir = f'./{task}/{model}/'
weight_file = f'./{model}_{task}_input_weights_fixedpt_scale_{scale}.inp'
tmp_dir = './tmp/'
label_file = f'./{task}Label.npy'
output_file = f'./output_{model}_{task}_{scale}.npy'
file_header = f'{task}File'
test_num = test_num_dict[task]

total_true_num = 0
acc_dict = dict()


def sub_get(start, end):
    global total_true_num
    label = np.load(label_file)
    prediction_list = []
    true_num = 0
    multi_thread_disabled = 0
    for i in range(start, end):
        """
        1. Convert AM and file to .inp
        2. Call a.out with input and weight
        3. Get weight, calculate label
        4. Compare, get acc
        """
        if start == 0 and end == test_num:
            multi_thread_disabled = 1
        if task in nlp_task:
            input_am_file_name = file_dir + file_header + f'{i:03}' + '_AM_dev' + '.npy'
            input_file_name = file_dir + file_header + f'{i:03}' + '_dev' + '.npy'
            tmp_fixed_am_file = file_dir + file_header + f'{i:03}' + '_AM_dev' + "_fixedpt_scale_" + f"{scale}" + ".inp"
            tmp_fixed_file = file_dir + file_header + f'{i:03}' + '_dev' + "_fixedpt_scale_" + f"{scale}" + ".inp"
            tmp_integrated_file = tmp_dir + 'data' + f'{i:03}' + '.inp'
            os.system(
                f"/home/ubuntu/miniconda3/envs/acc/bin/python ./convert_np_to_fixedpt.py --scale {scale} --inp {input_am_file_name}")
            os.system(
                f"/home/ubuntu/miniconda3/envs/acc/bin/python ./convert_np_to_fixedpt.py --scale {scale} --inp {input_file_name}")
            os.system(f"cat {tmp_fixed_file} {tmp_fixed_am_file} {weight_file}>{tmp_integrated_file}")
        if task in cv_task:
            input_file_name = file_dir + file_header + f'_{i:04}' + '.npy'
            tmp_fixed_file = file_dir + file_header + f'_{i:04}' + "_fixedpt_scale_" + f"{scale}" + ".inp"
            os.system(
                f"/home/ubuntu/miniconda3/envs/acc/bin/python ./convert_np_to_fixedpt.py --scale {scale} --inp {input_file_name}")
            tmp_integrated_file = tmp_dir + 'data' + f'{i:04}' + '.inp'
            os.system(f"cat {tmp_fixed_file} {weight_file}>{tmp_integrated_file}")
        cmd = binary_file
        with open(tmp_integrated_file) as f:
            ret, _ = subprocess.Popen(cmd, stdin=f, stdout=subprocess.PIPE).communicate()
            f.close()
        ret = int(ret)
        if task in input_restriction.keys():
            assert ret in input_restriction[task], f"[Error] Unexpected Output: {ret}"
        prediction_list.append(ret)
        if ret == label[i]:
            true_num += 1
            total_true_num += 1
        if multi_thread_disabled:
            print(
                f'No. {i} = {ret}, label = {label[i]}, acc = {true_num / (i + 1 - start) * 100}% ({true_num}/'
                f'{(i + 1 - start)})')
        else:
            print(f'{threading.Thread.getName(threading.currentThread())}, No. {i} = {ret}, label = {label[i]}, '
                  f'acc = {true_num / (i + 1 - start) * 100}% ({true_num}/{(i + 1 - start)})')
        os.remove(tmp_integrated_file)
    acc_dict[str(start)] = prediction_list


def get_loc_list(fullNum, threadNum):
    if threadNum == 0:
        return [0, fullNum]
    loc_list = []
    segment = int(fullNum / threadNum)
    pointer = 0
    while (pointer + segment) < fullNum:
        pointer += segment
        loc_list.append(pointer)
    if fullNum not in loc_list:
        loc_list.append(fullNum)
    return loc_list


if __name__ == '__main__':
    if (task not in nlp_task) and (task not in cv_task):
        raise NotImplementedError(f"{task} is not supported")
    loc_list = get_loc_list(test_num, thread_num)
    print("Segmentation for Multi-threading:")
    print(loc_list)
    thread_list = []
    if thread_num != 0:
        for i, locs in enumerate(loc_list):
            if i == 0:
                t = threading.Thread(name=("Threading" + str(i)), target=sub_get, args=(0, loc_list[i]))
            else:
                t = threading.Thread(name=("Threading" + str(i)), target=sub_get, args=(loc_list[i - 1], loc_list[i]))
            thread_list.append(t)
        for t in thread_list:
            t.start()
        for t in thread_list:
            t.join()
    else:
        sub_get(0, test_num)
    np.save(output_file, acc_dict)
    print(f"Final File Saved, Acc = {total_true_num / test_num * 100}% ({total_true_num}/{test_num})")
