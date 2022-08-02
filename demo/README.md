# Demo Section (for Result Reproducing)
This part contains examples for using Iron to make inference on Transformer.

## File Structure
- __plaintext_exp__: Demo for plaintext experiments, accuracy will be tested.
  - __cv__: Tasks in Computer Vision 
    - __cct_cifar100__: Image classification task on CIFAR100 task with CCT model
    - __cct_ImageNet__: Image classification task on ImageNet task with CCT model
  - __nlp__: Tasks in Natural Language Processing
    - __mrpc_tiny__: Sentence classification task on MRPC task with Bert-Tiny model
    - __sst2_tiny__: Sentence classification task on SST-2 task with Bert-Tiny model
    - __mnli_tiny__: Sentence classification task on MNLI task with Bert-Tiny model
    - __qnli_tiny__: Sentence classification task on QNLI task with Bert-Tiny model

## Specific Folder Contents
Inside the folder for each {task}_{model} pair, there are following contents typically.
- __A .cpp file__: It was the model converted by CryptFlow, which was named under the rule that, 
{model}\_{task}\_{BitWidth}\_cpp\_{ScaleFactor(optional)}\_0.cpp, where _BitWidth_ and _ScaleFactor_ are parameters for 
secure computation. If a .cpp file does not specify its 
_ScaleFactor_, that means it uses default setting that _SacleFactor_=12.
- __CmakeList.txt__: Used for compiling the .cpp file.
- __An .inp file__: It was the weight in fixed-point representation for model, which was named under the rule that, 
{model}\_{task}\_input_weights_fixedpt\_{ScaleFactor(optional)}\.inp, 
where _ScaleFactor_ is the parameter for secure computation. 
- __An .ezpc file (optional)__: This is the file used in CryptFlow, generated from original tensorflow/onnx model, used
by `EzPC/EzPC` module, whose output will be .cpp file. We attached this for structure integrity.

## Usage
For general workflow of transformer inferring, please refer to this 
[README.md](https://github.com/xingpz2008/Iron/blob/main/README.md) file.

To get the data in the format of Numpy Array file, you can run `img_processor_cifar100.py` in `/vit-pytorch` for 
cifar100 task, or `get_input_data.py` in `/bert` for all nlp task, respectively. 

Note: Because we used only parts of the ImageNet data, input for reproduction cannot be directly generated. You can 
download data from [here](). 

After getting the data, put the input files into `/test_src/{task}/{model}`. There will be a label file generated at the
end of file generation, which should be put at `/test_src/` as well as the weight file and the .cpp file.

The next step is to compile the .cpp file by g++ (or any available C++ compiler). The binary file should be named as
{task}_{model}, e.g. _cct_cifar100_ with no postfix. 

With everything ready, run `/test_src/get_plaintext_dev_acc.py` after setting parameters in the script. Typically, only
4 parameters in the given section should be modified. The script should handle everything without human interference.