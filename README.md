# Iron: Private Inference on Transformer
Source code and the implementation of the experiments in 2022 NIPS submission -  _Iron: Private Inference on 
Transformer_.

## Introduction
In the submission, we introduced _Iron_, a Transformer-based private inference framework which helps client and server 
hide their sensitive data from other participants. With the aid of our framework and protocols, the inference result can 
be calculated under the coordination between both parties without revealing any private data.

## Contents
This repository consists of the following parts:
- __EzPC__: A framework helps convert TensorFlow codes into MPC-friendly CPP codes.
- __bert__: BERT is a popular NLP model, applicable in various tasks, including sentence classification. Some 
modifications are applied in this part to removed operations which are incompatible with our crypto design. Some
useful scripts are also included in the folder.
- __bert_pure__: The original implementation of BERT, including training, fine-tuning, running inference and so on. 
Codes in this part are _almost_ identical with the original repository, refer to README.md in the folder.
- __test_src__: Scripts for getting experiment results.
- __demo__: Example data for using converted code under fixed point value.
- __tensorflow-vit__: Transformer model in CV Task, known as ViT. This folder contains Tensorflow codes to realize 
CCT(optimized model for ViT), and utilities for deploy them in fixed-point settings.
- __vit-pytorch__: Transformer model in CV Task, known as ViT. This folder contains PyTorch codes to realize 
CCT(optimized model for ViT), pre-trained models. We used weight in this folder for further experiments..


Note: Currently only experiment scripts for plaintext CPP/Tensorflow-implemented BERT/CCT(ViT) under different fixed
point encoding schemes are included.

## Installation
### Requirements
Main third-party packages and their recommended versions are listed below.

|                                 | Python | Tensorflow | Pytorch | Numpy  | CPP Compiler | Installed by Docker |
|:-------------------------------:|:------:|:----------:|---------|:------:|:------------:|:-------------------:|
|       __bert, bert_pure__       | 3.7.x  |   1.15.0   |         | 1.16.0 |              |                     |
|          __test_src__           | 3.7.x  |            |         | 1.16.0 |     g++      |                     |
|            __EzPC__             |        |            |         |        |    (g++)     |          √          |
| __vit-pytorch, tensorflow-vit__ | 3.9.x  |   1.15.0   | 1.10.0  | 1.16.0 |              |                     |


Special Note: We recommend you to build `EzPC` by simply run docker command `docker pull ezpc/ezpc:latest` in the 
terminal instead of building from source. If you wish to start from source, please refer to the steps in [EzPC 
repository](https://github.com/mpc-msri/EzPC/).

### Build
The build process for `EzPC` has been done when installing by Docker. No additional steps needed.
### Code Update
We have changed some codes in the `EzPC` so that a CNN-friendly framework can be used to process Transformer-based 
models. To apply these changes, just replace the two folders in the docker with the very folders under */EzPC* directory 
in this repository.

## Workflow
***NOTE: Here is a naive explanation of the workflow of our framework. Please refer to README.md in the folder of each 
part for detailed running instructions. To directly reproduce the experiment data, 
please refer to [README.md](https://github.com/xingpz2008/Iron/blob/main/demo/README.md).***

### Model Preparing
To test a BERT model in plaintext CPP codes with fixed point encoding, or to reproduce the results in the submission, 
the user need to fine-tune or train their model using codes in `bert_pure` first. (For CV tasks, this step can be 
skipped.)

Then `bert`(nlp task) or `tensorflow-vit`(cv task) freezes the graph and convert the model to a protocol 
buffer file _.pb_ with its architecture and weights.

After getting the converted file, copy it to the `EzPC` environment in docker (if the user didn't build from source). The 
model will be converted to a _.ezpc_ file written in intermediate programming language alone with a dumped weight file 
_.inp_ under fixed point setting. Some modifications are needed before converting an _.ezpc_ file to a _.cpp_ file. 

The last step for preparing the model is to compile the CPP code to an executable binary file. This can be done in 
either `EzPC` or `test_src` environment, as long as there is a g++ compiler.

### Data Preparing
Our framework support multiple GELU Benchmark datasets. However, the original data can not be used directly and some 
extra processing steps are compulsory.

(Assuming model fine-tune is finalized.)

To begin with, the user have to get the embedding output of the data. Embedding layers are excluded in our design, so we
use the same method to process it in either plaintext or crypto settings. This can be done by `bert` and two _.npy_ 
files should be generated. These are **embedding_outputs** and **input_masks** of the original data stored in the 
format of Numpy Array. The user can control the number of data to be processed, i.e. you can process all data at 
the push of a button. An extra _.npy_ file containing the **label** will be generated at the end of the process.

### Test the Model with Data
Copy the binary file and data files to `test_src` (and its subdirectory if needed). The only thing to do now is to set 
the parameters in the script. Stuffs like data type conversion, file concatenation will be handled automatically. The 
final result will be returned after the accomplishment of the program.

## Schedule & Progress
Here we listed some points to be handled in the future.
- **Directory re-organization**: The codes are not well organized and redundant code exists in the repository currently.
- **Code optimization**: Parameters should be handled by a parser rather than being explicitly written in scripts.
- **Other codes in submission**: To appear.

## Reference
[SIRNN: A Math Library for Secure RNN Inference](https://eprint.iacr.org/2021/459)  
Deevashwer Rathee, Mayank Rathee, Rahul Kranti Kiran Goli, Divya Gupta, Rahul Sharma, Nishanth Chandran, Aseem Rastogi  
*IEEE S&P 2021*

[CrypTFlow2: Practical 2-Party Secure Inference](https://eprint.iacr.org/2020/1002)  
Deevashwer Rathee, Mayank Rathee, Nishant Kumar, Nishanth Chandran, Divya Gupta, Aseem Rastogi, Rahul Sharma  
*ACM CCS 2020*

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, 
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby  


[Escaping the Big Data Paradigm with Compact Transformers](https://arxiv.org/abs/2104.05704)  
Ali Hassani, Steven Walton, Nikhil Shah, Abulikemu Abuduweili, Jiachen Li, Humphrey Shi  
*CVPR 2021*

`EzPC` in this repository is originally forked from [mpc-msri/EzPC](https://github.com/mpc-msri/EzPC)  
`bert` and `bert_pure` are originally forked from [google-research/bert](https://github.com/google-research/bert)  
`vit-pytorch` is originally forked from [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)  
Pre-trained models for CCT is contained in [SHI-Labs/Compact-Transformers](https://github.com/SHI-Labs/Compact-Transformers)  


## Disclaimer
This repository is a proof-of-concept prototype.