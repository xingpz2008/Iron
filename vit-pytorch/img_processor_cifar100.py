import os
import torch
from torchvision import transforms, datasets
import torch.utils.data as data
from vit_pytorch.cct_ import CCT_embedding
import numpy as np

file_path = ''

def create_loader():
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset = datasets.CIFAR100(
        "./data_cifar100", train=False, download=True, transform=val_transform
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)
    return val_loader


if __name__ == '__main__':
    cct_emb = CCT_embedding(
        img_size=32,
        embedding_dim=256,
        n_conv_layers=1,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        dropout=0.,
        pooling_stride=2,
        pooling_padding=1,
        num_layers=7,
        num_heads=4,
        mlp_radio=2.,
        num_classes=100,
        positional_embedding='sine',  # ['sine', 'learnable', 'none']
    )
    state_dict = torch.load(file_path)
    cct_emb.load_state_dict(state_dict, strict=False)
    loader = create_loader()
    label_list = []
    for idx, (val, target) in enumerate(loader):
        emb_res = cct_emb(val).detach().numpy()
        saved_val_file = 'cifar100File_' + f'{idx:04}' + '.npy'
        np.save(file='./data_cifar100/' + saved_val_file, arr=emb_res)
        label_list.append(int(target))
        print(f'Finished idx = {idx}, this label = {int(target)}')
    np.save(file='../data_cifar100/cifar100Label.npy', arr=label_list)
