# we used the precomputed min_max values from the original implementation: 
# https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/1901612d595e23675fb75c4ebb563dd0ffebc21e/src/datasets/mnist.py

import torch
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

from PIL import Image

from utils.utils import global_contrast_normalization


class MNIST_loader(data.Dataset):
    """This class is needed to processing batches for the dataloader."""
    def __init__(self, data, target, transform):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        """return transformed items."""
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y

    def __len__(self):
        """number of samples."""
        return len(self.data)


def get_mnist(args, data_dir='./data/mnist/'):
    """get dataloders"""
    # min, max values for each class after applying GCN (as the original implementation)
    min_max = [(-0.8826567065619495, 9.001545489292527),
                (-0.6661464580883915, 20.108062262467364),
                (-0.7820454743183202, 11.665100841080346),
                (-0.7645772083211267, 12.895051191467457),
                (-0.7253923114302238, 12.683235701611533),
                (-0.7698501867861425, 13.103278415430502),
                (-0.778418217980696, 10.457837397569108),
                (-0.7129780970522351, 12.057777597673047),
                (-0.8280402650205075, 10.581538445782988),
                (-0.7369959242164307, 10.697039838804978)]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: global_contrast_normalization(x)),
                                    transforms.Normalize([min_max[args.normal_class][0]],
                                                         [min_max[args.normal_class][1] \
                                                         -min_max[args.normal_class][0]])])
    train = datasets.MNIST(root=data_dir, train=True, download=True)
    test = datasets.MNIST(root=data_dir, train=False, download=True)

    x_train = train.data
    y_train = train.targets

    x_train = x_train[np.where(y_train==args.normal_class)]
    y_train = y_train[np.where(y_train==args.normal_class)]
                                    
    data_train = MNIST_loader(x_train, y_train, transform)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=0)
    
    x_test = test.data
    y_test = test.targets
    y_test = np.where(y_test==args.normal_class, 0, 1)
    data_test = MNIST_loader(x_test, y_test, transform)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=0)
    return dataloader_train, dataloader_test



def get_mvtec(data_dir):
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 이미지 크기 조정
        transforms.Grayscale(num_output_channels=1),  # 이미지를 1채널 흑백 이미지로 변환
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        transforms.Normalize(mean=[0.485], std=[0.229])  # 이미지 정규화
    ])
    
    # train dataset
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    train_dataset.targets = [0 if 'good' in os.path.basename(path) else 1 for path, _ in train_dataset.imgs]

    # test dataset
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
    test_dataset.targets = [0 if 'good' in os.path.basename(path) else 1 for path, _ in test_dataset.imgs]

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader
