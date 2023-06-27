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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



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

    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambda x: global_contrast_normalization(x)),
    #                                 transforms.Normalize([min_max[args.normal_class][0]],
    #                                                      [min_max[args.normal_class][1] \
    #                                                      -min_max[args.normal_class][0]])])
    
    # pretrained 모델 사용 위해
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
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




class MVTEC_loader(data.Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform

        self.image_paths, self.labels = self._load_data()

    def _load_data(self):
        image_paths = []
        labels = []
        classes = os.listdir(self.data_dir)

        for class_name in classes:

            class_dir = os.path.join(self.data_dir, class_name)
            
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    image_paths.append(image_path)
                    labels.append(0 if class_name == 'good' else 1)

        return image_paths, labels

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)
    
    

def get_mvtec(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = MVTEC_loader(os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = MVTEC_loader(os.path.join(data_dir, 'test'), transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # for path, target in zip(test_dataset.image_paths, test_dataset.labels):
    #     print("class path, index : ", path, target)
    
    
    return train_dataloader, test_dataloader



class MVTEC_multi_loader(data.Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform

        self.image_paths, self.labels = self._load_data()

    def _load_data(self):
        image_paths = []
        labels = []
        classes = os.listdir(self.data_dir)

        for class_name in classes:
            # print("class_name:", class_name)
            class_dir = os.path.join(self.data_dir, class_name)
            # print("class_dir:", class_dir)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    image_paths.append(image_path)

                    if 'bottle' in image_name:       label = 0 
                    elif 'cable' in image_name:      label = 1
                    elif 'capsule' in image_name:    label = 2
                    elif 'carpet' in image_name:     label = 3
                    elif 'grid' in image_name:       label = 4
                    elif 'hazelnut' in image_name:   label = 5
                    elif 'metal_nut' in image_name:  label = 6
                    elif 'leather' in image_name:    label = 7
                    elif 'pill' in image_name:       label = 8
                    elif 'screw' in image_name:      label = 9
                    elif 'tile' in image_name:       label = 10
                    elif 'toothbrush' in image_name: label = 11
                    elif 'transistor' in image_name: label = 12
                    elif 'wood' in image_name:       label = 13
                    else                     :       label = 14

                    labels.append(label)  
                    
        # class 별 개수 count
        from collections import Counter

        label_counts = Counter(labels)
        print(label_counts)
        
        return image_paths, labels

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)
    
def get_multi_mvtec(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = MVTEC_multi_loader(os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = MVTEC_multi_loader(os.path.join(data_dir, 'test'), transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # for path, target in zip(test_dataset.image_paths, test_dataset.labels):
    #     print("class path, index : ", path, target)
    
    return train_dataloader, test_dataloader