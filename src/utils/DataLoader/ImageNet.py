import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
import os
import random
from PIL import Image
from ..ImagePreporcessUitls import ImageNetPreProcess
from ...utils.ImagePreporcessUitls import BarlowTwinsTransform


def get_classification(config):
    # caffe file is downloaded from url: http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    caffe_root = config.DATASET["imagenet_caffe"]
    labels = OrderedDict()
    with open(os.path.join(caffe_root, "synsets.txt"), "r") as f:
        for i, line in enumerate(f):
            label = line.strip()
            labels[label] = i

    return labels


def get_train_list(config):
    # caffe file is downloaded from url: http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    caffe_root = config.DATASET["imagenet_caffe"]
    root_list = []
    label_list = []
    with open(os.path.join(caffe_root, "train.txt"), "r") as f:
        for line in f:
            root_list.append(line.split(" ")[0])
            label_list.append(int(line.split(" ")[1]))
    return root_list, label_list


def get_val_list(config):
    # caffe file is downloaded from url: http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    caffe_root = config.DATASET["imagenet_caffe"]
    val_list = []
    label_list = []
    with open(os.path.join(caffe_root, "val.txt"), "r") as f:
        for line in f:
            val_list.append(line.split(" ")[0])
            label_list.append(int(line.split(" ")[1]))

    return val_list, label_list

class BarlowTwinsTrainDataset(Dataset):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.imagenet_root = config.DATASET["imagenet_root"]
        
        self.root_list, self.label_list = get_train_list(config)
        self.root_list = [os.path.join(self.imagenet_root, x) for x in self.root_list]
        self.transform = BarlowTwinsTransform()

    def __getitem__(self, index):
        image = Image.open(self.root_list[index]).convert("RGB")
        return image, self.label_list[index]

    def __len__(self):
        return len(self.root_list)


def get_BarlowTwins_training_dataloader(config):

    dataset = torchvision.datasets.ImageFolder(config.DATASET['imagenet_root'], BarlowTwinsTransform())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config.DATALOADER["batch_size"]),
        shuffle=eval(config.DATALOADER["shuffle"]),
        num_workers=int(config.DATALOADER["num_workers"]),
        pin_memory=eval(config.DATALOADER["pin_memory"]),
    )
    return dataloader


class ImageNetTrainDataset(Dataset):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.imagenet_root = config.DATASET["imagenet_root"]
        
        self.root_list, self.label_list = get_train_list(config)
        self.root_list = [os.path.join(self.imagenet_root, x) for x in self.root_list]

        random.shuffle(self.root_list)

    def __getitem__(self, index):
        return ImageNetPreProcess(self.root_list[index]).squeeze(0), self.label_list[index]

    def __len__(self):
        return len(self.root_list)


class ImageNetValDataset(Dataset):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.imagenet_root = config.DATASET["imagenet_root"]
        
        self.root_list, self.label_list = get_val_list(config)
        self.root_list = [os.path.join(self.imagenet_root, x) for x in self.root_list]
        
        random.shuffle(self.root_list)

    def __getitem__(self, index):
        return ImageNetPreProcess(self.root_list[index]).squeeze(0), self.label_list[index]

    def __len__(self):
        return len(self.root_list)


def get_imagenet_training_dataloader(config):
    dataset = ImageNetTrainDataset(config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config.DATALOADER["batch_size"]),
        shuffle=eval(config.DATALOADER["shuffle"]),
        num_workers=int(config.DATALOADER["num_workers"]),
        pin_memory=eval(config.DATALOADER["pin_memory"]),
    )
    return dataloader


def get_imagenet_validation_dataloader(config):
    dataset = ImageNetValDataset(config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config.DATALOADER["batch_size"]),
        shuffle=eval(config.DATALOADER["shuffle"]),
        num_workers=int(config.DATALOADER["num_workers"]),
        pin_memory=eval(config.DATALOADER["pin_memory"]),
    )
    return dataloader


