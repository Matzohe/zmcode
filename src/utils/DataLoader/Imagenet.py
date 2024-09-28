import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import OrderedDict
import os

from ..ImagePreporcessUitls import ImageNetPreProcess


def get_classification(config):
    # caffe file is downloaded from url: http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    caffe_root = config.DATASET["imagenet_caffe"]
    labels = OrderedDict()
    with open(os.path.join(caffe_root, "synsets.txt"), "r") as f:
        for i, line in enumerate(f):
            label = line.strip()
            labels[label] = i

    return labels


class ImageNetDataset(Dataset):
    def __init__(self, config):
        pass

    def __getitem__():
        pass

    def __len__():
        pass
