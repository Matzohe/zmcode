import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
import PIL.Image as Image


class weed_dataset(Dataset):
    def __init__(self, weed_json_root, weed_image_root, data_preprocessor=None):
        self.weed_image_root = weed_image_root
        self.json_files = [os.path.join(weed_json_root, each) for each in os.listdir(weed_json_root)]
        if data_preprocessor is None:
            data_preprocessor = transforms.ToTensor()
        self.data_preprocessor = data_preprocessor

    def __getitem__(self, index):
        with open(self.json_files[index], 'r') as f:
            json_data = json.load(f)

        img_path = os.path.join(self.weed_image_root, json_data['imagePath'])
        info_list = []
        for each in json_data['shapes']:
            points = each['points']
            info_list.append(((int(points[0][0] + 0.5), int(points[0][1] + 0.5)), int(math.dist(points[0], points[1]) + 0.5)))

        return img_path, info_list

    def __len__(self):
        len(self.json_files)

