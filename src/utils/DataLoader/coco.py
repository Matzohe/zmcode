import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import json
import cv2
import numpy as np
from collections import OrderedDict
import math

from ..utils import ClipPreProcess

# coco_2017 dataset structure
# coco_2017
#   |- train2017
#   |- train_annotations
#       |-captions_train2017.json         
#       |-captions_val2017.json
#       |-instances_train2017.json
#       |-instances_val2017.json
#       |-person_keypoints_train2017.json
#       |-person_keypoints_val2017.json
#   |- test2017
#   |- test_annotations
#       |-image_info_test-dev2017.json
#       |-image_info_test2017.json
#   |- unlabeled2017


# annotations这个文件中，保存了几个字典，分别是'info', 'licenses', 'images', 'annotations', 'categories'
# 下面是train的信息
# 在caption文件中，没有categories, 在annotations中保存有图像id，annotations的id，以及对图片的描述 caption
# 在instances文件中，有categories属性，同时，annotation中，包含了category_id，可以在categories中找到category_id对应的类别，同时annotation中还包含了图像分割的segmentation的多边形信息
# 下面是test的信息
# test目前只下载了两个json文件，这两个文件给图像的标注为图像的类别，并没有更多的描述，字典为'info', 'licenses', 'images', 'categories'

class CoCoTrainDataset(Dataset):
    def __init__(self, config, target="instances", mission="category_id", preprocesser=ClipPreProcess):
        # Expect the structure of coco dataset is the same with the structure of coco_2017
        # target list include "instances", "captions", "keypoints", et.al, see config.cfg for more details
        self.coco_root = config.DATASET["coco"]
        self.batch_size = int(config.MODEL["batch_size"])
        self.coco_target = config.DATASET["coco_target"]
        self.mission = mission
        self.train_annotations = OrderedDict()
        self.preprocesser = preprocesser

        if target not in self.coco_target:
            raise NotImplementedError("loading coco dataset error, no such target named {}, see config.cfg for more details".format(target))

        if target == "instances":
            with open(os.path.join(self.coco_root, "train_annotations", "instances_train2017.json"), "r") as f:
                train_annotations = json.load(f)["annotations"]
                if mission not in train_annotations[0].keys():
                    raise NotImplementedError("loading coco dataset error, no such annotation key named {}".format(mission))
                
                for each in train_annotations:
                    self.train_annotations[each["image_id"]] = each

        elif target == "captions":
            with open(os.path.join(self.coco_root, "train_annotations", "captions_train2017.json"), "r") as f:
                train_annotations = json.load(f)["annotations"]
                if mission not in train_annotations[0].keys():
                    raise NotImplementedError("loading coco dataset error, no such annotation key named {}".format(mission))
                
                for each in train_annotations:
                    self.train_annotations[each["image_id"]] = each

        elif target == "keypoints":
            with open(os.path.join(self.coco_root, "train_annotations", "person_keypoints_train2017.json"), "r") as f:
                train_annotations = json.load(f)["annotations"]
                if mission not in train_annotations[0].keys():
                    raise NotImplementedError("loading coco dataset error, no such annotation key named {}".format(mission))
                
                for each in train_annotations:
                    self.train_annotations[each["image_id"]] = each

        else:
            raise NotImplementedError("Haven't add this target into the CoCoDataset, please edit coco.py")

    def __getitem__(self, index):
        image_list = []
        label_list = []
        for each in list(self.train_annotations.keys())[index * self.batch_size: (index + 1) * self.batch_size 
                                        if (index + 1) * self.batch_size < len(self.train_annotations) else len(self.train_annotations)]:
            
            file_name = os.path.join(self.coco_root, "train2017", "{:012}".format(each) + ".jpg")
            image_list.append(self.preprocesser(file_name))
            if self.mission == "category_id":
                label_list.append(self.train_annotations[each][self.mission])
            else:
                raise ValueError("Haven't process this misson's output structure, please edit coco.py")
        
        return torch.cat(image_list), torch.tensor(label_list)

    def __len__(self):
        return math.ceil(len(self.train_annotations) / self.batch_size)

      
class CoCoValDataset(Dataset):
    def __init__(self, config, target="instances", mission="category_id", preprocesser=ClipPreProcess):
        # Expect the structure of coco dataset is the same with the structure of coco_2017
        # target list include "instances", "captions", "keypoints", et.al, see config.cfg for more details
        self.coco_root = config.DATASET["coco"]
        self.batch_size = int(config.MODEL["batch_size"])
        self.coco_target = config.DATASET["coco_target"]
        self.mission = mission
        self.val_annotations = OrderedDict()

        if target not in self.coco_target:
            raise NotImplementedError("loading coco dataset error, no such target named {}, see config.cfg for more details".format(target))

        if target == "instances":
            with open(os.path.join(self.coco_root, "train_annotations", "instances_val2017.json"), "r") as f:
                val_annotations = json.load(f)["annotations"]
                if mission not in val_annotations[0].keys():
                    raise NotImplementedError("loading coco dataset error, no such annotation key named {}".format(mission))
                
                for each in val_annotations:
                    self.val_annotations[each["image_id"]] = each

        elif target == "captions":
            with open(os.path.join(self.coco_root, "train_annotations", "captions_val2017.json"), "r") as f:
                val_annotations = json.load(f)["annotations"]
                if mission not in val_annotations[0].keys():
                    raise NotImplementedError("loading coco dataset error, no such annotation key named {}".format(mission))
                
                for each in val_annotations:
                    self.val_annotations[each["image_id"]] = each

        elif target == "keypoints":
            with open(os.path.join(self.coco_root, "train_annotations", "person_keypoints_val2017.json"), "r") as f:
                val_annotations = json.load(f)["annotations"]
                if mission not in val_annotations[0].keys():
                    raise NotImplementedError("loading coco dataset error, no such annotation key named {}".format(mission))
                
                for each in val_annotations:
                    self.val_annotations[each["image_id"]] = each

        else:
            raise NotImplementedError("Haven't add this target into the CoCoDataset, please edit coco.py")

    def __getitem__(self, index):
        image_list = []
        label_list = []
        for each in list(self.train_annotations.keys())[index * self.batch_size: (index + 1) * self.batch_size 
                                        if (index + 1) * self.batch_size < len(self.val_annotations) else len(self.val_annotations)]:
            
            file_name = os.path.join(self.coco_root, "train2017", "{:012}".format(each) + ".jpg")
            image_list.append(self.preprocesser(file_name))
            if self.mission == "category_id":
                label_list.append(self.val_annotations[each][self.mission])
            else:
                raise ValueError("Haven't process this misson's output structure, please edit coco.py")
        
        return torch.from_numpy(np.concatenate(image_list, axis=0), dtype=torch.uint8), torch.tensor(label_list)

    def __len__(self):
        return math.ceil(len(self.val_annotations) / self.batch_size)
