import torch
from torch.utils.data import DataLoader, Dataset
from ..ImagePreporcessUitls import ClipPreProcess
import json
import os


class levirDataset(Dataset):
    def __init__(self, json_root, image_root, preprocess):
        with open(json_root, "r") as f:
            self.data = json.load(f)["images"]
        self.image_path_list = []
        self.image_discription = {}
        self.preprocess = preprocess
        for each in self.data:
            img_path = os.path.join(image_root, each["filepath"], each["filename"])
            self.image_path_list.append(img_path)
            sentence = each["sentences"]
            sentence_list = []
            for sent in sentence:
                sentence_list.append(sent["raw"][1:])
            self.image_discription[img_path] = sentence_list

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img_path = self.image_path_list[index]
        img = self.preprocess(img_path)
        sentence = self.image_discription[img_path]
        return img, sentence

        