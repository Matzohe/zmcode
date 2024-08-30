import torch
import torch.nn as nn
from torch.utils.data import Dataset
from ..utils import INIconfig

class CoCoDataset(Dataset):
    def __init__(self, coco):
        self.coco = coco
        self.ids = list(sorted(coco.anns.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

