from src.utils.DataLoader.NSDDataLoader import NSDDataset
from src.utils.utils import INIconfig
import src.MultiModal.clip as clip
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class LinearClip2Brain(nn.Module):
    def __init__(self, config):
        super(LinearClip2Brain, self).__init__()
        self.config = config
        self.model, self.model_transform = clip.load(config.IMAGE_EMBEDDING['model_name'])
        self.embedding_dim = int(config.IMAGE_EMBEDDING["embedding_dim"])
        self.dataset = NSDDataset(config)
        self.voxel_number = None
        self.linear_layer = None
        
    def _initialize(self, subj=1, voxel_activation_roi="SELECTIVE_ROI"):
        avg_activation = self.dataset.load_avg_activation_value(subj, voxel_activation_roi)
        self.image_root_list = self.dataset.load_image_root(subj)
        self.voxel_num = avg_activation.shape[-1]

        del avg_activation
        
        self.linear_layer = nn.Linear(self.embedding_dim, self.voxel_num, bias=True)
        

    def _change_target_model(self, target_model, embedding_dim):

        self.model = target_model
        self.embedding_dim = embedding_dim

        self.linear_layer = nn.Linear(self.embedding_dim, self.voxel_num)

    def get_image_dataset(self):
        if self.image_root_list is None:
            raise ValueError("please use _initialize funtion first to get the image root list")
        


    def get_image_dataloader(self, subj=1, voxel_activation_roi="SELECTIVE_ROI"):
        pass
        


    def forward(self, image_embedding):
        return self.linear_layer(image_embedding)