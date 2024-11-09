from src.utils.DataLoader.NSDDataLoader import NSDDataset
from src.utils.DataLoader.RootListDataLoader import get_root_list_dataloader
from src.utils.ModelTrainer import ModelTrainer
from src.utils.utils import INIconfig
from tqdm import tqdm
import src.MultiModal.clip as clip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class LinearClip2Brain:
    def __init__(self, config):
        self.config = config
        self.device = config.TRAINING['device']
        self.lr = float(config.TRAINING['lr'])
        self.lr_decay_rate = float(config.TRAINING['lr_decay_rate'])
        self.batch_size = int(config.TRAINING['batch_size'])
        self.coco_root = config.DATASET['coco_root']
        self.model, self.model_transform = clip.load(config.IMAGE_EMBEDDING['model_name'])
        self.embedding_dim = int(config.IMAGE_EMBEDDING["embedding_dim"])
        self.dataset = NSDDataset(config)
        self.loss_function = nn.MSELoss()
        self.epochs = int(config.TRAINING['epochs'])

        self.individual_bool_list = None
        self.same_bool_list = None

        self.training_image_root_list = None
        self.val_image_root_list = None

        self.train_avg_activation = None
        self.val_avg_activation = None

        self.voxel_number = None
        self.linear_layer = None
        self.optimizer = None
        
    def _initialize(self, subj=1, voxel_activation_roi="SELECTIVE_ROI"):
        avg_activation = self.dataset.load_avg_activation_value(subj, voxel_activation_roi)

        self.individual_avg_activation = avg_activation[self.individual_bool_list]
        self.same_avg_activation = avg_activation[self.same_bool_list]

        self.individual_bool_list, self.same_bool_list = self.dataset.load_individual_and_same_image_bool(subj)

        image_root_list = self.dataset.load_image_root(subj)
        image_root_list = [os.path.join(self.cooc_root, each) for each in self.image_root_list]

        self.training_image_root_list = []
        self.val_image_root_list = []

        for i in range(len(image_root_list)):
            if self.individual_bool_list[i]:
                self.training_image_root_list.append(image_root_list[i])
            else:
                self.val_image_root_list.append(image_root_list[i])

        self.voxel_num = self.avg_activation.shape[-1]
        self.linear_layer = nn.Linear(self.embedding_dim, self.voxel_num, bias=True).to(self.device)
        self.optimizer = optim.AdamW(self.linear_layer.parameters(), lr=float(self.config.TRAINING['lr']), 
                                     weight_decay=float(self.config.TRAINING['weight_decay']))

    def _change_target_model(self, target_model, embedding_dim):

        self.model = target_model
        self.embedding_dim = embedding_dim

        self.linear_layer = nn.Linear(self.embedding_dim, self.voxel_num)

    def _setup_image_dataloader(self):
        if self.image_root_list is None:
            raise ValueError("please use _initialize funtion first to get the image root list")
        self.training_dataloader = get_root_list_dataloader(batch_size=self.batch_size, image_root_list=self.training_image_root_list, image_transform=self.model_transform)
        self.val_dataloader = get_root_list_dataloader(batch_size=self.batch_size, image_root_list=self.val_image_root_list, image_transform=self.model_transform)
    
    def linear_fitting(self, subj=1, voxel_activation_roi="SELECTIVE_ROI", summary_writter=None):

        self._initialize(subj, voxel_activation_roi)
        self._setup_image_dataloader()

        for epoch in range(self.epochs):
            new_lrate = self.lr * (self.lr_decay_rate ** (epoch / self.epochs))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lrate

            for i, images in tqdm(enumerate(self.training_dataloader), total=len(self.training_dataloader)):
                with torch.no_grad():
                    image_embeddings = self.model.encode_image(images.to(self.device))
                
                predict_activation = self.linear_layer(image_embeddings)
                try:
                    target = self.individual_avg_activation[i * self.batch_size: (i + 1) * self.batch_size].to(self.device)
                except:
                    target = self.individual_avg_activation[i * self.batch_size:].to(self.device)
                
                loss = self.loss_function(predict_activation, target).sum()
                if summary_writter is not None:
                    summary_writter.add_scalar("loss", loss.detach().cpu(), epoch * len(self.training_dataloader) + i)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        