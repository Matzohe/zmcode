from src.utils.DataLoader.NSDDataLoader import NSDDataset
from src.utils.DataLoader.RootListDataLoader import get_root_list_dataloader
from src.utils.utils import INIconfig, check_path
from src.utils.r2_score import r2_score
from ..clip2brain.utils.load_target_model import get_target_model
from ..clip2brain.utils.extract_target_layer_function import decoding_extract_target_layer_output, extract_image_embedding
from tqdm import tqdm
import src.MultiModal.clip as clip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class LinearBrain2CLIP:
    def __init__(self, config):
        self.config = config
        self.device = config.TRAINING['device']
        self.dtype = eval(config.TRAINING['dtype'])
        self.lr = float(config.TRAINING['lr'])
        self.lr_decay_rate = float(config.TRAINING['lr_decay_rate'])
        self.batch_size = int(config.TRAINING['batch_size'])
        self.coco_root = config.DATASET['coco']
        self.NSD_coco_root = config.DATASET['nsd_coco']
        # there is two activation root, one is the final image embedding,
        # and another is the middle hidden image embedding
        self.image_activation_save_root = config.NSD['image_activation_save_root']
        self.image_same_activation_save_root = config.NSD['image_same_activation_save_root']
        self.middle_activation_save_root = config.NSD['middle_activation_save_root']
        self.middle_same_activation_save_root = config.NSD['middle_same_activation_save_root']

        self.from_coco_split = eval(config.NSD['from_coco_split'])
        self.model, self.model_transform = get_target_model(config.IMAGE_EMBEDDING['model_name'], device=self.device)
        self.model_name = config.IMAGE_EMBEDDING['model_name'].replace("/", "-")
        self.embedding_dim = int(config.IMAGE_EMBEDDING["embedding_dim"])
        self.dataset = NSDDataset(config)
        self.loss_function = nn.MSELoss().to(self.device)
        self.epochs = int(config.TRAINING['epochs'])

        self.linear_save_root = config.NSD['linear_save_root']
        self.middle_layer_linear_save_root = config.NSD['middle_layer_linear_save_root']

        self.calcutale_type = eval(config.TRAINING['calcutale_type'])
        self.r2_save_root = config.NSD['r2_save_root']
        self.middle_layer_r2_save_root = config.NSD['middle_layer_r2_save_root']

        self.individual_bool_list = None
        self.same_bool_list = None

        self.training_image_root_list = None
        self.val_image_root_list = None

        self.train_avg_activation = None
        self.val_avg_activation = None

        self.voxel_number = None
        self.linear_layer = None
        self.optimizer = None
        
    def _initialize(self, subj=1, voxel_activation_roi="ventral_visual_pathway_roi"):

        self.model_name = self.config.IMAGE_EMBEDDING['model_name'].replace("/", "-")
        
        avg_activation = self.dataset.load_avg_activation_value(subj, voxel_activation_roi)
        self.individual_bool_list, self.same_bool_list = self.dataset.load_individual_and_same_image_bool(subj)

        self.individual_avg_activation = avg_activation[self.individual_bool_list].to(self.device)
        self.same_avg_activation = avg_activation[self.same_bool_list].to(self.device)

        image_root_list = self.dataset.load_image_root(subj)
        if self.from_coco_split:
            image_root_list = [os.path.join(self.coco_root, each) for each in image_root_list]
        else:
            image_root_list = [os.path.join(self.NSD_coco_root, each) for each in image_root_list]

        self.training_image_root_list = []
        self.val_image_root_list = []

        for i in range(len(image_root_list)):
            if self.individual_bool_list[i]:
                self.training_image_root_list.append(image_root_list[i])
            else:
                self.val_image_root_list.append(image_root_list[i])

        self.voxel_num = avg_activation.shape[-1]
        self.linear_layer = nn.Linear(self.voxel_num, self.embedding_dim, bias=True).to(self.device)
        self.optimizer = optim.AdamW(self.linear_layer.parameters(), lr=float(self.config.TRAINING['lr']), 
                                     weight_decay=float(self.config.TRAINING['weight_decay']))

    def _change_target_model(self, target_model, embedding_dim):

        self.model = target_model
        self.embedding_dim = embedding_dim

        self.linear_layer = nn.Linear(self.embedding_dim, self.voxel_num)
        self.optimizer = optim.AdamW(self.linear_layer.parameters(), lr=float(self.config.TRAINING['lr']), 
                                     weight_decay=float(self.config.TRAINING['weight_decay']))

    # when get multi target layer, we can use this function to change the linear layer
    def _change_linear_layer(self, embedding_dim):
        self.linear_layer = nn.Linear(self.voxel_num, embedding_dim).to(self.device)
        self.optimizer = optim.AdamW(self.linear_layer.parameters(), lr=float(self.config.TRAINING['lr']), 
                                     weight_decay=float(self.config.TRAINING['weight_decay']))

    def _setup_image_dataloader(self):
        if self.training_image_root_list is None:
            raise ValueError("please use _initialize funtion first to get the training image root list")
        self.training_dataloader = get_root_list_dataloader(batch_size=self.batch_size, image_root_list=self.training_image_root_list, image_transform=self.model_transform)
        self.val_dataloader = get_root_list_dataloader(batch_size=self.batch_size, image_root_list=self.val_image_root_list, image_transform=self.model_transform)

    def target_layer_linear_fitting(self, subj=1, voxel_activation_roi="ventral_visual_pathway_roi", target_layer=None, summary_writer=None):
        if target_layer is None:
            raise ValueError("target_layer can't be None")
        elif isinstance(target_layer, str):
            target_layer = [target_layer]

        self._initialize(subj, voxel_activation_roi)
        self._setup_image_dataloader()
        check_path(self.middle_activation_save_root.format(subj, self.model_name, ""))
        check_path(self.middle_same_activation_save_root.format(subj, self.model_name, ""))

        for each in target_layer:
            try:
                _ = torch.load(self.middle_activation_save_root.format(subj, self.model_name, each))
            except:
                compressed_save_list = decoding_extract_target_layer_output(self.model, self.model_name, target_layer, 
                                                                   self.training_dataloader, self.device, self.dtype)
                for i, each in enumerate(target_layer):
                    torch.save(compressed_save_list[i], self.middle_activation_save_root.format(subj, self.model_name, each))
                del compressed_save_list
                break
        
        for each in target_layer:
            try:
                _ = torch.load(self.middle_same_activation_save_root.format(subj, self.model_name, each))
            except:
                compressed_save_list = decoding_extract_target_layer_output(self.model, self.model_name, target_layer, 
                                                                   self.val_dataloader, self.device, self.dtype)
                for i, each in enumerate(target_layer):
                    torch.save(compressed_save_list[i], self.middle_same_activation_save_root.format(subj, self.model_name, each))

                del compressed_save_list
                break
        

        for _, each_layer in enumerate(target_layer):
            training_data = torch.load(self.middle_activation_save_root.format(subj, self.model_name, each_layer), map_location=self.device)
            training_data = torch.cat(training_data, dim=0).to(dtype=self.dtype)
            valid_data = torch.load(self.middle_same_activation_save_root.format(subj, self.model_name, each_layer), map_location=self.device)
            valid_data = torch.cat(valid_data, dim=0).to(dtype=self.dtype)

            self._change_linear_layer(training_data.shape[-1])
            for epoch in range(self.epochs):
                new_lrate = self.lr * (self.lr_decay_rate ** (epoch / self.epochs))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lrate

                for i in range(training_data.shape[0] // self.batch_size + 1):
                    
                    try:
                        if i * self.batch_size > training_data.shape[0]:
                            break

                        batch_embedding = training_data[i * self.batch_size: (i + 1) * self.batch_size]
                    except:
                        batch_embedding = training_data[i * self.batch_size: ]
                    target = self.individual_avg_activation[i * self.batch_size: i * self.batch_size + batch_embedding.shape[0]].to(self.device)
                    # in some condition, subj haven't see several images, and the result is Nanm we need to process this condition
                    if torch.isnan(target).any():
                        nan_embedding_list = (torch.isnan(target).sum(dim=-1) == 0)
                        batch_embedding = batch_embedding[nan_embedding_list]
                        target = target[nan_embedding_list]
                    target = target / torch.norm(target, dim=-1, keepdim=True)
                    predict_embedding = self.linear_layer(target)
                    loss = self.loss_function(predict_embedding, batch_embedding)
                    if summary_writer is not None:
                        summary_writer.add_scalar("subj{}_loss".format(subj), loss.detach().cpu(), epoch * len(self.training_dataloader) + i)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                
                # validate the performence of this epoch's trained model
                valid_loss = 0
                with torch.no_grad():
                    for i in range(valid_data.shape[0] // self.batch_size + 1):
                        
                        try:
                            if i * self.batch_size > valid_data.shape[0]:
                                break
                            batch_embedding = valid_data[i * self.batch_size: (i + 1) * self.batch_size]
                        except:
                            batch_embedding = valid_data[i * self.batch_size: ]

                        batch_embedding = batch_embedding.to(self.device)
                        target = self.same_avg_activation[i * self.batch_size: i * self.batch_size + batch_embedding.shape[0]].to(self.device)
                        if torch.isnan(target).any():
                            nan_embedding_list = (torch.isnan(target).sum(dim=-1) == 0)
                            batch_embedding = batch_embedding[nan_embedding_list]
                            target = target[nan_embedding_list]
                        target = target / torch.norm(target, dim=-1, keepdim=True)
                        predict_embedding = self.linear_layer(target)
                        
                        loss = self.loss_function(predict_embedding, batch_embedding).sum()

                        valid_loss += loss.detach().cpu()

                    valid_loss = valid_loss / valid_data.shape[0]

                print("epoch:", epoch, "    valid_loss:", valid_loss)

            model_save_root = self.middle_layer_linear_save_root.format(subj, self.model_name, each_layer,voxel_activation_roi)
            check_path(model_save_root)
            torch.save(self.linear_layer.state_dict(), model_save_root)
            