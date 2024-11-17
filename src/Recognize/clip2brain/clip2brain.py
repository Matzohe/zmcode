from src.utils.DataLoader.NSDDataLoader import NSDDataset
from src.utils.DataLoader.RootListDataLoader import get_root_list_dataloader
from src.utils.utils import INIconfig, check_path
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
        self.dtype = eval(config.TRAINING['dtype'])
        self.lr = float(config.TRAINING['lr'])
        self.lr_decay_rate = float(config.TRAINING['lr_decay_rate'])
        self.batch_size = int(config.TRAINING['batch_size'])
        self.coco_root = config.DATASET['coco']
        self.NSD_coco_root = config.DATASET['nsd_coco']
        self.image_activation_save_root = config.NSD['image_activation_save_root']
        self.image_same_activation_save_root = config.NSD['image_same_activation_save_root']
        self.from_coco_split = eval(config.NSD['from_coco_split'])
        self.model, self.model_transform = clip.load(config.IMAGE_EMBEDDING['model_name'], device=self.device)
        self.embedding_dim = int(config.IMAGE_EMBEDDING["embedding_dim"])
        self.dataset = NSDDataset(config)
        self.loss_function = nn.MSELoss().to(self.device)
        self.epochs = int(config.TRAINING['epochs'])
        self.linear_save_root = config.NSD['linear_save_root']
        self.calcutale_type = eval(config.TRAINING['calcutale_type'])

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
        self.individual_bool_list, self.same_bool_list = self.dataset.load_individual_and_same_image_bool(subj)

        self.individual_avg_activation = avg_activation[self.individual_bool_list]
        self.same_avg_activation = avg_activation[self.same_bool_list]

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
        self.linear_layer = nn.Linear(self.embedding_dim, self.voxel_num, bias=True).to(self.device)
        self.optimizer = optim.AdamW(self.linear_layer.parameters(), lr=float(self.config.TRAINING['lr']), 
                                     weight_decay=float(self.config.TRAINING['weight_decay']))

    def _change_target_model(self, target_model, embedding_dim):

        self.model = target_model
        self.embedding_dim = embedding_dim

        self.linear_layer = nn.Linear(self.embedding_dim, self.voxel_num)

    def _setup_image_dataloader(self):
        if self.training_image_root_list is None:
            raise ValueError("please use _initialize funtion first to get the training image root list")
        self.training_dataloader = get_root_list_dataloader(batch_size=self.batch_size, image_root_list=self.training_image_root_list, image_transform=self.model_transform)
        self.val_dataloader = get_root_list_dataloader(batch_size=self.batch_size, image_root_list=self.val_image_root_list, image_transform=self.model_transform)
    
    def linear_fitting(self, subj=1, voxel_activation_roi="SELECTIVE_ROI", summary_writer=None):

        self._initialize(subj, voxel_activation_roi)
        self._setup_image_dataloader()

        check_path(self.image_activation_save_root.format(subj))
        check_path(self.image_same_activation_save_root.format(subj))
        
        if not os.path.exists(self.image_activation_save_root.format(subj)):
            save_list = []
            for i, images in tqdm(enumerate(self.training_dataloader), total=len(self.training_dataloader)):
                with torch.no_grad():
                    image_embeddings = self.model.encode_image(images.to(self.device)).to(dtype=self.dtype)
                    image_embeddings = image_embeddings / torch.norm(image_embeddings, keepdim=True)
                save_list.append(image_embeddings.detach().cpu())
                break
            save_list = torch.cat(save_list, dim=0)
            torch.save(save_list, self.image_activation_save_root.format(subj))
            training_data = save_list
        else:
            training_data = torch.load(self.image_activation_save_root.format(subj))


        if not os.path.exists(self.image_same_activation_save_root.format(subj)):
            save_list = []
            for i, images in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader)):
                with torch.no_grad():
                    image_embeddings = self.model.encode_image(images.to(self.device)).to(dtype=self.dtype)
                    image_embeddings = image_embeddings / torch.norm(image_embeddings, keepdim=True)
                save_list.append(image_embeddings.detach().cpu())
                break
            save_list = torch.cat(save_list, dim=0)
            torch.save(save_list, self.image_same_activation_save_root.format(subj))
            valid_data = save_list
        else:
            valid_data = torch.load(self.image_same_activation_save_root.format(subj))

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

                batch_embedding = batch_embedding.to(self.device)
                predict_activation = self.linear_layer(batch_embedding)
                target = self.individual_avg_activation[i * self.batch_size: i * self.batch_size + batch_embedding.shape[0]].to(self.device)
                loss = self.loss_function(predict_activation, target)
                if summary_writer is not None:
                    summary_writer.add_scalar("subj{}_loss".format(subj), loss.detach().cpu(), epoch * len(self.training_dataloader) + i)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                break
            
            # validate the performence of this epoch's trained model
            valid_loss = 0
            for i in range(valid_data.shape[0] // self.batch_size + 1):
                
                try:
                    if i * self.batch_size > valid_data.shape[0]:
                        break
                    batch_embedding = valid_data[i * self.batch_size: (i + 1) * self.batch_size]
                except:
                    batch_embedding = valid_data[i * self.batch_size: ]

                batch_embedding = batch_embedding.to(self.device)

                predict_activation = self.linear_layer(batch_embedding)

                target = self.same_avg_activation[i * self.batch_size: i * self.batch_size + batch_embedding.shape[0]].to(self.device)
                loss = self.loss_function(predict_activation, target).sum()

                valid_loss += loss.detach().cpu()
                break

            print("epoch:", epoch, "    valid_loss:", valid_loss)

            model_save_root = self.linear_save_root.format(subj, voxel_activation_roi)
            check_path(model_save_root)
            torch.save(self.linear_layer.state_dict(), model_save_root)
            raise RuntimeError()
        