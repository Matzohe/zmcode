from ...utils.DataLoader.NSDDataLoader import NSDDataset
from ...utils.DataLoader.RootListDataLoader import get_root_list_dataloader
from ...utils.utils import INIconfig, check_path
from ...utils.r2_score import r2_score
from ..clip2brain.utils.load_target_model import get_target_model
from ..clip2brain.utils.extract_target_layer_function import decoding_extract_target_layer_output, extract_image_embedding
from ...MultiModal import clip as clip
from .neighbor_extractor import seed_point_search
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class VisualPathAnalysis:
    def __init__(self, config):
        self.config = config
        self.device = config.TRAINING['device']
        self.middle_layer_linear_save_root = config.NSD['middle_layer_linear_save_root']
        # whether get the data from coco raw dataset, or get the image from the NSD image file
        self.from_coco_split = eval(config.NSD['from_coco_split'])
        self.coco_root = config.DATASET['coco']
        self.NSD_coco_root = config.DATASET['nsd_coco']
        self.dtype = eval(config.TRAINING['dtype'])
        self.batch_size = int(config.TRAINING['batch_size'])

        self.similarity_save_root = config.BRAIN2CLIP['similarity_save_root']
        self.is_eight_neighbors = eval(config.BRAIN2CLIP['is_eight_neighbors'])

        self.middle_activation_save_root = config.NSD['middle_activation_save_root']
        self.middle_same_activation_save_root = config.NSD['middle_same_activation_save_root']
        self.model_name = config.IMAGE_EMBEDDING['model_name'].replace("/", "-")
        self.dataset = NSDDataset(config)
        self.calcutale_type = eval(config.TRAINING['calcutale_type'])
        
        self.target_layer = None
        self.target_roi = None
        self.analyse_subj = None
        self.target_weight = None
        self.val_model_embedding = None
        self.roi_mask = None

        self.val_image_root_list = None
        self.val_avg_activation = None
        self.voxel_num = None
        self.index2coodinate = OrderedDict()
        self.coodinate2index = OrderedDict()

        self.individual_bool_list = None
        self.same_bool_list = None
    
    def _initialize(self, *, subj=1, voxel_activation_roi="ventral_visual_pathway_roi", target_layer):
        # extract the target layer's trained linear weight
        try:
            model_state_dict = torch.load(self.middle_layer_linear_save_root.format(subj, self.model_name, target_layer, voxel_activation_roi), map_location=self.device)
            self.target_weight = model_state_dict["weight"]
        except:
            raise ValueError("No such file or directory: '{}'".format(self.middle_layer_linear_save_root.format(subj, self.model_name, target_layer, voxel_activation_roi)))
        # extract the target layer's embeddings
        try:
            self.val_model_embedding = torch.load(self.middle_same_activation_save_root.format(subj, self.model_name, target_layer, voxel_activation_roi), map_location=self.device)
        except:
            raise ValueError("No such file or directory: '{}'".format(self.middle_same_activation_save_root.format(subj, self.model_name, target_layer, voxel_activation_roi)))

        self.target_layer = target_layer
        self.analyse_subj = subj
        self.analyse_roi = voxel_activation_roi
        # extract the fMRI data and each data's coodinate
        avg_activation = self.dataset.load_avg_activation_value(subj, voxel_activation_roi)
        voxel_coodinate = self.dataset.load_mask_coordinate(subj, voxel_activation_roi)
        self.roi_mask = self.dataset.load_unflattened_mask(subj, voxel_activation_roi)

        self.index2coodinate = OrderedDict()
        self.coodinate2index = OrderedDict()
        for i in range(len(voxel_coodinate)):
            self.index2coodinate[i] = (voxel_coodinate[i][0], voxel_coodinate[i][1], voxel_coodinate[i][2])
            self.coodinate2index[(voxel_coodinate[i][0], voxel_coodinate[i][1], voxel_coodinate[i][2])] = i
            
        self.individual_bool_list, self.same_bool_list = self.dataset.load_individual_and_same_image_bool(subj)
        # we only need the valid data here
        self.val_avg_activation = avg_activation[self.same_bool_list].to(self.device)

        image_root_list = self.dataset.load_image_root(subj)
        if self.from_coco_split:
            image_root_list = [os.path.join(self.coco_root, each) for each in image_root_list]
        else:
            image_root_list = [os.path.join(self.NSD_coco_root, each) for each in image_root_list]

        self.val_image_root_list = []
        for i in range(len(image_root_list)):
            if not self.individual_bool_list[i]:
                self.val_image_root_list.append(image_root_list[i])
        
        self.voxel_num = avg_activation.shape[-1]

    def load_target_layer(self, target_layer):
        self.target_layer = target_layer
        self.linear_weight = torch.load(self.middle_layer_linear_save_root.format(self.analyse_subj, self.model_name, self.target_layer, self.analyse_roi)).to(device=self.device)
    
    def extract_target_voxels(self, *, subj=1, voxel_activation_roi="ventral_visual_pathway_roi", target_layer, target_image_id=0):
        self._initialize(subj=subj, voxel_activation_roi=voxel_activation_roi, target_layer=target_layer)

        self.val_model_embedding = torch.cat(self.val_model_embedding, dim=0)
        
        if torch.isnan(self.val_avg_activation).any():
            nan_embedding_list = (torch.isnan(self.val_avg_activation).sum(dim=-1) == 0)
            self.val_model_embedding = self.val_model_embedding[nan_embedding_list]
            self.val_avg_activation = self.val_avg_activation[nan_embedding_list]

        self.val_avg_activation = self.val_avg_activation / torch.norm(self.val_avg_activation, dim=-1, keepdim=True)
        target_image_model_embedding = self.val_model_embedding[target_image_id]
        target_fMRI_signal = self.val_avg_activation[target_image_id].view(-1)
        target_fMRI_activation = (target_fMRI_signal * self.target_weight).T
        voxel_selected_count = torch.zeros_like(target_fMRI_signal).to(device=self.device)
        for i in tqdm(range(len(target_fMRI_signal)), total=len(target_fMRI_signal)):
            seed_point = self.index2coodinate[i]
            seed_point_count = seed_point_search(
                mask=self.roi_mask,
                seed_point=seed_point,
                target_model_embedding=target_image_model_embedding,
                fMRI_activation=target_fMRI_activation,
                coodinate2index=self.coodinate2index,
                is_eight_neighbors=self.is_eight_neighbors,
                )
            voxel_selected_count += seed_point_count.to(device=self.device)
        return voxel_selected_count

    def extract_similarity(self, *, subj=1, voxel_activation_roi="ventral_visual_pathway_roi", target_layer):
        self._initialize(subj=subj, voxel_activation_roi=voxel_activation_roi, target_layer=target_layer)
        # the fMRI activation extracted: self.val_avg_activation
        # the model embedding extracted: self.val_model_embedding
        # the target trained linear weight: self.target_weight
        
        self.val_model_embedding = torch.cat(self.val_model_embedding, dim=0)
        # check if there is any nan value in the fMRI data
        if torch.isnan(self.val_avg_activation).any():
            nan_embedding_list = (torch.isnan(self.val_avg_activation).sum(dim=-1) == 0)
            self.val_model_embedding = self.val_model_embedding[nan_embedding_list]
            self.val_avg_activation = self.val_avg_activation[nan_embedding_list]
        self.val_momdel_embedding = self.val_model_embedding / torch.norm(self.val_model_embedding, dim=-1, keepdim=True)
        self.val_avg_activation = self.val_avg_activation / torch.norm(self.val_avg_activation, dim=-1, keepdim=True)
        similarity_list = torch.zeros(size=(self.val_avg_activation.shape[0], self.val_avg_activation.shape[1]))
        for i in tqdm(range(self.val_avg_activation.shape[-1]), total=self.val_avg_activation.shape[-1]):
            compute_weight = self.target_weight[:, i].view(1, -1)
            contributions = self.val_avg_activation[:, i].view(-1, 1) @ compute_weight
            similarity_list[:, i] = (self.val_model_embedding * contributions).sum(dim=-1)
        
        save_path = self.similarity_save_root.format(subj, self.model_name, target_layer, voxel_activation_roi)
        check_path(save_path)
        torch.save(similarity_list, save_path)
