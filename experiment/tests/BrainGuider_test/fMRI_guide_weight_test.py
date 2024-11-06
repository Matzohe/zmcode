import torch
from sklearn.metrics.pairwise import cosine_similarity
from src.Recognize.BrainGuider.utils.voxel_representation import voxel_representation
from src.utils.utils import INIconfig
import numpy as np
import matplotlib.pyplot as plt
import os

def fMRI_guide_weight_test(config_root="config/brainGuide_config.cfg"):

    subj = 1

    config = INIconfig(config_path=config_root)
    roi_root = config.ROI['roi_root']
    roi_list = eval(config.ROI['roi_list'])

    weight_root = config.WEIGHT['weight_root']
    weight = torch.from_numpy(np.load(weight_root)).T
    # weight = torch.load("testDataset/BrainSCUBA/adapted_weight/adapted_weight.pt")
    weight = weight / torch.norm(weight, dim=1, keepdim=True)

    image_root_list = config.NSD['image_root_list'].format(subj)
    coco_root = config.DATASET['coco']
    image_root_list = [os.path.join(coco_root, each) for each in image_root_list]

    save_name_list = eval(config.BRAIN['save_name_list'])
    voxel_activation = torch.load(config.NSD['zscore_avg_activation_save_root'].format(subj, save_name_list[0]))

    weight = voxel_representation(weight, voxel_activation)[0]

    ranged_weight = []
    label_index_place = []

    for i, each in enumerate(roi_list):
        roi_index = torch.from_numpy(np.loadtxt(roi_root.format(each))).view(-1)
        roi_index = roi_index > 0
        label_index_place.append(roi_index.sum())
        if i > 0:
            label_index_place[i] += label_index_place[i-1]

        roi_weight = weight[roi_index]
        ranged_weight.append(roi_weight)
    
    ranged_weight = torch.cat(ranged_weight, dim=0)
    
    similarity_matrix = cosine_similarity(ranged_weight, ranged_weight) - torch.eye(ranged_weight.shape[0]).numpy()

    plt.figure()
    plt.imshow(similarity_matrix, cmap="inferno", vmin=-1, vmax=1)
    plt.xticks(label_index_place, labels=roi_list, rotation=45)
    plt.yticks(label_index_place, labels=roi_list)
    plt.colorbar()
    plt.show()