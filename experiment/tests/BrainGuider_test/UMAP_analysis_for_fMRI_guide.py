import torch
from sklearn.metrics.pairwise import cosine_similarity
from src.Recognize.BrainGuider.utils.voxel_representation import voxel_representation
from src.utils.utils import INIconfig, check_path
from src.utils.Drawer.HTML3DDrawer import HTML3DDrawer
from src.Projection_utils.UMAP_analysis import UMAPVisualize, UMAPVisualizeSecializedFor4D, UMAPVisualizeSecializedFor5D
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image

color_list = torch.tensor([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],]) / 255.
# FFA: 红，EBA：绿，RSC：蓝，VWFA：黄，FOOD：粉


def UMAP_analysis_for_fMRI_guide_test(config_root="config/brainGuide_config.cfg"):

    subj = 1

    config = INIconfig(config_path=config_root)

    weight_root = config.WEIGHT['weight_root']
    weight = torch.from_numpy(np.load(weight_root)).T
    # weight = torch.load("testDataset/BrainSCUBA/adapted_weight/adapted_weight.pt")

    weight = weight / torch.norm(weight, dim=1, keepdim=True)

    image_root_list = torch.load(config.NSD['image_root_save_root'].format(subj))
    coco_root = config.DATASET['coco']
    image_root_list = [os.path.join(coco_root, each) for each in image_root_list]

    save_name_list = eval(config.BRAIN['save_name_list'])
    voxel_activation = torch.load(config.NSD['zscore_avg_activation_save_root'].format(subj, save_name_list[0]))
    voxel_activation = voxel_activation / torch.norm(voxel_activation, dim=1, keepdim=True)
    all_weight = weight * voxel_activation[2048].view(-1, 1)
    image_embedding = torch.from_numpy(np.load(config.WEIGHT['image_embedding_root'])).squeeze(1)
    print(voxel_activation[2048])
    # all_weight = voxel_representation(weight, voxel_activation)

    weight = torch.cat([all_weight, image_embedding], dim=0)
    color_index = torch.zeros(size=[len(weight), 3])
    color_index[: len(all_weight)] = color_list[0]
    color_index[len(all_weight): ] = color_list[1]
    low_dimensional_data, _ = UMAPVisualize(weight, n_components=3, show=False)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(low_dimensional_data[:, 0], low_dimensional_data[:, 1], c=color_index, s=0.5)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    plt.show()
