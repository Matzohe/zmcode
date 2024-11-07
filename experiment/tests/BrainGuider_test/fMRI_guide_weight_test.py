import torch
import torch.nn.functional as F
from src.utils.cosine_similarity import cosine_similarity
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


def fMRI_guide_weight_test(config_root="config/brainGuide_config.cfg"):

    subj = 1

    config = INIconfig(config_path=config_root)
    roi_root = config.ROI['roi_root']
    roi_list = eval(config.ROI['roi_list'])

    weight_root = config.WEIGHT['weight_root']
    weight = torch.from_numpy(np.load(weight_root)).T
    # weight = torch.load("testDataset/BrainSCUBA/adapted_weight/adapted_weight.pt")

    # weight = weight / torch.norm(weight, dim=1, keepdim=True)

    image_root_list = torch.load(config.NSD['image_root_save_root'].format(subj))
    coco_root = config.DATASET['coco']
    image_root_list = [os.path.join(coco_root, each) for each in image_root_list]

    save_name_list = eval(config.BRAIN['save_name_list'])
    voxel_activation = torch.load(config.NSD['zscore_avg_activation_save_root'].format(subj, save_name_list[0]))
    voxel_activation = voxel_activation / torch.norm(voxel_activation, dim=1, keepdim=True)
    voxel_activation = voxel_activation.to(dtype=torch.float32)
    all_weight = weight * voxel_activation[1025].view(-1, 1)
    print(voxel_activation[1025])
    # all_weight = voxel_representation(weight, voxel_activation)
    for img_idx in tqdm(range(1025, 1026), total=1):
        weight = all_weight
        ranged_weight = []
        label_index_place = []
        color_index = torch.zeros(size=[len(weight), 3])
        for i, each in enumerate(roi_list):
            roi_index = torch.from_numpy(np.loadtxt(roi_root.format(subj, each))).view(-1)
            roi_index = roi_index > 0
            label_index_place.append(roi_index.sum())
            color_index[roi_index] = color_list[i]
            if i > 0:
                label_index_place[i] += label_index_place[i-1]

            roi_weight = weight[roi_index]
            ranged_weight.append(roi_weight)
        
        ranged_weight = torch.cat(ranged_weight, dim=0).to(dtype=torch.float32)

        low_dimensional_data, _ = UMAPVisualize(weight, n_components=3, show=False)
        check_path("experiment/output/BrainGuider/weight_html/fig_{}_interactive_3d_plot.html".format(img_idx))
        HTML3DDrawer(low_dimensional_data, color_index, save_path="experiment/output/BrainGuider/weight_html/fig_{}_interactive_3d_plot.html".format(img_idx))

        similarity_matrix = cosine_similarity(ranged_weight, ranged_weight)

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(121)
        cax = ax.imshow(similarity_matrix, cmap="inferno")
        ax.set_xticks(label_index_place)
        ax.set_xticklabels(roi_list)
        ax.tick_params(axis='x', rotation=45)
        ax.set_yticks(label_index_place)
        ax.set_yticklabels(roi_list)
        plt.colorbar(cax, ax=ax)

        ax = fig.add_subplot(122)
        img = Image.open(image_root_list[img_idx])
        ax.imshow(img)
        ax.axis('off')
        check_path("experiment/output/BrainGuider/weight_visualization/fig_{}_cos_similarity.png".format(img_idx))
        plt.savefig("experiment/output/BrainGuider/weight_visualization/fig_{}_cos_similarity.png".format(img_idx), dpi=1000)
        plt.close()