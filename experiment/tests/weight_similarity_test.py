import torch
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.utils import INIconfig
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def weight_similarity_tests():
    config = INIconfig()
    
    roi_root = config.BRAINDIVE['roi_root']
    roi_list = eval(config.BRAINDIVE['roi_list'])

    weight_root = config.BRAINDIVE['weight_root']
    weight = torch.from_numpy(np.load(weight_root)).T
    # weight = torch.load("testDataset/BrainSCUBA/adapted_weight/adapted_weight.pt")

    weight = weight / torch.norm(weight, dim=1, keepdim=True)

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


def pearson_similarity_tests():
    config = INIconfig()
    
    roi_root = config.BRAINDIVE['roi_root']
    roi_list = eval(config.BRAINDIVE['roi_list'])

    weight_root = config.BRAINDIVE['weight_root']
    weight = torch.from_numpy(np.load(weight_root)).T
    # weight = torch.load("testDataset/BrainSCUBA/adapted_weight/adapted_weight.pt")

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

    metrix_1 = ranged_weight - torch.mean(ranged_weight, dim=1, keepdim=True)
    metrix_2 = ranged_weight - torch.mean(ranged_weight, dim=1, keepdim=True)

    similarity_matrix = torch.matmul(metrix_1, metrix_2.T)

    metrix_x = torch.sqrt(torch.sum(metrix_1 * metrix_1, dim=1)).view(-1, 1)
    metrix_y = torch.sqrt(torch.sum(metrix_2 * metrix_2, dim=1)).view(1, -1)
    
    similarity_matrix = similarity_matrix / (metrix_x @ metrix_y) - torch.eye(ranged_weight.shape[0]).numpy()

    plt.figure()
    plt.imshow(similarity_matrix, cmap="inferno", vmin=-1, vmax=1)
    plt.xticks(label_index_place, labels=roi_list, rotation=45)
    plt.yticks(label_index_place, labels=roi_list)
    plt.colorbar()
    plt.show()

