from src.Projection_utils.UMAP_analysis import UMAPVisualize, UMAPVisualizeSecializedFor4D, UMAPVisualizeSecializedFor5D
from src.utils.utils import INIconfig
from src.utils.Drawer.HTML3DDrawer import HTML3DDrawer
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import torch


color_list = torch.tensor([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],]) / 255.


def Two_Dim_UMAP_tests():
    config = INIconfig()

    weight_root = config.BRAINDIVE['weight_root']
    weight = torch.from_numpy(np.load(weight_root)).T

    roi_root = config.BRAINDIVE['roi_root']
    roi_list = eval(config.BRAINDIVE['roi_list'])

    color_index = torch.zeros(size=[len(weight), 3])
    for i, each in enumerate(roi_list):
        roi = torch.from_numpy(np.loadtxt(roi_root.format(each))).view(-1)
        for j in range(len(weight)):
            if roi[j] > 0:
                color_index[j] = color_list[i]

    low_dimensional_data, _ = UMAPVisualize(weight, show=False)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(low_dimensional_data[:, 0], low_dimensional_data[:, 1], c=color_index, s=0.5)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    plt.show()


def Three_Dim_UMAP_tests():
    config = INIconfig()

    # weight_root = config.BRAINDIVE['weight_root']
    # weight = torch.from_numpy(np.load(weight_root)).T
    weight_root = "/Users/a1/PythonProgram/zmcode/testDataset/BrainSCUBA/adapted_weight/adapted_weight.pt"
    weight = torch.load(weight_root)

    roi_root = config.BRAINDIVE['roi_root']
    roi_list = eval(config.BRAINDIVE['roi_list'])

    color_index = torch.zeros(size=[len(weight), 3])
    for i, each in enumerate(roi_list):
        roi = torch.from_numpy(np.loadtxt(roi_root.format(each))).view(-1)
        for j in range(len(weight)):
            if roi[j] > 0:
                color_index[j] = color_list[i]

    low_dimensional_data, _ = UMAPVisualize(weight, n_components=3, show=False)

    HTML3DDrawer(low_dimensional_data, color_index)
    
def four_Dim_UMAP_tests():
    config = INIconfig()

    weight_root = config.BRAINDIVE['weight_root']
    weight = torch.from_numpy(np.load(weight_root)).T

    roi_root = config.BRAINDIVE['roi_root']
    roi_list = eval(config.BRAINDIVE['roi_list'])

    color_index = torch.zeros(size=[len(weight), 3])
    for i, each in enumerate(roi_list):
        roi = torch.from_numpy(np.loadtxt(roi_root.format(each))).view(-1)
        for j in range(len(weight)):
            if roi[j] > 0:
                color_index[j] = color_list[i]

    low_dimensional_data, _ = UMAPVisualize(weight, n_components=4, show=False)

    UMAPVisualizeSecializedFor4D(low_dimensional_data, color_index)


def five_Dim_UMAP_tests():
    config = INIconfig()

    weight_root = config.BRAINDIVE['weight_root']
    weight = torch.from_numpy(np.load(weight_root)).T

    roi_root = config.BRAINDIVE['roi_root']
    roi_list = eval(config.BRAINDIVE['roi_list'])

    color_index = torch.zeros(size=[len(weight), 3])
    for i, each in enumerate(roi_list):
        roi = torch.from_numpy(np.loadtxt(roi_root.format(each))).view(-1)
        for j in range(len(weight)):
            if roi[j] > 0:
                color_index[j] = color_list[i]

    low_dimensional_data, _ = UMAPVisualize(weight, n_components=5, show=False)

    UMAPVisualizeSecializedFor5D(low_dimensional_data, color_index)
