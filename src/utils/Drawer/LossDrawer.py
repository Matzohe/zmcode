import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from typing import List, Tuple
import os

# ===============================================================================
# Utils for loss drawing
# ===============================================================================

# process loss data downloading from torchvision
def getTensorBoardData(data_root: str) -> Tuple[List, List]:
    data = pd.read_csv(data_root)
    step = data['Step'].values.tolist()
    loss = data['Value'].values.tolist()
    return step, loss

# get different model's loss and steps
def getMultiLossData(data_root: str) -> Tuple[List, List]:
    if not os.path.isfile(data_root):
        raise ValueError("To use this function, you need to put a file in which there exists a lot of csv file downloading from tensorboard")
    
    path_list = [os.path.join(data_root, each) for each in os.listdir(data_root)]

    step_list = []
    loss_list = []
    for each in path_list:
        step, loss = getTensorBoardData(each)
        step_list.append(step)
        loss_list.append(loss)

    return step_list, loss_list

# plot losses in one figure
def plotLoss(step: List, loss: List, config, save=False):
    figsize = eval(config.DRAWER["figsize"])
    labels = eval(config.DRAWER['loss_name'])
    colors = eval(config.DRAWER['colors'])
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(step[0], List):
        for i in range(len(step)):
            ax.plot(step[i], loss[i], label=labels[i], colors=colors[i])
    else:
        ax.plot(step, loss, label=labels[0], colors=colors[0])

    ax.legend()
    ax.set_xlabel(config.DRAWER['x_label'])
    ax.set_ylabel(config.DRAWER['y_label'])
    if save:
        fig.savefig(config.DRAWER['save_path'], dpi=int(config.DRAWER['dpi']))
    else:
        plt.show()
