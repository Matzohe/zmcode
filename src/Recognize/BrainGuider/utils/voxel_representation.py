import torch
import numpy as np


def voxel_representation(voxel_weight, voxel_activation):
    """_summary_

    Args:
        voxel_weight (torch.tensor): the ridge regression or linear weight of clip2brain
        voxel_activation (torch.tensor): the fMRI activation of a sertain person

    Returns:
        adapted_weight: the weight addapted by the fMRI activation
    """
    assert voxel_activation.shape[-1] == voxel_weight.shape[0]

    voxel_activation = voxel_activation.unsqueeze(-1)

    voxel_weight = voxel_weight.unsqueeze(0)

    return voxel_activation * voxel_weight