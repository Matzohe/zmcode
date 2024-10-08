import numpy as np
import torch
from tqdm import tqdm

# Specialize for NSD Datasets
def zscore_by_run(mat, run_n=480):
    from scipy.stats import zscore

    run_n = np.ceil(
        mat.shape[0] / 62.5
    )  # should be 480 for subject with full experiment\

    zscored_mat = np.zeros(mat.shape)
    index_so_far = 0
    for i in tqdm(range(int(run_n)), desc="NSD dataset zscore processing..."):
        if i % 2 == 0:
            zscored_mat[index_so_far : index_so_far + 62, :] = zscore(
                mat[index_so_far : index_so_far + 62, :]
            )
            index_so_far += 62
        else:
            zscored_mat[index_so_far : index_so_far + 63, :] = zscore(
                mat[index_so_far : index_so_far + 63, :]
            )
            index_so_far += 63

    return zscored_mat


def ev(data, biascorr=True):
    """
    Computes the amount of variance in a voxel's response that can be explained by the
    mean response of that voxel over multiple repetitions of the same stimulus.

    If [biascorr], the explainable variance is corrected for bias, and will have mean zero
    for random datasets.

    Data is assumed to be a 2D matrix: time x repeats.
    """
    ev = 1 - torch.var(data.T - torch.nanmean(data, axis=1)) / torch.var(data)
    if biascorr:
        return ev - ((1 - ev) / (data.shape[1] - 1.0))
    else:
        return ev