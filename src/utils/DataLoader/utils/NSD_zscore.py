import numpy as np
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