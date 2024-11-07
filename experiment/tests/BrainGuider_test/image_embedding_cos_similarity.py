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


def image_embedding_cos_similarity_test(config_root="config/brainGuide_config.cfg"):

    subj = 1

    config = INIconfig(config_path=config_root)

    image_embedding = torch.from_numpy(np.load("testDataset/BrainSCUBA/embeddings_visualization/clip_visual_resnet.npy")).squeeze(1)

    similarity_matrix = cosine_similarity(image_embedding, image_embedding)

    plt.imshow(similarity_matrix, cmap='inferno')
    plt.colorbar()
    plt.show()