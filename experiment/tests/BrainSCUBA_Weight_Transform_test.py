from src.Recognize.BrainSCUBA.weight_attention import WeightTransform
from src.utils.utils import INIconfig 
from src.Projection_utils.UMAP_analysis import UMAPVisualize
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils.DataLoader.fake_news import FakeNewsDataset


def BrainSCUBA_Weight_Transform_test():
    config = INIconfig()
    color_list = torch.tensor([[0, 255, 255], [255, 255, 0], [255, 0, 255]], dtype=torch.float32) / 255.
    image_embedding_root = config.BRAINSCUBA['image_embedding_root']
    text_embedding_root = config.BRAINSCUBA['text_embedding_root']
    voxel_weight_root = config.BRAINSCUBA['voxel_weight_root']
    
    image_embeddings = torch.from_numpy(np.load(image_embedding_root)).to(dtype=torch.float32).squeeze(1)
    text_embeddings = torch.from_numpy(np.load(text_embedding_root)).to(dtype=torch.float32)
    voxel_weight = torch.from_numpy(np.load(voxel_weight_root)).to(dtype=torch.float32).T
    
    adapted_weight = WeightTransform(image_embeddings, voxel_weight)
    UMAP_inform = torch.concat((image_embeddings, adapted_weight, voxel_weight), dim=0)
    print(UMAP_inform.shape)
    UMAP_inform = UMAP_inform / torch.norm(UMAP_inform, dim=1, p=2).view(-1, 1)
    low_dimensional_data, _ = UMAPVisualize(UMAP_inform, show=False)

    color_index = torch.zeros(size=(low_dimensional_data.shape[0], 3))
    color_index[: image_embeddings.shape[0]] = color_list[0]
    color_index[image_embeddings.shape[0]: image_embeddings.shape[0] + adapted_weight.shape[0]] = color_list[1]
    color_index[image_embeddings.shape[0] + adapted_weight.shape[0]:] = color_list[2]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(low_dimensional_data[:, 0], low_dimensional_data[:, 1], c=color_index, s=0.5)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    plt.show()
    
    torch.save(adapted_weight, 'adapted_weight.pt')
