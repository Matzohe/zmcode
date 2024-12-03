import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchextractor as tx
from typing import List
from tqdm import tqdm
import numpy as np


# extract target layer's output with the help of torchextractor
def extract_target_layer_output(model: torch.nn.Module, model_name: str, 
                                target_layer: List, dataloader: DataLoader, 
                                device: str, dtype: str) -> List:

    model_visual = tx.Extractor(model, target_layer)
    compressed_features = [[] for _ in range(len(target_layer))]

    if "clip" in model_name:
        for _, images in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad():
                text_embedding = torch.randint(size=(images.shape[0], 77), low=0, high=1000).to(device=device)
                _, image_features = model_visual(images.to(device=device), text_embedding)
            
            for k, f in enumerate(image_features.values()):
                f = f / torch.norm(f, dim=-1, keepdim=True)
                if len(f.size()) > 3:
                    tmp = nn.functional.adaptive_avg_pool2d(f.data, (f.shape[-2], f.shape[-1]))
                    compressed_features[k].append(tmp.view(images.shape[0], -1).detach())
                else:
                    if "ViT" in model_name:
                        compressed_features[k].append(f.data[0, :, :].view(images.shape[0], -1).detach())
                    else:
                        compressed_features[k].append(f.data.view(images.shape[0], -1).detach())
                
    else:
        for _, images in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad():
                _, image_features = model_visual(images.to(device=device)).to(dtype=dtype)

            for k, f in enumerate(image_features.values()):
                f = f / torch.norm(f, dim=-1, keepdim=True)
                if len(f.size()) > 3:
                    tmp = nn.functional.adaptive_avg_pool2d(f.data, (f.shape[-2], f.shape[-1]))
                    compressed_features[k].append(tmp.squeeze().cpu().view(f.shape[0], -1))
                else:
                    if "ViT" in model_name:
                        compressed_features[k].append(f.data[0, :, :].view(images.shape[0], -1))
                        raise RuntimeError()
                    else:
                        compressed_features[k].append(f.data.view(f.data.shape[0], -1))

    return compressed_features


# extract the last layer image embeddings
def extract_image_embedding(model: torch.nn.Module, dataloader: DataLoader, 
                            device: str, dtype: str) -> torch.Tensor:
    if "clip" in model._get_name():
          save_list = []
          for _, images in tqdm(enumerate(dataloader), total=len(dataloader)):
              with torch.no_grad():
                  image_embeddings = model.encode_image(images.to(device)).to(dtype=dtype)
                  image_embeddings = image_embeddings / torch.norm(image_embeddings, dim=-1, keepdim=True)
              save_list.append(image_embeddings.detach().cpu())
          save_list = torch.cat(save_list, dim=0)
          return save_list

    else:
         save_list = []
         for _, images in tqdm(enumerate(dataloader), total=len(dataloader)):
             with torch.no_grad():
                 image_embeddings = model(images.to(device=device)).to(dtype=dtype)
                 image_embeddings = image_embeddings / torch.norm(image_embeddings, dim=-1, keepdim=True)
             save_list.append(image_embeddings.detach().cpu())
         save_list = torch.cat(save_list, dim=0)
         return save_list
