import os
import torch
import pandas as pd
from torchvision import datasets, transforms, models
from typing import Tuple
from ....MultiModal import clip



def get_target_model(model_name: str, device: str) -> Tuple[torch.nn.Module, callable]:
    """
    returns target model in eval mode and its preprocess function
    target_name: supported options - {resnet18_places, resnet18, resnet34, resnet50, resnet101, resnet152}
                 except for resnet18_places this will return a model trained on ImageNet from torchvision
                 
    To Dissect a different model implement its loading and preprocessing function here
    """
    if model_name == 'resnet50':
        target_model = models.resnet50(pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()

    elif model_name == 'clip_resnet50':
        target_model ,preprocess=clip.load("RN50",device=device)
        target_model.eval()

    elif "vit_b" in model_name:
        target_name_cap = model_name.replace("vit_b", "ViT_B")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(model_name))
    
    elif model_name == "clip_ViT-B_32":
        target_model ,preprocess=clip.load("ViT-B/32",device=device)
        target_model.eval()
        
    elif model_name == "clip_ViT-B_16":
        target_model ,preprocess=clip.load("ViT-B/16",device=device)
        target_model.eval()

    else:
        raise ValueError("Target model {} not supported".format(model_name))
    
    target_model.eval()
    return target_model, preprocess

def get_resnet_imagenet_preprocess() -> transforms.Compose:
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

def get_alexnet_imagenet_preprocess() -> transforms.Compose:
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(227),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess
