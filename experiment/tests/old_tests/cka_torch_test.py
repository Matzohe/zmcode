from torch_cka import CKA
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from src.utils.CKA import CKA_Function
from src.utils.Drawer.CKADrawer import CKA_Image_Saver
import numpy as np
from src.utils.utils import INIconfig
from src.MultiModal import clip
from src.utils.DataLoader.clip_dataloader import get_ClipDataloader 
from tqdm import tqdm
import torchextractor as tx
import torch


def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()


def cka_test():
    # load test models
    config = INIconfig('config.cfg')
    device = "cuda:2"
    # model1 = models.resnet50(pretrained=True)
    # model2 = models.resnet50(pretrained=True)
    model1, preprocess = clip.load("ViT-B/32",device=device)
    model1 = model1.eval().to(device=device)
    model2 = clip.load("ViT-B/16",device=device)[0].eval().to(device=device)

    model_1_layer = []
    model_2_layer = []
    for i in range(12):
        model_1_layer.append("visual.transformer.resblocks.{}.attn".format(i))
        model_1_layer.append("visual.transformer.resblocks.{}.mlp.c_fc".format(i))
        model_1_layer.append("visual.transformer.resblocks.{}.mlp.c_proj".format(i))
        model_2_layer.append("visual.transformer.resblocks.{}.attn".format(i))
        model_2_layer.append("visual.transformer.resblocks.{}.mlp.c_fc".format(i))
        model_2_layer.append("visual.transformer.resblocks.{}.mlp.c_proj".format(i))
    model_visual = tx.Extractor(model1, model_1_layer)
    # prepare for the dataloader
    dataloader = get_ClipDataloader(config.DATASET["nsd_coco"], preprocess, int(config.DATALOADER["batch_size"]), device=device)
    data_matrix = torch.zeros(size=())
    
    model_1_feature = {}
    model_2_feature = {}
    

    for _, data in tqdm(dataloader, total=len(dataloader)):
        with torch.no_grad():
            _, model_features = model_visual(data.to(device=device))
            for k, f in enumerate(model_features.values()):
                model_1_feature[model_1_layer[k]] = f
                model_2_feature[model_1_layer[k]] = f

        for k, f in enumerate(model_features.values()):



    CKA_Image_Saver(output, config)