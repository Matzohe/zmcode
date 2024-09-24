import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
from src.utils.utils import INIconfig
from src.utils.PostHocUtils.CKA_Analysis import cfaAnalysis
from src.utils.DataLoader.cifar import data_loader

def cfa_Test():
    config = INIconfig("config.cfg")
    model_A = models.resnet18(pretrained=True).eval().to(device=config.INFERENCE['device'])
    model_B = models.resnet34(pretrained=True).eval().to(device=config.INFERENCE['device'])
    _, val_dataloader = data_loader(config)
    print(config.CFA)
    cfa = cfaAnalysis(model_A, model_B, config)
    cfa(val_dataloader)