import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
from src.utils.utils import INIconfig
from src.utils.PostHocUtils.CFA_Analysis import cfaAnalysis
from src.utils.DataLoader.cifar import data_loader

def cfa_Test():
    config = INIconfig("config.cfg")
    model_A = models.resnet18(pretrained=True)
    model_B = models.resnet34(pretrained=True)
    _, val_dataloader = data_loader(config)
    cfa = cfaAnalysis(model_A, model_B, model_A_layers=["conv1", "layer1", "layer2", "layer3", "layer4"], model_B_layers=["conv1", "layer1", "layer2", "layer3", "layer4"], kernel="linear")
    output = cfa.forward(val_dataloader)
    output = torch.cat(output, dim=0).permute(1, 2, 0).numpy()
    im = Image.fromarray(output)
    im.show()