from torch_cka import CKA
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from src.utils.CKA import CKA_Function
from src.utils.Drawer.CKADrawer import CKA_Image_Saver
import numpy as np
from src.utils.utils import INIconfig


def cka_test():
    # load test models
    config = INIconfig('config.cfg')
    model1 = models.resnet18(pretrained=True)
    model2 = models.resnet34(pretrained=True)

    # prepare for the dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFAR10(root='testDataset/cifar', train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    output = CKA_Function(model1, model2, dataloader, "ResNet18", "ResNet34")
    CKA_Image_Saver(output, config)