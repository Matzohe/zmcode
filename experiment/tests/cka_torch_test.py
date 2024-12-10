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


def cka_test():
    # load test models
    config = INIconfig('config.cfg')

    # model1 = models.resnet50(pretrained=True)
    # model2 = models.resnet50(pretrained=True)
    model1, preprocess = clip.load("ViT-B/32",device="cpu")
    model1 = model1.eval().to(device="mps")
    model2 = clip.load("ViT-B/16",device="mps")[0].eval().to(device="cpu")
    model_1_layer = []
    model_2_layer = []
    for i in range(12):
        model_1_layer.append("visual.transformer.resblocks.{}.attn".format(i))
        model_1_layer.append("visual.transformer.resblocks.{}.mlp.c_fc".format(i))
        model_1_layer.append("visual.transformer.resblocks.{}.mlp.c_proj".format(i))
        model_2_layer.append("visual.transformer.resblocks.{}.attn".format(i))
        model_2_layer.append("visual.transformer.resblocks.{}.mlp.c_fc".format(i))
        model_2_layer.append("visual.transformer.resblocks.{}.mlp.c_proj".format(i))
    # prepare for the dataloader
    dataloader = get_ClipDataloader(config.DATASET["nsd_coco"], preprocess, int(config.DATALOADER["batch_size"]), device="cpu")
    
    output = CKA_Function(model1, model2, dataloader, "clip-ViT-B_32", "clip-ViT-B_32", device="cpu")
    CKA_Image_Saver(output, config)