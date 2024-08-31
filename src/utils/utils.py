import torch
import torch.nn as nn
from dataclasses import dataclass, make_dataclass, field
from typing import List, Dict
import configparser
import numpy as np
import torch
import cv2
from torchvision import transforms
from PIL import Image

# ==================================================
# Config Related Functions
# ==================================================
@dataclass
class INICfg:
    # An example of INI config dataclass
    CfgUsage = "Read Local Config.cfg file and save all the info in this dataclass"
    sections: List = field(default_factory=list) # List of section names

    # each section has a dict, which contains all the information in the section.
    # The Property's name is the same as the section's name in INI .cfg file
    section_name: Dict = field(default_factory=dict)  # {"keys": "values in section"}  


# Use to Process INI stile config file
def INIconfig(config_path: str) -> dataclass:
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)

    # get all the sections
    sections = config_parser.sections()
    # put the information in a =info list, then make a dataclass by make_dataclass
    info = [
        ("sections", List, field(default_factory=list)),
    ]

    for each in sections:
        info.append((each, Dict[str, str], field(default_factory=dict)))

    Config = make_dataclass("Config", info)
    return Config(sections=sections, **{section: {key: value for key, value in config_parser[section].items()} for section in sections})


# ==================================================
# Useful nn.Module Class for building the model
# ==================================================
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# ==================================================
# Useful functions for building the model
# ==================================================


# ==================================================
# Functions for Image Segmentation
# ==================================================


# ==================================================
# Functions for Image Preprocessing
# ==================================================


# TODO: Transform the Image package to OpenCv, which is faster
def ClipPreProcess(image_path: str) -> torch.Tensor:
    # CLIP image preprocess function
    # The hyperparameters in the Normalize is the average and standard deviation of ImageNet

    image = Image.open(image_path)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    return preprocess(image).unsqueeze(0)