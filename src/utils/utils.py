import torch
import torch.nn as nn
from dataclasses import dataclass, make_dataclass, field
from typing import List, Dict
import configparser
import numpy as np
import torch
import cv2
import random
from torchvision import transforms
from PIL import Image
import os

# ==================================================
# utils to check several things
# ==================================================

def check_path(path):
    path = "/".join(path.split("/")[ :-1])
    if not os.path.exists(path):
        os.makedirs(path)

def check_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        try:
            data = torch.tensor(data)
            return data
        except ValueError:
            print("Current data type not support to transform to torch.Tensor")


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


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ==================================================
# Functions for Image Segmentation
# ==================================================
