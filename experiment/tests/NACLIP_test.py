from src.CV.NACLIP import NACLIP
from src.utils.utils import INIconfig
from src.utils.ImagePreporcessUitls import SegmentPreProcess
from src.utils.Drawer.NACLIPDrawer import NACLIP_Drawer
from PIL import Image
import torch
import numpy as np

def NACLIP_test():
    # self.data_preprocessor = SegDataPreProcessor(mean=[122.771, 116.746, 104.094], std=[68.501, 66.632, 70.323], rgb_to_bgr=True)
    config = INIconfig()
    root = "/data/guoyuan/ImageNet/ILSVRC2012_img_train/n01440764/n01440764_18.JPEG"
    img = SegmentPreProcess(root)
    device = config.INFERENCE['device']
    img = img.to(device)
    with torch.no_grad():
        model = NACLIP(config)
        output = model.predict(img)
    NACLIP_Drawer(output)

