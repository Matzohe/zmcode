from src.CV.NACLIP import NACLIP
from src.utils.utils import INIconfig
from src.utils.ImagePreporcessUitls import SegmentPreProcess
from PIL import Image
import numpy as np

def NACLIP_test():
    # self.data_preprocessor = SegDataPreProcessor(mean=[122.771, 116.746, 104.094], std=[68.501, 66.632, 70.323], rgb_to_bgr=True)
    config = INIconfig()
    root = "/Users/a1/Downloads/fig.png"
    img = SegmentPreProcess(root)

    model = NACLIP(config)
    output = model.forward_slide(img)
    print(output)