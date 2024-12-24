import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np


def result_viewer(path, image_path="/Users/a1/PythonProgram/zmcode/testDataset/weeddetection/train/train/images"):
    seg, labels = torch.load(path)
    image_path = os.path.join(image_path, path.split("/")[-1])
    image = cv2.imread(image_path.replace("_classification.pt", ".png"), cv2.COLOR_BGR2RGB)
    image = (image * 255).astype(np.uint8)
    seg = (seg == 4).to(dtype=torch.uint8).cpu().squeeze(0).numpy()
    
    




if __name__ == "__main__":
    result_viewer("/Users/a1/PythonProgram/zmcode/experiment/output/weed_detect/pred_sem_seg/0_classification.pt")