from ...NACLIP import NACLIP
from ....utils.utils import INIconfig, check_path
from ....utils.ImagePreporcessUitls import SegmentPreProcess
from .weed_dataloader import weed_dataset
import torch
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def load_NACLIP_model(config):
    model = NACLIP(config, labels=['background', "soil", "plant roots", "dead plant", 'green weed'])
    return model

def extract_weed_likelyhood(config):
    model = load_NACLIP_model(config)
    weed_pred_test_sem_seg_save_root = config.DATASET['weed_pred_test_sem_seg_save_root']
    check_path(weed_pred_test_sem_seg_save_root + "/")
    training_dataset = weed_dataset(config.DATASET['test_weed_json_root'], config.DATASET['test_weed_image_root'])
    for img_root, label in tqdm(training_dataset, total=len(training_dataset)):
        img = SegmentPreProcess(img_root)
        img = img.to(config.INFERENCE['device'])
        with torch.no_grad():
            output = model.predict(img)
        seg_logits = output[0]['pred_sem_seg']

        save_info = [seg_logits, label]
        save_name = img_root.split('/')[-1]
        save_name = save_name.replace('.png', '_classification.pt')
        torch.save(save_info, os.path.join(weed_pred_test_sem_seg_save_root, save_name))

