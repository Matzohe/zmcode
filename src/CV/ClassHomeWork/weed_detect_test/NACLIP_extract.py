from ...NACLIP import NACLIP
from src.utils.utils import INIconfig
from src.utils.ImagePreporcessUitls import SegmentPreProcess
from .weed_dataloader import weed_dataset
import torch
import cv2
import matplotlib.pyplot as plt

def load_NACLIP_model(config):
    model = NACLIP(config, labels=['background', "soil", "plant roots", "dead plant", 'green weed'])
    return model

def extract_weed_likelyhood(config):
    model = load_NACLIP_model(config)
    training_dataset = weed_dataset(config.DATASET['train_weed_json_root'], config.DATASET['train_weed_image_root'])
    for img_root, label in training_dataset:
        img = SegmentPreProcess(img_root)
        img = img.to(config.INFERENCE['device'])
        with torch.no_grad():
            output = model.predict(img)
        seg_logits = output[0]['seg_logits']

        img = cv2.imread(img_root)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for circle, r in label:
            cv2.circle(img, circle, r, color=(255, 0, 0), thickness=1)
        ax = plt.subplot(121)
        ax.imshow(img)

        ax = plt.subplot(122)
        ax.imshow(seg_logits[4], cmap='grey')
        plt.show()
        break

    