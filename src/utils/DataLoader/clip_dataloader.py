import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image
import os

class ClipDataset(Dataset):
    def __init__(self, image_root, image_transform, device):
        self.device = device
        self.image_root = [os.path.join(image_root, each) for each in os.listdir(image_root)]
        self.image_transform = image_transform

    def __len__(self):
        return len(self.image_root)

    def __getitem__(self, index):
        image_root = self.image_root[index]
        image = Image.open(image_root)
        image = self.image_transform(image).to(device=self.device)


        return image
    
def get_ClipDataloader(image_root, image_transform, batch_size, device="mps", num_workers=4):
    dataset = ClipDataset(image_root, image_transform, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader