import torch
import torch.nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class NSDImageDataset(Dataset):
    def __init__(self, image_root_list, image_transform):
        super().__init__()
        self.image_root_list = image_root_list
        self.image_transform = image_transform

    def __len__(self):
        return len(self.image_root_list)
    
    def __getitem__(self, index):
        image_root = self.image_root_list[index]
        img = Image.open(image_root)

        return self.image_transform(img)


def get_NSD_image_dataloader(batch_size, image_root_list, image_transform, pin_memory=True, num_workers=0):
    dataset = NSDImageDataset(image_root_list, image_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
    return dataloader
