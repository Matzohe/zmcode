import torch
from PIL import Image
from torchvision import transforms


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