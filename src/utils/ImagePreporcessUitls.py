import torch
from PIL import Image, ImageOps, ImageFilter
import random
from torchvision import transforms


# ==================================================
# Functions for Image Preprocessing
# ==================================================


# TODO: Transform the Image package to OpenCv, which is faster
def ClipPreProcess(image_path: str) -> torch.Tensor:
    # CLIP image preprocess function
    # The hyperparameters in the Normalize is the average and standard deviation of ImageNet

    image = Image.open(image_path)
    if image.mode == "L":
        image = image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    img = preprocess(image).unsqueeze(0)
    img = img[:, [2, 1, 0], :, :]
    return img

def SegmentPreProcess(image_path: str) -> torch.Tensor:
    # Segment image preprocess function
    # The hyperparameters in the Normalize comming from NACLIP:https://github.com/sinahmr/NACLIP.git

    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((122.771 / 255., 116.746 / 255., 104.094 / 255.), (68.501 / 255., 66.632 / 255., 70.323 / 255.))
    ])

    return preprocess(image).unsqueeze(0)

def ImageNetPreProcess(image_path: str) -> torch.Tensor:
    # ImageNet image preprocess function
    # The hyperparameters in the Normalize is the average and standard deviation of ImageNet

    # TODO: bug here, need to check linux's web
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return preprocess(image).unsqueeze(0)


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img




class BarlowTwinsTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2
    

# TODO: Need to fill the Transform Code
class BrainSAILTransform:
    def __init__(self, config):

        self.config = config
        self.transform_number = int(config.BRAINSAIL['transform_number'])
        self.y_threashould = float(config.BRAINSAIL['y_threashould'])
        self.x_threashould = float(config.BRAINSAIL['x_threashould'])
        self.transform_list = [
            transforms.Compose([transforms.Lambda(lambda crop: transforms.functional.affine(crop, 
                                                                                            angle=0, 
                                                                                            translate=(int(crop.shape[-1] * x_rate), int(crop.shape[-2] * y_rate)), 
                                                                                            scale=1, 
                                                                                            shear=0))]) 
                                                                                            for x_rate, y_rate in zip(self.x_rate_list, self.y_rate_list)
        ]
        self.anti_transform_list = []

    # print(new_image.shape)
    # # 定义平移变换，假设你想水平和垂直方向上都平移10个像素
    # transform = transforms.Compose([
    #     transforms.RandomAffine(degrees=0, translate=(0.2, 0)),  # 10%的平移
    # ])

    # # 应用变换
    # transformed_image = transform(image)

    # # 显示原图和变换后的图像
    # image.show()
    # transformed_image.show()