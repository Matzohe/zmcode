import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm


def compute_normalized_correlation(image1, image2):
    image1 = image1 - torch.mean(image1, dim=0, keepdim=True)
    image2 = image2 - torch.mean(image2, dim=0, keepdim=True)
    return torch.sum(image1 * image2) / (torch.sqrt(torch.sum(image1 * image1)) * torch.sqrt(torch.sum(image2 * image2)))

image = cv2.imread('/Users/a1/Documents/Image/yande.re 1053218 bikini bondage flandre_scarlet gekidoku_shoujo ke-ta loli panty_pull swimsuits touhou wings.jpg')
image = torch.from_numpy(image) / 255.
image = image.permute(2, 0, 1)
image = F.interpolate(image.unsqueeze(0), size=(512, 512), mode='bilinear')[0]
sub_image = image[:, 256: 328, 256: 328]


output_image_list = np.zeros(shape=(image.shape[1] - sub_image.shape[1] + 1, image.shape[2] - sub_image.shape[2] + 1))

for i in tqdm(range(image.shape[1] - sub_image.shape[1] + 1), total=image.shape[1] - sub_image.shape[1] + 1):
    for j in range(image.shape[2] - sub_image.shape[2] + 1):
        sub_image_part = image[:, i: i + sub_image.shape[1], j: j + sub_image.shape[2]]

        output_image_list[i, j] = compute_normalized_correlation(sub_image, sub_image_part)

image = image.permute(1, 2, 0)
sub_image = sub_image.permute(1, 2, 0)
# 打印比较结果

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor((image * 255.).to(dtype=torch.uint8).numpy(), cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor((sub_image * 255.).to(dtype=torch.uint8).numpy(), cv2.COLOR_BGR2RGB))
plt.title('Small Image')

plt.subplot(1, 3, 3)
plt.imshow(output_image_list)
plt.title('Similarity Map')

plt.show()
