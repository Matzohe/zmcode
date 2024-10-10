import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_normalized_correlation(image1, image2):
    image1 = image1 - torch.mean(image1)
    image2 = image2 - torch.mean(image2)
    return torch.sum(image1 * image2) / (torch.sqrt(torch.sum(image1 * image1)) * torch.sqrt(torch.sum(image2 * image2)))

image = cv2.imread('/Users/a1/Documents/Image/yande.re 1053218 bikini bondage flandre_scarlet gekidoku_shoujo ke-ta loli panty_pull swimsuits touhou wings.jpg')
sub_image = image[400: 600, 700: 1000, :]


output_image_list = np.zeros(shape=(image.shape[0] - sub_image.shape[0] + 1, image.shape[1] - sub_image.shape[1] + 1))

for i in range(image.shape[0] - sub_image.shape[0] + 1):
    for j in range(image.shape[1] - sub_image.shape[1] + 1):
        sub_image_part = image[i: i + sub_image.shape[0], j: j + sub_image.shape[1], :]

        output_image_list[i, j] = compute_normalized_correlation(sub_image, sub_image_part)


# 打印比较结果

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(sub_image, cv2.COLOR_BGR2RGB))
plt.title('Gaussian Filtered Image')

plt.subplot(1, 3, 3)
plt.imshow(output_image_list)
plt.title('Mean Filtered Image')

plt.show()
