# 高斯核滤波，以及均值核滤波

import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_filter(image):
    # 高斯核的大小
    kernel_size = 3
    # 高斯核的标准差
    sigma = 1.0
    # 创建高斯核
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    # 计算高斯卷积
    convolved_image = cv2.filter2D(image, -1, kernel)
    return convolved_image

def mean_filter(image):
    # 均值核的大小
    kernel_size = 3
    # 创建均值核
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    # 计算均值卷积
    convolved_image = cv2.filter2D(image, -1, kernel)
    return convolved_image

# 加载图像
image = cv2.imread('/Users/a1/Documents/Image/yande.re 1053218 bikini bondage flandre_scarlet gekidoku_shoujo ke-ta loli panty_pull swimsuits touhou wings.jpg')

# 高斯滤波
filtered_image = gaussian_filter(image)

# 均值滤波
mean_filtered_image = mean_filter(image)

# 结果展示
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Gaussian Filtered Image')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(mean_filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Mean Filtered Image')

plt.show()
