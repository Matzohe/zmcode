import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(title, image):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

# 1. 载入图像，进行傅里叶变换，显示频谱图像
image = cv2.imread('/Users/a1/Documents/Image/CG_Death_2.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Could not open or find the image!")
    exit()

# 执行傅里叶变换
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)  # 将零频率分量移动到频谱中心

# 计算频谱图像
magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
magnitude_spectrum = np.log(magnitude_spectrum + 1)  # 取对数以增强可视化

show_image("Magnitude Spectrum", magnitude_spectrum)

# 2. 去除频谱图像中的高频部分，进行反变换
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# 创建掩模，去除高频部分
mask = np.zeros((rows, cols, 2), np.uint8)
r = 30  # 低频区域的半径
cv2.circle(mask, (ccol, crow), r, (1, 1), thickness=-1)

# 应用掩模
filtered_dft = dft_shift * mask

# 反变换
filtered_idft = cv2.idft(np.fft.ifftshift(filtered_dft))
filtered_image = cv2.magnitude(filtered_idft[:, :, 0], filtered_idft[:, :, 1])

show_image("Filtered Image (High Frequencies Removed)", filtered_image)

# 3. 去除频谱图像中的低频部分，进行反变换
# 创建掩模，去除低频部分
mask_high = np.ones((rows, cols, 2), np.uint8)
cv2.circle(mask_high, (ccol, crow), r, (0, 0), thickness=-1)

# 应用掩模
filtered_dft_high = dft_shift * mask_high

# 反变换
filtered_idft_high = cv2.idft(np.fft.ifftshift(filtered_dft_high))
filtered_image_high = cv2.magnitude(filtered_idft_high[:, :, 0], filtered_idft_high[:, :, 1])

show_image("Filtered Image (Low Frequencies Removed)", filtered_image_high)
