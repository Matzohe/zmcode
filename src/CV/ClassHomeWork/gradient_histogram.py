import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_gradient_histogram(block, bins=9):
    # 计算梯度大小和方向
    gradient_x = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)  # x方向的梯度
    gradient_y = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)  # y方向的梯度

    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)  # 梯度大小
    angle = np.arctan2(gradient_y, gradient_x) * 180 / np.pi  # 梯度方向（角度）

    # 将角度范围调整为0~180度
    angle = np.mod(angle, 180)

    # 使用梯度大小作为权重，计算直方图
    hist, bin_edges = np.histogram(angle, bins=bins, range=(0, 180), weights=magnitude)

    return hist, bin_edges

def display_results(image, histograms, bin_edges):
    # 显示原图
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # 显示四个block的梯度方向直方图
    for i, (hist, _) in enumerate(histograms):
        plt.subplot(3, 2, i+2)
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')
        plt.title(f'Block {i+1} Gradient Histogram')
        plt.xlabel('Gradient Direction (Degrees)')
        plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.show()

def main(image_path, bins=9):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to load image.")
        return

    # 将图像分为2*2的块
    height, width = image.shape
    block_height = height // 2
    block_width = width // 2

    histograms = []
    bin_edges = None

    # 计算每个block的梯度方向直方图
    for i in range(2):
        for j in range(2):
            block = image[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width]
            hist, bin_edges = calculate_gradient_histogram(block, bins)
            histograms.append((hist, bin_edges))

    # 显示原图和梯度直方图
    display_results(image, histograms, bin_edges)

# 运行主程序，输入图像路径
image_path = "/Users/a1/Documents/Image/CG_Death_2.png"
main(image_path, bins=180)  # 可以调整bins的数量
