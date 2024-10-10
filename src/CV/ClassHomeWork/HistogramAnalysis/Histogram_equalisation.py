import cv2
import matplotlib.pyplot as plt

def equalize_hist_yuv(image):
    # 转换为 YUV 颜色空间
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # 对 Y 通道进行直方图均衡化
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
    
    # 将 YUV 图像转换回 RGB
    equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    
    return equalized_image

def compute_rgb_histogram(image):
    # 分离 R, G, B 通道
    channels = cv2.split(image)
    colors = ('b', 'g', 'r')  # OpenCV 中通道的顺序是 BGR

    histograms = {}
    for i, color in enumerate(colors):
        # 计算直方图
        hist = cv2.calcHist([channels[i]], [0], None, [256], [0, 256])
        histograms[color] = hist

    return histograms


# 加载图像
image = cv2.imread('/Users/a1/Documents/Image/yande.re 1053218 bikini bondage flandre_scarlet gekidoku_shoujo ke-ta loli panty_pull swimsuits touhou wings.jpg')

# 对 Y 通道进行直方图均衡化
equalized_image = equalize_hist_yuv(image)

# 显示原图和均衡化后的图像的直方图
origin_hist = compute_rgb_histogram(image)
equalized_hist = compute_rgb_histogram(equalized_image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for color in origin_hist.keys():
    plt.plot(origin_hist[color], color=color)
    plt.xlim([0, 256])
plt.title('Original Histogram')

plt.subplot(1, 2, 2)
for color in equalized_hist.keys():
    plt.plot(equalized_hist[color], color=color)
    plt.xlim([0, 256])
plt.title('Equalized Histogram')

plt.show()

plt.close()


# 显示原图和均衡化后的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))
plt.title('Equalized Image (YUV)')

plt.show()
