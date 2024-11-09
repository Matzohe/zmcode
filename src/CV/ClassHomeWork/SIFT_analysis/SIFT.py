import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

# 读取一幅图像，将图像进行变化之后作为第二张图像
img1 = cv2.imread('/Users/a1/Documents/Image/yande.re 910898 dress eve_(rurudo) nopan possible_duplicate rurudo rurudot see_through skirt_lift summer_dress wet wings.png', cv2.IMREAD_GRAYSCALE)  # 查询图像
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=70)
])
img2 = img_transform(img1).squeeze(0) * 255
img2 = img2.to(torch.uint8).numpy()
# 创建SIFT检测器
sift = cv2.SIFT_create()

# 检测关键点和描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

img1_with_kp = cv2.drawKeypoints(img1, kp1, None)
img2_with_kp = cv2.drawKeypoints(img2, kp2, None)

# 使用KNN进行特征匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(121)
# ax.imshow(img1_with_kp, cmap='gray')
# ax.axis('off')
# ax = fig.add_subplot(122)
# ax.imshow(img2_with_kp, cmap='gray')
# ax.axis('off')
ax = fig.add_subplot(111)
ax.imshow(matched_img)
ax.axis('off')
ax.set_title("SIFT Feature Matching")
plt.show()


