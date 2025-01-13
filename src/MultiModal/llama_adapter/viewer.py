import cv2
import matplotlib.pyplot as plt
from PIL import Image
img = Image.open("/Volumes/T7/zmcode/testDataset/Levir-CC-dataset/images/test/A/test_000105.png")
plt.imshow(img)
# 添加标题
plt.title("some villas are built along the road . generated: a road is built on the bare land .")
plt.show()