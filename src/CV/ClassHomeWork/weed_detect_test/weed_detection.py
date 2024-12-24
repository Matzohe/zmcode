import torch
import torch.nn as nn
import cv2
import os
import torchvision
from torchvision import transforms
import csv
from patch_detection import get_patch_info



class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(35*35*3, 1600),
            nn.ReLU(),
            nn.Linear(1600, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        output = self.model(x)
        return torch.softmax(output, dim=-1)


model = torch.load('model.pt')
seg_info_root_list = [os.path.join("experiment/output/weed_detect/test_pred_sem_seg", each) for each in os.listdir("experiment/output/weed_detect/test_pred_sem_seg")]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((35, 35))
])
all_data = []

for root in seg_info_root_list:
    img_list, rectangle_list = get_patch_info(root)
    img_name = root.split('/')[-1].replace("_classification.pt", "")

    for i, each in enumerate(img_list):
        each = transform(each)
        classes = model(each.view(1, -1))
        new_data = [int(img_name), classes.argmax(dim=-1).view(-1).item(), rectangle_list[i][0][0], rectangle_list[i][0][1], rectangle_list[i][0][2], rectangle_list[i][0][3]]
        all_data.append(new_data)

all_data = sorted(all_data, key=lambda x: x[0])
for i in range(len(all_data)):
    all_data[i] = [i + 1] + all_data[i]
with open("submission.csv", 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID","image_id","class_id","x_min","y_min","width","height"])
    writer.writerows(all_data)
    for i in range(5000 - len(all_data) - 1):
        writer.writerow([i + len(all_data) + 1, 99999, 9, 0, 0, 0, 0])
id = -1
img = None
for each in all_data:
    if each[1] != id:
        if img is not None:
            cv2.imwrite("experiment/output/weed_detect/test_output_img/rectangled_{}.png".format(str(id)), img)
        img_root = "testDataset/weeddetection/test/test/images"
        img_root = os.path.join(img_root, str(each[1]) + ".png")
        img = cv2.imread(img_root)
        cv2.rectangle(img, (int(each[3]), int(each[4])), (int(each[3] + each[5]), int(each[4] + each[6])), (0, 0, 255), 2)
        id = each[1]
    else:
        cv2.rectangle(img, (int(each[3]), int(each[4])), (int(each[3] + each[5]), int(each[4] + each[6])), (0, 0, 255), 2)
    
        
