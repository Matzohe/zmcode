import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import torchvision
import cv2
import torchvision.transforms as transforms
import json
import math
from tqdm import tqdm


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

def extraction():
    json_root = "testDataset/weeddetection/train/train/labels"
    json_root_list = [os.path.join(json_root, each) for each in os.listdir(json_root)]
    final_image_list = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((35, 35))
    ])
    for each in json_root_list:
        with open(each, 'r') as f:
            data = json.load(f)
            img_name = data['imagePath']
            if '/' in img_name:
                img_name = img_name.split('/')[-1]
            image_path = os.path.join("testDataset/weeddetection/train/train/images", img_name)
            image = cv2.imread(image_path)
            for info in data['shapes']:
                classes = info['label']
                if classes == 'mq':
                    classes = 1
                else:
                    classes = 0
                points = info['points']
                r = int(math.dist(points[0], points[1]) + 0.5)
                x_min = max(0, min(int(points[0][0] + 0.5) - r, image.shape[1] - 1))
                x_max = max(0, min(int(points[0][0] + 0.5) + r, image.shape[1] - 1))
                y_min = max(0, min(int(points[0][1] + 0.5) - r, image.shape[0] - 1))
                y_max = max(0, min(int(points[0][1] + 0.5) + r, image.shape[0] - 1))
                area_image = image[y_min: y_max, x_min: x_max, :]
                final_image_list.append([transform(area_image), classes])

    torch.save(final_image_list, "training_info.pt")

class newDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index][0].view(-1), self.data[index][1]

    def __len__(self):
        return len(self.data)


def training():
    data = torch.load("training_info.pt")
    dataset = newDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = classifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    epochs = 10
    for i in tqdm(range(epochs)):
        for img, label in dataloader:
            output = model(img)
            label = torch.tensor(label, dtype=torch.long).view(-1)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model, "model.pt")
    

def evaluate():
    model = torch.load("model.pt")
    data = torch.load("training_info.pt")
    dataset = newDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    correct = 0
    total = 0
    zero_sum = 0
    one_sum = 0
    with torch.no_grad():
        for img, label in dataloader:
            output = model(img)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            zero_sum += (predicted == 0).sum().item()
            one_sum += (predicted == 1).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
    print("zero_sum: ", zero_sum, "one_sum: ", one_sum)

if __name__ == '__main__':
    training()
    evaluate()


