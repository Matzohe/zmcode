import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = "mps"

class CNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.batch_norm = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batch_norm_2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)
        self.gelu = nn.GELU()
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(self.gelu(self.batch_norm(self.conv1(x))))
        x = self.pool(self.gelu(self.batch_norm_2(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4)
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def main():
    model = CNN_model().to(device=device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    # 导入FashionMNIST数据集
    train_dataset = FashionMNIST(root='/Users/a1/PythonProgram/zmcode/testDataset/mnist', train=True, download=True, transform=transform)
    test_dataset = FashionMNIST(root='/Users/a1/PythonProgram/zmcode/testDataset/mnist', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels_mask = labels > 4
            labels[labels_mask] = 255
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                labels_mask = labels > 4
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels_mask.sum().item()
                correct += (predicted == labels).sum().item()
            print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    torch.save(model.state_dict(), 'cnn_model.pt')

def eval():
    model = CNN_model().to(device=device)
    model.load_state_dict(torch.load('cnn_model.pt'))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    test_dataset = FashionMNIST(root='/Users/a1/PythonProgram/zmcode/testDataset/mnist', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels_mask = labels > 4
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_mask.sum().item()
            correct += (predicted == labels).sum().item()
        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    

if __name__ == '__main__':
    eval()