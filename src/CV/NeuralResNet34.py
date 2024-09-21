import torch
import torch.nn as nn

from ..utils.MyIdea.NeuralResidual import NeuralResnet34BottleNeck


class NeuralResNet34(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = eval(config.RESNET34['layers'])
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 224 / 4 = 56
        self.inplanes = int(config.RESNET34['inplanes'])
        self.planes = eval(config.RESNET34['planes'])
        self.gama = float(config.MYRES['gama'])
        self.image_size = eval(config.MYRES['image_size'])
        self.layer1 = self._make_layer(self.planes[0], self.layer[0], self.image_size[0], stride=1)  # 56
        self.layer2 = self._make_layer(self.planes[1], self.layer[1], self.image_size[1], stride=2)  # 56 / 2 = 28
        self.layer3 = self._make_layer(self.planes[2], self.layer[2], self.image_size[2], stride=2)  # 28 / 2 = 14
        self.layer4 = self._make_layer(self.planes[3], self.layer[3], self.image_size[3], stride=2)  # 14 / 2 = 7
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, 1000)

    def _make_layer(self, planes, num_bottleneck, image_size, stride=1):
        layers = []
        layers.append(NeuralResnet34BottleNeck(self.inplanes, planes, image_size, gama=self.gama, stride=stride))
        for _ in range(1, num_bottleneck):
            layers.append(NeuralResnet34BottleNeck(planes, planes, image_size, gama=self.gama))
        self.inplanes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1((out, x))
        out = self.layer2((out[0], x))
        out = self.layer3((out[0], x))
        out = self.layer4((out[0], x))

        out = self.avgpool(out[0]).view(out[0].shape[0], -1)
        out = self.fc(out)

        return out
