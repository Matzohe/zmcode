import torch
import torch.nn as nn

from ..utils.ResNetBottleNeck import Resnet34Bottleneck


class ResNet34(nn.Module):
    def __init__(self, config):
        self.layer = eval(config.RESNET34['layers'])
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = int(config.RESNET34['inplanes'])
        self.planes = eval(config.RESNET34['planes'])
        self.layer1 = self._make_layer(self.planes[0], self.layer[0])
        self.layer2 = self._make_layer(self.planes[1], self.layer[1])
        self.layer3 = self._make_layer(self.planes[2], self.layer[2])
        self.layer4 = self._make_layer(self.planes[3], self.layer[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, 1000)

    def _make_layer(self, planes, num_bottleneck):
        layers = []
        layers.append(Resnet34Bottleneck(self.inplanes, planes))
        for _ in range(1, num_bottleneck):
            layers.append(Resnet34Bottleneck(planes, planes))
        self.inplanes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)  
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
