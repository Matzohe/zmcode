import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class NewResnet34BottleNeck(nn.Module):
    def __init__(self, inplanes, planes, gama=0.5, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride > 1 or inplanes != planes:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes))
            ]))
        else:
            self.downsample = nn.Identity()

        self.ImageProcess = nn.Sequential(
            OrderedDict([
                ('resize', nn.AvgPool2d(stride) if stride > 1 else nn.Identity()),
                ('conv', nn.Conv2d(3, planes, 3, stride=1, padding=1, bias=False)),
                ('bn', nn.BatchNorm2d(planes)),
            ])
        )
        self.gama = gama
        
    def forward(self, x, input_image):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        identity = self.downsample(x)
        out += self.gama * self.ImageProcess(input_image)
        out += identity
        out = self.relu(out)
        return out
