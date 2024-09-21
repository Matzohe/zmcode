import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class NeuralResnet34BottleNeck(nn.Module):
    def __init__(self, inplanes, planes, image_size, gama=0.5, stride=1):
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
                ('conv', nn.Conv2d(3, planes, 1, stride=1, bias=False)),
                ('bn', nn.BatchNorm2d(planes)),
            ])
        )
        self.gama = gama
        self.image_size = image_size
        
    def forward(self, input_info):
        x = input_info[0]
        input_image = input_info[1]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        identity = self.downsample(x)
        out = out + self.gama * self.ImageProcess(F.interpolate(input_image, self.image_size))
        out = out + identity
        out = self.relu(out)
        return (out, input_image)
