# -*- coding: utf-8 -*-
# @Time : 20-6-3 下午5:14
# @Author : zhuying
# @Company : Minivision
# @File : MultiFTNet.py
# @Software : PyCharm

import torch
from torch import nn
from models.MobileFaceNetPruned import MobileFaceNetPv4


class FTGenerator(nn.Module):

    def __init__(self, in_channels=48, out_channels=1):
        super(FTGenerator, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_channels, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MultiFTNet(nn.Module):

    def __init__(self, img_channel=3, num_classes=3, embedding_size=128, conv6_kernel=(5, 5)):
        super(MultiFTNet, self).__init__()
        self.img_channel = img_channel
        self.num_classes = num_classes
        self.model = MobileFaceNetPv4(embedding_size=embedding_size, conv6_kernel=conv6_kernel, num_classes=num_classes, img_channel=img_channel)
        self.FTGenerator = FTGenerator(in_channels=128)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2_dw(x)
        x = self.model.conv_23(x)
        x = self.model.conv_3(x)
        x = self.model.conv_34(x)
        x = self.model.conv_4(x)
        x1 = self.model.conv_45(x)
        x1 = self.model.conv_5(x1)
        x1 = self.model.conv_6_sep(x1)
        x1 = self.model.conv_6_dw(x1)
        x1 = self.model.conv_6_flatten(x1)
        x1 = self.model.linear(x1)
        x1 = self.model.bn(x1)
        x1 = self.model.drop(x1)
        cls = self.model.prob(x1)

        if self.training:
            dep = self.FTGenerator(x)
            return cls, dep
        else:
            return cls


if __name__ == '__main__':

    net = MultiFTNet()
    print(net)

    input = torch.randn((2, 3, 80, 80))

    cls, dep = net(input)

    input = torch.randn((1, 3, 80, 80))
