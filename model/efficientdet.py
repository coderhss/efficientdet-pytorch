# -*- coding:utf-8 -*-
# Author: huashuoshuo
# Data: 2019/12/17 10:53


import torch
import numpy as np
import torch.nn as nn
from .BiFPN import BiFPN
# from .RetinaHead import RetinaHead

# class ConvBlock(nn.Module):
#     def __init__(self):
#         super().__init__()
class ConvBlock(nn.Module):
    """

    """
    def __init__(self, inp, oup, k_size, stride=1, padding=0):
        super().__init__()
        # Conv2d = get_same_padding_conv2d
        self.conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=k_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(num_features=oup)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.norm(self.conv(x))
        return self.act(x)

from model import EfficientNet
# from .RetinaHead import RetinaHead
class EfficientDet(nn.Module):
    """

    """
    def __init__(self, args):
        super().__init__()

        self.inp = 64
        self.oup = 64
        self.bifpn_repeat = 2
        print(args.backbone)
        self.backbone = EfficientNet.from_pretrained(args)
        # self.backbone.get_list_features()
        self.tail = nn.ModuleList([ConvBlock(320, self.oup, 3, 2, 1), ConvBlock(self.oup, self.oup, 3, 2, 1)])
        self.channel_same = self.change_channel(self.backbone.get_list_feature()[-3:])
        self.BiFPN_first = BiFPN(oup=self.oup, first=True)
        self.BiFPN = nn.ModuleList()
        for i in range(self.bifpn_repeat-1):
            self.BiFPN.append(BiFPN(oup=self.oup, first=False))

    def forward(self, inputs):
        features_in = self.extra(inputs)
        features_out = self.BiFPN_first(features_in)
        for i, bifpn in enumerate(self.BiFPN):
            features_out = bifpn(features_out)
        return features_out


    def extra(self, img):
        x = self.backbone(img)[-3:]
        # before_fpn = self.channel_same(x[-5:])
        # print(x[-1].shape)
        # print(self.tail)
        # tail = [tail_conv(x[-1]) for i, tail_conv in enumerate(self.tail)]
        for i, tail_conv in enumerate(self.tail):
            x.append(tail_conv(x[-1]))


        before_fpn = [
            conv(x[i])
            for i, conv in enumerate(self.channel_same)]

        before_fpn.extend(x[-2:])

        return before_fpn

    def change_channel(self, channel):
        convs = nn.ModuleList()
        for i in range(len(channel)):
            conv = ConvBlock(channel[i], self.oup, k_size=1, stride=1, padding=0)
            convs.append(conv)
        return convs