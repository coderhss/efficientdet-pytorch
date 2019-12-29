# -*- coding:utf-8 -*-
# Author: huashuoshuo
# Data: 2019/12/17 14:36

import torch
import torch.nn as nn
import torch.functional as F
from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)

class ConvBlock(nn.Module):
    """

    """
    def __init__(self, inp, oup, k_size, stride=1, padding=0, group=1):
        super().__init__()
        # Conv2d = get_same_padding_conv2d
        self.conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=k_size, stride=stride, padding=padding, bias=False, groups=group).cuda()
        self.norm = nn.BatchNorm2d(num_features=oup).cuda()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.norm(self.conv(x))
        # print(self.conv)
        x = self.conv(x)
        return self.act(x)


class BiFPN(nn.Module):
    """

    """
    def __init__(self,oup, first=True):
        super().__init__()
        # self.features_in = features_in
        self.oup = oup
        # self.dw_conv = ConvBlock(oup, oup, k_size=3, stride=1, padding=1, group=oup)
        # self.pw_conv = ConvBlock(oup, oup, k_size=1, stride=1, padding=0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.first = first
        self.conv_gen()
        self.w_gen()
    def forward(self, features_in):
        # self.tail(x)
        # P3_in, P4_in, P5_in, P6_in, P7_in = features_in

        features_out = self.top_down(features_in)
        return features_out

    def conv_gen(self):
        # P3_in, P4_in, P5_in, P6_in, P7_in = features_in
        if not self.first:
            self.P3_in_conv = ConvBlock(self.oup, self.oup, k_size=1, stride=1, padding=0)
            self.P4_in_conv = ConvBlock(self.oup, self.oup, k_size=1, stride=1, padding=0)
            self.P5_in_conv = ConvBlock(self.oup, self.oup, k_size=1, stride=1, padding=0)
            self.P6_in_conv = ConvBlock(self.oup, self.oup, k_size=1, stride=1, padding=0)
            self.P7_in_conv = ConvBlock(self.oup, self.oup, k_size=1, stride=1, padding=0)

        # upsample
        self.P6_td_conv = ConvBlock(self.oup, self.oup, k_size=3, stride=1, padding=1, group=self.oup)
        self.P5_td_conv = ConvBlock(self.oup, self.oup, k_size=3, stride=1, padding=1, group=self.oup)
        self.P4_td_conv = ConvBlock(self.oup, self.oup, k_size=3, stride=1, padding=1, group=self.oup)
        self.P3_out_conv = ConvBlock(self.oup, self.oup, k_size=3, stride=1, padding=1, group=self.oup)

        # downsample
        self.P4_out_conv = ConvBlock(self.oup, self.oup, k_size=3, stride=1, padding=1, group=self.oup)
        self.P5_out_conv = ConvBlock(self.oup, self.oup, k_size=3, stride=1, padding=1, group=self.oup)
        self.P6_out_conv = ConvBlock(self.oup, self.oup, k_size=3, stride=1, padding=1, group=self.oup)
        self.P7_out_conv = ConvBlock(self.oup, self.oup, k_size=3, stride=1, padding=1, group=self.oup)

    def w_gen(self):
        self.P6_td_add = wAdd(2)
        self.P5_td_add = wAdd(2)
        self.P4_td_add = wAdd(2)
        self.P3_out_add = wAdd(2)
        self.P4_out_add = wAdd(3)
        self.P5_out_add = wAdd(3)
        self.P6_out_add = wAdd(3)
        self.P7_out_add = wAdd(2)

    def top_down_no_w(self, features_in):
        P3_in, P4_in, P5_in, P6_in, P7_in = features_in
        if not self.first:
            P3_in = self.P3_in_conv(P3_in)
            P4_in = self.P4_in_conv(P4_in)
            P5_in = self.P5_in_conv(P5_in)
            P6_in = self.P6_in_conv(P6_in)
            P7_in = self.P7_in_conv(P7_in)

        # upsample
        P7_U = self.Resize()(P7_in)
        P6_td = P7_U + P6_in
        P6_td = self.P6_td_conv(P6_td)
        P6_U = self.Resize()(P6_td)
        P5_td = P6_U + P5_in
        P5_td = self.P5_td_conv(P5_td)
        P5_U = self.Resize()(P5_td)
        P4_td = P5_U + P4_in
        P4_td = self.P4_td_conv(P4_td)
        P4_U = self.Resize()(P4_td)
        P3_out = P4_U + P3_in
        P3_out = self.P3_out_conv(P3_out)

        # downsample
        P3_D = self.pool(P3_out)
        P4_out = P3_D + P4_td + P4_in
        P4_out = self.P4_out_conv(P4_out)
        P4_D = self.pool(P4_out)
        P5_out = P4_D + P5_td + P5_in
        P5_out = self.P5_out_conv(P5_out)
        P5_D = self.pool(P5_out)
        P6_out = P5_D + P6_td + P6_in
        P6_out = self.P6_out_conv(P6_out)
        P6_D = self.pool(P6_out)
        P7_out = P6_D + P7_in
        P7_out = self.P7_out_conv(P7_out)
        return [P3_out, P4_out, P5_out, P6_out, P7_out]

    def top_down(self, features_in):
        P3_in, P4_in, P5_in, P6_in, P7_in = features_in
        if not self.first:
            P3_in = self.P3_in_conv(P3_in)
            P4_in = self.P4_in_conv(P4_in)
            P5_in = self.P5_in_conv(P5_in)
            P6_in = self.P6_in_conv(P6_in)
            P7_in = self.P7_in_conv(P7_in)

        # upsample
        P7_U = self.Resize()(P7_in)
        P6_td = self.P6_td_add([P6_in, P7_U])
        P6_td = self.P6_td_conv(P6_td)
        P6_U = self.Resize()(P6_td)
        P5_td = self.P5_td_add([P5_in, P6_U])
        P5_td = self.P5_td_conv(P5_td)
        P5_U = self.Resize()(P5_td)
        P4_td = self.P4_td_add([P4_in, P5_U])
        P4_td = self.P4_td_conv(P4_td)
        P4_U = self.Resize()(P4_td)
        P3_out = self.P3_out_add([P3_in, P4_U])
        P3_out = self.P3_out_conv(P3_out)

        # downsample
        P3_D = self.pool(P3_out)
        P4_out = self.P4_out_add([P3_D, P4_td, P4_in])
        P4_out = self.P4_out_conv(P4_out)
        P4_D = self.pool(P4_out)
        P5_out = self.P5_out_add([P4_D, P5_td, P5_in])
        P5_out = self.P5_out_conv(P5_out)
        P5_D = self.pool(P5_out)
        P6_out = self.P6_out_add([P5_D, P6_td, P6_in])
        P6_out = self.P6_out_conv(P6_out)
        P6_D = self.pool(P6_out)
        P7_out = self.P7_out_add([P6_D, P7_in])
        P7_out = self.P7_out_conv(P7_out)

        return [P3_out, P4_out, P5_out, P6_out, P7_out]



    def Resize(self, scale=2, mode='nearest'):
        upsample = nn.Upsample(scale_factor=scale, mode=mode)
        return upsample

    # def get_weight(self):


class wAdd(nn.Module):
    """

    """
    def __init__(self, num_in):
        super().__init__()
        self.epsilon = 1e-4
        self.w = nn.Parameter(torch.Tensor(num_in).fill_(1 / num_in))

    def forward(self, inputs):
        # len(inputs)
        num_in = len(inputs)
        # w = nn.Parameter(torch.Tensor(num_in).fill_(1 / num_in))
        w = self.w.cuda()
        # x = [w[i] * inputs[i] for i in range(num_in)]
        x = 0
        # print(w[0])
        for i in range(num_in):
            x += w[i] * inputs[i]
        x /= (torch.sum(w) + self.epsilon)
        # x = x.cuda()
        return x
        # x = torch.sum(x)



