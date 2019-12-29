# -*- coding:utf-8 -*-
# Author: huashuoshuo
# Data: 2019/12/18 16:37

import torch
import torch.nn as nn
from model.BiFPN import ConvBlock
import model.losses as losses
from model.efficientdet import EfficientDet
from pycocotools.coco import COCO as COCO
from model.anchors import Anchors
# from lib.nms.pth_nms import pth_nms
import torchvision.ops as ops
from model.util import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes, Filter_boxes
def nms(bbox, score, thresh):
    # bbox, score = dets
    return ops.nms(boxes=bbox, scores=score, iou_threshold=thresh)
    # return pth_nms(dets, thresh)


class Reg(nn.Module):
    """

    """
    def __init__(self, inp, oup, depth, num_anchor):
        super().__init__()
        self.inp = inp
        self.oup = oup
        self.D = depth
        self.reg = nn.ModuleList()
        self.num_anchors = num_anchor

        for i in range(self.D):
            self.reg.append(ConvBlock(inp=self.inp, oup=self.oup, k_size=3, stride=1, padding=1))
        # self.retina_cls = nn.Conv2d(self.oup, self.num_anchors * self.num_class, 3, padding=1)
        self.retina_reg = nn.Conv2d(self.oup, self.num_anchors * 4, 3, padding=1)
    def forward(self, x):
        reg = x
        for conv in self.reg:
            reg = conv(reg)

        reg = self.retina_reg(reg)

        reg = reg.permute(0, 2, 3, 1)
        return reg.contiguous().view(reg.shape[0], -1, 4)

class Cls(nn.Module):
    """

    """
    def __init__(self, inp, oup, depth, num_anchor, num_class):
        super().__init__()
        self.inp = inp
        self.oup = oup
        self.D = depth
        self.cls = nn.ModuleList()
        self.num_anchors = num_anchor
        self.num_class = num_class
        for i in range(self.D):
            self.cls.append(ConvBlock(inp=self.inp, oup=self.oup, k_size=3, stride=1, padding=1))
        self.retina_cls = nn.Conv2d(self.oup, self.num_anchors * self.num_class, 3, padding=1)
        self.act = nn.Sigmoid()
    def forward(self, x):
        cls = x
        for conv in self.cls:
            cls = conv(cls)
        cls = self.retina_cls(cls)
        cls = self.act(cls)

        cls = cls.permute(0, 2, 3, 1)

        batch_size, width, height, channel = cls.shape

        out = cls.view(batch_size, width, height, self.num_anchors, self.num_class)
        return out.contiguous().view(cls.shape[0], -1, self.num_class)


class RetinaHead(nn.Module):
    """

    """
    def __init__(self, parser, num_classes=80, num_anchor=9, is_demo=False):
        super().__init__()
        depth = 3
        inp = oup = 64

        self.regression = Reg(inp, oup, depth-1, num_anchor)
        self.classification = Cls(inp, oup, depth-1, num_anchor, num_classes)
        self.FocalLoss = losses.FocalLoss()
        self.anchors = Anchors()
        self.EfficientDet = EfficientDet(parser)
        self.regressBoxes = BBoxTransform()
        self.is_demo = is_demo
        self.clipBoxes = ClipBoxes()
    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        features = self.EfficientDet(img_batch)
        regression = torch.cat([self.regression(feature) for feature in features], dim=1)
        classification = torch.cat([self.classification(feature) for feature in features], dim=1)
        anchors = self.anchors(img_batch)

        # self.FocalLoss(classification, regression, anchors, annotations)
        if self.training:
            return self.FocalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            if self.is_demo:
                return transformed_anchors, classification, scores

            scores_over_thresh = (scores>0.01)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0).cuda(), torch.zeros(0).cuda(), torch.zeros(0, 4).cuda()]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]
            # print(transformed_anchors.shape, scores.shape)

            # anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5)
            # print(transformed_anchors[0, :, :])
            anchors_nms_idx = nms(transformed_anchors[0, :, :], scores[0, :, 0], 0.45)
            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]






