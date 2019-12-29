# -*- coding:utf-8 -*-
# Author: huashuoshuo
# Data: 12/26/19 2:12 PM
import torch
import torch.nn as nn
from model.util import Filter_boxes
import os
import argparse
from RetinaHead import RetinaHead
import skimage.io
import skimage
import skimage.transform
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import time
from model.util import num2name
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='/home/huashuoshuo/bishe/imges/6.jpg')
    parser.add_argument('--weight_path', type=str, default='./weights/retinanet_15.pth')
    parser.add_argument('--backbone', type=str, default='efficientnet-b0')
    parser.add_argument('--backbone_pretrained', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0.35)

    parser = parser.parse_args()
    with torch.no_grad():
        efficientdet = RetinaHead(parser, is_demo=True)
        # efficientdet = torch.nn.DataParallel(efficientdet).cuda()
        efficientdet = efficientdet.cuda()
        state_dict = torch.load(parser.weight_path)
        efficientdet.load_state_dict(state_dict)

        # img read
        img = skimage.io.imread(parser.img_path)
        img_input, scale1, scale2= preprocessing(img)
        efficientdet.eval()
        img_input = img_input.cuda()
        time_start = time.time()
        # for i in range(1000):
        boxes, classification, scores = efficientdet(img_input)
        boxes, scores, labels= Filter_boxes(parser)([boxes, classification, scores])

        time_stop = time.time()
        print('time:', time_stop-time_start)
        # scores = scores.cpu().numpy()
        # labels = labels.cpu().numpy()
        # boxes = boxes.cpu().numpy()

        # print(boxes)
        # print(np.shape(img))
        text_thickness = 1
        thickness = 2
        scale = 0.4
        line_type = 8
        for i in range(np.shape(boxes)[0]):
            box = boxes[i].cpu().numpy()
            score = scores[i].cpu().numpy()
            for j in range(np.shape(box)[0]):
                p1 = (int(box[j][0]/scale2), int(box[j][1]/scale1))
                p2 = (int(box[j][2]/scale2), int(box[j][3]/scale1))
                cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
                s = '%s/%.1f%%' % (num2name[labels[i]+1], score[j] * 100)
                text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)

                if (p2[0] - p1[0] < 1) or (p2[1] - p1[1] < 1):
                    continue
                # p1 = (p1[0] - text_size[1], p1[1])

                cv2.rectangle(img, (p1[0], p1[1]),
                              (p1[0] + text_size[0], p1[1] + text_size[1]), (0, 0, 255), -1)

                cv2.putText(img, s, (p1[0], p1[1] + 2*baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                            text_thickness, line_type)
        plt.imshow(img)
        plt.show()
    # print(scores, labels)



    return

def preprocessing(img):

    img = img.astype(np.float32) / 255.0
    # normalize
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
    img = (img - mean) / std
    # resize
    rows, cols, cns = np.shape(img)
    scale1 = 512 / rows
    scale2 = 512 / cols
    img_input = skimage.transform.resize(img, (512, 512))
    img_input = torch.from_numpy(img_input)
    img_input = img_input.unsqueeze(0)
    img_input = img_input.permute(0, 3, 1, 2).float()
    return img_input, scale1, scale2


def box_filter(scores, labels, boxes):
    scores = scores.cpu()
    labels = labels.cpu()
    boxes = boxes.cpu()

    return


if __name__=='__main__':
    main()
