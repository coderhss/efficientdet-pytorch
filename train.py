# -*- coding:utf-8 -*-
# Author: huashuoshuo
# Data: 2019/12/19 14:57

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from model.efficientdet import EfficientDet
from model.RetinaHead import RetinaHead
import coco_eval
import argparse
from tensorboardX import SummaryWriter
import cv2 as cv2
import matplotlib.pyplot as plt

# writer = SummaryWriter('log')

os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2, 3'
def main(arg=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--coco_path', type=str, default='/home/hoo/Dataset/COCO')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--epoches', type=int, default=50)
    parser.add_argument('--phi', type=int, default=0)
    parser.add_argument('--backbone', type=str, default='efficientnet-b0')
    parser.add_argument('--backbone_pretrained', type=bool, default=True)
    parser.add_argument('--EfficientDet_pretrained', type=bool, default=False)
    parser.add_argument('--pretrained', type=str, default='./weights/retinanet_1.pth')
    parser.add_argument('--batch_size', type=int, default=24)

    parser = parser.parse_args(arg)
    dataset_train = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    # print(dataset_train.num_classes())
    dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=16, collate_fn=collater, batch_sampler=sampler)


    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the Model

    efficientdet = RetinaHead(parser)



    efficientdet = torch.nn.DataParallel(efficientdet).cuda()
    if parser.EfficientDet_pretrained:
        state_dict = torch.load(parser.pretrained)
        # print(state_dict)
        efficientdet.module.load_state_dict(state_dict)

    efficientdet.training = True

    optimizer = optim.Adam(efficientdet.parameters(), lr=1e-3)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 5, 7, 9, 11, 13, 15, 17, 19], gamma=0.5)

    for epoch_num in range(parser.epoches):
        efficientdet.train()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
                break
            # try:
                # print(data)
                optimizer.zero_grad()
                # print(np.shape(data['annot']))
                classification_loss, regression_loss = efficientdet([data['img'].cuda().float(), data['annot']])
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss
                if bool(loss==0):
                    continue
                loss.backward()

                torch.nn.utils.clip_grad_norm_(efficientdet.parameters(), 0.1)
                optimizer.step()
                epoch_loss.append(float(loss))
                print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss)))

                if iter_num % 200 == 199:
                    niter = epoch_num * len(dataloader_train) + iter_num
                    # print(loss)
                    writer.add_scalar('Train/Loss', loss, niter)
                    writer.add_scalar('Train/Reg_Loss', regression_loss, niter)
                    writer.add_scalar('Train/Cls_Loss', classification_loss, niter)


                del classification_loss
                del regression_loss
            # except Exception as e:
                # print(e)
            # continue
                # if iter_num == 20:
                #     break

        # print('Evaluating dataset')
        mAP = coco_eval.evaluate_coco(dataset_val, efficientdet)
        # writer.add_scalar('Test/mAP', mAP, epoch_num)
        print('Save Model')
        # torch.save(efficientdet.module.state_dict(), './weights/retinanet_{}.pth'.format(epoch_num))
        # scheduler.step(np.mean(epoch_loss))
        scheduler.step(epoch=epoch_num)
# writer.close()


if __name__ == '__main__':
    main()