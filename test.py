# -*- coding:utf-8 -*-
# Author: huashuoshuo
# Data: 12/25/19 2:51 PM

from __future__ import print_function

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np
import json
import os

import torch

def test(dataset):
    coco_true = dataset.coco
    coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()