# efficientdet-pytorch
![image](https://github.com/coderhss/efficientdet-pytorch/blob/master/img/2.png)
![image](https://github.com/coderhss/efficientdet-pytorch/blob/master/img/1.png)
![image](https://github.com/coderhss/efficientdet-pytorch/blob/master/img/3.jpg)
![image](https://github.com/coderhss/efficientdet-pytorch/blob/master/img/4.png)

Pytorch implementtation of EfficientDet object detection as described in [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)

This implementation is a very simple version without many data augmentation.

The EfficientNet code are borrowed from the [A PyTorch implementation of EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch),if you want to train EffcicientDet from scratch,you should load the efficientnet pretrained parameter. use

```
python train.py --coco_path '/home/hoo/Dataset/COCO' --backbon 'efficientnet-b0' --backbone_pretrained True
```

and the efficientnet pretrainied parameter will be download and load automatically, and start to train.

I've only trained efficientdet-d0 so far,and without many data augmentation.if you want to load efficientnet pretrained parameter,use

```
python train.py --coco_path '/home/hoo/Dataset/COCO' --backbone 'efficientnet-b0' --backbone_pretrained False --EfficientDet_pretrained True --pretrained './weights/efficientdet_0.pth'
```
|      Model      |  mAP  |                         pre_trained                          |
| :-------------: | :---: | :----------------------------------------------------------: |
| efficientdet-d0 | 25.9% | [download](https://drive.google.com/open?id=1UgQp9wqtc1O_EabU9O6NWNG6B8imYmv_) |

**QQ-group: 607724770(Torch交流群)**

## Acknowledgements
- The EfficientNet code are borrowed from the [A PyTorch implementation of EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)
- The code of RetinaNet are borrowed from the [Pytorch implementation of RetinaNet object detection.](https://github.com/yhenon/pytorch-retinanet)
