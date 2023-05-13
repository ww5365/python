# -*- coding: utf-8 -*-
"""
# @file name  : transforms_methods_1.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-09-11 10:08:00
# @brief      : transforms方法(一)
"""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
path_lenet = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "lenet.py"))
path_tools = os.path.abspath(os.path.join(BASE_DIR, "..", "tools", "common_tools.py"))
assert os.path.exists(path_lenet), "{}不存在，请将lenet.py文件放到 {}".format(path_lenet, os.path.dirname(path_lenet))
assert os.path.exists(path_tools), "{}不存在，请将common_tools.py文件放到 {}".format(path_tools, os.path.dirname(path_tools))

import sys
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__) + os.path.sep + "..")
sys.path.append(hello_pytorch_DIR)

from tools.my_dataset import RMBDataset
from tools.common_tools import set_seed, transform_invert

set_seed(1)  # 设置随机种子

# 参数设置
MAX_EPOCH = 1
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
rmb_label = {"1": 0, "100": 1}


class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作

    椒盐噪声：脉冲噪声  是一种随机出现的白点或黑点
    信噪比：SNR  signal-noise rate  噪声元素占图片元素的比列

    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))    # 2020 07 26 or --> and
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.]) # 0 原图像 90%  1 盐噪声

            '''
            choice(a, size=None, replace=True, p=None)
            a: 给定的一维数组且必须是一维数组。可以有两种给值，给定一个数组，例如a=[1,2,3,4]
               给定一个正值，此时就相当于从列表np.arange(a)中随机抽取D个整数
            size: 表示想要得到的数的个数，
                  如果size=3,表示得到3个数。
                  如果size=(3,2)，表示得到3*2个数，且形状为3*2
            replace: 表示是否重复选取，默认表示可以重复选择.
            p: 表示列表中某数被选取的概率，默认为正态分布
            '''

            mask = np.repeat(mask, c, axis=2)

            '''
            np.repeat(a, repeats, axis=None)
            对a数组中的对应元素进行重复复制
        
            a=np.array([[10,20],[30,40]])  
            np.repeat(a, [3,2], axis = 0)

            [10,20]
            [10,20]
            [10,20]
            [30,40]
            [30,40]

            '''

            img_[mask == 1] = 255   # 盐噪声 白色
            img_[mask == 2] = 0     # 椒噪声 黑色
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img


# ============================ step 1/5 数据 ============================
split_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "rmb_split"))
if not os.path.exists(split_dir):
    raise Exception(r"数据 {} 不存在, 回到lesson-06\1_split_dataset.py生成数据".format(split_dir))
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 1 CenterCrop
    # transforms.CenterCrop(512),     # 512

    # 2 RandomCrop
    # transforms.RandomCrop(224, padding=16),
    # transforms.RandomCrop(224, padding=(16, 64)),
    # transforms.RandomCrop(224, padding=16, fill=(255, 0, 0)),
    # transforms.RandomCrop(512, pad_if_needed=True),   # pad_if_needed=True
    # transforms.RandomCrop(224, padding=64, padding_mode='edge'),
    # transforms.RandomCrop(224, padding=64, padding_mode='reflect'),  # 鏡像
    # transforms.RandomCrop(1024, padding=1024, padding_mode='symmetric'),

    # 3 RandomResizedCrop
    # transforms.RandomResizedCrop(size=224, scale=(0.5, 0.5)),

    # 4 FiveCrop
    # transforms.FiveCrop(112),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 5 TenCrop
    # transforms.TenCrop(112, vertical_flip=False),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 1 Horizontal Flip
    # transforms.RandomHorizontalFlip(p=1),

    # 2 Vertical Flip
    # transforms.RandomVerticalFlip(p=0.5),

    # 3 RandomRotation
    # transforms.RandomRotation(90),
    # transforms.RandomRotation((90), expand=True),
    # transforms.RandomRotation(30, center=(0, 0)),
    # transforms.RandomRotation(30, center=(0, 0), expand=True),   # expand only for center rotation


    # 1 Pad
    # transforms.Pad(padding=32, fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 64), fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='symmetric'),

    # 2 ColorJitter  图像的亮度，对比度，饱和度，色相
    # transforms.ColorJitter(brightness=0.5),
    # transforms.ColorJitter(contrast=0.5),
    # transforms.ColorJitter(saturation=0.5),
    # transforms.ColorJitter(hue=0.3),

    # 3 Grayscale  灰度图
    # transforms.Grayscale(num_output_channels=3),

    # 4 Affine  仿射变换  旋转 平移 缩放 错切 翻转
    # transforms.RandomAffine(degrees=30),
    # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), fillcolor=(255, 0, 0)),
    # transforms.RandomAffine(degrees=0, scale=(0.7, 0.7)),
    # transforms.RandomAffine(degrees=0, shear=(0, 0, 0, 45)),
    # transforms.RandomAffine(degrees=0, shear=90, fillcolor=(255, 0, 0)),

    # 5 Erasing  对图像进行随机的遮挡
    # transforms.ToTensor(),
    # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254/255, 0, 0)),
    # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='1234'),

    # 1 RandomChoice  一组transforms操作中，随机选择
    # transforms.RandomChoice([transforms.RandomVerticalFlip(p=1), transforms.RandomHorizontalFlip(p=1)]),

    # 2 RandomApply  执行一组transforms操作
    # transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=45, fillcolor=(255, 0, 0)),
    #                         transforms.Grayscale(num_output_channels=3)], p=0.5),
    # 3 RandomOrder  对一组transforms操作打乱顺序
    # transforms.RandomOrder([transforms.RandomRotation(15),
    #                         transforms.Pad(padding=32),
    #                         transforms.RandomAffine(degrees=0, translate=(0.01, 0.1), scale=(0.9, 1.1))]),

    # 自定义transform 
    # 1: 仅接收一个参数，返回一个参数  2：注意上下游的输入和输出
    
    # 
    # class Compose(object):
    #    def __call__(self, img):
    #        for t in self.transforms:
    #            img = t(img)
    #            return img

    # 通过类实现多个参数传入：
    # class  YourTransforms(object):
    #    def __init__(self, ...):
    #        ...
    #    def __call__(self, img):
    #        ...
    #        return img


    # eg：增加椒盐噪声
    AddPepperNoise(0.9, p=0.5),

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std) # 数据标准化处理
])

# 构建MyDataset实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)


# ============================ step 5/5 训练 ============================
for epoch in range(MAX_EPOCH):
    for i, data in enumerate(train_loader):

        inputs, labels = data   # B C H W 

        img_tensor = inputs[0, ...]     # C H W  3 * 224 * 224 张量
        img = transform_invert(img_tensor, train_transform)  # 
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()

        # bs, ncrops, c, h, w = inputs.shape
        # for n in range(ncrops):
        #     img_tensor = inputs[0, n, ...]  # C H W
        #     img = transform_invert(img_tensor, train_transform)
        #     plt.imshow(img)
        #     plt.show()
        #     plt.pause(1)
