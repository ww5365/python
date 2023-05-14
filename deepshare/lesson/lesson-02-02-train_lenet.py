# -*- coding: utf-8 -*-
"""
# @file name  : train_lenet.py
# @author     : 参考 https://github.com/TingsongYu
# @date       : 2019-09-07 10:08:00 2023
# @brief      : 人民币分类模型训练
"""

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"]="true"
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt

# if __name__ == '__main__':
#     if __package__ is None:
#         import sys
#         from os import path
#         sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
#         import api
#     else:dataloader
#         import ..api.api
print(BASE_DIR)

path_lenet = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "lenet.py"))
path_tools = os.path.abspath(os.path.join(BASE_DIR, "..", "tools", "common_tools.py"))
assert os.path.exists(path_lenet), "{}not exisits, please put lenet.py file to {}".format(path_lenet, os.path.dirname(path_lenet))
# assert Expression,[arguments] 表达式返回false 时, 直接抛出异常终止继续执行
assert os.path.exists(path_tools), "{}not exisits, please put common_tools.py to {}".format(path_tools, os.path.dirname(path_tools))


import sys
WORK_DIR = os.path.abspath(os.path.dirname(__file__)+ os.path.sep + "..") #\deepshare\  直接算出上层目录是啥了   
WORK_DIR2 = os.path.join(BASE_DIR, "..")  #\deepshare\lesson\..
sys.path.append(WORK_DIR)

from model.lenet import LeNet
from tools.my_dataset import RMBDataset
from tools.common_tools import transform_invert, set_seed

set_seed()  # 设置随机种子
rmb_label = {"1": 0, "100": 1}

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 16
# BATCH_SIZE = 2
LR = 0.01
log_interval = 10
val_interval = 1

# ============================ step 1/5 数据 ============================
split_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "rmb_split"))
if not os.path.exists(split_dir):
    raise Exception(r"train data {} not exisit!".format(split_dir))
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

'''
torchvision 库简介
https://zhuanlan.zhihu.com/p/476220305

torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；

transforms.CenterCrop 对图片中心进行裁剪 
transforms.ColorJitter 对图像颜色的对比度、饱和度和零度进行变换
transforms.FiveCrop  对图像四个角和中心进行裁剪得到五分图像
transforms.Grayscale  对图像进行灰度变换
transforms.Pad  使用固定值进行像素填充
transforms.RandomAffine  随机仿射变换 
transforms.RandomCrop  随机区域裁剪
transforms.RandomHorizontalFlip  随机水平翻转
transforms.RandomRotation  随机旋转
transforms.RandomVerticalFlip  随机垂直翻转

transforms.Compose: 图像进行各种转换操作，并用函数compose将这些转换操作组合起来

Normalize: 标准化，加快训练模型的收敛速度 和 提高泛化能力

'''

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomGrayscale(p=0.9),  # 红色的一百元不能正确分类，因为红色和训练数据中1元的纸币看起来的像，所以使用灰度图来训练，弱化色彩的干扰
    transforms.ToTensor(),  # 转化成张量，并且把0~255数据进行归一化处理，除以255归一化
    transforms.Normalize(norm_mean, norm_std),   # 对数据进行逐channel的标准化  (x - mean) /std
])

valid_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# 构建MyDataset实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

print("-----------------------")

for data, label in train_loader:

    print("data shape: {} label: {}".format(data.shape,label))


# exit(0)

# ============================ step 2/5 模型 ============================

net = LeNet(classes=2)
net.initialize_weights()

# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数

# ============================ step 4/5 优化器 ============================
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)                        # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)     # 设置学习率下降策略 ？

# ============================ step 5/5 训练 ============================
train_curve = list()
valid_curve = list()

for epoch in range(MAX_EPOCH):

    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train()
    for i, data in enumerate(train_loader):  # 从dataloader中获取1个batchsize大小的样本数据, 常用的dataloader的循环处理的方式

        # forward
        inputs, labels = data
        outputs = net(inputs)   # 进入nn.Module 中__call__函数,会调用forward函数 

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()

        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
            loss_mean = 0.

    scheduler.step()  # 更新学习率

    # validate the model
    if (epoch+1) % val_interval == 0:

        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().sum().numpy()

                loss_val += loss.item()

            loss_val_epoch = loss_val / len(valid_loader)
            valid_curve.append(loss_val_epoch)
            # valid_curve.append(loss.item())    # 20191022改，记录整个epoch样本的loss，注意要取平均
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val_epoch, correct_val / total_val))


train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)
valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
valid_y = valid_curve

plt.plot(train_x, train_y, label='Train')
plt.plot(valid_x, valid_y, label='Valid')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
plt.show()

# ============================ inference ============================

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# test_dir = os.path.join(BASE_DIR, "test_data")

test_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "inference_data"))

test_data = RMBDataset(data_dir=test_dir, transform=valid_transform)
valid_loader = DataLoader(dataset=test_data, batch_size=1)

for i, data in enumerate(valid_loader):
    # forward
    inputs, labels = data
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)

    rmb = 1 if predicted.numpy()[0] == 0 else 100
    print("模型获得{}元".format(rmb))

    img_tensor = inputs[0, ...]  # C H W
    img = transform_invert(img_tensor, train_transform)
    plt.imshow(img)
    plt.title("LeNet got {} Yuan".format(rmb))
    plt.show()
    plt.pause(0.5)
    plt.close()
