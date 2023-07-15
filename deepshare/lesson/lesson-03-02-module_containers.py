# -*- coding: utf-8 -*-
"""
# @file name  : module_containers.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-09-20 10:08:00
# @brief      : 模型容器——Sequential, ModuleList, ModuleDict
"""
import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict


# ============================ Sequential
class LeNetSequential(nn.Module):
    def __init__(self, classes):
        super(LeNetSequential, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),)

        '''
        nn.Sequential 也是Module, 也是有那8个有序词典 : 会将这6个子网络加入到_module参数中
        '''

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes),)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


class LeNetSequentialOrderDict(nn.Module):
    def __init__(self, classes):
        super(LeNetSequentialOrderDict, self).__init__()

        self.features = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(3, 6, 5),
            'relu1': nn.ReLU(inplace=True),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),

            'conv2': nn.Conv2d(6, 16, 5),
            'relu2': nn.ReLU(inplace=True),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2),
        }))
        #参数的个数：1个  有序词典 OrderedDict

        self.classifier = nn.Sequential(OrderedDict({
            'fc1': nn.Linear(16*5*5, 120),
            'relu3': nn.ReLU(),

            'fc2': nn.Linear(120, 84),
            'relu4': nn.ReLU(inplace=True),

            'fc3': nn.Linear(84, classes),
        }))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


# net = LeNetSequential(classes=2)
# net = LeNetSequentialOrderDict(classes=2)
#
# fake_img = torch.randn((4, 3, 32, 32), dtype=torch.float32)
#
# output = net(fake_img)
#
# print(net)
# print(output)


# ============================ ModuleList
# 适用： 构建大量重复的子网络时，可以考虑

class ModuleList(nn.Module):
    def __init__(self):
        super(ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(20)]) # 构建20层的全连接层

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
        return x


# net = ModuleList()
#
# print(net)
#
# fake_data = torch.ones((10, 10))
#
# output = net(fake_data)
#
# print(output)


# ============================ ModuleDict
# 适用： 构建的可以选择的子网络

class ModuleDict(nn.Module):
    def __init__(self):
        super(ModuleDict, self).__init__()
        self.choices = nn.ModuleDict({
            'conv': nn.Conv2d(10, 10, 3),
            'pool': nn.MaxPool2d(3)
        })

        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'prelu': nn.PReLU()
        })

        '''
        relu: rectified linear unit  纠正的线性单元
        prelu: parametric rectified linear unit  参数化的纠正的线性单元  当x<0时，y = ax a很小
        '''

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x


net = ModuleDict()

fake_img = torch.randn((4, 10, 32, 32))

output = net(fake_img, 'conv', 'relu')

print(output)




# 4 AlexNet

'''
特点：
1. 采用relu 激活函数： 替换饱和激活函数， 减轻梯度消失
2. LRN：Local Response Normalization  数据归一化， 减轻梯度消失
3. dropout： 增强全连接层的鲁棒性，提升泛化能力
4. data augmentation： 数据增强  eg：色彩修改 TenCrop

论文：《ImageNet Classification with Deep Convolutional Neural Networks》

'''

alexnet = torchvision.models.AlexNet()





