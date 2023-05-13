# -*- coding: utf-8 -*-
"""
# @file name  : lenet.py
# @author     : yts3221@126.com
# @date       : 2019-08-21 10:08:00
# @brief      : lenet模型定义
"""
import torch.nn as nn
import torch.nn.functional as F

'''
lesson 3-1

关注模型构建的两个要素：
    1. 构建子模块
    2. 拼接子模块

所有模型都是继承自nn.Module的

几个重要的类：

nn.parameter: 张量的子类， 标识可学习的参数，eg：weight， bias
nn.Module: 所有网络层基类， 管理网络属性
nn.functional: 函数的具体实现， eg： 卷积， 池化， 激活函数等
nn.init： 参数的初始化


重点关注：nn.Module 类属性

parameters: 存储管理nn.Parameter类  OrderedDict() 有序字典
modules：存储管理nn.Module类
buffers： 存储管理缓冲属性，如BN层中的running_mean
***_hooks: 存储管理钩子函数

nn.Module 总结

* 1个module可以包含多个子module ： LeNet中包含2个卷积子module，3 个全连接子module
* 1个module相当于1个运算，必须要实现forward()函数
* 每个module都有8个字典管理他的属性


'''

class LeNet(nn.Module):

    def __init__(self, classes) -> object: # 构建子模块 子module
        super(LeNet, self).__init__()  # 调用lenet的父类的__init__,构建8个有序字典：paramters modules
        self.conv1 = nn.Conv2d(3, 6, 5)
        # conv1是nn.Module子网络，有8个属性字段，最终要的属性字段
        # _parameters  :  保存了weights 和 bias  是nn.Paramter类型（继承于张量）
        # _modules ： 在conv1子网络属性，没有子网络了，所以是空的

        self.conv2 = nn.Conv2d(6, 16, 5)
        # 构建子网络时, 先判断赋值的数据类型
        # 是nn.parameter类型，存储到parameters有序字典中；
        # 是nn.Module类，会被存储到LeNet这个网络的modules有序字典属性中, 这里是 conv1: Conv2d()  conv2: Con2d()
        #  
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

        # nn.Linear(): 全连接层  torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        # 使用参考：https://blog.csdn.net/zhaohongfei_358/article/details/122797190 


    def forward(self, x):  # 拼接子模块 
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()

class LeNet2(nn.Module):
    def __init__(self, classes):
        super(LeNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

class LeNet_bn(nn.Module):
    def __init__(self, classes):
        super(LeNet_bn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(num_features=6)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(num_features=120)

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = F.max_pool2d(out, 2)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.bn3(out)
        out = F.relu(out)

        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)
                m.bias.data.zero_()





