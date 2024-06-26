# -*- coding: utf-8 -*-
"""
# @file name  : bn_and_initialize.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-11-01
# @brief      : bn与权值初始化
"""
import torch
import numpy as np
import torch.nn as nn

import sys, os
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__) + os.path.sep + "..")
sys.path.append(hello_pytorch_DIR)

from tools.common_tools import set_seed

set_seed(1)  # 设置随机种子


class MLP(nn.Module):
    def __init__(self, neural_num, layers=100):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(neural_num) for i in range(layers)])
        self.neural_num = neural_num

    def forward(self, x):

        for (i, linear), bn in zip(enumerate(self.linears), self.bns):
            x = linear(x)  # 全连接
            x = bn(x)  # 输入数据的批处理
            x = torch.relu(x)

            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))  # 前向传播什么时候结束？输入数据的方差为NAN时，not a number
                break

            """
            情况1：不bn + method1 权重初始化方式， x.std 会sqrt(neural_num)的倍数，变大；数据尺度变的很大，分布不一致;
            情况2：不bn + method2 凯明方法  relu   数据尺度基本不变
            情况3:  bn + 不加权重初始化方法   数据尺度更稳定，不变
            """
            print("layers:{}, std:{} ".format(i, x.std().item())) 

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):

                # method 1
                # nn.init.normal_(m.weight.data, std=1)    # normal: mean=0, std=1  初始化模型的权重，不合理方差，x数据变大

                # method 2 kaiming
                nn.init.kaiming_normal_(m.weight.data)


neural_nums = 256
layer_nums = 50
batch_size = 16

net = MLP(neural_nums, layer_nums)
net.initialize()

inputs = torch.randn((batch_size, neural_nums))  # normal: mean=0, std=1

print("input shape: {}".format(inputs.shape))  # 16 * 256

output = net(inputs)
print(output)



















