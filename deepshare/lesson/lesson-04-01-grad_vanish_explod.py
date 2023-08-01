# -*- coding: utf-8 -*-
"""
# @file name  : grad_vanish_explod.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-09-30 10:08:00
# @brief      : 
# 梯度消失与爆炸实验
# 权重初始化的方法
"""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import torch
import random
import numpy as np
import torch.nn as nn
path_tools = os.path.abspath(os.path.join(BASE_DIR, "..", "tools", "common_tools.py"))
assert os.path.exists(path_tools), "{}不存在，请将common_tools.py文件放到 {}".format(path_tools, os.path.dirname(path_tools))

import sys
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__) + os.path.sep + "..")
sys.path.append(hello_pytorch_DIR)

from tools.common_tools import set_seed

set_seed(1)  # 设置随机种子


class MLP(nn.Module):
    def __init__(self, neural_num, layers):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.neural_num = neural_num

    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)  # x * weight.T  
            before_relu_x = x

            x = torch.relu(x)

            # x = torch.tanh(x)

            # print("layer:{}, std:{} x:{} befor_relu_x:{}".format(i, x.std(), x, before_relu_x))
            
            print("layer:{}, std:{}".format(i, x.std()))

            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                '''
                为什么经过多层网络后，方差会不断增大？
                每次扩大为：sqrt(n) 倍  n 是神经元的个数

                所以想让网络层的std保持不变：w的标准差为1/sqrt(n)
                '''


                break
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                
                # nn.init.normal_(m.weight.data)  #标准正态分布，0 均值， 1标准差  后面std越来越大，导致nan出现
                # nn.init.normal_(m.weight.data, std=np.sqrt(1/self.neural_num))    # normal: mean=0, std=1/sqrt(n)  保持后面的网络标准差稳定；但也会发现std越来越小，可能导致梯度消失
                # print("init weight: {}".format(m.weight.data))

                '''
                xavier 初始化

                方差一致性：保持数据尺度维持在前档的范围  通常方差为1  这样的话，即使传播多层，方差也是1

                激活函数： 饱和激活函数  sigmod tanh

                前向和后向 都保持方差为1

                n_i * D(w) = 1
                n_(i+1) * D(w) = 1

                D(w) = 2/(n_i + n_(i+1))   前后传播神经元个数

                主要针对饱和激活函数

                '''

                # xavier  手动
                # a = np.sqrt(6 / (self.neural_num + self.neural_num))   
                # tanh_gain = nn.init.calculate_gain('tanh')
                # a *= tanh_gain
                # nn.init.uniform_(m.weight.data, -a, a)

                # xavier torch提供的方法
                # nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)

                '''
                kaiming
                D(w) = 2/n_i
                '''

                # kaiming 初始化方法 手动
                # nn.init.normal_(m.weight.data, std=np.sqrt(2 / self.neural_num))

                # kaiming pytorch方法
                nn.init.kaiming_normal_(m.weight.data)

# flag = 0
flag = 1

if flag:
    layer_nums = 100
    neural_nums = 256
    batch_size = 16

    # layer_nums = 2
    # neural_nums = 4
    # batch_size = 3

    net = MLP(neural_nums, layer_nums)
    net.initialize()

    inputs = torch.randn((batch_size, neural_nums))  # normal: mean=0, std=1


    print("inputs: {}".format(inputs))
    
    output = net(inputs)

    print("outputs: {}".format(output))

# ======================================= calculate gain =======================================

flag = 0
# flag = 1

if flag:

    x = torch.randn(10000)
    out = torch.tanh(x)

    gain = x.std() / out.std()
    print('gain:{}'.format(gain))

    tanh_gain = nn.init.calculate_gain('tanh')
    print('tanh_gain in PyTorch:', tanh_gain)

