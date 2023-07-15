# -*- coding: utf-8 -*-
"""
# @file name  : nn_layers_others.py
# @author     : tingsongyu
# @date       : 2019-09-25 10:08:00
# @brief      : 其它网络层
"""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import torch
import random
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
path_tools = os.path.abspath(os.path.join(BASE_DIR, "..", "tools", "common_tools.py"))
assert os.path.exists(path_tools), "{}不存在，请将common_tools.py文件放到 {}".format(path_tools, os.path.dirname(path_tools))

import sys
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

from tools.common_tools import set_seed

set_seed(1)  # 设置随机种子

# ================================= load img ==================================
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lena.png")
img = Image.open(path_img).convert('RGB')  # 0~255

# convert to tensor
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0)    # C*H*W to B*C*H*W

# ================================= create convolution layer ==================================

# ================ maxpool
# flag = 1
flag = 0
if flag:
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2))   # input:(i, o, size) weights:(o, i , h, w)
    img_pool = maxpool_layer(img_tensor)
    print("maxpool : img_tensor:{} shape: {} img_pool: {}  shape: {}".format(img_tensor, img_tensor.shape, img_pool, img_pool.shape))

# ================ avgpool
# flag = 1
flag = 0
if flag:
    avgpoollayer = nn.AvgPool2d((2, 2), stride=(2, 2))   # input:(i, o, size) weights:(o, i , h, w)
    img_pool = avgpoollayer(img_tensor)

# ================ avgpool divisor_override

'''
torch.nn.AvgPool2d(kernel_size , stride=None , padding=0 , ceil_mode=False , count_include_pad=True , divisor_override=None )

注意：
输入尺寸是(N,C,H,W),输出尺寸是(N,C,H_out, W_out) 4个维度，试了下：c,h,w  3个维度也可以运算

kernel_size：池化核的尺寸大小
stride：窗口的移动步幅，默认与kernel_size大小一致
padding：在两侧的零填充宽度大小
divisor_override：如果被指定，则除数会被代替成divisor_override。
如果不指定该变量，则平均池化的计算过程其实是在一个池化核内，将元素相加再除以池化核的大小，也就是divisor_override默认为池化核的高×宽；
如果该变量被指定，则池化过程为将池化核内元素相加再除以divisor_override。

'''

# flag = 1
flag = 0
if flag:
    img_tensor = torch.ones((1, 4, 4))
    print(img_tensor)
    avgpool_layer = nn.AvgPool2d((2, 2), stride=(2, 2), divisor_override=3)
    img_pool = avgpool_layer(img_tensor)

    print("raw_img:\n{} shape: \n{} pooling_img:\n{} shape:\n{}".format(img_tensor, img_tensor.shape, img_pool, img_pool.shape))


# ================ max unpool

'''
class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助

'''

flag = 1
# flag = 0
if flag:
    # pooling
    img_tensor = torch.randint(high=5, size=(1, 2, 4, 4), dtype=torch.float)
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
    img_pool, indices = maxpool_layer(img_tensor)

    print("img_tensor:{} \n img_pool: {}\n indices:{}".format(img_tensor, img_pool, indices))

    # unpooling
    img_reconstruct = torch.randn_like(img_pool, dtype=torch.float)  # 重新构造了：1 * 2 * 2 * 2 的向量
    maxunpool_layer = nn.MaxUnpool2d((2, 2), stride=(2, 2))  
    img_unpool = maxunpool_layer(img_reconstruct, indices)  # 使用indices恢复成：1 * 2 * 4 * 4 的向量

    print("raw_img:\n{}\nimg_pool:\n{}".format(img_tensor, img_pool))
    print("------------------")
    print("img_reconstruct:\n{}\nimg_unpool:\n{}".format(img_reconstruct, img_unpool))


# ================ linear
'''
nn.Linear
功能：对一维信号进行线性组合
主要参数：
in_features: 输入节点数  有几个神经元
out_features: 输出节点数  
bias: 是否需要偏置

公式： y = xW^T + bias

'''

flag = 1
# flag = 0
if flag:
    inputs = torch.tensor([[1., 2, 3]])

    print("inputs: {} shape:{}".format(inputs, inputs.shape))
    linear_layer = nn.Linear(3, 4)
    linear_layer.weight.data = torch.tensor([[1., 1., 1.],
                                             [2., 2., 2.],
                                             [3., 3., 3.],
                                             [4., 4., 4.]])

    linear_layer.bias.data.fill_(0.5)
    output = linear_layer(inputs)
    print(inputs, inputs.shape)
    print(linear_layer.weight.data, linear_layer.weight.data.shape)
    print(output, output.shape)

'''
激活函数

nn.Sigmoid()
nn.Tanh()   # 双曲正切

nn.ReLU()
nn.PReLU()
nn.RReLU()

'''

# ================================= visualization ==================================
# print("池化前尺寸:{}\n池化后尺寸:{}".format(img_tensor.shape, img_pool.shape))
# img_pool = transform_invert(img_pool[0, 0:3, ...], img_transform)
# img_raw = transform_invert(img_tensor.squeeze(), img_transform)
# plt.subplot(122).imshow(img_pool)
# plt.subplot(121).imshow(img_raw)
# plt.show()
