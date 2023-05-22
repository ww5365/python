# -*- coding: utf-8 -*-
"""
# @file name  : nn_layers_convolution.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-09-23 10:08:00
# @brief      : 学习卷积层
"""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
path_tools = os.path.abspath(os.path.join(BASE_DIR, "..", "tools", "common_tools.py"))
assert os.path.exists(path_tools), "{}不存在，请将common_tools.py文件放到 {}".format(path_tools, os.path.dirname(path_tools))

import sys
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

from tools.common_tools import transform_invert, set_seed

set_seed(2)  # 设置随机种子

# ================================= load img ==================================
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lena.png")
img = Image.open(path_img).convert('RGB')  # 0~255

# print("before transform img: {}".format(img, img.shape))  # 512 * 512 

# convert to tensor
img_transform = transforms.Compose([transforms.ToTensor()])

img_tensor = img_transform(img)
print("img_tensor shape: {} \n value: {}".format(img_tensor.shape, img_tensor)) # 3 * 512 * 512
img_tensor.unsqueeze_(dim=0)    # C*H*W to B*C*H*W
print("unsqueeze img_tensor shape: {} \n value: {}".format(img_tensor.shape, img_tensor)) # 3 * 512 * 512


# ================================= create convolution layer ==================================

# ================ 2d

'''
nn.Conv2d
功能： 对多个二维信号进行二维卷积
参数：
in_channels: 输入通道数
out_channels: 输出通道数，等价卷积核数
kernel_size: 卷积核尺寸
stride：步长
padding：填充个数
dilation: 空洞卷积
groups： 分组卷积设置
bias： 偏置

'''

flag = 1
# flag = 0
if flag:
    conv_layer = nn.Conv2d(3, 1, 3)   # input:(i, o, size) weights:(o, i , h, w)
    nn.init.xavier_normal_(conv_layer.weight.data)

    # calculation
    print("tensor shape: {}".format(img_tensor.shape))
    img_conv = conv_layer(img_tensor)  # 1 * 1 * 510 * 510
    print("conv tensor shape: {}".format(img_conv.shape))

# ================ transposed
# flag = 1
flag = 0   

if flag:
    conv_layer = nn.ConvTranspose2d(3, 1, 3, stride=2)   # input:(i, o, size)
    nn.init.xavier_normal_(conv_layer.weight.data)

    # calculation
    img_conv = conv_layer(img_tensor)   # 1 * 1 * 1025 * 1025 没有明白为什么？ stride=2  扩展了img_conv的维度到1025

# ================================= visualization ==================================
print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv.shape))

'''
Tensor的任意维度上进行切片操作，PyTorch已经封装好了两个运算符:和...，它们的用法如下：
:常用于对一个维度进行操作，基本的语法形式是：start:end:step。单独使用:代表全选这个维度，start和end为空分别表示从头开始和一直到结束，step的默认值是1。
...用于省略任意多个维度，可以用在切片的中间，也可以用在首尾。
'''

t = img_conv[0, 0:1, ...]

print("t shape: {}".format(t.shape))

img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)  # 第0 维度取0行索引  第1维度取0~1行，剩下维度取任意多个维度

img_raw = transform_invert(img_tensor.squeeze(), img_transform)

plt.subplot(122).imshow(img_conv, cmap='gray')
plt.subplot(121).imshow(img_raw)
plt.show()
