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

set_seed(3)  # 设置随机种子

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
    img_conv = conv_layer(img_tensor)

# ================ transposed
# flag = 1
flag = 0
if flag:
    conv_layer = nn.ConvTranspose2d(3, 1, 3, stride=2)   # input:(i, o, size)
    nn.init.xavier_normal_(conv_layer.weight.data)

    # calculation
    img_conv = conv_layer(img_tensor)


# ================================= visualization ==================================
print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv.shape))
img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)
img_raw = transform_invert(img_tensor.squeeze(), img_transform)

plt.subplot(122).imshow(img_conv, cmap='gray')
plt.subplot(121).imshow(img_raw)
plt.show()



