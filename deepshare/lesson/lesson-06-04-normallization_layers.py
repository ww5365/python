# -*- coding: utf-8 -*-
"""
# @file name  : bn_and_initialize.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-11-03
# @brief      : pytorch中常见的 normalization layers
"""
import torch
import numpy as np
import torch.nn as nn
import sys, os
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__) + os.path.sep + "..")
sys.path.append(hello_pytorch_DIR)
from tools.common_tools import set_seed


set_seed(1)  # 设置随机种子

# ======================================== nn.layer norm
flag = 1
# flag = 0
if flag:
    batch_size = 2
    num_features = 3

    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)  # 2D
    feature_maps = torch.stack([feature_map * (i + 1) for i in range(num_features)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps for i in range(batch_size)], dim=0)  # 4D

    # feature_maps_bs shape is [8, 6, 3, 4],  B * C * H * W
    # ln = nn.LayerNorm(feature_maps_bs.size()[1:], elementwise_affine=True)
    # ln = nn.LayerNorm(feature_maps_bs.size()[1:], elementwise_affine=False)
    # ln = nn.LayerNorm([6, 3, 4])
    # ln = nn.LayerNorm([3, 2, 2])
    ln = nn.LayerNorm(2)
    # ln = nn.LayerNorm([6, 3])

    feature_maps_bs[0:1,0:1,0:1,0:1] = 5
    output = ln(feature_maps_bs)

    ## NLP任务中：seq_len * batch_size * dim

    corpus = [[[1,3],[2,3]],[[2,4],[2,4]],[[2,3],[5,6]]]  # 3 * 2 * 2

    x = torch.tensor(corpus).to(torch.float32)

    ln = nn.LayerNorm(2)  # 按照最后一个维度2，即2个元素进行规范化
    # ([[[-1.0000,  1.0000],
    #          [-1.0000,  1.0000]],

    #         [[-1.0000,  1.0000],
    #          [-1.0000,  1.0000]],

    #         [[-1.0000,  1.0000],
    #          [-1.0000,  1.0000]]]

    # ln = nn.LayerNorm([2,2])   # 按照 2*2的维度，4个元素整体进行规范化

    # ([[[-1.5075,  0.9045],
    #      [-0.3015,  0.9045]],

    #     [[-1.0000,  1.0000],
    #      [-1.0000,  1.0000]],

    #     [[-1.2649, -0.6325],
    #      [ 0.6325,  1.2649]]]

    print("test layerNorm\n: {}".format(ln(x)))


    print("Layer Normalization")
    print(ln.weight.shape)
    print(feature_maps_bs[0, ...])
    print(output[0, ...])

# ======================================== nn.instance norm 2d
# flag = 1
flag = 0
if flag:

    batch_size = 2
    num_features = 3
    momentum = 0.3

    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)    # 2D
    feature_maps = torch.stack([feature_map * (i + 1) for i in range(num_features)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps for i in range(batch_size)], dim=0)  # 4D

    print("Instance Normalization")
    print("input data:\n{} shape is {}".format(feature_maps_bs, feature_maps_bs.shape))

    instance_n = nn.InstanceNorm2d(num_features=num_features, momentum=momentum, affine=True)

    feature_maps_bs[0:1,0:1,0:1,0:1] = 5

    for i in range(1):
        outputs = instance_n(feature_maps_bs)

        print(outputs)
        print("\niter:{}, running_mean.shape: {}".format(i, instance_n.running_mean))
        # print("iter:{}, running_var.shape: {}".format(i, instance_n.running_var.shape))
        print("iter:{}, weight.shape: {} weight: {}".format(i, instance_n.weight.shape, instance_n.weight))
        print("iter:{}, bias.shape: {} {}".format(i, instance_n.bias.shape, instance_n.bias))


# ======================================== nn.grop norm
# flag = 1
flag = 0
if flag:

    batch_size = 2
    num_features = 4
    num_groups = 2   # 3 Expected number of channels in input to be divisible by num_groups

    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)    # 2D
    feature_maps = torch.stack([feature_map * (i + 1) for i in range(num_features)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps * (i + 1) for i in range(batch_size)], dim=0)  # 4D

    gn = nn.GroupNorm(num_groups, num_features)
    outputs = gn(feature_maps_bs)

    print("Group Normalization")
    print(gn.weight.shape)
    print(outputs[0])
