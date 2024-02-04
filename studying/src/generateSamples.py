#!C:\Users\BoBo\PycharmProjects\grad 
# -*- coding:utf-8 -*-
# @Time  : 2019/8/20 11:20
# @Author: YiFei
# @File  : generateSamples.py
# 根据给定的模型生成样本
# import tensorflow as tf

import numpy as np
import os
import matplotlib.pyplot as plt

num_units = 3

def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-x))
    # ds=s(1-s)
    return s


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def initWmatrix():  # kernel的每num_units列分别为Wi,Wc,Wf,Wo
    Wf = np.array(
        [[0.05, 0.7, 0.1, 0.3, 1, 0, 0.6], [0.05, 0.25, 0.1, 0.35, 0, 1, 1], [0.15, 0.75, 0.1, 0.45, 0, 0, 1]])
    Wi = np.array([[0.45, 0.4, 0.7, 0.3, 1, 1, 0], [0.05, 0.4, 0.75, 0.35, 1, 1, 1], [0.04, 0.1, 0.75, 0.45, 1, 0, 1]])
    Wc = np.array(
        [[0.15, 0.8, 0.6, 0.35, 0.2, 1, 0], [0.25, 0.1, 0.65, 0.3, 3, 1, 1], [0.07, 0.1, 0.75, 0.45, 0, 1, 0]])
    Wo = np.array([[0.5, 0.5, 0.75, 0.4, 1, 1, 0], [0.15, 0.9, 0.6, 0.5, 0, 1, 0], [0.2, 0.4, 0.75, 0.45, 1, 1, 0]])
    return Wf.T, Wi.T, Wc.T, Wo.T


def initBmatrix():  # kernel的每num_units列分别为Wi,Wc,Wf,Wo
    Bf = np.array([[0.15, 0.75, 0.1]])
    Bi = np.array([[0.05, 0.1, 0.7]])
    Bc = np.array([[0.25, 0.1, 0.15]])
    Bo = np.array([[0.7, 0.8, 0.2]])
    return Bf, Bi, Bc, Bo


forget_bias = 0


def forward(x_pre, h_pre, c_pre, Wi, Wc, Wf, Wo, Bi, Bc, Bf, Bo):  # 传入行矩阵
    
    # 两个矩阵的列合并 np.column_stack 参考：https://blog.csdn.net/IMWTJ123/article/details/103124169
    
    xh = np.column_stack((x_pre, h_pre))  # 行矩阵合并为行

    print("forward xh: {}".format(xh))

    # ft = xh.dot(Wf) + Bf
    # ft = ft+forget_bias
    # ft = sigmoid(ft)

    # np.dot(a,b) 一维向量是内积运算; 矩阵的话是举证乘积运算

    ft = sigmoid(xh.dot(Wf) + Bf + forget_bias)
    print("wf: {}  bf: {}  ft: {}".format(Wf, Bf, ft))

    it = sigmoid(xh.dot(Wi) + Bi)
    ot = sigmoid(xh.dot(Wo) + Bo)
    ct_ = tanh(xh.dot(Wc) + Bc)

    # np.multiply() 和 * : 两个数组进行对应位置的乘积（element-wise product）输出的结果与参与运算的数组或者矩阵的大小一致

    ct = np.multiply(ft, c_pre) + np.multiply(it, ct_)
    ht = np.multiply(ot, tanh(ct))


    return ct, ht


# numpy.random.randn(d0,d1,…,dn)
# randn函数返回一个或一组样本，具有标准正态分布。
# dn表格每个维度
# 返回值为指定维度的array

N = 1  # 30000

X = np.random.randn(N, 3, 4)
label_list = []
val_list = []
samples_X = []
ch0 = 0
ch1 = 0
ch2 = 0

for i in range(X.shape[0]):
    Wf, Wi, Wc, Wo = initWmatrix()
    Bf, Bi, Bc, Bo = initBmatrix()
    h_pre = np.zeros((1, num_units))
    c_pre = h_pre
    x_pre = np.array([X[i, 0, :]])
    c1, h1 = forward(x_pre, h_pre, c_pre, Wi, Wc, Wf, Wo, Bi, Bc, Bf, Bo)
    x_pre = np.array([X[i, 1, :]])
    c2, h2 = forward(x_pre, h1, c1, Wi, Wc, Wf, Wo, Bi, Bc, Bf, Bo)
    x_pre = np.array([X[i, 2, :]])
    c2, h2 = forward(x_pre, h2, c2, Wi, Wc, Wf, Wo, Bi, Bc, Bf, Bo)
    ratio = 1.0
    if h2[0, 0] > ratio * (h2[0, 1] + h2[0, 2]):
        samples_X.append(X[i].tolist())
        val_list.append(h2[0].tolist())
        label_list.append([0])
        ch0 = ch0 + 1

    elif h2[0, 1] > ratio * (h2[0, 0] + h2[0, 2]):
        samples_X.append(X[i].tolist())
        label_list.append([1])
        val_list.append(h2[0].tolist())
        ch1 = ch1 + 1

    elif h2[0, 2] > ratio * (h2[0, 0] + h2[0, 1]):
        samples_X.append(X[i].tolist())
        label_list.append([2])
        val_list.append(h2[0].tolist())
        ch2 = ch2 + 1

plt.figure()
plt.plot(label_list, marker='*', c='r', alpha=0.2)  # 点之间画直线 【划线不能有参数s，alpha为透明通道1为不透明，接近0则更透明】
plt.show()
print('ch0,ch1,ch2', ch0, ch1, ch2)
print(val_list[0:2])
np.savez('Samples.npz', x=samples_X, y=label_list, y_val=val_list)
