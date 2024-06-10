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

    print("forward xh: {} shape: {} \n".format(xh, xh.shape))

    # ft = xh.dot(Wf) + Bf
    # ft = ft+forget_bias
    # ft = sigmoid(ft)

    # np.dot(a,b) 一维向量是内积运算; 矩阵的话是举证乘积运算

    ft = sigmoid(xh.dot(Wf) + Bf + forget_bias)
    print("wf: {}  shape: {} \n".format(Wf, Wf.shape))
    print("bf: {}  shape: {}\n".format(Bf, Bf.shape))
    print("ft: {}  shape: {}".format(ft, ft.shape)) # 1*3

    it = sigmoid(xh.dot(Wi) + Bi)
    ot = sigmoid(xh.dot(Wo) + Bo)
    ct_ = tanh(xh.dot(Wc) + Bc)

    # np.multiply() 和 * : 两个数组进行对应位置的乘积（element-wise product）输出的结果与参与运算的数组或者矩阵的大小一致

    ct = np.multiply(ft, c_pre) + np.multiply(it, ct_)

    print("c_pre: {}  shape: {} \n".format(c_pre, c_pre.shape))
    print("it: {}  shape: {}\n".format(it, it.shape))
    print("ct_: {}  shape: {}".format(ct_, ct_.shape))
    print("ct: {}  shape: {}".format(ct, ct.shape))  # 1*3

    ht = np.multiply(ot, tanh(ct))
    
    print("ot: {}  shape: {}".format(ot, ot.shape))
    print("ht: {}  shape: {}".format(ht, ht.shape))  # 1*3


    return ct, ht


# numpy.random.randn(d0,d1,…,dn)
# randn函数返回一个或一组样本，具有标准正态分布。
# dn表格每个维度
# 返回值为指定维度的array

N = 3000  # 30000

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

    print("==================================")
    print("\n c2: {}  h2: {}\n".format(c2, h2))   # c2: 1 * 3   h2: 1 * 3  二维


    if h2[0, 0] > ratio * (h2[0, 1] + h2[0, 2]):
        samples_X.append(X[i].tolist())  # 已选样本数 * 3 * 4  代表3个时间序列，每个时间序列输入的维度是：4
        val_list.append(h2[0].tolist())  # 已选样本数 * 3
        label_list.append([0])  # 已选样本数 * 1
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



print("samples: {}".format(samples_X))
print("val list: {}".format(val_list))
print("label list: {}".format(label_list))


plt.figure()
plt.plot(label_list, marker='*', c='r', alpha=0.2)  # 点之间画直线 【划线不能有参数s，alpha为透明通道1为不透明，接近0则更透明】
plt.show()
print('ch0,ch1,ch2', ch0, ch1, ch2)
print(val_list[0:2])


# def savez(file, *args, **kwds):
# savez()函数：以未压缩的.npz格式将多个数组保存到单个文件中。
# .npz格式：以压缩打包的方式存储文件，可以用压缩软件解压。
# savez()函数：第一个参数是文件名，其后的参数都是需要保存的数组，也可以使用关键字参数为数组起一个名字，非关键字参数传递的数组会自动起名为arr_0, arr_1, …。
# savez()函数：输出的是一个压缩文件（扩展名为.npz），其中每个文件都是一个save()保存的.npy文件，文件名对应于数组名。
# load()自动识别.npz文件，并且返回一个类似于字典的对象，可以通过数组名作为关键字获取数组的内容。

np.savez(r'.\Samples.npz', x=samples_X, y=label_list, y_val=val_list)

print("the samples_x len: {}".format(len(samples_X)))
