#!C:\Users\BoBo\PycharmProjects\grad
# -*- coding:utf-8 -*-
# @Time  : 2019/8/22 15:46
# @Author: YiFei
# @File  : LstmBP.py
"""
自编写反向传播训练模型【在LstmBP_gradD.py 梯度下降基础上，修改为Adam方法训练参数】
优化目标为L = 1/2*(h_last - y)^2​
"""
# import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

forget_bias = 0
h_dimens = 3  # 隐含层维度为3
x_dimens = 4  # 输入向量维度为4
time_steps = 3  # 时间序为2
batchs = 700  # 批量样本
learn_rate = 0.03  # adam学习率
##############################################################################
'''
最终得到全部的样本数据中输入为 X，输出为labels_vecs（分类标签），y_val(输出的值)
'''
c = np.load('Samples.npz')
y = c['y']  # 分类标签
y_val = c['y_val']  # 输出值，用于回归拟合
X = c['x']
##############################################################################
'''
前向传播的基本函数
'''


def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-x))
    # ds=s(1-s)
    return s


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


##############################################################################
'''
初始化矩阵参数，并计算前向传播的各个时间值
'''
Wf = np.random.rand(h_dimens, h_dimens + x_dimens)
Wi = np.random.rand(h_dimens, h_dimens + x_dimens)
Wc = np.random.rand(h_dimens, h_dimens + x_dimens)
Wo = np.random.rand(h_dimens, h_dimens + x_dimens)
Bf = np.random.rand(h_dimens, 1)
Bi = np.random.rand(h_dimens, 1)
Bc = np.random.rand(h_dimens, 1)
Bo = np.random.rand(h_dimens, 1)

F = np.zeros((batchs, h_dimens, time_steps))  # F矩阵[batch,h_dimens,steps]
I = np.zeros((batchs, h_dimens, time_steps))  # I矩阵[batch,h_dimens,steps]
O = np.zeros((batchs, h_dimens, time_steps))  # O矩阵[batch,h_dimens,steps]
C_ = np.zeros((batchs, h_dimens, time_steps))  # C_矩阵[batch,h_dimens,steps]
C = np.zeros((batchs, h_dimens, time_steps))  # C矩阵[batch,h_dimens,steps]
H = np.zeros((batchs, h_dimens, time_steps))  # H矩阵[batch,h_dimens,steps]
HX = np.zeros((batchs, h_dimens + x_dimens, time_steps))  # H矩阵[batch,h_dimens+x_dimens,steps]


def batch_forward(X):  # X[batch,steps,x_dimens]
    Xt = np.transpose(X, (0, 2, 1))  # 转换为[batch,x_dimens,steps]
    for i in range(batchs):
        h_pre = np.zeros((h_dimens, 1))
        c_pre = h_pre
        for j in range(time_steps):
            x_pre = Xt[i, :, j]
            x_pre = x_pre.reshape((len(x_pre), 1))
            xh = np.row_stack((h_pre, x_pre))
            ft = sigmoid(Wf.dot(xh) + Bf + forget_bias)
            it = sigmoid(Wi.dot(xh) + Bi)
            ot = sigmoid(Wo.dot(xh) + Bo)
            ct_ = tanh(Wc.dot(xh) + Bc)
            ct = np.multiply(ft, c_pre) + np.multiply(it, ct_)
            ht = np.multiply(ot, tanh(ct))
            F[i, :, j] = ft.reshape(-1)
            I[i, :, j] = it.reshape(-1)
            O[i, :, j] = ot.reshape(-1)
            C_[i, :, j] = ct_.reshape(-1)
            C[i, :, j] = ct.reshape(-1)
            H[i, :, j] = ht.reshape(-1)
            h_pre = ht
            c_pre = ct
    HX = np.column_stack((H, Xt))
    return F, I, O, C_, C, H, HX


########################################################################################
'''
计算反向传播的各个时间值
'''
Dh = np.zeros((batchs, h_dimens, time_steps))  # h的偏导数矩阵[batch,h_dimens,steps]
Dc = np.zeros((batchs, h_dimens, time_steps))  # c的偏导数矩阵[batch,h_dimens,steps]


def back_forward(Y):  # Y[batch,h_dimens]
    for i in range(batchs):
        ft = F[i, :, -1]
        it = I[i, :, -1]
        ot = O[i, :, -1]
        ct_ = C_[i, :, -1]
        ct = C[i, :, -1]
        ht = H[i, :, -1]

        dh = ht - Y[i, :]  # 公式（3）
        dc = dh * ot * (1 - tanh(ct) * tanh(ct))  # 公式（9） # *代表hadamard积
        Dh[i, :, -1] = dh
        Dc[i, :, -1] = dc
        for j in range(0, time_steps - 1):
            ct_1 = C[i, :, -2 - j]
            dh = np.dot(Wo.T, (tanh(ct) * (1 - ot) * ot * dh))
            dh = dh + np.dot(Wf.T, ct_1 * (1 - ft) * ft * dc)
            dh = dh + np.dot(Wi.T, ct_ * (1 - it) * it * dc)
            dh = dh + np.dot(Wc.T, it * (1 - ct_ * ct_) * dc)
            dh = dh[0:h_dimens]  # 公式（10_12）
            Dh[i, :, -2 - j] = dh  # 公式（10_12）

            ot_1 = O[i, :, -2 - j]
            dc = dc * ft + dh * ot_1 * (1 - tanh(ct_1) * tanh(ct_1))  # 公式(13_14)
            Dc[i, :, -2 - j] = dc

            ft = F[i, :, -2 - j]
            it = I[i, :, -2 - j]
            ot = O[i, :, -2 - j]
            ct_ = C_[i, :, -2 - j]
            ct = C[i, :, -2 - j]
    return Dh, Dc


def dMatrix():
    dBf = np.zeros((batchs, h_dimens, time_steps))
    dBi = np.zeros((batchs, h_dimens, time_steps))
    dBc = np.zeros((batchs, h_dimens, time_steps))
    dBo = np.zeros((batchs, h_dimens, time_steps))
    dWf = np.zeros((batchs, time_steps, h_dimens, h_dimens + x_dimens))
    dWi = np.zeros((batchs, time_steps, h_dimens, h_dimens + x_dimens))
    dWc = np.zeros((batchs, time_steps, h_dimens, h_dimens + x_dimens))
    dWo = np.zeros((batchs, time_steps, h_dimens, h_dimens + x_dimens))
    HX0 = np.zeros((HX.shape[0], HX.shape[1], HX.shape[2] + 1))  # 将h_pre增加进来
    C0 = np.zeros((C.shape[0], C.shape[1], C.shape[2] + 1))  # 将c_pre增加进来
    for i in range(HX.shape[0]):
        HX0[i, :, 1:] = HX[i]
        C0[i, :, 1:] = C[i]
    for i in range(batchs):
        for j in range(time_steps):
            dh = Dh[i, :, j].reshape(-1, 1)
            dc = Dc[i, :, j].reshape(-1, 1)
            ot = O[i, :, j].reshape(-1, 1)
            ct = C[i, :, j].reshape(-1, 1)
            hxt = np.zeros((h_dimens + x_dimens, 1))
            hxt[0:h_dimens] = HX0[i, 0:h_dimens, j].reshape(-1, 1)
            hxt[h_dimens:] = HX0[i, h_dimens:, j + 1].reshape(-1, 1)
            dBo[i, :, j] = (dh * tanh(ct) * ot * (1 - ot)).reshape(-1)  # ∂h/∂bo
            dWo[i, j, :, :] = np.dot(dBo[i, :, j].reshape(-1, 1), hxt.T)  # ∂h/∂wo

            ct_1 = C0[i, :, j].reshape(-1, 1)
            ft = F[i, :, j].reshape(-1, 1)
            dBf[i, :, j] = (dc * ct_1 * ft * (1 - ft)).reshape(-1)  # ∂c/∂bf
            dWf[i, j, :, :] = np.dot(dBf[i, :, j].reshape(-1, 1), hxt.T)  # ∂c/∂wf

            it = I[i, :, j].reshape(-1, 1)
            ct_ = C_[i, :, j].reshape(-1, 1)
            dBi[i, :, j] = (dc * ct_ * it * (1 - it)).reshape(-1)  # ∂c/∂bi
            dWi[i, j, :, :] = np.dot(dBi[i, :, j].reshape(-1, 1), hxt.T)  # ∂c/∂wi

            dBc[i, :, j] = (dc * it * (1 - ct_ * ct_)).reshape(-1)  # ∂c/∂bc
            dWc[i, j, :, :] = np.dot(dBc[i, :, j].reshape(-1, 1), hxt.T)  # ∂c/∂wc
    return dBf, dBi, dBc, dBo, dWf, dWi, dWc, dWo


def grad_of_Batchs(dBf, dBi, dBc, dBo, dWf, dWi, dWc, dWo):
    '''根据所有的差值矩阵，得到每个batch样本对单个变量的偏导数'''
    dbf = np.sum(dBf, 0)  # batch合一
    dbf = np.sum(dbf, 1)  # time_steps合一

    dbi = np.sum(dBi, 0)  # batch合一
    dbi = np.sum(dbi, 1)  # time_steps合一

    dbc = np.sum(dBc, 0)  # batch合一
    dbc = np.sum(dbc, 1)  # time_steps合一

    dbo = np.sum(dBo, 0)  # batch合一
    dbo = np.sum(dbo, 1)  # time_steps合一

    dwf = np.sum(dWf, 0)  # batch合一
    dwf = np.sum(dwf, 0)  # time_steps合一

    dwi = np.sum(dWi, 0)  # batch合一
    dwi = np.sum(dwi, 0)  # time_steps合一

    dwc = np.sum(dWc, 0)  # batch合一
    dwc = np.sum(dwc, 0)  # time_steps合一

    dwo = np.sum(dWo, 0)  # batch合一
    dwo = np.sum(dwo, 0)  # time_steps合一

    dbf = dbf / batchs
    dbf = dbf.reshape((len(dbf), 1))
    dbi = dbi / batchs
    dbi = dbi.reshape((len(dbi), 1))
    dbc = dbc / batchs
    dbc = dbc.reshape((len(dbc), 1))
    dbo = dbo / batchs
    dbo = dbo.reshape((len(dbo), 1))
    return dbf, dbi, dbc, dbo, dwf, dwi, dwc, dwo


########################################################################################
'''参数更新方法'''


def updata(lr, dbf, dbi, dbc, dbo, dwf, dwi, dwc, dwo):
    '''根据偏导数更新模型内参数'''
    Wf_ = Wf - lr * dwf
    Wi_ = Wi - lr * dwi
    Wc_ = Wc - lr * dwc
    Wo_ = Wo - lr * dwo

    Bf_ = Bf - lr * dbf
    Bi_ = Bi - lr * dbi
    Bc_ = Bc - lr * dbc
    Bo_ = Bo - lr * dbo
    return Wf_, Wi_, Wc_, Wo_, Bf_, Bi_, Bc_, Bo_


def Adamgrad(dB, t, m, v, lr=learn_rate, beta1=0.9, beta2=0.999, epsilon=1e-08):
    m = beta1 * m + (1 - beta1) * dB
    v = beta2 * v + (1 - beta2) * (dB ** 2)
    mb = m / (1 - beta1 ** t)  # t is step number
    vb = v / (1 - beta2 ** t)
    detB = lr * mb / (np.sqrt(vb) + epsilon)
    return m, v, detB


########################################################################################

'''
开始训练
'''
# batch_size = 1500
no_of_batchs = int(len(X)) // batchs
epoch = 5000
print(len(X))
######[关于adam算法的一些参数设置-start]#####
t = 0  # adam迭代的步数
dbf_m, dbi_m, dbc_m, dbo_m = np.zeros((h_dimens, 1)), np.zeros((h_dimens, 1)), np.zeros((h_dimens, 1)), np.zeros(
    (h_dimens, 1))
dbf_v, dbi_v, dbc_v, dbo_v = dbf_m, dbi_m, dbc_m, dbo_m
dwf_m, dwi_m, dwc_m, dwo_m = np.zeros((h_dimens, h_dimens + x_dimens)), np.zeros(
    (h_dimens, h_dimens + x_dimens)), np.zeros((h_dimens, h_dimens + x_dimens)), np.zeros(
    (h_dimens, h_dimens + x_dimens))
dwf_v, dwi_v, dwc_v, dwo_v = dwf_m, dwi_m, dwc_m, dwo_m
######[关于adam算法的一些参数设置-end]#####
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batchs):
        t = t + 1
        ptr_end = ptr + batchs
        batch_x, batch_y = X[ptr:ptr_end], y_val[ptr:ptr_end]
        F, I, O, C_, C, H, HX = batch_forward(batch_x)  # 正向传播求中间变量
        Dh, Dc = back_forward(batch_y)  # 反向传播求Dh,Dc
        dBf, dBi, dBc, dBo, dWf, dWi, dWc, dWo = dMatrix()  # 反向传播由Dh,Dc求各个权值矩阵和偏置的所有batch和不同时刻的值
        # 权值矩阵和偏置需要将所有时刻的累加和，并且batch范围内求平均
        dbf, dbi, dbc, dbo, dwf, dwi, dwc, dwo = grad_of_Batchs(dBf, dBi, dBc, dBo, dWf, dWi, dWc, dWo)
        dbf_m, dbf_v, dbf = Adamgrad(dbf, t, dbf_m, dbf_v)
        dbi_m, dbi_v, dbi = Adamgrad(dbi, t, dbi_m, dbi_v)
        dbc_m, dbc_v, dbc = Adamgrad(dbc, t, dbc_m, dbc_v)
        dbo_m, dbo_v, dbo = Adamgrad(dbo, t, dbo_m, dbo_v)

        dwf_m, dwf_v, dwf = Adamgrad(dwf, t, dwf_m, dwf_v)
        dwi_m, dwi_v, dwi = Adamgrad(dwi, t, dwi_m, dwi_v)
        dwc_m, dwc_v, dwc = Adamgrad(dwc, t, dwc_m, dwc_v)
        dwo_m, dwo_v, dwo = Adamgrad(dwo, t, dwo_m, dwo_v)
        Wf, Wi, Wc, Wo, Bf, Bi, Bc, Bo = updata(1, dbf, dbi, dbc, dbo, dwf, dwi, dwc, dwo)
    if i % 5 == 0:  # 通过全局数据查看准确率
        H2D = H[:, :, -1]  # 取最后时刻的batch范围内的样本的h输出，注意需要转置从而才能满足维度为[batch，h_dimens]
        sumdet = np.sum(np.square(H2D - batch_y)) / batchs / 2
        print('the square of error:', sumdet)
    if i % 50 == 0:
        print('\nWf:\n', Wf,'\nWi:\n', Wi, '\nWc:\n', Wc,  '\nWo:\n', Wo, '\n')
        print('\nBf:\n', Bf.T,'\nBi:\n', Bi.T, '\nBc:\n', Bc.T,  '\nBo:\n', Bo.T, '\n')
