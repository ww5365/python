# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import random

def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-x))
    # ds=s(1-s)
    return s

if __name__ == '__main__':

    print("-"*20)
    # numpy 版本
    print(np.__version__)

    # 创建： list(list1,list2,...) ->ndarray: 二维数组
    li = [[1.00, 2, 3, 4], [5, 6, 7, 8]]
    arr1 = np.array(li)
    print(li)
    print("li -> array : \n")
    print(arr1)

    print("array -> li : \n")
    # list <-> ndarray 数据类型转换
    li2 = arr1.tolist()

    # 索引方式
    print(li2[1][:])   # list ndarray支持的索引方式,获取第2行
    print(arr1[1, :])  # ndarray支持此种索引方式，获取第2行
    print(arr1[:, -1])  # ndarray支持此种索引方式
    print(arr1[:, :-1])  # ndarray支持此种索引方式


    # 查看ndarray类型： 基本属性
    print(arr1.shape)
    print(arr1.dtype)
    print(arr1.size)

    # 操作函数：reshape 改变数组的形状
    print("before arr1 shape:\n", arr1)
    arr2 = arr1.reshape((4, 2), order='F')  # 按照列优先的方式，本质要进行深拷贝
    # arr2 = arr1.reshape((4, 2))  # 按照行优先取数，不进行深拷贝
    print("after reshape arr2:\n", arr2)
    arr1[0][0] = 8.0
    print("arr1 update:\n", arr1)
    print("arr2 update or not:\n", arr2)

    # range 和 arrange 区别? arrange步长可以为小数, 且是numpy库来提供
    li2 = list(range(10))
    print(li2)
    arr3 = np.arange(0, 4, 0.5)
    print(arr3)

    # 等差数列
    arr4 = np.linspace(0, 1, 11)
    print("linesapce arr4:\n", arr4)

    # 等比数列
    # a1 = 10^0,.. a5 = q^4 = 10^4 -> q = 10 -> 1,10,100,1000,10000
    arr6 = np.logspace(0, 4, 5)
    print("logspace arr6:\n", arr6)

    # 特殊数组的生成数据接口
    
    arr7 = np.zeros((2, 3))
    print("zeros:\n", arr7)
    arr8 = np.eye(4)  # 对角线上全部为1
    print("eye: \n", arr8)
    arr9 = np.diag([2, 3, 4, 6])
    print("diag:\n", arr9)

    # 随机数生成
    # 直接使用random库，只能生成1个随机数
    arr10 = random.random()  # 生成随机浮点数[0,1]
    arr11 = random.randint(1, 2)  # 生成[1,2]之间的1个数
    print(arr10, arr11)

    # 生成数组随机数？ numpy 提供接口
    print("===========np.random==========")
    arr12 = np.random.random(4)  # 生成一维，4个0-1之间的小数, 随机
    arr13 = np.random.rand(2, 3)  # 生成2*3维数据,float，0~1均匀分布
    arr14 = np.random.randn(3, 2)  # 生成3*2维标准正态分布的数据
    print("test for module random")
    print("{} \n {} \n {} \n".format(arr12, arr13, arr14))

    # 切片

    # #list的切片
    li3 = [1, 2, 3, 4, 5, 6, 7]
    # list倒置
    print(li3[::-1])  # 从最后一个开始，按照步长1来返回数据
    li3.reverse()  # 直接将list中数逆置
    print(li3)

    # 二维索引切片
    arr15 = np.array([[1, 2], [3, 4], [5, 6]])  # shape: (3, 2)
    print("arr15:", arr15[1:3, 1:2])  # 行：[1,3)  列：[1,2)
    print("arr15 reshape: {}".format(arr15.reshape(-1).shape)) # shape: (6,)
    print("arr15 reshape2: {}".format(arr15.reshape(-1, 1).shape)) # shape: (6,1)


    # 三维索引切片
    arr30 = np.random.randint(low=1,high=10,size=(2,4,3)) # 2*4*3
    arr30_res = arr30[0,:,-1] # shape: (4,) 是1维向量, 取第1个矩阵的最后1列
    arr30_res2 = arr30_res.reshape((len(arr30_res), 1))  # shape: (4, 1)  reshape后变成二维
    h_pre = np.zeros((3,1))  # shape: (3,1)
    res = np.row_stack((h_pre, arr30_res2))  # 行合并  M*1  +  N*1 => (M+N)*1  => (7, 1)
    
    res2 = res.reshape(-1)  # (7,1) => (7,) 变成1维, 拉直
    print("res2 {}  shape: {}".format(res2, res2.shape))

    print("arr30: {} \n shape: {}\n".format(arr30, arr30.shape))
    print("arr30_res: {} \n shape: {}\n len: {}".format(arr30_res, arr30_res.shape, len(arr30_res)))
    print("arr30_res2: {} \n shape: {}\n".format(arr30_res2, arr30_res2.shape))
    print("res: {} \n shape: {}\n  reshape(-1): {}\n".format(res, res.shape, res.reshape(-1))) # shape: (7, 1)
    print("sigmod res: {}".format(1.0/(1.0 + np.exp(-res))))  # shape: (7, 1)
    
    # 切片赋值  
    hx = np.zeros((2,7,3)) # shape: (2,7,3) res2:shape:(7,)
    print("init hx: {}\n".format(hx))
    hx[0,:,2] = res2  # (7,) 赋值给第1个矩阵的最后一列
    print("after hx: {}\n".format(hx))
    other = np.array([1,2,3])
    hx[0,0,:] = other  # (3,)赋值给第1个矩阵的第一行
    print("after hx 2: {}\n".format(hx))


    # 改变数组的形状
    li4 = [i for i in range(10)]  # list
    print(li4)
    li5 = np.arange(12)  # array
    print("li5: ", type(li5), li5)
    arr16 = li5.reshape(3, 4, order='C')
    print(arr16)
    print("ndarray dimension: ", arr16.ndim)

    # ndarray数组的展开  flatten
    li7 = arr16.flatten('C')  # 默认就是c，行优先，理解：行的维度铺展开
    li8 = arr16.flatten('F')  # 列优先，理解：列的维度优先展开
    print(li7)
    print(li8)

    # 数组拼接
    li9 = np.arange(20)
    li10 = np.arange(21, 41)
    arr17 = li9.reshape(4, 5)
    arr18 = li10.reshape(4, 5)
    print("arr17: ", arr17)
    print("arr18: ", arr18)

    arr19 = np.concatenate((arr17, arr18), axis=0)
    arr20 = np.concatenate((arr17, arr18), axis=1)

    print("arr19: ", arr19)
    print("arr20: ", arr20)

    # 排序
    li11 = np.arange(12)
    li12 = sorted(li11, reverse=True)

    print("li11: ", li11)
    print("li12: ", li12)

    # 数组相乘 multiply, *, dot
    '''
    multiply: 数组和矩阵的点乘 elem-wise
    * : 数组和矩阵也是点乘elem-wise
    dot : 数组或矩阵乘法
    @ : 数组或矩阵乘法
    '''
    arr21 = np.arange(4).reshape(2, 2)
    arr22 = np.arange(4, 8).reshape(2, 2)

    print(arr21)
    print(arr22)

    print("*运算: ", arr21*arr22)
    print("multiply: ", np.multiply(arr21, arr22))  # 对应位置相乘
    print("dot: ", np.dot(arr21, arr22))  # 矩阵乘法，1维数组就是点积

    li12 = np.array([1, 2, 3])
    print(np.dot(li12, li12))  # 点积

    # dot 矩阵乘法的实列
    arr31 = np.array([[0.25498618, 0.93752152]])  # shape: (1,2)
    arr32 = np.array([[-0.10251392], [ 0.03656583]]) # shape: (2,1)
    arr33 = np.array([[0.49076024]]) # shape: (1,1)
    arr34 = arr31.dot(arr32) + arr33   #   ft:[[0.83426776]

    arr35 = sigmoid(arr34)

    print("arr34: {} \n shape: {} \n arr35: {}\n shape:{}".format(arr34, arr34.shape, arr35, arr35.shape))

    # * 数组的点乘，hadamard积

    arr36 = np.array([1,2,3])
    arr37 = np.array([4,5,6])

    arr38 = arr36 * arr37

    print("*乘法运算符：{} shape: {}\n".format(arr38, arr38.shape))


    # 几个重要的函数
    arr24 = np.array([1, 2, 3, 8, 3, 2])
    print(np.unique(arr24))  # 去重后排序

    arr23 = np.arange(6).reshape(2, 3)
    print("arr23: ", arr23)

    print(np.repeat(arr23, 3, axis=0))  # 以行为维度，每行复制3次
    print(np.repeat(arr23, 3, axis=1))  # 以列为维度，每列复制3次
    print(np.tile(arr23, 3))  # 以列为维度，所有列整体复制3次

    print("np.sum 0: ", np.sum(arr23, axis=0))  # 以行为维度，求和
    print(np.sum(arr23, axis=1))  # 以列为维度，求和

    print(np.mean(arr23))  # 求所有元素的均值
    print(np.std(arr23))  # 标准差 var 的平方根
    print(np.var(arr23))  # 方差
    print(np.var(arr23, axis=0))  # 方差 按照行理解，均值是每个列求均值后计算方差
    print(np.var(arr23, axis=1))  # 方差 按照列计算
    print(np.argmax(arr23))

    print(np.argmin(arr23))  # 全数组中的最小值，索引值
    # axis = 0 表示列维度，结果行结果  返回对应列中，哪行的数据最大, 索引值;每列都是从0开始
    print(np.argmax(arr23, axis=0))
    print(np.argmin(arr23, axis=1))

    print(np.cumsum(arr23))  # 累积和，斐波那契数列

    # ndarray多维数据的拼接

    '''
    np.hstack将参数元组的元素数组按水平方向进行叠加

    np.column_stack: 
    二维：M * N + M * T => M * (N + T) 等价np.hstack , np.concatenate(,axis=1)
    三维：(M,N,Q) + (M,T,Q) => (M, N+T, Q)

    np.raw_stack:  
    二维：M * N + T * N => (M + T) * N 等价np.vstack
    三维：(M,N,Q) + (T,N,Q) => (M+T, N, Q)

    '''

    # 二维矩阵 
    arr1 = np.array([[1,3], [2,4] ])
    arr2 = np.array([[1,4], [2,6] ])
    res = np.hstack((arr1, arr2))
    res2 = np.column_stack((arr1, arr2))
    print("hstack: {}".format(res))
    #[[1 3 1 4]
    #[2 4 2 6]
    print("column stack: {}".format(res2))

    # 三维矩阵
    arr40 = np.arange(0, 24).reshape((2,3,4))
    arr41 = np.arange(24, 48).reshape((2,3,4))

    print("arr40:\n {} \n arr41:\n {}\n".format(arr40, arr41))  # shape: (2,3,4)

    arr42 = np.column_stack((arr40, arr41)) # shape: (2,6,4)
    arr43 = np.row_stack((arr40, arr41)) # shape: (4,3,4)

    print("arr40 + arr41 column: \n {} \n shape: {} \n".format(arr42, arr42.shape)) # shape: (2,6,4)
    print("arr40 + arr41 raw : \n {} \n shape: {} \n".format(arr43, arr43.shape)) # shape: (2,6,4)













