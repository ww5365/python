                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           # -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import shutil  # shutil模块是对os模块的补充，主要针对文件的拷贝、删除、移动、压缩和解压操作
import random  # 使用random.shuffle(x)

from torch.utils.data import IterableDataset

from torch.utils.data import DataLoader


import torch.nn as nn

torch.manual_seed(1)

# random.seed(1)


def lesson02_extra():

    '''
    什么是代器数据？

    for x  in list_var : 这种for循环支持的迭代器数据

    迭代器数据:
    
    每个元素的获取，本质都会主动调：iter 和 next 函数

    '''

    li = [1,3,5,7,9]
    for elem in li:
        print(elem)

    # 上面的迭代器数据的实质

    li_iter = iter(li)
    elem = next(li_iter, None)
    while elem is not None:
        print(elem)
        elem = next(li_iter, None)


    '''
    IterableDataset:  接口，需要实现
    __init__
    __iter__
    __next__

    '''

    class MyDataset(IterableDataset):
        
        def __init__(self):
            print("...init...")
        
        def __iter__(self):
            print("...iter...")
            self.n = 1
            return self

        def __next__(self):

            print('next...') #获取下个元素
            x = self.n
            self.n += 1

            if x >= 100 :
                raise StopIteration
            return x

    dl = DataLoader(MyDataset(), batch_size = 5)

    # 可迭代对象在初始化中会调用一次​​__init__​​​方法
    # 在获取迭代器时会调用一次​​__iter__​​​方法，之后在获取元素时，每获取一个元素都会调用一次​​__next__​​方法

    for i, data in enumerate(dl):
        print(i, data)

    d1 = random.uniform(0, 1)

    mask = np.random.choice([0,1,2], size= (2,3,1), p=[0.9, 0.05, 0.05])
    mask2 = np.repeat(mask, 3, axis=2)

    print("d1 type: {}".format(d1))
    print("mask type: {}".format(mask))
    print("mask2 type: {}".format(mask2))

def lesson03_extra():

    '''
    nn.Linear  全连接网络使用

    参考：
    https://blog.csdn.net/zhaohongfei_358/article/details/122797190
    https://zhuanlan.zhihu.com/p/557253923

    '''

    input = torch.randn(10, 3)  # 10个样本，每个样本3个特征

    print("input: {} dtype:{} \n".format(input, input.dtype))

    model = nn.Linear(3, 7)  # 3个特征(神经元)， 输出7个特征
    output = model(input)

    for para in model.parameters():
        print("modle parameters: {} \n".format(para)) 
        # 有两个参数：W ： 7 * 3 tensor 运算时转置
        # bias：1 * 7 

    print("output:{} \n".format(output))
    print(output.shape)



if __name__ == '__main__':

    # lesson02_extra()
    lesson03_extra()

    t1 = torch.tensor([-1.5256, -0.7502, -0.6540])
    t2 = torch.tensor([-0.3065,  0.1698, -0.1667])

    t3 = torch.matmul(t1, t2)

    print("t3: {}".format(t3 + 0.0843))


