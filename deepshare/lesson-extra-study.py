# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import shutil  # shutil模块是对os模块的补充，主要针对文件的拷贝、删除、移动、压缩和解压操作
import random  # 使用random.shuffle(x)

from torch.utils.data import IterableDataset

from torch.utils.data import DataLoader


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

    # 可迭代对象在初始化中会调用一次​​__init__​​​方法，在获取迭代器时会调用一次​​__iter__​​​方法，之后在获取元素时，每获取一个元素都会调用一次​​__next__​​方法
    
    for i, data in enumerate(dl):
        print(i, data)

    


if __name__ == '__main__':

    lesson02_extra()


