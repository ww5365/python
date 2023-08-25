                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           # -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import shutil  # shutil模块是对os模块的补充，主要针对文件的拷贝、删除、移动、压缩和解压操作
import random  # 使用random.shuffle(x)
import numpy as np
import torch.optim as optim

from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import string 
import unicodedata


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

    # view 改变shape 共享内存
    t1 = torch.tensor([1.,2.,3.,4.,5.,6.]) # [6, ] 
    t1_view = t1.view(t1.size()[0], -1) # shape : [6, 1] 
    t1_view[0][0] += 10
    print("tensort: t: {} view: {} shape: {} {}".format(t1, t1_view, t1_view.shape, t1_view.size()))

    # flatten 拉平维度
    t2 = torch.randn(2,4,2)
    t22 = torch.flatten(t2)
    print("t2: {} t22:{}".format(t2.shape, t22.shape)) # [2,4,2] -> [16,]


def lesson06_extra():

    # torch.linespace 生成的数的维度
    t1 = torch.linspace(-1,1,10).unsqueeze_(dim=1)  #  linspace返回[10,] unsqueeze后得到[10, 1]
    print("tensor t1: {} shape: {}".format(t1, t1.shape))


    # Modulist

    net = nn.ModuleList([nn.Linear(10, 10) for i in range(3)])

    for idx, m in enumerate(net.modules()):

        print("idx: {} module:{}".format(idx, m))  # 总共4个模型 nn.ModuleList 1, nn.Linear: 3

def lesson07_extra():

    # maxpool2d 使用
    t1 = torch.tensor([[[1.,1.,2.,4.],[5.,6.,7.,8.],[3.,2.,1.,0.],[1.,2.,3.,4.]]])
    print(t1, t1.shape)
    maxPool = nn.MaxPool2d(2,2)   # 主要验证以下的两种参数设置方式有什么不同？ 从现在看结果都是一致的
    maxPool3 = nn.MaxPool2d((2,2))
    t2 = maxPool(t1)
    t3 = maxPool3(t1)
    print(t2)
    print(t3)

    # 张量切片
    print("省略号: {}".format(t1[0,...]))  # 省略号的作用? 切片出第1个维度的所有值

    # torch.max 使用
    print("max: {}".format(torch.max(t1, 1))) #  返回两组tensor,第1组：按照dim = 1，每行的最大值；第2组：每行最大值的索引位置

    # tensor变脸直接计算sum
    print(t1[0,...].sum())  # 计算所有元素的和，并返回tensor值
    print(t1[0,...].sum().numpy())  # 将tensor转为numpy, 共享内存的

    # np.argsort

    li = [5, 4, 8, 1]
    li2 = np.argsort(li)[::-1] # 值从大到小的下标索引[3,1,0,2]
    li2_str = ",".join(map(str, li2)) # 把list中所有整数值转成字符串，并使用“,”拼接
    print("li2: {} {}".format(li2, li2_str))

    # 创建一个大小为（1, 1, 4, 4）的张量
    x = torch.tensor([[
        [[1, 2, 0, 3], [2, 1, 1, 0], [0, 1, 2, 1], [0, 1, 0, 2]]
        ]])

    x = x.to(torch.float32)

    # 使用大小为（2, 2）的池化窗口进行最大池化
    pool = F.max_pool2d(x, kernel_size=2)
    print("shape: {} {} pool: {}".format(x.shape, pool.shape, pool))

    pool2 = pool.view(pool.size()[0], -1)

    print("pool2: {} {}".format(pool2, pool2.shape))



def lesson08_extra():

    # string.ascii_letters string.digits
    str_letter = string.ascii_letters + string.digits
    print("all chars and digitals: {}".format(str_letter))

    # 文件名.扩展名 把扩展名取出来
    path = r"c:\test\file1.txt"
    print("path: {} {} ".format(path, os.path.basename(path)))
    print("ext name: {}".format(os.path.splitext(os.path.basename(path))[1]))
    
    # unicodedata.normalize 把UNICODE字符串转换为普通格式的字符串，具体格式支持NFC、NFKC、NFD和NFKD格式
    #  aä 实际是等价表述，不同的码表示，normalize之后，归一a
    print(unicodedata.normalize('NFD', u'aあä').encode('ascii', 'ignore'))  # aa

    title = u"Klüft skräms inför på fédéral électoral große"
    print(unicodedata.normalize('NFKD', title))


    #\xa0、\u3000等字符的解释及去除
    s = 'T-shirt\xa0\xa0短袖圆领衫,\u3000体恤衫\xa0买一件\t吧'
    print(unicodedata.normalize('NFKD', s)) # T-shirt  短袖圆领衫, 体恤衫 买一件 吧 

    # unicodedata.category(chr) 判断字符类型
    print(unicodedata.category('四'))  #Lo  letter other
    print(unicodedata.category('8')) #Nd  Number, Decimal Digit
    print(unicodedata.category('a')) #Ll    Letter, Lowercase

    # 负对数似然损失函数（Negative Log Likelihood）
    t1 = torch.tensor([[1.,1.,1.],[2.,3.,4.],[5.,1.,3.]])
    nl1 = nn.NLLLoss(reduction="sum")  # softmax(x) + log(x) + nn.NLLLoss ====> nn.CrossEntropyLoss
    loss = nl1(t1,torch.tensor([1, 0, 2]))  # -(1, 2, 3)/3

    print("loss: {}".format(loss))

    print(random.randint(0, 3))
    
    # tensor的topk

    t2 = torch.tensor([2., 1.2, 3., 4.])

    print(t2.topk(1))  # 返回两个tensor，(values=tensor([4.]),indices=tensor([3])) 第1个是最大值的张量，第2个是最大值下标的张量


if __name__ == '__main__':

    # lesson02_extra()
    # lesson03_extra()

    # lesson06_extra()

    # lesson07_extra()

    lesson08_extra()


    exit(0)

    t1 = torch.tensor([-1.5256, -0.7502, -0.6540])
    t2 = torch.tensor([-0.3065,  0.1698, -0.1667])

    # 向量内积  
    t3 = torch.dot(t1, t2)
    t33 = torch.matmul(t1, t2)
    t333 = t1 * t2   # 所有元素并行运算,相当于 torch.mul(t1, t2)  对位相乘

    print("t3: {}  t33: {} t333: {}".format(t3, t33, t333))

    # 列表表达式
    li = ["test.jpg", "test2.doc", "test3.xml"]
    print(li)
    li2 = list(filter(lambda x : x.endswith(".jpg"), li))
    print(li2)

    for i in range(3):
        print("range: {}".format(i))

    print(np.arange(1, 3))
    print(np.arange(1, 3) * 2)
    
    # transpose  转置操作

    '''
    torch.transpose(input, dim0, dim1, out=None) → Tensor

    input (Tensor) – 输入张量，必填
    dim0 (int) – 转置的第一维，默认0，可选
    dim1 (int) – 转置的第二维，默认1，可选

    注意：
    1. dim 不区分数的大小  transpose(0, 2) 等价 transpose(2, 0)
    2. 只操作2个维度的数据交换，torch.transpose(x)合法， x.transpose()合法
    2. 返回值，是copy新数返回，不是原位的操作
    '''

    x = torch.randn(2,3,1)
    print("x tensor: {}".format(x))
    x1 = x.transpose(2,0)  # 0维和2维 交换数据 2 * 3 * 1 -》 1 * 3 * 2
    x2 = x.transpose(0,2)
    print("x1 tensor: {}".format(x1))
    print("x2 tensor: {}".format(x2))

    t = torch.randn(size=(2,2,3,4))
    print("t tesnsor: {}".format(t))  # 2*2*3*4

    t1 = t[0:1, 0:2, ...]
    print("t1 tesnsor: {} shape:{}".format(t1, t1.shape))  # 1*2*3*4
    
    t2 = t[0, 0:2, ...]
    print("t2 tesnsor: {} shape:{}".format(t2, t2.shape))  # 2*3*4 会自动的squeeze()

    arr = np.array([[ 0.1103, -2.2590,  0.6067, -0.1383],
        [ 0.8310, -0.2477, -0.8029,  0.2366],
        [ 0.2857,  0.6898, -0.6331,  0.8795]])

    print("mean: {}".format(np.mean(arr)))

    print("std: {}".format(np.std(arr)))

    weight = np.array([[ 1.8793, -0.0721,  0.1578, -0.7735],
        [ 0.1991,  0.0457,  0.1530, -0.4757],
        [-0.1110,  0.2927, -0.1578, -0.0288],
        [ 2.3571, -1.0373,  1.5748, -0.6298]])

    print("weight mean: {}".format(np.mean(weight)))

    print("weight std: {}".format(np.std(weight)))

    print("矩阵乘法: {}".format(np.dot(arr, weight.T)))

    li3 = [-2,1,-2,1,-1.8,1.1,-2.2,1.3]
    np1 = np.array(li3)

    print("np1 mean: {}  var: {}".format(np.mean(np1), np.var(np1)))    



    
