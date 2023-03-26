# -*- coding: utf-8 -*-

import torch
import numpy as np


def lesson02():

    '''
    张量拼接

    torch.cat(tensors=, dim=0, out=None)
    torch.stack(tensors=, dim=0, out=None)

    tensors: 张量序列
    dim：要拼接的维度

    区别： cat 不会扩张张量的维度， stack会； 

    '''

    t = torch.ones((2,3))

    print("t: {}".format(t))

    t1 = torch.cat([t, t], dim = 0)  # 0行，1 列  扩张行

    t2 = torch.cat([t, t], dim = 1)  # 0行，1 列  扩张列
    print("t1: {}  t1.shape: {} \n t2:{} t2.shape:{}".format(t1, t1.shape, t2, t2.shape))


    t3 = torch.stack([t,t], dim = 0)
    t4 = torch.stack([t,t], dim = 1)
    t5 = torch.stack([t,t], dim = 2)

    t6 = torch.stack([t,t, t], dim = 0)

    print("t3: {}  t3.shape: {} \n t4:{} t4.shape:{}".format(t3, t3.shape, t4, t4.shape))
    print("t5: {}  t5.shape: {} ".format(t5, t5.shape))
    print("t6: {}  t6.shape: {} ".format(t6, t6.shape))



    '''
    张量分割
    
    torch.chunk(input, chunks, dim=0) 
    chunks: 切分的份数 平均切分
    返回张量列表， 不能整除，最后一个张量小于其它张量


    torch.split(tensor, split_size_or_sections, dim = 0)

    split_size_or_sections : 为int时，标识切分的每份长度；为list时，标识按照list中元素大小来切分，总数要等于要切分的那维的维度


    '''

    tt = torch.ones((2, 5))
    list_tensor = torch.chunk(tt, 2, dim = 1)   # 返回两个张量， 按照dim=1来切分;第一个应该是2*3 第二个是：2*2

    for idx, t in enumerate(list_tensor):
        print("the {} , tensor:{} shape:{}".format(idx, t, t.shape))

    
    list_tensor2 = torch.split(tt, 2, dim = 1)   # 每份长度为2，前两个是2*2 但最后一个：2 * 1

    for idx, t in enumerate(list_tensor2):
        print("the {} , tensor:{} shape:{}".format(idx, t, t.shape))


    '''
    张量索引

    torch.index_select(input, dim, index, out=None)

    功能：在dim上，按照index索引数据
    


    '''



'''

增加了google的样本数据，效果没有样本的banlance效果明显？ 原因

样本banlacne的方法？
样本不平衡，会造成什么问题？
答：样本不平衡，在我们构建模型时，看不出来什么问题。往往能得到很高的accuracy，为什么？假设y=1占比1%，模型整体把测试集预测为0，
https://cloud.tencent.com/developer/article/1947624

实验的结果onebox
特征分析的excel


特征穿越 ： https://zhuanlan.zhihu.com/p/402812843  
Traning-servering skew

模型膨胀 ： 

'''
        '''
    矩阵运算

    torch.mul(input, other)
    对位乘法： 点乘
    
    torch.mm(input, mat2)
    矩阵乘法

    torch.matmul(input, other)


    torch.add(input=, other=, alpha = )

    功能：input + other * alpha 



    '''

    t11 = torch.randint(10, (2, 3))

    t12 = torch.randint(5, (2,3))

    t13 = torch.mul(t11, t12)  # t11, t12的维度必须相同

    print("t11: {} \n t12:{}\n t13:{}".format(t11, t12, t13))


    t12_1 = t12.t()
    t14 = torch.mm(t11, t12_1)  # t11的列，必须和t12的行相同： m * n  n * q => m * q

    print("t11: {} \n t12_1:{}\n t14:{}".format(t11, t12_1, t14))


    t15 = torch.ceil(torch.rand(3) * 10)  # 1d 向量： 2个元素
    t16 = torch.ceil(torch.rand(3) * 10) 
    t17 = torch.randint(10, size=(1,2))  # 矩阵： 1*2

    t18 = torch.matmul(t15, t16)  # 向量的点积  标量

    print("t15: {} \n t15 size:{}\n t16:{} t16 size:{} t18:{} t18 size:{}".format(t15, t15.shape, t16,  t16.shape, t18, t18.shape))

    t19 = torch.rand(2,4)
    t20 = torch.rand(4,3) ###维度也要对应才可以乘  矩阵乘法
    print(torch.matmul(t19,t20),'\n',torch.matmul(t19,t20).size())


    t21 = torch.ceil(torch.rand(4) * 10)  # 1d 向量： 4个元素
    t22 = torch.ceil(torch.rand(4, 3) * 10)   # 2d 矩阵
    t23 = torch.matmul(t21, t22)

    ### python 中的广播机制，处理一些维度不同的tensor结构进行相乘操作， 1d * 2d 处理过程如下：
    ### 扩充x =>(,4) 
    ### 相乘x(,4) * y(4,3) =>(,3) 
    ### 去掉1D =>(3)

    print("t21: {} \n t21 size:{}\n t22:{} t22 size:{} t23:{} t23 size:{}".format(t21, t21.shape, t22,  t22.shape, t23, t23.shape))


    t24 = torch.ceil(torch.rand(4, 3) * 10)  # 2d 矩阵： 4 * 3
    t25 = torch.ceil(torch.rand(3) * 10)   # 1d 向量 
    t26 = torch.matmul(t24, t25)  ## 进行点积运算，4个元素的向量

    print("t24: {} \n t24 size:{}\n t25:{} t25 size:{} t26:{} t26 size:{}".format(t24, t24.shape, t25,  t25.shape, t26, t26.shape))



    t27 = torch.rand((2, 3))

    t28 = torch.rand((2,3))

    t29 = torch.add(t27, t28, alpha= 2)

    
    print("t27: {} \n t27 size:{}\n t28:{} t28 size:{} t29:{} t29 size:{}".format(t27, t27.shape, t28,  t28.shape, t29, t29.shape))

    
    
def torch_multiply():

    '''
    参考：https://www.cnblogs.com/HOMEofLowell/p/15963140.html
    '''

    t1 = torch.tensor([1, 2, -1])
    t2 = torch.tensor([1, 2, 1])
    print("t2: {} dtype:{} shape:{}".format(t2, t2.dtype, t2.shape))

    t3 = torch.matmul(t1, t2)   # 结果是个标量 等价：torch.dot 点积运算,输入两个向量元素个数相同
    print("t3: {} dtype:{} shape:{}".format(t3, t3.dtype, t3.shape))

    t33 = torch.dot(t1, t2)
    print("t33: {} dtype:{} shape:{}".format(t33, t33.dtype, t33.shape))

    t4 = torch.unsqueeze(t3, dim=0) # 1个元素向量, 1维张量  升维操作
    print("t4: {} dtype:{} shape:{}".format(t4, t4.dtype, t4.shape))
    t5 = torch.unsqueeze(t4, dim=1) # 1*1个元素矩阵， 2维张量
    print("t4: {} dtype:{} shape:{}".format(t5, t5.dtype, t5.shape))

    t6 = torch.mul(t1, t2)  # 对位相乘，这个没有对应的mamul()运算的
    print("t6: {} dtype:{} shape:{}".format(t6, t6.dtype, t6.shape))

    
    t7 = torch.tensor([[1,2,1],[2,2,1]])  # 2*3 
    t8 = torch.tensor([[1,2,2],[2,2,1],[1,1,1]]) # 3*3
    t9 = torch.mm(t7, t8)  # 得到： 2 * 3  向量

    print("t9: {} dtype:{} shape:{}".format(t9, t9.dtype, t9.shape))

    t10 = torch.matmul(t7, t8)  # 等价：torch.mm 矩阵的乘法，满足 (m * n)  (n * q)

    print("t10: {} dtype:{} shape:{}".format(t10, t10.dtype, t10.shape))




if __name__ == '__main__':

    lesson02()
