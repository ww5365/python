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
    



if __name__ == '__main__':

    lesson02()
