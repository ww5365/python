#-*- coding: utf-8 -*-

import torch
import numpy as np


def lesson01():
    
    # 创建张量 torch.tensor
    arr = np.ones((3,3))  # 创建3*3的ndarray
    
    print("np datatype: ", arr.dtype)
    t1 = torch.tensor(arr, device='cpu')  # torch.float64  和 ndarray数据类型相同
    print(t1)
    
    # 使用numpy创建张量  torch.from_numpy(ndarray)

    arr2 = np.array([[1,2,3],[4,5,6]])
    
    t2 = torch.from_numpy(arr2)  # t2 和 arr2 是共享内存的
    
    print(t2)  # torch.int32  
    
    arr2[0,1] = 9
    print(t2)  # 因为共享内存，t2也跟着发生变化
    
    
    '''
    依据数值创建：
    torch.zeros(*size, out, layout, device, requries_grad)
    
    size:(3,4)
    out:接收创建的张量
    layout： 内存布局？strided sparse_coo
    device: gpu/cpu
    requires_grad: 是否计算梯度
    
    torch.zeros_like(input, layout, device, requries_grad)
    
    input: 依据input的形状创建张量
    layout： 内存布局？strided sparse_coo
    device: gpu/cpu
    requires_grad: 是否计算梯度
    
    
    同理类似：
    
    torch.ones
    torch.ones_like
    
    torch.full
    torch.full_like
    
    torch.full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    fill_value: 是要填充的值
    
    '''
    out_t = torch.tensor([1])
    t3 = torch.ones((3,3), out=out_t)
    
    print(t3)
    print(out_t)
    print(id(t3), id(out_t))  ## t3 和 out_t 是同一个，内存地址是一样的
    
    t4 = torch.full((3,3), 188)  ## 创建3*3的全部是188的张量 
    print(t4)
    
    # 创建等差的1维张量, [start, end)  
    # torch.arange(start = 0, end, step = 1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    
    
    t5 = torch.arange(1,10)  # 1 ~ 9
    
    print(t5)
    
    # 创建均分的1维张量 : (end - start)/(step - 1)   
    # torch.linspace(start = 0, end, steps = 2, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) steps 是数列的长度
    
    t6 = torch.linspace(2, 10, 6)  # 张量长度为6
    
    print(t6)
    
    # 创建对数均分的1维张量: logspace(start, end, base, steps=100, out=None)
    # steps: 数列长度  数列中的值： base^(start + (end-start)/(steps-1))
    t7 = torch.logspace(1, 10, 10)  # t7的值是x，x取log后，得到1~10的10个点
    print(t7) # tensor([1.0000e+01, 1.0000e+02, 1.0000e+03, 1.0000e+04, 1.0000e+05, 1.0000e+06, 1.0000e+07, 1.0000e+08, 1.0000e+09, 1.0000e+10])

    # torch.eye(n, m, out=Noe, dtype=None, layout, device, reqires_grad)   n: 张量行 m：张量的列 默认方阵
    
    t8 = torch.eye(4)
    print(t8)   # 4*4 的对角阵

    '''
    依概率分布来创建张量
    '''

    # torch.normal(mean, std)  高斯分布, 使用mean均值和std标准差，来获取张量

    mean = torch.arange(1, 5, dtype=torch.float)
    std =  torch.arange(1, 5, dtype=torch.float)
    t9 = torch.normal(mean, std)  # 有4个数，第1个数是从均值和方差为(1,1)的正态分布中采样的
    print(t9)

    mean = 0.0  # 标量
    std = 1.0
    t10 = torch.normal(mean, std,size=(4,))  # 都是标量的情况下，设置size参数，生成了长度为4的张量
    print(t10)

    # 标准正态分布，采样获取张量
    # torch.randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    # torch.randn_like(input, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)

    t11 = torch.randn(4)
    print("t11: {}".format(t11))


    #  均匀分布，采样获取张量
    # [0,1)区间上生成均匀分布
    # torch.rand(*size, out=None ,dtype=None, layout=, device=None, requires_grad=False)
    # torch.rand_like(input=, ..)

    # [low, high)区间上生成整数的均匀分布
    # torch.randint(low=0, high=, size=, out=None, dtype=None, layout=, device=None, requires_grad=False)
    # torch.randint_like(input, high, ..)

    t12 = torch.rand(4)  # 长度为4的，[0, 1)的均匀分布
    print(t12)

    t13 = torch.randint(1, 10, size=(10,))  # [1,10) 均匀分布，长度为10
    print(t13)

    # torch.randperm(n)  生成0~n-1的随机全排列
    t14 = torch.randperm(10)  
    print(t14)

    # torch.bernoulli(input, *, generator=None, out=None)  以input为概率，生成伯努利概率分布
    # input是float概率值，输出size和input相同，每个元素是0/1

    t15 = torch.tensor([0.3, 0.6, 0.2, 0.8])
    print(t15)
    t16 = torch.bernoulli(t15)
    print(t16)
    
    
    
    



if __name__ == '__main__':
    
    lesson01()