# -*-coding:utf8 -*-
import sys
import numpy as np

from multiprocessing import Array, Value

if __name__ == '__main__':

    '''
    0、numpy 基础：

    是 Python 语言的一个扩展程序库，支持大维度数组与矩阵运算；
    主要用于数组计算，包含：
    一个强大的N维数组对象ndarray
    广播功能函数
    整合 C/C++/Fortran 代码的工具
    线性代数、傅里叶变换、随机数生成等功能

    数组创建：参考https://www.runoob.com/numpy/numpy-array-creation.html
    1、list -> numpy
    2、

    '''
    # list转numpy ndarray
    li = [[1, 2], [3, 4], [5, 6]]
    np1 = np.array(li)
    print("list to numpy: \n", np1, type(li), type(np1))

    '''
    #  empty, zeros, ones
    numpy.empty(shape, dtype = float, order = 'C')
    shape: 数组形状
    dtype： int np(int16, int32, int64) float(float32 float64) complex 
    order: C 语言行优先； F fortan是列优先;
    
    '''
    arr1 = np.empty((3, 4), dtype=np.int16, order='C')
    print("np.empty:\n", arr1)   # 随机内存的值

    arr2 = np.zeros((2, 3), order='C')
    print("np.zeros\n", arr2)   # zeros 数组

    shape_demo = (3, 4)
    arr3 = np.ones(shape=shape_demo, dtype=complex)
    print("np.ones\n", arr3)   # ones 数组

    '''
    范围：创建ndarray
    numpy.arange(start, stop, step, dtype)
    start: 默认是0
    
    
    '''

    arr4 = np.arange(10)
    print("np.arrange:\n", arr4)

    arr5 = np.arange(10, 20, 2, dtype=np.float)
    print("np.arrange float:\n", arr5)

    '''
    1、 使用numpy产出：随机数

        np.random.randint
        np.random.random
        np.random.uniform
    '''

    # randint(low=x, high=y, size=z or (m,n), d) 产出[low, high)之间的随机值，产出个数z或（m,n）

    indcies1 = np.random.randint(low=0, high=4, size=10)
    indcies2 = np.random.randint(low=0, high=4, size=(2, 3))

    print("np.random.randint: ", indcies1)
    print("np.random.randint matrix：", indcies2)

    tmp1 = np.random.random(3)  # float类型随机数
    tmp2 = np.random.uniform(low=0 / 10, high=1 / 10,
                             size=5)  # [0, 0.1)之间的均匀分布

    print("np.random.random: ", tmp1)
    print("np.random.uniform: ", tmp2)

    '''
    2、numpy类型数组，转为ctypes类型
    '''
    # ctypes 类型使用: c的数组类型; Array实现线程安全
    tmp = np.ctypeslib.as_ctypes(tmp2)
    print("ctypes tmp:", tmp)  # <c_double_Array_5 object at 0x119c8cb00>
    print("dir tmp:", dir(tmp), tmp._type_)
    tmp = Array(tmp._type_, tmp, lock=False)
    # <c_double_Array_5 object at 0x11a4ce050> 实际都是c double array类型，不过Array后，可以控制进线程的安全性；
    print("Array tmp: ", tmp)
    print("Array tmp[0]: ", tmp[0])

    '''
    3、numpy axis 的使用
    Axis就是数组层级
    设axis=i，则Numpy沿着第i个下标变化的方向进行操作 相当于sql中group by
    
    axis=0，表示沿着第 0 轴进行操作，即对每一列进行操作；axis=1，表示沿着第1轴进行操作，即对每一行进行操作
    
    '''

    num1 = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])

    print("numpy array: ", num1)

    print("numpy sum axis=1：", np.sum(num1, axis=1))  # 按照(x_i,j)j,行统计
    print("numpy sum axis=0：", np.sum(num1, axis=0))  # 按照(x_i,j)i,列统计
    '''
    4、numpy 计算内积 和矩阵乘法 dot
    '''

    num2 = np.array([[1, 2, 3, 1], [1, 1, 0, 0]])
    num3 = np.array([[0, 1, 2, 3], [0, 1, 2, 0]])
    print("numpy array num2:", num2)

    num3 = num3.transpose()
    dot_res2 = np.dot(num2, num3)
    print("numpy dot res2:", dot_res2)
    print("max pooling: ", dot_res2.max())
    '''
    5、计算范数： 矩阵或向量
    '''
    print(np.linalg.norm(num2))  # 二阶范数
    print(np.linalg.norm(num3))
    print(np.linalg.norm(num2, axis=0))  # 保留行维度，按照列计算范数

    print(dot_res2 / (np.linalg.norm(num2) * np.linalg.norm(num3)))
