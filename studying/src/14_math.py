# -*- encoding: utf-8 -*-

import math
import cmath
from decimal import Decimal


'''
浮点数的运算

1、Python math 模块提供了许多对浮点数的数学运算函数。
Python cmath 模块包含了一些用于复数运算的函数。
cmath 模块的函数跟 math 模块函数基本一致，区别是 cmath 模块运算的是复数，math 模块运算的是数学运算。

2、decimal Decimal : 高精度的浮点运算模块

'''

if __name__ == '__main__':

    # math提供的函数和常量

    print(dir(math))

    # 使用math进行算术运算
    x = math.pi
    y = math.e
    t = math.tau  # tau = 2pi
    print("pi and  e  tau:", x, y, t)

    x2 = math.fabs(-1.000202)  # 返回绝对值,类型：float
    print("x2: ", x2)

    x3 = math.log(y)  # 默认使用e作为底数
    print("x3: ", x3)
