# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd

from functools import reduce


def foo():
    '''
    yield how to work?
    类似return 但又有不同; 有yield的函数，是生成器了；使用next()/send()调用执行

    ref:
    https://blog.csdn.net/mieleizhi0522/article/details/82142856

    '''
    print("start fun ----")
    while True:
        res = yield 4
        print("res:", res)


def lambda_use():
    '''
    1、lambda 表达式： 创建匿名函数
    lambda [arg1 [,arg2,.....argn]]:expression

    2、函数：
    def fun():
        statement
    lambda和函数区别：
    1、简洁，可读性，效率？为什么效率高？
    2、lamda是一次性的？在特定作用域内，类似局部变量，会被释放；
    '''

    # lambda + map :

    '''
    map()接收一个函数f和一个或多个序列list，并通过把函数f依次作用在list的每个元素上，得到一个新的迭代器（Python2是列表）并返回。

    语法：
    map(function, iterable, ...)
    function -- 函数，如果iterable有多个序列，则function的参数也对应有多个
    iterable -- 一个或多个序列
    
    参考：https://www.cnblogs.com/gdjlc/p/11483646.html
    
    '''

    li = list(map(lambda x, y: x * y, range(1, 10), range(1, 5)))
    print("lamda_use:", li)

    '''
    
    reduce()用传给 reduce 中的函数 function（必须有两个参数）先对集合中的第 1、2 个元素进行操作，得到的结果再与第三个数据用 function 函数运算，最后得到一个结果。
    在Python2中reduce()是内置函数，Pytho3移到functools 模块 : from functools import reduce

    语法：
    
    reduce(function, iterable[, initializer])
    function -- 函数，有两个参数,必须有两个参数
    iterable -- 可迭代对象
    initializer -- 可选，初始参数
    
    '''

    res = reduce(lambda x, y: x + y, [2, 3, 4])  # 2+3+4

    print("reduce result: ", res)

    # lambda + filter :
    # filter (function, sequence)
    # 对 sequence 中的item依次执行function(item)，
    # 将结果为 True 的 item 组成一个 List/String/Tuple（取决于 sequence 的类型）并返回;

    li2 = list(filter(lambda x: x % 3 == 0, range(1, 10)))
    print("li2", li2)

    # lambda + dataFrame + apply
    # df.apply(function)
    # DataFrame.apply() 函数则会遍历每一个元素，对元素运行指定的 function
    # 参考：https://www.jianshu.com/p/4fdd6eee1b06

    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    df = pd.DataFrame(matrix, columns=list('xyz'), index=['a', 'b', 'c'])
    print(df)

    # 对某列求平方
    df1 = df.apply(lambda x: (x * x) if x.name in ['x', 'y'] else x)
    print(df1)

    # 对某行求平方
    df2 = df.apply(lambda x: (x * x) if x.name == 'a' else x, axis=1)
    print(df2)


if __name__ == '__main__':

    # test yield
    g = foo()
    # print(g)
    print(next(g))
    print('*' * 20)
    print(next(g))

    # test lambda
    lambda_use()
