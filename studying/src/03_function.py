# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import types

from functools import reduce


'''
函数参数：
1、可变对象： 列表，字典
   不可变对象：整数，字符串，元组; 
   如 fun(a)，传递的只是 a 的值，没有影响 a 对象本身。
   如果在 fun(a) 内部修改 a 的值，则是新生成一个 a 的对象。
'''


def test(val: int):

    print("before id val: ", id(val))
    val = 10
    print("after id val: ", id(val))


def foo():
    '''
    yield how to work?
    类似return 但又有不同; 有yield的函数，是生成器了；使用next()/send()调用执行

    ref:
    https://blog.csdn.net/mieleizhi0522/article/details/82142856


    带有 yield 的函数就是一个 generator，它和普通函数不同，生成一个 generator 看起来像函数调用，
    但不会执行任何函数代码，直到对其调用 next()（在 for 循环中会自动调用 next()）才开始执行。
    虽然执行流程仍按函数的流程执行，但每执行到一个 yield 语句就会中断，并返回一个迭代值，下次执行时从 yield 的下一个语句继续执行。
    看起来就好像一个函数在正常执行的过程中被 yield 中断了数次，每次中断都会通过 yield 返回当前的迭代值。

    ref:https://www.runoob.com/w3cnote/python-yield-used-analysis.html

    '''
    print("start fun ----")
    while True:
        res = yield 4
        print("res:", res)


def foo2():
    '''
    1、iter(object[, sentinel])
    iter() 函数用来生成迭代器
    返回：迭代器对象
    参数：
    object -- 支持迭代的集合对象。
    sentinel -- 如果传递了第二个参数，则参数 object 必须是一个可调用的对象（如，函数）
    此时，iter 创建了一个迭代器对象，每次调用这个迭代器对象的__next__()方法时，都会调用 object。

    2、next(iterable[, default])
    iterable -- 可迭代对象
    default -- 可选，用于设置在没有下一个元素时返回该默认值，如果不设置，又没有下一个元素则会触发 StopIteration 异常。

    https://www.runoob.com/python/python-func-next.html

    '''

    li = [2, 3, 4, 5, 6]
    li_iterator = iter(li)   # 列表转成迭代器对象
    print(type(iter(li)))  # <class 'list_iterator'>
    print(type(li_iterator))

    while True:
        try:
            # 获取下一个值
            elem = next(li_iterator)
            print(elem)
        except StopIteration:
            # 遇到stopiteration就退出循环
            break


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


def fun(arg1, *arg2, **arg3):
    '''
    任意个数的参数导入
    单引号： 以元祖形式导入参数; 函数使用时，表示解压参数列表：li = [1, 2]  fun(*li) -> fun(1,2)
    双引号： 以字典形式的导入参数
    '''
    print(arg1)
    print(arg2)
    print(arg3)


if __name__ == '__main__':

    print('--' * 30)
    # 生成器相关测试
    # test yield
    g = foo()
    # print(g)
    print(next(g))

    print(type(g))  # <class 'generator'>
    print(isinstance(g, types.GeneratorType))  # 迭代器类型 true

    print('--' * 30)
    foo2()
    print('--' * 30)

    # test  参数
    test(15)

    # test lambda
    print('--' * 30)
    lambda_use()

    # 函数的参数
    fun(1, 2, 3, 4, e=5, f=6)
