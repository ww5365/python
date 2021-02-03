# -*- coding: utf-8 -*-

import os
import builtins
import logging

'''
参考:https://www.runoob.com/python3/python3-namespace-scope.html
'''

total = 0  # 全局变量


def sum(arg1=0, arg2=0):
    '''
    @description: 怎么让全局变量在函数中生效？ global
    '''
    global total  # 函数中使用全局变量
    total = arg1 + arg2   # 是一个局部变量
    print("local variant in funciton: ", total)
    return total


def outer():
    '''
    @description: 怎么让外层的变量在嵌套的函数中生效? nolocal
    @param {type}
    path: 数据文件所在目录
    @return:list  包含session_id, role, conten
    '''
    num = 10

    def inner():
        nonlocal num
        num = 100
        print("inner num: ", num)

    inner()
    print("outer num: ", num)


if __name__ == '__main__':

    print("begin to run file -- %s" % (__file__))

    # 打印：pyton3中预定义的哪些变量
    print("python builtins variants:%s" % (dir(builtins)))

    # 变量作用域类型：local -> enclosing(非局部，函数嵌套时，外层函数作用域) ->global -> builtin

    # #全局变量total 是否生效
    print("sum result: {} total: {}".format(sum(10, 20), total))

    # #global nonlocal 变量
    outer()
