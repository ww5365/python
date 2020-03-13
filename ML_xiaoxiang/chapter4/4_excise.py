# -*- coding=utf8 -*-
import numpy as np
import os
import sys
sys.path.append('/Users/wangwei69/workspace/github/python/ml_xiaoxiang/chapter4')
import exciselib as elib

if __name__ == '__main__':
    print "hello: %s"%("word")

    print __file__,__name__

    ##引入其它 pyton 文件中的函数
    elib.test()

    ##数据类型转换：numpy 提供的能力
    L = [1.1,2.2,3.3]
    a = np.array(L, dtype= float)
    print a.dtype, np.shape(a)
    b = a.astype(np.int)
    print b.dtype
    print a, b

    ##produce random digits,range:[0,1)
    V = np.random.rand(2,3) ## 2*3 dimension matrix
    print V

    ##produce regular digitals 
    X1 = np.arange(0, 10, 0.5) ##产生[0,10]之间，步长为0.5，浮点数数组
    print "X1=",X1










