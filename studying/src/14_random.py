# -*-coding:utf8 -*-
import sys
import random
import uuid

'''
基本数据类型
常用数据类型

参考：
https://www.pythonf.cn/read/128945


'''


if __name__ == '__main__':

    '''
    1、random 模块使用
    '''

    # 获取64位整数
    for i in range(10):
        long_value = random.getrandbits(64)
        print("random i:{seq} value:{value}".format(seq=i, value=long_value))

    print("*" * 20)
    print(uuid.uuid1().int >> 64)
