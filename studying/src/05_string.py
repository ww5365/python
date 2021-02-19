# -*- coding: utf-8 -*-

import os
import sys

if __name__ == '__main__':

    # 常用函数使用

    str0 = ' this is a test ! '
    print(str0.strip().strip('!'))  # 去除首位空格以及!号

    # 打印单引号
    str1 = "'test string'"
    print(str1)

    # 使用rindex进行字符串的截取
    str1 = 'sz-t-xt'
    pos = str1.rindex('-')
    print('rindex pos', pos)
    print("befor pos:", str1[:pos], str1[pos + 1:].upper())

    # 替换字符串中某个字符

    print("before replace: ", str1)
    str1 = str1.replace("-", ' ')
    print("after replace: ", str1)

    # 内置函数的使用:
    # type(变量): 返回某个变量的类型；只认继承类自己；
    # isinstance(变量，类型): 判断某个变量类型,  返回True/False；继承类变量属于某个父类

    li1 = []
    li1.append(1)
    li1.append(2)
    print("type:isinstance", type(li1) == list, isinstance(li1, list))

    # format
    str2 = 'I Love {}'
    print(str2.format("China"))

    str2 = 'I Love {country}'
    print(str2.format(country="China"))

    # 字符串之前加上r ，表示这个字符串是raw string ,比如，字符串里面的: \n 是两个字符，不是换行符
    str3 = r'd:\n\files\t\test'
    print("str3=", str3)

    # 多行字符串,使用三引号，保留了所见即所得的格式

    str4 = '''
    select * from talble1
    where col1 = "test"
    and   col2 = 3
    '''

    print('str4=', str4)

    '''
    字符串编码:
    在Python 3版本中，所有的字符串都是使用Unicode编码的字符串序列。
    Python 3最重要的新特性之一是对字符串和二进制数据流做了明确的区分。
    文本总是 Unicode，由 str 类型表示，二进制数据则由 bytes 类型表示。
    Python 3 不会以任意隐式的方式混用 str 和 bytes ，你不能拼接字符串和字节流，
    也无法在字节流里搜索字符串（反之亦然），
    也不能将字符串传入参数为字节流的函数（反之亦然）。
    
    encode : 字符编码 -》 字节编码
    decode : 字节编码-》 字符编码
    '''
    print(sys.getdefaultencoding())

    s = '中国'  # Python3中定义的字符串（str）默认就是 unicode字符串
    print(s, type(s))  # 中国 <class 'str'>
    # s1 = s.decode("utf-8")  # AttributeError: 'str' object has no attribute 'decode' ← Python3中字符串不再有decode方法

    s2 = s.encode("gbk")  # 将字符串（str）用 “gbk字符编码” 编码为 “gbk字符编码的字节”
    print(s2, type(s2))  # b'\xd6\xd0\xb9\xfa' <class 'bytes'>
    s22 = s2.decode('gbk')  # 将 “字节串” 用 “gbk字符编码” 解码 为 “字符串（str)”
    print(s22, type(s22))  # 中国 <class 'str'>

    s3 = s.encode("utf-8")
    print(s3, type(s3))  # b'\xe4\xb8\xad\xe5\x9b\xbd' <class 'bytes'>
    s33 = s.encode("utf-8").decode('utf-8')
    print(s33, type(s33))  # 中国 <class 'str'>

    # 字符串分割： split partition
    line = "wang  wei nihao  a 北京"
    tokens = line.split(
        ' ')  # 默认split()也是空格分割；但和split(' ')有区别，对于连续空格情况，这个只考虑1个，其它空格保留
    for token in tokens:
        print("line token: %s" % token)

    line = "en"
    arr = line.split('-')  # 源字符串没有-情况下，只有1个结果
    print(arr[0])

    line = "wangwei&hello&world"
    arr = line.partition("&")  # 把字符串分割成：[左边, 自身(分割符), 右边] 三部分
    print(arr)
