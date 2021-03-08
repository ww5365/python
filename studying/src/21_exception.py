# -*- encoding: utf-8 -*-

import os
import sys


'''

完整的异常处理语句：

    try:
       执行代码
    except IOError as e：
       发生IO异常时的执行语句
    except (NameError, TypeError, RuntionError):
       发生运行时异常，类型异常，命名异常 处理语句
    except:
       发生其他异常时的处理语句
    else：
       未发生异常时，处理语句
    finally：
       无论发不发生异常，都会执行的异常处理语句

跑出异常：

    raise Exception(参数)
    raise 异常类继承自 Exception 类，可以直接继承，或者间接继承类

'''


def divide(x, y):
    '''
    @des  定义除法运算
    '''

    try:
        result = x/y
    except ZeroDivisionError as e:
        print("division zero!")
    else:
        # 这里还有可能抛出异常，比如，xy是str，抛出TypeError; 此时finally语句还是会执行
        print("result is: ", result)
    finally:
        print("finally clause")


class MyError(Exception):
    '''
    @desc  自己的异常处理类，继承了Exception
    '''

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)  # 返回一个对象的 string 格式


if __name__ == '__main__':

    print("python system parameter:", sys.argv)

    divide(12, 0)
    divide(12, 2)
    #divide(12, '2')

    # 自己抛出自己的异常

    try:
        raise MyError(2*2)
    except MyError as e:
        print("my except value: ", e.value)
