# -*- coding: utf-8 -*-
import os
from functools import wraps

'''
decorator: 装饰器

将函数fun名称作为参数fun_var，传给装饰器函数； 
装饰器函数中再wrap一个内部函数，此内部wrapper再次使用fun_var实现具体功能;
再次调用fun，此时最终使用的是wrapper函数来实现功能


装饰器的总结：可以修改其它函数功能的函数

'''


def a_new_decorator(a_func):
    '''
    @wraps接受一个函数来进行装饰，并加入了复制函数名称、注释文档、参数列表等等的功能。
    这可以让我们在装饰器里面访问在装饰之前的函数的属性。
    '''
    @wraps(a_func)  # 如果不加@wraps(a_func),会是wrapper
    def wrapper():

        print("before excute a_func")
        a_func()
        print("after excute a_func")

    return wrapper  # 返回wrapper函数函数名


@a_new_decorator
def a_func_requring_decoration():
    print("now I am in func requring decoration!")


'''
装饰器的应用场景：logging
'''


def logit(func):
    @wraps(func)
    def with_logging(*args, **kwargs):
        print(func.__name__ + " was called!")
        return func(*args, **kwargs)
    return with_logging


@logit
def add_func(x):
    res = x + x
    return res


'''
装饰器类：

1、记录日志在文件中
2、同时发送记录信息到email

'''


class logit2(object):
    def __init__(self, log_file='out.log'):  # 初始化构造函数
        self.log_file = log_file

    def __call__(self, func):  # 实现：类的实例(对象)，可以像函数一样来调用，并修改类成员变量的值
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_str = func.__name__ + " was called"
            print(log_str)

            with open(self.log_file, 'a') as f:
                f.write(log_str + '\n')

            # 发送通知
            self.notify()
            # 核心是要执行这个函数，但之前我们给增加了额外的功能，记录了相关的日志信息
            return func(*args, **kwargs)  # 函数计算结果return
        return wrapper  # wrapper函数名的return

    def notify(self):
        # logit 只打印日志，不做别的事情
        print("logit only log str into file:[%s]" % (self.log_file))
        pass


# 装饰器是一个类的对象：因为重载了__call__()，所以会自动调用logit().__call__(myfunc1)
@logit2()
def my_func1():
    pass


class logit2_email(logit2):
    '''
    继承：logit2来发送邮件通知
    '''

    def __init__(self, email='admin@test.com', *args, **kwargs):
        self.email = email
        super(logit2_email, self).__init__(*args, **kwargs)

    def notify(self):
        print("logit send msg to logfile and email!!")


@logit2_email()
def my_func2():
    pass


if __name__ == '__main__':

    print("*" * 20)

    # 因为是装饰器函数，所以会自动调用：
    # a_func_requring_decoration = a_new_decorator(a_func_requring_decoration)
    # a_func_requring_decoration()
    a_func_requring_decoration()
    print(a_func_requring_decoration.__name__)  # 如果不加@wraps(a_func),会是wrapper;
    res = add_func(3)
    print("add_func res: ", res)

    # 装饰器的类的使用
    my_func1()

    # 继承类：继承wrapper
    my_func2()
