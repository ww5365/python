# -*- coding: utf8 -*-

import sys
'''
1、python中_,__,__varname__,varname 下划线的前后缀，修饰变量和方法,作用？

object     # public
__object__ # special, python system use, user should not define like it
__object   # private (name mangling during runtime),类似c++种private
 _object   # obey python coding convention, consider it as private， 类似c++种protected

另注：python 没有严格的private，通过：_ClassName__object 也可访问私有成员 
根据python的约定，应该将其视作private，而不要在外部使用它们，（如果你非要使用也没辙），良好的编程习惯是不要在外部使用它。

也叫：私有变量的mangling   相当于 c++ 中宏定义： python实际讲私有变量，重替换为：_ClassName__object

'''


class Foo():
    def public(self):
        print("public fun")

    def _half_private(self):
        print("_half_private")

    def __full_private(self):
        print("__full_private")


class A(object):
    def __init__(self):
        print("A__init__")
        self.__private()  # name mangling后，_A__private()
        self.public()

    def __private(self):
        print("A.__private")

    def public(self):
        print("A.public")


class B(A):
    '''
    类的继承
    '''

    def __private(self):
        print("B.__private")

    def public(self):
        print("B.public")


def test1():

    # 类成员，访问权限
    f = Foo()
    f.public()
    f._half_private()
    # f.__full_private() ##error 无权限
    f._Foo__full_private()  # 不建议的访问方式

    # 看私有变量的mangling
    print("*" * 30)
    b = B()
    '''
    输出什么? 
    __init__实例进行初始化；调用基类A中的__init__,之后__private使用A的，public使用类B的。
    '''

    # 查看类B中的所有成员
    print("*" * 30)
    print(" ".join(dir(B)))  # _A__private _B__private 私有变慢


'''
self  cls

实例方法,我们知道在类里每次定义方法的时候都需要绑定这个实例,就是foo(self, x),为什么要这么做呢?
因为实例方法的调用离不开实例,我们需要把实例自己传给函数,调用时：a.foo(x)， 其实是foo(a, x)
类方法一样,只不过它传递的是类而不是实例.

对于classmethod，它的第一个参数不是self，是cls，它表示这个类本身。

@staticmethod和@classmethod 

区别可以参考：
https://blog.csdn.net/helloxiaozhe/article/details/79940321
'''


class Foo2(object):
    def __init__(self, title):  # 实例初始化，也是一个instance函数
        self.title = title

    def instance_method(self):  # 实例方法， 必须self
        print("instance method")

    @classmethod
    def class_method_create(cls, title):  # 修饰器: @classmethod，cls:指代类本身
        foo2 = cls(title=title)  # 使用类Foo2 来构造对象
        return foo2

    @staticmethod
    def static_method_create(title):  # 不需要cls self
        foo2 = Foo2(title)  # 在静态方法中，使用Foo2来构造对象
        return foo2

    @property
    def closed(self):
        return True

    def property_use(self):
        if self.closed:
            print("property 修饰器?: true")
        else:
            print("property 修饰器?: false")


def test2():
    obj1 = Foo2("use init create instance")
    obj2 = Foo2.class_method_create("use class method create instance")
    obj3 = Foo2.static_method_create("use static method create instance")

    print("--" * 30)
    print("function test2")

    print(obj1.title)
    print(obj2.title)
    print(obj3.title)

    obj1.property_use()


'''
__metaclass__ __new__ __init__ :

__new__是一个静态方法,而__init__是一个实例方法.
__new__方法会返回一个创建的实例,而__init__什么都不返回.
只有在__new__返回一个cls的实例时后面的__init__才能被调用.
当创建一个新实例时调用__new__,初始化一个实例时用__init__.
__metaclass__是创建类时起作用.
所以我们可以分别使用__metaclass__,__new__和__init__来分别在类创建,实例创建和实例初始化的时候做一些小手脚.


'''

if __name__ == '__main__':

    print("--" * 30)

    test1()

    test2()

    print("--" * 30)
