# -*- coding: utf8 -*-

import sys
import types

'''
* 类基础
1、在类的内部，使用 def 关键字来定义一个方法，类方法必须包含参数self，且为第一个参数，self代表的是类的实例。
2、类属性 和 实例属性
   实例属性： 构造函数中使用self的变量
   类属性： 类和实例同时拥有
3、类的专有方法：
    __init__ : 构造函数，在生成对象时调用
    __del__ : 析构函数，释放对象时使用
    __repr__ : 打印，转换
    __setitem__ : 按照索引赋值
    __getitem__: 按照索引获取值
    __len__: 获得长度
    __cmp__: 比较运算
    __call__: 函数调用
    __add__: 加运算
    __sub__: 减运算
    __mul__: 乘运算
    __truediv__: 除运算
    __mod__: 求余运算
    __pow__: 乘方
4、继承：
   class DerivedClassName(modname.BaseClassName):  模块名.基类名
   class DerivedClassName(Base1, Base2, Base3): 多继承，基类中重名函数调用顺序，从左到右

5、Python 子类继承父类构造函数说明
      
   如果在子类中需要父类的构造方法就需要显式地调用父类的构造方法，或者不重写父类的构造方法。
   子类不重写 __init__，实例化子类时，会自动调用父类定义的 __init__。
   如果重写了__init__ 时，实例化子类，就不会自动调用父类已经定义的 __init__
   如果重写了__init__ 时，要继承父类的构造方法，可以使用 super 关键字：
        super(子类名，self).__init__(参数1，参数2，....)
        父类名称.__init__(self,参数1，参数2，...)
    
'''


class Father(object):
    count = 1  # 类属性

    def __init__(self, name):
        self.name = name  # 实例属性
        print("father init")

    def getName(self):  # 类成员函数，至少有一个参数；第一个参数是类的实例：self
        return 'Father ' + self.name


class Son(Father):

    def __init__(self, name):
        super(Son, self).__init__(name)   # super(): 函数
        print("son init")
        self.name = name

    def __getitem__(self, idx):
        # 这个类的对象，可以通过[idx]来获取下标idx的值
        if idx < len(self.name):
            return self.name[idx]
        return ""
    def getName(self):
        return 'Son ' + self.name


def test():
    son = Son("wangwei")  # 定义类的对象
    print(son.getName())

    print("class property count: ", Son.count)  # 类属性,通过类访问
    print("class property count: ", son.count)  # 类属性，通过类实例来访问
    print("instance property name: ", son.name)  # 实例属性，通过实例来访问
    print("__getitem__ use idx: ", son[1])


'''

* 获取对象的信息
type(): 获取对象（普通变量，类，函数名）类型
isinstance(): 获取对象的类型判断，特别针对类的继承，可以识别子类对象是否属于父类
dir(): 列出类或对象中的属性，方法
getattr():
setattr():
hasattr(): hasattr(son, 'name')  #判断son对象是否有name属性

'''
def test3():
    print(type(test))  # 获取函数类型
    print(type(Son))  # 获取类的类型
    if type(abs) == types.BuiltinFunctionType:
        print("abs is builtin function!")
    print(dir(Son))  # 使用dir把类中属性和方法都列出来
    son = Son('www')
    print(dir(son))  # 比类多一个name实例属性
    print("son type:", type(son))
    if type(son) == Father:
        print("son is Father type")
    else:
        print("son is not  Father type")

    if isinstance(son, Father):  # 可以判断出子类对象，是父类的一种类型
        print("son is also class Father type")

    if hasattr(son, 'name'):
        print("instance son has attr name")

'''
类的访问权限：

python中_,__,__varname__,varname 下划线的前后缀，修饰变量和方法,作用？

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

实例方法:  self
我们知道在类里每次定义方法的时候都需要绑定这个实例,就是foo(self, x),为什么要这么做呢?
因为实例方法的调用离不开实例,我们需要把实例自己传给函数,调用时：a.foo(x)， 其实是foo(a, x)

类方法:    cls  classmethod
对于classmethod，它的第一个参数不是self，是cls，它表示这个类本身。

静态方法： 无 staticmethod


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
    test()
    print("--" * 30)
    test1()
    print("--" * 30)
    test2()
    print("--" * 30)
    test3()
    print("--" * 30)
