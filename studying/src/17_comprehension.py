# -*- coding: utf-8 -*-
import os
import sys
import types

from collections.abc import Iterable
from collections.abc import Iterator

'''
python中的语法糖，推导式：

列表(list)推导式`
字典(dict)推导式
集合(set)推导式
ref: https://www.jianshu.com/p/0a269715a742

'''


def test_comprehension_use():
    '''

    [x for x in data if condition]  # 此处if主要起条件判断作用，data数据中只有满足if条件的才会被留下，最后统一生成为一个数据列表。
    [exp1 if condition else exp2 for x in data] # 此处if...else主要起赋值作用，当data中的数据满足if条件时将其做exp1处理，否则按照exp2处理，最后统一生成为一个数据列表

    使用()生成generator：将推导式的[]改成()即可得到生成器

    '''

    # 求(x,y),其中x是0-5之间的偶数，y是0-5之间的奇数组成的元祖列表

    li = [(x, y) for x in range(5) if x %
          2 == 0 for y in range(5) if y % 2 == 1]
    print(li)

    # 得到生成器
    tuple_itor = (i for i in range(10) if i % 3 == 0)
    for elem in tuple_itor:
        print(elem)

    '''
    字典推导式：
    {key:value for key,value in existing_data_structure }: 字典推导和列表推导的使用类似，只不过中括号改成大括号
    
    '''
    # 字典的kv互换
    di1 = {"a": 10, "b": 11, "c": 13}
    di2 = {v: k for k, v in di1.items() if k in ['a', 'b']}
    print(di2)

    '''
    集合推导式：
    基本格式：{ expr for value in collection if condition }
    
    '''
    names = [['Tom', 'Billy', 'Jefferson', 'Andrew', 'Wesley', 'Steven', 'Joe'], [
        'Alice', 'Jill', 'Ana', 'Wendy', 'Jennifer', 'Sherry', 'Eva']]

    # 用嵌套列表实现
    name_set = {name for lst in names for name in lst if name.count(
        'e') >= 2}  # 注意遍历顺序，这是实现的关键

    print(name_set, type(name_set))

# ['Jefferson', 'Wesley', 'Steven', 'Jennifer']


'''
yield :  使用

1、函数f中使用yield  -》 generator function
   f : 生成器函数 
   f(VAR): 返回迭代器对象：iterable对象
   iterable对象：有next()函数，如果被用在for循环中，next会自动调用
   
   程序执行逻辑： 遇到yield, 中断，返回迭代值；下一轮调用，会从中断初开始执行；
   
'''


def fab(max: int):
    # 计算斐波那契数列
    n, a, b = 0, 0, 1

    while n < max:
        yield b  #
        a, b = b, a+b
        n += 1


'''

Iterable: 可迭代对象

作用于for循环的对象都是Iterable类型
list、dict、str虽然是Iterable，却不是Iterator，可以使用iter()函数变成迭代器Iterator
可迭代对象可用于 for 循环 及各种 以 iterable 为形参的函数/方法 中 (如 zip()、map()、enumerate() 等)
for 循环的本质即：对可迭代对象调用 iter() 返回一个关于该对象的迭代器，然后不断调用 next() 迭代/遍历元素

判断是否为可迭代对象
isinstance([], Iterable)




Iterator 迭代器

迭代器 (iterator) 是一种用于表示 一连串数据流 的对象
迭代器对象要求支持 迭代器协议 —— 对象须同时支持/实现 __iter__() 方法和 __next__() 方法

迭代器必为可迭代对象但可迭代对象不一定是迭代器, 换言之，只有迭代器有 __next__() 方法，而可迭代对象没有

1、普通的可迭代的数据类型：转成迭代器对象
iter(object[, sentinel])
参数：
object - - 支持迭代的集合对象。
sentinel - - 如果传递了第二个参数，则参数 object 必须是一个可调用的对象（如，函数），此时，iter 创建了一个迭代器对象，每次调用这个迭代器对象的__next__()方法时，都会调用 object。
打开模式
返回值 ： 迭代器。

2、next()函数：

next(iterable[, default])
参数说明：
iterable -- 可迭代对象
default -- 可选，用于设置在没有下一个元素时返回该默认值，如果不设置，又没有下一个元素则会触发 StopIteration 异常。
'''


def test_iterable():
    # 普通的可迭代的数据类型：转成迭代器对象
    lst = [1, 2, 3, 4]
    for i in iter(lst):  # 自动调用next
        print(i)

    it = iter([1, 2, 3, 4, 5])
    while True:
        try:
            elem = next(it)
            print('while iterator:', elem)
        except StopIteration:
            #  遇到StopIteration退出循环
            break
    
    # 迭代器必定是可迭代对象，但可迭代对象，不一定是迭代器，因为没有next()

    print(isinstance(lst, Iterable), isinstance(lst, Iterator))  # lst 是可迭代对象, 但还不是迭代器  (True, False) 
    
    iterator = iter(lst)  # 返回可迭代对象 lst 的迭代器, 并由变量 iterator 指向/保存该迭代器
    
    print(isinstance(iterator, Iterable), isinstance(iterator, Iterator))  # iterator 既是可迭代对象, 也是迭代器 True True


if __name__ == '__main__':

    test_comprehension_use()

    # fab判断是否为： 生成器类型？迭代器对象？

    print("fab types: ", isinstance(fab, types.GeneratorType))  #False
    print("fab(2) types: ", isinstance(fab(2), types.GeneratorType)) #True

    print("fab instance: ", isinstance(fab, Iterable)) #False
    print("fab(2) instance: ", isinstance(fab(2), Iterable)) # True

    print("fab返回迭代器对象---\n")

    for val in fab(5):
        print(val)
    
    print("fab返回迭代器对象---\n")
    # 测试迭代器对象的使用
    test_iterable()
