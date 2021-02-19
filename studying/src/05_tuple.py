# -*- encoding: utf-8 -*-

import os

'''
参考：
https://www.runoob.com/python3/python3-tuple.html
'''


def tuple_use():
    '''
      元组 (x,y) 
      tuple与list类似，不同之处在于tuple中的元素不能进行修改。
      而且tuple使用小括号()，list使用方括号[]。
      * 比列表操作速度快
      * 对数据写保护
      * 可用于字符串格式化中
      * 可作为字典的key
    '''
    print("-" * 30)

    # tuple定义：() 小括号 元素不修改
    tup1 = (1, 2, 3, 4, 5)
    print(type(tup1))
    tup2 = (3,)  # 注意单个元素，需要后面加上逗号，不然就成了()运算符

    # 访问元祖： 切片 类似list
    print(tup1[1:])  # 从第2个元素到结尾
    print(tup1[0::2])  # 间隔取元素
    print(tup1[-1::-1])  # 倒叙输出

    # 修改元祖 ：  虽然元祖的元素不可修改，但可以连接组合 删除整个元祖
    tup3 = tup1 + (889,) + tup2  # 元祖连接
    print(tup3)

    del tup2  # 删除整个tup2元祖
    try:
        print(tup2)  # 抛异常
    except Exception as e:
        print("raise exception: ", e)

    # 类型转换： 把可迭代类型转成tuple tuple(iterable):
    li1 = ['go', 'python', 'c++']
    tup4 = tuple(li1)  # list-> tuple
    print("list->tuple: ", tup4)

    enum_obj = enumerate(li1)   # 下面有对 enumerate 介绍
    print("enum obj： ", type(enum_obj))  # enumerate 对象  <class 'enumerate'>

    for idx, value in enum_obj:
        print("enum obj content: ", idx, value)

    '''
    zip:用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。 
    python3中返回的是：一个对象，可以手动转成list
    '''
    li1 = [1, 2, 4, 5, 2]
    li2 = ["test1", "test2", "test4", "test5", "test2"]

    tuple_list = zip(li1, li2)
    # 一个可迭代对象, zip类的对象 ： <class 'zip'>
    print("tuple_list type:", type(tuple_list))

    for key, value in zip(li1, li2):  # 一个可迭代对象，对每个元素(是一个元祖)
        print("key:value=[%d:%s]" % (key, value))  # key:value=[1:test1]

    '''
    enumerate 接口说明:
    enumerate(iterable, start=0)
    iterable - a sequence, an iterator, or objects that supports iteration
    start (optional) - enumerate() starts counting from this number. If start is omitted, 0 is taken as start.
    RETURN: adds counter to an iterable and returns it. The returned object is a enumerate object.
    
    You can convert enumerate objects to list and tuple using list() and tuple() method respectively.
    '''
    for idx, value in enumerate(zip(li1, li2)):
        print("enumerate key:value=[%d:%s]" %
              (idx, value))  # key:value=[0:(1, 'test1')]


def namedtuple_use():
    '''
    collections 库提供了namedtuple 结构，作用？
    因为元组的局限性：不能为元组内部的数据进行命名，所以往往我们并不知道一个元组所要表达的意义.
    所以在这里引入了 collections.namedtuple 这个工厂函数，来构造一个带字段名的元祖
    '''
    pass


if __name__ == "__main__":

    tuple_use()
