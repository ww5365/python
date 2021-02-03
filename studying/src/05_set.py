# -*- encoding: utf-8 -*-

import os


def test_set_use():
    '''
    set
    集合（set）是一个无序的不重复元素序列。
    可以使用大括号 { } 或者 set() 函数创建集合
    注意：创建一个空集合必须用 set() 而不是 { }，因为 { } 是用来创建一个空字典

    重复的key，只保留一个

    '''

    # 定义set
    set1 = set()
    set2 = {'wang', 'wei'}  # 集合初始化，和定义一个空dict不同

    # 增加单个元素
    set1.add('test')  # 只能添加1个元素

    # 增加多个set元素
    set1.update(set2)  # 也可以添加多个元素，且参数可以是列表，元组，字典等
    # 仅添加key到set中，update和之前重复的,仅保留1个，比如wang
    set1.update({'wang': 5, 'num2': 6})
    print(set1)

    # 删除
    set1.remove("wei")  # key不存在的话，会抛出异常
    set1.discard("we")  # key不存在的话，不会抛出异常
    print(set1)

    # 查询
    elem = "wang"
    if elem in set1:
        print(elem)

    # 遍历
    print("------traversal-------")
    for elem in set1:
        print(elem)

    # 推导式 表达式
    set5 = {x for x in set1 if x in {'num2', 'test', 'wei'}}
    print("--------set5----------")
    print(set5)

    # 集合的或交补
    set3 = set("abcd")
    set4 = set("cdef")  # {'c','d','e','f'}

    print(set3 | set4)
    print(set3 & set4)
    print(set3 - set4)
    print(set3 ^ set4)  # 异或

    print("--------intersetction----------")
    # 多个集合的差集，交集
    set6 = set3.intersection(set4)  # 交集，set3不会变化，返回交集结果
    print(set3)
    print(set6)
    set3.intersection_update(set4)  # 在set3基础上计算更新，set3的值发生变化，无返回值
    print(set3)

    print("--------difference----------")
    set7 = set4.difference(set3, set6)  # 差集, 返回新集合 set4 - set3 -set6
    print(set7)


if __name__ == "__main__":
    test_set_use()
