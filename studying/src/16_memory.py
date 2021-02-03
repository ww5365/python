# -*- coding: utf-8 -*-

import copy


def test1():
    '''
    直接赋值： 变量赋值传递时的引用和拷贝
             字符串，数值，元组均为静态变量: 拷贝
             列表，字典为动态变量： 引用

    深拷贝: copy 模块的 deepcopy 方法，完全拷贝了父对象及其子对象,两者完全独立
    浅拷贝 :  b = a.copy(): 浅拷贝, a 和 b 是一个独立的对象，但他们的子对象还是指向统一对象（是引用）
    '''
    a = [1, 2, 3, 4, ['a', 'b']]  # 原始对象
    b = a  # 赋值，传对象的引用
    c = copy.copy(a)  # 对象拷贝，浅拷贝
    d = copy.deepcopy(a)  # 对象拷贝，深拷贝

    a.append(5)  # 修改对象a
    a[4].append('c')  # 修改对象a中的['a', 'b']数组对象

    print("--" * 30)
    print("funtion test1")

    print("a=", a)  # a = [1,2,3,4,['a', 'b', 'c'],5]
    print("b=", b)  # b -> a = [1,2,3,4,['a', 'b', 'c'],5]
    print("c=", c)  # c和a平行两片内存区域，子对象[a,b,c]，两者是共享的
    print("d=", d)  # a和d完全平行两片区域，互不影响

    # 查看是否为同一个对象，使用id可以看
    print("id(a),id(b), id(c), id(d)", id(a), id(b), id(c), id(d))
    print("id(a[4]),id(c[4])", id(a[4]), id(c[4]))


if __name__ == "__main__":

    test1()
