# -*- coding: utf-8 -*-
import os
import re

if __name__ == '__main__':

    '''
    1、 list 常见的使用

    * 元素可重复，类型可不同

    * list是最常用的数据类型,关注
      切片
      列表表达式
      排序
    '''

    '''
    列表表达式:
    [想要的值 值满足的表达式条件]
    '''

    # list 取出符合条件的元素的下标indices
    li = [1, 2, 3, 3, 4, 5, 7, 8, 10]
    indices = [idx for idx in range(len(li)) if li[idx] % 2 == 0]  # 列表表达式
    print("indices:", indices)

    li = [x for x in range(10)]
    print(li)
    for idx in li:
        print(idx)

    order_level_id = ['order_{}_id'.format(idx) for idx in range(10)]
    print(order_level_id)

    # 列表表达式?
    is_assitance = 0
    aStr = 'customer' if str(is_assitance) == '0' else 'assitance'
    print(aStr)

    # list 填充操作？
    vocab_size = 10
    parent = [1] * (2 * vocab_size - 2)
    print("parent: ", parent)

    li2 = [''] * 3
    print(li2)

    # list 中冒号使用说明 [start:end:step] 取[start,end)之间的元素，同时step为步长 (注意end是闭区间，不取这个位置上数)
    li = [0, 1, 2, 3, 4]
    print(li[:: 2])
    print(li[1: -2])  # 从第2个元素到倒数第2个元素

    # 切片
    line = "I want to study python"
    line = line.strip().split()
    print(line, line[-1])

    # list 遍历方法1
    for i in li:
        print("序号：%s   值：%s" % (li.index(i) + 1, i))

    # 方法2
    for i in range(len(li)):
        print("序号：%s   值：%s" % (i + 1, li[i]))

    # 方法3
    for i, val in enumerate(li):
        print("序号：%s   值：%s" % (i + 1, val))

    # 删除list中元素
    print("befor del list: ", li)

    # 使用index 删除某个索引位置的元素，del[start:end) pop(index)
    del li[0:1]  # 删除[0,1)第1位置的元素
    print("after del list: ", li)
    li.pop(0)
    print("after pop list: ", li)

    # 按照某个元素值删除元素
    li.remove(3)
    print("after remove list: ", li)

    # 全部清空 clear
    li.clear()
    print("after clear list: ", li)

    # join 用法
    line2 = ['#tag', '测试数据']
    line2 = re.sub(r'[#]*', '', "\t".join(line2))  # 正则匹配
    print("join and re.sub test: ", line2)

    # extend 用法
    '''
    extend:
        函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表)

    返回值：

        无， 但是会将一个链表的内容添加到另一个链表的后面
    '''

    aList = [1, 3, 'a']
    bList = ['xi', 2009]
    aList.extend(bList)

    print("---" * 30)
    print(aList, bList)

    '''
    append: list.append(obj) 添加到列表末尾的对象
    该方法无返回值，但是会修改原来的列表。
    '''
    aList.append('appendtest')
    print(aList)

    '''
    list(tuple) : 把元素转成list
    
    '''

    atuple = (1, 2, 3, 4)
    aList = list(atuple)
    print(aList)

    '''
    列表的排序:
    alist.sort(key, reverse) : 原列表中进行排序，本身被修改
    key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
    reverse -- 排序规则，reverse = True 降序， reverse = False 升序（默认）。
    
    更高效的方法：
    sort: 更改原列表中的值
    sorted:不更改原列表中的元素值,  返回新的一个列表
    
    '''

    bList = ['name1', 'name2', 'name3', 'name4']
    atuple = zip(bList, aList)
    cList = list(atuple)  # 按照第2个元素降序排列
    print("clist: ", cList)

    # 排序方法1
    cList2 = sorted(cList, key=lambda x: x[1], reverse=True)
    print(cList)
    print(cList2)

    # 排序方法2
    cList.sort(key=lambda x: x[1], reverse=True)
    print(cList)
