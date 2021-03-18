# -*- encoding: utf-8 -*-
import os
import json


def find_dict(dict_data, key_target):
    '''
    @describe: 根据key_target是否在嵌套dict中，存在的话，返回对应的value；不存在，返回None；
               嵌套dict中的key不能重复；否则，只能返回最外层匹配的key；
               层级遍历；
    '''
    queue = [dict_data]
    while len(queue) > 0:
        data = queue.pop()
        print("recursive data: ", data)
        for key, value in data.items():
            print("recursive data222: ", key, value)
            if key == key_target:
                return value
            elif type(value) == dict:
                queue.append(value)
    return None


if __name__ == '__main__':
    print("dict use example")

    # dict 初始化
    dict1 = dict()
    dict2 = {}
    # key只能是不可变的数据类型：str, int, tuple
    dict3 = {'id': 12, 'name': 'ww', 'class': 5}
    print(dir(dict()))  # 查看哪些成员

    # 查
    print(dict1.get('id', 0))
    print(dict1.get('ids', 0))
    # print(dict1['id'])  # key 不存在会抛异常; get不会抛异常

    # 更新
    dict1.update(dict3)
    dict2['id'] = 13
    print("dict1", dict1)
    print("dict3", dict3)

    # 轮询： items() 返回[(key, val),..] 以列表返回元祖数组
    dict2['id1'] = 14
    dict2['id2'] = 10

    print("-----type of dict.items -----")
    print(type(dict2.items()))
    print(dict2.items())
    item_list = list(dict2.items())
    print(item_list)

    print("-----type of dict.keys() -----")  # 返回一个迭代器，可以使用 list() 来转换为列表
    print(type(dict2.keys()))  # <class 'dict_keys'>
    print(list(dict2.keys()))  # 转成list

    # 遍历方法1
    for key, val in dict2.items():
        print("dict2 key:val = ", key, val)

    # 遍历方法2
    for x in dict2:
        print("dict2 key:val 22 = ", x, dict2[x])

    # 遍历方法3
    for x in dict2.values():
        print("x in dict2 value: ", x)

    # 获取dict 的 key 和 value
    dict6 = {'name': 'haohao'}
    (key, value), = dict6.items()  # 不要遗漏后面的逗号
    print("key:value = %s:%s" % (key, value))

    # 字典推导式
    dict7 = {k: v for k, v in zip(range(10), range(1, 11))}
    print("dict7:", dict7)

    # 对dict按照value排序; sorted返回list

    res = sorted(dict2.items(), key=lambda x: x[1], reverse=True)
    for key, val in dict2.items():
        print("dict2 sorted key:val", key, val)  # dict2中的值不会发生变化
    for key, val in res:
        print("res sorted key:val", key, val)

    print("---------del dict-------------")
    # 字典的删除
    print("origin dict: ", dict2)

    # 删除字典给定键 key 所对应的值，返回值为被删除的值。key值必须给出。 否则，返回default值。
    del_val = dict2.pop('id', 0)
    print("pop dict:", del_val, dict2)

    del dict2['id1']  # 删除对应的key值，不会返回值
    print("del dict['id1']:", dict2)

    dict2.clear()  # 清空dict,但dict 还存在
    print("clear dict: ", dict2)
    del dict2  # 删除dict ，dict不存在了，访问会报错
    # print("del dict: ", dict2)  # NameError

    # dict转成json
    dict4 = {}
    dict4['id'] = [1, 2, 3, 4]  # key对应的：list 的值
    dict4['name'] = 'ww'
    dict4['doc'] = {'doc1': 'test1', "doc2": 'test2'}  # key对应的：dict 的值
    print(dict4)
    print(json.dumps(dict4))  # 将dict转成json字符串，保存输出; 注意dumps 有个s，标识字符串

    # json字符串加载到dict中
    json_str = r'{"items":[{"project":"en.wikipedia"}]}'
    dict5 = json.loads(json_str)   # 将json字符串转成dict
    print(dict5.get('items', []))

    # 获取嵌套dict中key对应的值
    print(dict4)

    print(find_dict(dict4, "docx"))
