import sys
import os
import collections
import math
import numpy as np
from multiprocessing import Array


g_types_dict = {
    # 原始数据是10位的编码不足的后面都是0补充，使用方便仅保留4位或8位
    "0101": ["restaurant"],
    "0102": ["café", "pub"],
    "0502": ["tourist attraction"],
    "0601": ["hospital", "polyclinic"],
    "0702": ["hotel", "motel"],
    "0801": ["airport"],
    "0802": ["train station", "railway station"],
    "0911": ["education institution", "educational institute"],
    "08020040": ["metro", "subway", "tube", "underground"],
    "08030010": ["bus stop"],
    "08030020": ["taxi stand"],
    "08030030": ["tram stop"],
    "08030040": ["coach station", "coach stop"],
    "08070010": ["terminal", "airport terminal"],
    "11060010": ["parking lot", "car park", "car parking", "car lot",
                 "parking facility", "parking space", "parking"],
    "11060040": ["parking garage"],
}


def is_choose_type(types: str):
    print("types 4 and 8:[%s][%s] " % (types[0:4], types[0:8]))

    return types[0:4] in g_types_dict or types[0:8] in g_types_dict


if __name__ == '__main__':

    # 前4或8位是否在dict的key中

    code_type = '0103030000'

    if is_choose_type(code_type):
        print("is choose type")
    else:
        print("not choose types")

    # _ 代表什么？不关心的变量，以忽略特定的值
    test1, _ = 'bbc', 4

    # 文件读写方式1

    fi = open('./README.md', 'r')
    lineNo = 0
    for line in fi:
        line = line.strip()
        lineNo += 1
        print("lineNo:%d content:%s" % (lineNo, line))
    file_bytes = fi.tell()  # 除了1行，每行的换行符计算1个字节

    print("file bytes: %d" % file_bytes)

    # 存放多个dict？使用collections
    c1 = collections.defaultdict(dict)
    c1["sec1"] = {'key1', 'val1'}
    c1["sec2"] = {'key2', 'val2'}
    print(c1)
    print(c1['sec3'])  # 访问没有元素不会抛异常

    # 定义并初始化字典
    result_dict = {1: 4}
    for key in result_dict.keys():
        print(key)

    li = [1, 2, 3, 34]
    # enumerate 枚举迭代器； 逆序访问数组或字符串;
    for index, key in enumerate(li[::-1]):
        print("index:%d value:%d" % (index, key))

    # 直接两个list进行cat，相加
    count = [t for t in li] + [1e15] * (5 - 1)
    print(count)

    for idx in range(4):
        print(idx)

    # math.power(x,y) x^y
    count = 100
    norm = math.pow(count, 0.5)
    print(norm)

    # 随机数产出
    # randint(low=x, high=y, size=z or (m,n), d) 产出[low, high)之间的随机值，产出个数z或（m,n）

    indcies1 = np.random.randint(low=0, high=4, size=10)
    indcies2 = np.random.randint(low=0, high=4, size=(2, 3))

    print(indcies1)
    print(indcies2)

    tmp1 = np.random.random(3)
    tmp2 = np.random.uniform(low=0 / 10, high=1 / 10, size=5)

    print(tmp1)
    print(tmp2)

    # ctypes 类型使用
    tmp = np.ctypeslib.as_ctypes(tmp2)
    print("ctypes tmp:", tmp)  # <c_double_Array_5 object at 0x119c8cb00>
    print("dir tmp:", dir(tmp), tmp._type_)
    tmp = Array(tmp._type_, tmp, lock=False)
    print(
        "Array tmp:", tmp
    )  # <c_double_Array_5 object at 0x11a4ce050> 实际都是c double array类型，不过Array后，可以控制进线程的安全性；
    print(tmp[0])
