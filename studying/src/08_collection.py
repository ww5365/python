# -*- coding: utf-8 -*-
from collections import Counter
from collections import OrderedDict
import pathlib
from collections import deque
import sys
import time


def deque_use_test():
    '''
    deque是双边队列（double-ended queue），具有队列和栈的性质。
    1、在list的基础上增加了移动、旋转和增删等，比list更高效
    2、线程安全
    '''

    d = deque([])
    d.append('a')  # 最右边添加一个元素a
    d.appendleft('b')  # 最左边添加一个元素b
    d.extend(['c', 'd'])  # 最右边添加多个元素：b a c d
    d.extendleft(['e', (2, 3), 'f'])  # 最左边添加多个元素：f e b a c d
    print("deque 1: ", d)

    print(d.pop())  # 最右边元素取出，并返回d
    print(d.popleft())  # 最左边元素取出，并返回f
    print("deque 2: ", d)

    try:
        print("type: ", type((2, 3)))
        d.remove((2, 3))   # 删除, 若元素不存在的，会抛异常
        d.remove('r')   # 删除, 若元素不存在的，会抛异常
        print("deque 3: ", d)
    except (ValueError) as e:
        print("exsits element not in deque: %s" % (e))

    d.rotate(-1)
    print("rotate: ", d)  # 循环左移1位

    d.reverse()
    print("reverse: ", d)  # 列表逆值

    for i in range(len(d)):  #顺序访问
        print("deque elem ", d[i])


def deque_use_test2(length=50, speed=1, direction=1) -> None:
    '''
    使用deque实现跑马灯
    length : 总长度
    speed ： 移动速度，默认每秒
    directiuon : 移动方向, 默认右移
    '''

    if direction == 1:
        array = '>'
    else:
        array = '<'

    que = deque([array])
    que.extend(['-'] * (length - 1))
    while True:
        print('%s' % ''.join(que))  # deque 也可以看做list使用join, 打印：>--------

        if direction == 1:
            que.rotate(speed)  # 右移
        else:
            que.rotate(-1 * speed)

        # 屏幕刷新
        sys.stdout.flush()
        time.sleep(0.1)


if __name__ == '__main__':

    print("begin to test collections")

    # deque的使用
    deque_use_test()
    # deque_use_test2()

    # ref: https://docs.python.org/3.7/library/collections.html

    # collections Counter 初始化
    li = ["wang", "wei", "wang", "da", "da"]
    cnt = Counter()
    for word in li:
        cnt[word] += 1
    print(cnt)  # Counter({'wang': 2, 'da': 2, 'wei': 1})

    cnt3 = Counter(li)
    print(cnt3)  # 效果同上
    print(cnt3.most_common(2))  # return list: [('wang', 2), ('da', 2)]

    cnt2 = Counter(cats=3, dogs=2, birds=-1)
    print(cnt2)  # Counter({'cats': 3, 'dogs': 2, 'birds': -1})
    cnt2 = sorted(cnt2.elements())  # elements()会把<0的元素去掉
    print(cnt2)  # ['cats', 'cats', 'cats', 'dogs', 'dogs']

    ord_dict = OrderedDict()

    ord_dict['china'] = '12,34'
    ord_dict['other'] = '13,34'

    print(ord_dict)

    for country, cordinate in ord_dict.items():
        print(country, cordinate)
