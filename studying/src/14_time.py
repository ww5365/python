# -*- coding: utf-8 -*-

from time import sleep
from datetime import datetime


if __name__ == '__main__':

    start = datetime.now()  # 获取当前时间, 精确到微妙 ： 2020-09-24 10:32:38.778414
    print(start)
    sleep(1)
    print(datetime.now() - start)
