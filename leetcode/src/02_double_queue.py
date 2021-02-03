# -*- coding: utf-8 -*-
'''
1、实现单链表的双端队列
参考：https://blog.csdn.net/weixin_43790276/article/details/104033394

'''


class Node(object):
    def __init__(self, data):
        self.data = data
        self.next = None
