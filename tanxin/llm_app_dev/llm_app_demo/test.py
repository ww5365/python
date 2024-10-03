# -*- coding: utf-8 -*-


if __name__ == '__main__':
    
    li = [[1,5],[2,4],[3,3],[4,2],[5,1]]  # 按照第1个元素从高到低进行排序
    
    li2 = sorted(li, key = lambda x : x[0], reverse=True)  # 返回值，放到新的list中，原来的list不影响
    
    print("{}".format(li))
    
    print("{}".format(li2))
    
    