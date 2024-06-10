# -*- encoding:utf-8 -*-

'''
Created on Aug 24, 2018

@author: wangwei69
'''

import sys

##最长公共子序列函数
def lcs(a, b):
    """lcs"""
    lena = len(a)
    lenb = len(b)
    c = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    flag = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    
    print("c:", c)
    for i in range(lena):
        for j in range(lenb):
            if a[i] == b[j]:
                c[i + 1][j + 1] = c[i][j] + 1
                flag[i + 1][j + 1] = 'ok'
            elif c[i + 1][j] > c[i][j + 1]:
                c[i + 1][j + 1] = c[i + 1][j]
                flag[i + 1][j + 1] = 'left'
            else:
                c[i + 1][j + 1] = c[i][j + 1]
                flag[i + 1][j + 1] = 'up'
    return c, flag


def printLcs(flag, a, i, j, list):
    """lcs璁＄��"""
    if i == 0 or j == 0:
        return
    if flag[i][j] == 'ok':
        printLcs(flag, a, i - 1, j - 1, list)
        #print(a[i - 1])
        list.append(a[i - 1])
    elif flag[i][j] == 'left':
        printLcs(flag, a, i, j - 1, list)
    else:
        printLcs(flag, a, i - 1, j, list)
        
        
        
if __name__ == '__main__':
    
    name = "王(伟你)好"
    query = "王好(啊)"
    
    name_new = name.replace("（", "(").replace("）", ")")
    
    query_new = query.replace("（","(").replace("）",")")
    
    print(name_new, query_new)
    
    c,flag = lcs(name_new, query_new)
    list_res = []
    printLcs(flag,name_new,len(name_new),len(query_new),list_res)
    
    print ("res:", list_res)
    
        
        
            