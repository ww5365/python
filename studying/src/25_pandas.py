# -*- encoding: utf-8 -*-

import os
import numpy as np
import pandas as pd


if __name__ == '__main__':

    # 创建dataFrame
    li1 = ['a', 'b', 'c', 'd']
    li2 = [5, 6, 7, 8]
    li3 = [li1, li2]

    print("two dimension: ", li3)

    # 由python的数据类型来创建dataFrame
    # #dict  -> pandas.core.frame.DataFrame dict的key作为列名，value作为列的值
    df1 = pd.DataFrame({"col1": li1, "col2": li2}, index=[1, 2, 3, 4])
    print(df1, type(df1))

    # #list -> DataFrame  list[list1,list2,...] 二维数组, 每行是list1，list2...
    df1_1 = pd.DataFrame(li3, columns=['col1', 'clo2', 'clo3', 'clo4'])
    print("df1_1: ", df1_1)

    # #numpy类型创建dataFrame： numpy.ndarray -> dataFrame
    np1 = np.arange(1, 13)
    print(np1, type(np1))
    np2 = np1.reshape((3, 4))  # ndarray
    print(np2, type(np2))
    df2 = pd.DataFrame(np2, index=[2, 3, 4], columns=[
        "col1", 'col2', 'col3', 'col4'])  # 行索引值是2,3,4

    print("df2 ", df2)

    '''
    reset_index : 重置索引
    参考：https://zhuanlan.zhihu.com/p/110819220
    
    DataFrame.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')
    level：数值类型可以为：int、str、tuple或list，默认无，仅从索引中删除给定级别。默认情况下移除所有级别。控制了具体要还原的那个等级的索引 。
    drop：当指定drop=False时，则索引列会被还原为普通列；否则，经设置后的新索引值被会丢弃。默认为False。
    inplace：输入布尔值，表示当前操作是否对原数据生效，默认为False。
    col_level：数值类型为int或str，默认值为0，如果列有多个级别，则确定将标签插入到哪个级别。默认情况下，它将插入到第一级。
    col_fill：对象，默认‘’，如果列有多个级别，则确定其他级别的命名方式。如果没有，则重复索引名。
    '''
    df2_new1 = df2.reset_index()  # 会多出一列index
    print("df2_new1: ", df2_new1)
    df2_new2 = df2.reset_index(drop=True)  # 重置索引列，同时index一列被丢掉
    print("df2_new2: ", df2_new2)

    '''
    pivot_table: 
    透视表pivot_table()是一种进行分组统计的函数，参数aggfunc决定统计类型。
    pivot_table(data, values=None, index=None, columns=None,aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All')
    pivot_table有四个最重要的参数index、values、columns、aggfunc
    
     
    '''
    print('*' * 40)
    df2_new1.loc[1, 'col1'] = 1
    print(df2_new1)

    '''
    df2_new1:
    
           index  col1  col2  col3  col4
    0      2     1     2     3     4
    1      3     1     6     7     8
    2      4     9    10    11    12
    '''
    # col1列作为索引，进行默认统计,mean;col1的第1，2行的值都为1，要进行聚合统计
    df3 = pd.pivot_table(df2_new1, index=['col1'])
    print(df3)

    # col1列作为索引，进行默认统计,mean;col1的第1，2行的值都为1，要进行聚合统计;只取col2，col3 列进行计算
    df3_2 = pd.pivot_table(df2_new1, index=[
                           'col1'], values=['col2', 'col3'])
    print(df3_2)

    # col1列作为索引进行统计;col1的第1，2行的值都为1，计算sum，mean统计；只取col2，col3 列进行计算
    df3_3 = pd.pivot_table(df2_new1, index=[
                           'col1'], values=['col2', 'col3'], aggfunc=[np.sum, np.mean])
    print(df3_3)
