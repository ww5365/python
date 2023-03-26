#-*-coding: utf-8-*-

import numpy as np
from sklearn import preprocessing


def zScore():
    X = np.array([[1., -1., 2.],  #创建一个原始数据：3x3
              [2., 0., 0.],
              [0., 1., -1.]])
    scaler = preprocessing.StandardScaler().fit(X)
    print(scaler.mean_)  # 计算每一列的均值
    print(scaler.var_)  # 计算每一列的方差
    print(scaler.transform(X))  # 计算x的每个原始对应的zScore值
    
    

if __name__ == "__main__":
    zScore()