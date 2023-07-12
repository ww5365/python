# -*- coding: utf-8 -*-
import json
import os
import sys
import re
import pandas as pd
import random
import datetime
import time

import numpy as np
from sklearn.model_selection import KFold
random.seed(123)
random_state = random.randint(0, 100)

from sklearn import datasets
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

import lightgbm as lgb 


from sklearn.metrics import mean_absolute_error # mas 绝对值误差均值
# from sklearn.preprocessing import Imputer # 数据的预处理，一般是特征缩放和特征编码

from sklearn.impute import SimpleImputer as Imputer



'''
实例重点参考

1. https://blog.51cto.com/u_15476879/4872788

'''
def lgb_classifier_model_demo():

    ## iris数据集 
    iris = datasets.load_iris()

    print("===============================")
    print(type(iris))

    print(iris.feature_names)
    print(iris.target_names)
    print("------------------------------------------------------")
    print(iris.data, iris.data.shape, type(iris.data))
    print("------------------------------------------------------")
    print(iris.target, iris.target.shape)

    print("------------------------------------------------------")

    X = iris.data
    y = iris.target

    X_sepal = X[:, :2]
    print("X_sepal: {}".format(X_sepal))

    # 原始数据的画图，看类别分布情况
    plt.scatter(X_sepal[:, 0], X_sepal[:, 1], c=y, cmap=plt.cm.gnuplot) # 按照标签类别c=y,使用不同colormap进行绘图
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    # plt.show()

    X_petal = X[:, 2:4]
    plt.scatter(X_petal[:, 0], X_petal[:, 1], c=y, cmap=plt.cm.gnuplot)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')


    # 基于LightGBM原生接口的多分类的实例

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

    # 转换为Dataset数据格式
    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_test, label=y_test)

    # 参数
    params = {
        'learning_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'max_depth': 4,
        'objective': 'multiclass',   # 目标函数，这个和xgboost中的参数不同
        'num_class': 3,
    }

    # 模型训练
    gbm = lgb.train(params, train_data, valid_sets=[validation_data])

    # 模型预测
    y_pred = gbm.predict(X_test)
    
    print("y_pred: {}".format(y_pred))

    y_pred = [list(x).index(max(x)) for x in y_pred]
    print(y_pred)

    # 模型评估
    print(accuracy_score(y_test, y_pred))

    '''

    准召率，roc，AUC 可参考：https://www.cnblogs.com/limingqi/p/11729572.html


    '''


def lgb_regression_model_demo():

    ## 房价预测数据集 ： https://www.kaggle.com/c/house-prices-advanced-regression-techniques

    '''
    
    该房价预测的训练数据集中一共有81列，第一列是Id，最后一列是label，中间79列是特征。
    这79列特征中，有43列是类别型变量，33列是整数变量，3列是浮点型变量。
    训练数据集中存在缺失值missing value

    '''

    root = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(root,"data", "demo", "train.csv")

    print(data_dir)
    
    # 1.读文件
    data = pd.read_csv(data_dir)
    # 2.切分数据输入：特征  输出：预测目标变量
    y = data.SalePrice
    X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
    # 3.切分训练集、测试集,切分比例7.5 : 2.5
    train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25)

    print(train_X)
    
    # 4.空值处理，默认方法：使用特征列的平均值进行填充

    '''
    fit是计算矩阵缺失值外的相关值的大小，以便填充其他缺失数据矩阵时进行使用
    transform是对矩阵缺失值进行填充,填充值是fit计算得到
    fit_transform是上述两者的结合体

    另外还可参考：https://bbs.huaweicloud.com/blogs/298795
    '''
    my_imputer = Imputer()
    train_X = my_imputer.fit_transform(train_X)
    test_X = my_imputer.transform(test_X)


    # # 5.转换为Dataset数据格式
    lgb_train = lgb.Dataset(train_X, train_y)
    lgb_eval = lgb.Dataset(test_X, test_y, reference=lgb_train)
    # 6.参数
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'regression',  # 目标函数
        'metric': {'l2', 'auc'},  # 评估函数
        'num_leaves': 31,  # 叶子节点数
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    # 7.调用LightGBM模型，使用训练集数据进行训练（拟合）
    # Add verbosity=2 to print messages while running boosting
    my_model = lgb.train(params, 
                        lgb_train, 
                        num_boost_round=20, 
                        valid_sets=lgb_eval, 
                        early_stopping_rounds=5)

    # 8.使用模型对测试集数据进行预测
    predictions = my_model.predict(test_X, num_iteration=my_model.best_iteration)

    # 9.对模型的预测结果进行评判（平均绝对误差）
    print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))


if __name__ == "__main__":

    # lgb_classifier_model_demo()
    lgb_regression_model_demo()
