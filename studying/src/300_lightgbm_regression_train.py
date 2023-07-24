'''
prob的回归，进行优化调参

'''
#-*- coding: utf-8 -*-
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import random
import os 
from lightgbm import log_evaluation, early_stopping
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint


callbacks = [log_evaluation(period=1), early_stopping(stopping_rounds=10)]
#其中period=1指每迭代1次打印一次log；stopping_rounds=15指如果验证集的误差在15次迭代内没有降低，则停止迭代。

dataDelimiter='\t'
#data_dir = "D:\Disk_F\work\工作内容\\202210-202212\时效性\泛时效性\分档预测\\"
# data_dir = "D:\Disk_F\work\工作内容\工具脚本\SparkleSearchTool\SparkleSearchTool\onlineDebug\output2\\"


root_path = os.path.dirname(os.path.abspath(__file__))
print("root_path: {}".format(root_path))

sample_dir = os.path.join(root_path, "data", "petal_data")

# sample_file = os.path.join(sample_dir, "dataset_with_query-with_p75_p50_v1.xlsx")
sample_file = os.path.join(sample_dir, "dataset_with_query-with_p75_p50_20w_with_new_feature_all_v1.xlsx")

data = pd.read_excel(sample_file)
Y = data["prob"]
X = data.drop(["query", "baidu_pt_distribute", "baidu_pt_prob_distribute", "label", "prob"], axis=1)
X["intent"] = X["intent"].astype("category")  # 作为类别特征

random_state = random.randint(0, 10000)   # [0, 10000]之间的1个随机数 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=random_state)


X_train = X_train.reset_index(drop=True)   # drop=True 把原来的索引列去掉
Y_train = Y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)


print(len(X_train), len(X_test))


lgb_model = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    max_bin=100,
    learning_rate=0.01,
    bagging_fraction=0.95,
    bagging_freq=5,
    bagging_seed=66,
    feature_fraction_seed=66,
    boosting='gbdt',
    n_jobs=25,
    verbose=0,
    early_stopping_rounds=50
)


param_dict = {
    "num_leaves": sp_randint(5, 40),
    "min_data_in_leaf": sp_randint(5, 64),
    "min_sum_hessian_in_leaf": np.linspace(0, 10, 30),
    "feature_fraction": np.linspace(0.55, 1, 30),
    'lambda_l1': np.linspace(0, 10, 30),
    'lambda_l2': np.linspace(0, 10, 30),
    "min_gain_to_split": np.linspace(0., 1, 30),
    "n_estimators": sp_randint(200, 600)   # 返回[200, 600)中每个整数的概率分布,均匀分布；RandomizedSearchCV 方法要求传入的数据为一个分布或列表
}

random_search = RandomizedSearchCV(
    lgb_model,
    param_distributions=param_dict,
    n_iter=30,
    cv=5,
    verbose=1,
    n_jobs=-1
)


'''
scikit-learn: 提供了2个参数搜索的能力：
GridSearchCV ： 网格搜索，；穷举搜索：在所有候选的参数选择中，通过循环遍历，尝试每一种可能性，表现最好的参数就是最终的结果
RandomizedSearchCV ： 随机搜索
https://blog.csdn.net/xiaohutong1991/article/details/107946291
'''

reg_cv = random_search.fit(
    X_train, 
    Y_train, 
    eval_set=[(X_train, Y_train),(X_val, Y_val)],
    verbose=200
)

print("best params: {}".format(reg_cv.best_params_)) 

# y_pred = reg_cv.predict(X_test)
# roc_auc_score(Y_test, y_pred)


#Feature importance for top 50 predictors
predictors = [x for x in X_train.columns]
feat_imp = pd.Series(reg_cv.best_estimator_.feature_importances_, predictors).sort_values(ascending=False)
feat_imp = feat_imp[0:50]
plt.rcParams['figure.figsize'] = 20, 5
feat_imp.plot(kind='bar', title='Feature Importance')
plt.ylabel('Feature Importance Score')
