# -*- coding: utf-8 -*-
import pandas as pd
import lightgbm as lgb
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

canceData = load_breast_cancer()
X = canceData.data
y = canceData.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.2)

# 数据转换
print('数据转换')
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(
    X_test, y_test, reference=lgb_train, free_raw_data=False)

# 设置初始参数--不含交叉验证参数
print('设置参数')
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'learning_rate': 0.1
}

# # 交叉验证(调参)
# print('交叉验证')
# max_auc = float('0')
# best_params = {}

# # 准确率
# print("调参1：提高准确率")
# for num_leaves in range(5, 100, 5):
#     for max_depth in range(3, 8, 1):
#         params['num_leaves'] = num_leaves
#         params['max_depth'] = max_depth

#         cv_results = lgb.cv(
#             params,
#             lgb_train,
#             seed=1,
#             nfold=5,
#             metrics=['auc'],
#             early_stopping_rounds=10,
#             verbose_eval=True
#         )

#         mean_auc = pd.Series(cv_results['auc-mean']).max()
#         boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()

#         if mean_auc >= max_auc:
#             max_auc = mean_auc
#             best_params['num_leaves'] = num_leaves
#             best_params['max_depth'] = max_depth
# if 'num_leaves' and 'max_depth' in best_params.keys():
#     params['num_leaves'] = best_params['num_leaves']
#     params['max_depth'] = best_params['max_depth']

# # 过拟合
# print("调参2：降低过拟合")
# for max_bin in range(5, 256, 10):
#     for min_data_in_leaf in range(1, 102, 10):
#         params['max_bin'] = max_bin  # 表示​​feature​​​将存入的​​bin​​的最大数量
#         params['min_data_in_leaf'] = min_data_in_leaf   # 叶子可能具有的最小记录数 

#         cv_results = lgb.cv(
#             params,
#             lgb_train,
#             seed=1,
#             nfold=5,
#             metrics=['auc'],
#             early_stopping_rounds=10,
#             verbose_eval=True
#         )

#         mean_auc = pd.Series(cv_results['auc-mean']).max()
#         boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()

#         if mean_auc >= max_auc:
#             max_auc = mean_auc
#             best_params['max_bin'] = max_bin
#             best_params['min_data_in_leaf'] = min_data_in_leaf
# if 'max_bin' and 'min_data_in_leaf' in best_params.keys():
#     params['min_data_in_leaf'] = best_params['min_data_in_leaf']
#     params['max_bin'] = best_params['max_bin']

# print("调参3：降低过拟合")
# for feature_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
#     for bagging_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
#         for bagging_freq in range(0, 50, 5):
#             params['feature_fraction'] = feature_fraction  # 在每次迭代中随机选择80％的特征来建树
#             params['bagging_fraction'] = bagging_fraction  # 每次迭代时用的数据比例 用于加快训练速度和减小过拟合
#             params['bagging_freq'] = bagging_freq  # ​​bagging​​​的频率，k意味着每k次迭代执行​​bagging​​

#             cv_results = lgb.cv(
#                 params,
#                 lgb_train,
#                 seed=1,
#                 nfold=5,
#                 metrics=['auc'],
#                 early_stopping_rounds=10,  # 如果一次验证数据的一个度量在最近的​​early_stopping_round​​ 回合中没有提高，模型将停止训练
#                 verbose_eval=True
#             )

#             mean_auc = pd.Series(cv_results['auc-mean']).max()
#             boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()

#             if mean_auc >= max_auc:
#                 max_auc = mean_auc
#                 best_params['feature_fraction'] = feature_fraction
#                 best_params['bagging_fraction'] = bagging_fraction
#                 best_params['bagging_freq'] = bagging_freq

# if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in best_params.keys():
#     params['feature_fraction'] = best_params['feature_fraction']
#     params['bagging_fraction'] = best_params['bagging_fraction']
#     params['bagging_freq'] = best_params['bagging_freq']


# print("调参4：降低过拟合")
# for lambda_l1 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
#     for lambda_l2 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.4, 0.6, 0.7, 0.9, 1.0]:
#         params['lambda_l1'] = lambda_l1
#         params['lambda_l2'] = lambda_l2  # https://www.jianshu.com/p/a86b39f0b151  这个里面讲了l1和l2正则分析
#         cv_results = lgb.cv(
#             params,
#             lgb_train,
#             seed=1,
#             nfold=5,
#             metrics=['auc'],
#             early_stopping_rounds=10,
#             verbose_eval=True
#         )

#         mean_auc = pd.Series(cv_results['auc-mean']).max()
#         boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()

#         if mean_auc >= max_auc:
#             max_auc = mean_auc
#             best_params['lambda_l1'] = lambda_l1
#             best_params['lambda_l2'] = lambda_l2
# if 'lambda_l1' and 'lambda_l2' in best_params.keys():
#     params['lambda_l1'] = best_params['lambda_l1']
#     params['lambda_l2'] = best_params['lambda_l2']

# print("调参5：降低过拟合2")
# for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#     params['min_split_gain'] = min_split_gain   # 执行节点分裂的最小增益。默认设置为0

#     cv_results = lgb.cv(
#         params,
#         lgb_train,
#         seed=1,
#         nfold=5,
#         metrics=['auc'],
#         early_stopping_rounds=10,
#         verbose_eval=True
#     )

#     mean_auc = pd.Series(cv_results['auc-mean']).max()
#     boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()

#     if mean_auc >= max_auc:
#         max_auc = mean_auc

#         best_params['min_split_gain'] = min_split_gain
# if 'min_split_gain' in best_params.keys():
#     params['min_split_gain'] = best_params['min_split_gain']

# print(best_params)
# 完成参数的选择
##############################################################################################

print("###" * 30)
random_state = 10

folds = KFold(n_splits=5, shuffle=True, random_state=random_state)  
val_ac = 0
val_ac_loose = 0
for fold, (trn_idx, val_idx) in enumerate(folds.split(X_train)):  #只需传入数据，不需要传入y标签，切分返回索引list
    
    if fold == 1:
        print("x_train idx:{} x_train: {}".format(len(val_idx), X_train[val_idx]))

    lgb_train = lgb.Dataset(X_train[trn_idx], label=y_train[trn_idx])
    lgb_val = lgb.Dataset(X_train[val_idx], label=y_train[val_idx])
    clf = lgb.train(params,
                    lgb_train,
                    # valid_sets=[lgb_train, lgb_val],   这里感觉有问题，修改为lgb_val
                    valid_sets=[lgb_val],
                    verbose_eval=20,
                    early_stopping_rounds=60,
                    num_boost_round=4800)

    Y_pred = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    print("Y_pred: {}  len Y_pred:{} type:{} ".format(Y_pred, len(Y_pred), type(Y_pred)))

