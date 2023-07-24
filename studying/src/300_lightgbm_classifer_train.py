import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import random
import os 
from lightgbm import log_evaluation, early_stopping

callbacks = [log_evaluation(period=1), early_stopping(stopping_rounds=10)]
#其中period=1指每迭代1次打印一次log；stopping_rounds=15指如果验证集的误差在15次迭代内没有降低，则停止迭代。

dataDelimiter='\t'
#data_dir = "D:\Disk_F\work\工作内容\\202210-202212\时效性\泛时效性\分档预测\\"
# data_dir = "D:\Disk_F\work\工作内容\工具脚本\SparkleSearchTool\SparkleSearchTool\onlineDebug\output2\\"


root_path = os.path.dirname(os.path.abspath(__file__))
print("root_path: {}".format(root_path))

sample_dir = os.path.join(root_path, "data", "petal_data")

# sample_file = os.path.join(sample_dir, "dataset_with_query-with_p75_p50_v1.xlsx")
sample_file = os.path.join(sample_dir, "dataset_with_query-with_p75_p50_20w_with_new_feature_all.xlsx")

data = pd.read_excel(sample_file)
Y = data["label"]
X = data.drop(["query", "baidu_pt_distribute", "baidu_pt_prob_distribute", "label", "prob"], axis=1)
X["intent"] = X["intent"].astype("category")  # 作为类别特征

random_state = random.randint(0, 10000)   # [0, 10000]之间的1个随机数 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)
X_train = X_train.reset_index(drop=True)   # drop=True 把原来的索引列去掉
Y_train = Y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)


print(len(X_train), len(X_test))


# print(X_train)
# print(Y_train)
# X_train = X_train.apply(lambda x: (x - np.mean(x)) / np.std(x))
# X_test = X_test.apply(lambda x: (x - np.mean(x)) / np.std(x))

# 为lightgbm准备Dataset格式数据

lgb_train = lgb.Dataset(X_train, Y_train)

### 设置初始参数--不含交叉验证参数
print('设置参数')
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'nthread': 4,
    'learning_rate': 0.1,
    'num_class': 5,
    "verbosity": -1,
    'max_bin': 255,
    'feature_pre_filter': False
    # 'device': 'gpu'
}


# ############### test : lgb.cv  ##################
# '''
# lightgbm.cv() 
# 会同时在每个round训练K个模型，并记录每个round中最优的模型的metric。
# 这样指定的early stopping通常会更早停止，这样避免过拟合，而且训练速度会更快一些。
# '''
# params['num_leaves'] = 5
# params['max_depth'] = 5
# cv_results = lgb.cv(
#     params,
#     lgb_train,
#     seed=1,
#     nfold=5,  # n折交叉验证
#     metrics=['multi_logloss'],
#     # early_stopping_rounds= 10,    # 早停决策，如果一个验证集的度量在10次循环中没有提升，则停止训练
#     # verbose_eval=True
#     callbacks=callbacks   ## 使用callbacks替换early_stopping_rounds,这样不会报warning了
#     )
    
# print("cv_results: {} len: {}".format(cv_results, len(cv_results['multi_logloss-mean'])))

# mean_loss = pd.Series(cv_results['multi_logloss-mean']).min()
# boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmin()

# print("loss: {}, boost_rounds: {}".format(mean_loss, boost_rounds))
# exit(0)

# #### end test



### 交叉验证(调参)
print('交叉验证')
min_loss = float('1000')
best_params = {}

# 准确率
print("调参1：提高准确率")
for num_leaves in range(5, 100, 5):
    for max_depth in range(3, 8, 1):
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth

        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=1,
            nfold=5,  
            metrics=['multi_logloss'],
            early_stopping_rounds=10,
            verbose_eval=True
        )

        mean_loss = pd.Series(cv_results['multi_logloss-mean']).min()
        boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmin()

        print("调参1： num_leves: {} depth: {}  cv_result: {}".format(num_leaves, max_depth, cv_results))

        if mean_loss <= min_loss:
            min_loss = mean_loss
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth
            
if 'num_leaves' and 'max_depth' in best_params.keys():
    params['num_leaves'] = best_params['num_leaves']
    params['max_depth'] = best_params['max_depth']



# 过拟合
print("调参2：降低过拟合")
for min_data_in_leaf in range(1, 102, 10):
    params['min_data_in_leaf'] = min_data_in_leaf

    cv_results = lgb.cv(
        params,
        lgb_train,
        seed=1,
        nfold=5,
        metrics=['multi_logloss'],
        early_stopping_rounds=10,
        verbose_eval=True
    )

    mean_loss = pd.Series(cv_results['multi_logloss-mean']).min()
    boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmin()

    if mean_loss <= min_loss:
        min_loss = mean_loss
        best_params['min_data_in_leaf'] = min_data_in_leaf

print("调参3：降低过拟合")
for feature_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
    for bagging_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for bagging_freq in range(0, 50, 5):
            params['feature_fraction'] = feature_fraction  #指定训练每棵树时要采样的特征百分比，它存在的意义也是为了避免过拟合
            params['bagging_fraction'] = bagging_fraction  # 指定用于训练每棵树的训练样本百分比
            params['bagging_freq'] = bagging_freq  #每建立多少棵树，就进行一次bagging

            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                metrics=['multi_logloss'],
                early_stopping_rounds=10,
                verbose_eval=True
            )

            mean_loss = pd.Series(cv_results['multi_logloss-mean']).min()
            boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmin()

            if mean_loss <= min_loss:
                min_loss = mean_loss
                best_params['feature_fraction'] = feature_fraction
                best_params['bagging_fraction'] = bagging_fraction
                best_params['bagging_freq'] = bagging_freq

if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in best_params.keys():
    params['feature_fraction'] = best_params['feature_fraction']
    params['bagging_fraction'] = best_params['bagging_fraction']
    params['bagging_freq'] = best_params['bagging_freq']

print("调参4：降低过拟合")
for lambda_l1 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    for lambda_l2 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.4, 0.6, 0.7, 0.9, 1.0]:
        params['lambda_l1'] = lambda_l1
        params['lambda_l2'] = lambda_l2
        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=1,
            nfold=5,
            metrics=['multi_logloss'],
            early_stopping_rounds=10,
            verbose_eval=True
        )

        mean_loss = pd.Series(cv_results['multi_logloss-mean']).min()
        boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmin()

        if mean_loss <= min_loss:
            min_loss = mean_loss
            best_params['lambda_l1'] = lambda_l1
            best_params['lambda_l2'] = lambda_l2
if 'lambda_l1' and 'lambda_l2' in best_params.keys():
    params['lambda_l1'] = best_params['lambda_l1']
    params['lambda_l2'] = best_params['lambda_l2']

print("调参5：降低过拟合2")
for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    params['min_split_gain'] = min_split_gain

    cv_results = lgb.cv(
        params,
        lgb_train,
        seed=1,
        nfold=5,
        metrics=['multi_logloss'],
        early_stopping_rounds=10,
        verbose_eval=True
    )

    mean_auc = pd.Series(cv_results['multi_logloss-mean']).min()
    boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmin()

    if mean_loss <= min_loss:
        min_loss = mean_loss

        best_params['min_split_gain'] = min_split_gain
if 'min_split_gain' in best_params.keys():
    params['min_split_gain'] = best_params['min_split_gain']

print("=====================================best params========================================")
print(best_params, boost_rounds)
print("=======================================best params======================================")
# fp = open("./params.txt", 'w+')
# fp.write(best_parms)
# fp.close()


