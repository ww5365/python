from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import operator
import matplotlib.pylab as plt
import xlsxwriter as xw
import random
import math
import os


def save_data_xlsx(dst, sheet_name, data):
    workbook = xw.Workbook(dst)
    worksheet1 = workbook.add_worksheet(sheet_name)
    worksheet1.activate()
    for i, val in enumerate(data):
        row = "A" + str(i + 1)
        worksheet1.write_row(row, val)
    workbook.close()


def accuracy_score_new(predict, label, offset):
    # print("========================y_pred len: {},  content: {}".format(len(predict), predict))
    label = label.reset_index(drop=True)
    # print("label len: {}  label: {}".format(len(label), label))
    countAc = 0

    countLevelAll = {}
    countLevelAc = {}
    for i in range(len(predict)):
        if predict[i] in countLevelAll.keys():
            countLevelAll[predict[i]] += 1
        else:
            countLevelAll[predict[i]] = 1

        if abs(predict[i] - label[i]) <= offset:
            countAc += 1
            if predict[i] in countLevelAc.keys():
                countLevelAc[predict[i]] += 1
            else:
                countLevelAc[predict[i]] = 1

    levelResult = {}
    for key, val in countLevelAll.items():
        levelResult[key] = countLevelAc[key] * 1.0 / val

    allResult = countAc * 1.0 / len(predict)

    return (allResult, levelResult)


def predictByStrategy(X_test):
    X_test_list = np.array(X_test).tolist()
    ret_label = []
    for i in range(len(X_test_list)):
        ret_label.append(predictByStrategySingleSample(X_test_list[i]))
    return ret_label


def predictByStrategySingleSample(data):
    web_all = data[8]
    news_all = data[13]
    web_90_ratio = data[19]
    web_180_ratio = data[20]
    web_360_ratio = data[21]
    news_90_ratio = data[23]
    news_180_ratio = data[24]
    news_360_ratio = data[25]
    web_month_ave_90 = data[30]
    label = 3
    if web_all >= 30 and web_90_ratio >= 0.17 or news_90_ratio >= 0.26 and news_all >= 10:
        # print("111")
        label = 0
    elif web_all >= 60 and web_180_ratio >= 0.26 or news_180_ratio >= 0.34 and news_all >= 20:
        # print("222")
        label = 1
    elif web_all >= 120 and web_360_ratio >= 0.37 or news_360_ratio >= 0.45 and news_all >= 40:
        # print("333")
        label = 2
    return label


def predictByModelCombineStrategy(X_test, Y_pred):
    X_test_list = np.array(X_test).tolist()
    Y_pred_combine = [label for label in Y_pred]
    for i in range(len(Y_pred)):
        data = X_test_list[i]
        web_all = data[8]
        news_all = data[13]
        web_90_ratio = data[19]
        web_180_ratio = data[20]
        web_360_ratio = data[21]
        news_90_ratio = data[23]
        news_180_ratio = data[24]
        news_360_ratio = data[25]
        if Y_pred[i] == 3:
            if web_all >= 120 and web_360_ratio >= 0.37 and news_360_ratio >= 0.45 and news_all >= 40:
                Y_pred_combine[i] = 2
        elif Y_pred[i] == 2:
            if web_all >= 30 and web_90_ratio >= 0.17 and news_90_ratio >= 0.26 and news_all >= 10:
                Y_pred_combine[i] = 0
            elif web_all >= 60 and web_180_ratio >= 0.26 and news_180_ratio >= 0.34 and news_all >= 20:
                Y_pred_combine[i] = 1
        if Y_pred[i] == 0:
            if not (web_all >= 30 and web_90_ratio >= 0.17 or news_90_ratio >= 0.26 and news_all >= 10):
                Y_pred_combine[i] = 1
    return Y_pred_combine


def getModelParams():

    # params = {'num_leaves': 25,   # 树的最大叶子节点数
    #           'max_depth': 6,   # 树的最大深度，当模型过拟合时,可以考虑首先降低
    #           'max_bin': 255,
    #           'min_data_in_leaf': 81,
    #           'feature_fraction': 0.6,
    #           'bagging_fraction': 0.9,
    #           'bagging_freq': 10,
    #           'lambda_l1': 0.7,
    #           'lambda_l2': 0.001,
    #           'min_split_gain': 0.1,
    #           'boosting_type': 'gbdt',
    #           'objective': 'multiclass',
    #           'metric': 'multi_logloss',
    #           'nthread': 4,
    #           'learning_rate': 0.1,
    #           'num_class': 5,
    #           "verbosity": -1
    #           }

    params = {'num_leaves': 95,   # 树的最大叶子节点数
              'max_depth': 7,   # 树的最大深度，当模型过拟合时,可以考虑首先降低
              'max_bin': 255,
              'min_data_in_leaf': 71,
              'feature_fraction': 0.7,
              'bagging_fraction': 1.0,
              'bagging_freq': 45,
              'lambda_l1': 1.0,
              'lambda_l2': 0.7,
              'min_split_gain': 0.1,
              'boosting_type': 'gbdt',
              'objective': 'multiclass',
              'metric': 'multi_logloss',
              'nthread': 4,
              'learning_rate': 0.1,
              'num_class': 5,
              "verbosity": -1
              }

    # params = {'num_leaves': 70,   # 树的最大叶子节点数
    #           'max_depth': 7,   # 树的最大深度，当模型过拟合时,可以考虑首先降低
    #           'max_bin': 255,
    #           'min_data_in_leaf': 101,
    #           'feature_fraction': 0.7,
    #           'bagging_fraction': 1.0,
    #           'bagging_freq': 45,
    #           'lambda_l1': 1.0,
    #           'lambda_l2': 0.001,
    #           'min_split_gain': 0.1,
    #           'boosting_type': 'gbdt',
    #           'objective': 'multiclass',
    #           'metric': 'multi_logloss',
    #           'nthread': 4,
    #           'learning_rate': 0.1,
    #           'num_class': 5,
    #           "verbosity": -1
    #           }

    return params

def getGenericTimelinessProb(Y_pred, Y_pred_combine, Y_pred_prob):
    ret_prob = []
    for i in range(len(Y_pred_combine)):
        model_prob_list = Y_pred_prob[i]
        model_label = Y_pred[i]
        combine_label = Y_pred_combine[i]
        secondMaxIndex = -1
        secondMax = 0
        for j in range(len(model_prob_list)):
            if j != model_label and secondMax < model_prob_list[j]:
                secondMax = model_prob_list[j]
                secondMaxIndex = j
        prob_diff = model_prob_list[model_label] - model_prob_list[secondMaxIndex]
        direction = 1 if secondMaxIndex < model_label else -1
        gt_prob = ((len(model_prob_list) - model_label) * 2 - 1) / (len(model_prob_list) * 2) \
                  + prob_diff * direction / (len(model_prob_list) * 2)
        gt_prob += (model_label - combine_label) / len(model_prob_list)
        ret_prob.append(gt_prob)
    return ret_prob

def sum_list(data, start, end):
    ret = 0
    for i in range(start, end):
        ret += data[i]
    return ret

def getGenericTimelinessProbV2(Y_pred, Y_pred_combine, Y_pred_prob):
    ret_prob = []
    for i in range(len(Y_pred_combine)):
        model_prob_list = Y_pred_prob[i]
        model_label = Y_pred[i]
        combine_label = Y_pred_combine[i]

        if model_label == (len(model_prob_list) - 1):
            prob_other = 1 - model_prob_list[-1]
            gt_prob = prob_other / 3
            # print("model_label: " + str(model_label) + "\tgt_prob: " + str(gt_prob) + "\tmodel predict: " +
            #       ",".join([str(round(prob, 2)) for prob in model_prob_list]))
        else:
            prob_sum = sum_list(model_prob_list, 0, model_label + 1)
            x1 = 1 - (len(model_prob_list) - (model_label + 1)) / (len(model_prob_list) - model_label)
            y1 = 1 - (model_label + 1) / len(model_prob_list)
            x2 = 1
            y2 = 1 - model_label / len(model_prob_list)
            a = (y2 - y1) / (x2 - x1)
            b = (y1 * x2 - y2 * x1) / (x2 - x1)
            gt_prob = a * prob_sum + b
            # print("model_label: " + str(model_label) + "\tgt_prob: " + str(gt_prob) + "\tx1: " + str(x1) + "\ty1: " +
            #       str(y1) + "\tx2: " + str(x2) + "\ty2: " + str(y2)  + "\tmodel predict: " +
            #       ",".join([str(round(prob, 2)) for prob in model_prob_list]))

        gt_prob += (model_label - combine_label) / len(model_prob_list)
        ret_prob.append(gt_prob)
    return ret_prob

if __name__=="__main__":
    dataDelimiter = '\t'
    # data_dir = "D:\Disk_F\work\工作内容\\202210-202212\时效性\泛时效性\分档预测\\训练样本\\"
    # data_dir = "D:\Disk_F\work\工作内容\工具脚本\SparkleSearchTool\SparkleSearchTool\onlineDebug\output2\\"

    root_path = os.path.dirname(os.path.abspath(__file__))
    print("root_path: {}".format(root_path))

    data_dir = os.path.join(root_path, "data", "petal_data")

    # region = "eg"
    model_data_dir = os.path.join(root_path, "data", "model")

    # model_data_dir = "d:\\workspace\\my_codehub"

    data = pd.read_excel(data_dir + '\\dataset_with_query-with_p75_p50_20w_with_new_feature_all_v1.xlsx')
    Y = data["label"]
    X = data.drop(["label"], axis=1)
    X["intent"] = X["intent"].astype("category")  # 作为类别特征

    random_state = random.randint(0, 10000)

    X_train_with_query, X_test_with_query, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                                              random_state=random_state)
    X_train = X_train_with_query.drop(["query", "baidu_pt_distribute", "baidu_pt_prob_distribute", "prob"], axis=1).reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    X_test = X_test_with_query.drop(["query", "baidu_pt_distribute", "baidu_pt_prob_distribute", "prob"], axis=1).reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)

    # X_train = X_train.apply(lambda x: (x - np.mean(x)) / np.std(x))
    # X_test = X_test.apply(lambda x: (x - np.mean(x)) / np.std(x))

    params = getModelParams()
    folds = KFold(n_splits=5, shuffle=True, random_state=random_state)  
    val_ac = 0
    val_ac_loose = 0

    level_ac = {}
    level_ac_loose = {}

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X_train)):  #只需传入数据，不需要传入y标签，切分返回索引list
        lgb_train = lgb.Dataset(X_train.iloc[trn_idx], label=Y_train[trn_idx])
        lgb_val = lgb.Dataset(X_train.iloc[val_idx], label=Y_train[val_idx])
        clf = lgb.train(params,
                        lgb_train,
                        # valid_sets=[lgb_train, lgb_val],   #这里感觉有问题，修改为lgb_val
                        valid_sets=[lgb_val],
                        verbose_eval=20,
                        early_stopping_rounds=60,
                        num_boost_round=4800)

        Y_pred = clf.predict(X_train.iloc[val_idx], num_iteration=clf.best_iteration)

        Y_pred = [list(x).index(max(x)) for x in Y_pred]   # list.index(x) 寻址元素x的在list中的索引位置

        (tmp_all_ac, tmp_level_ac) = accuracy_score_new(Y_pred, Y_train[val_idx], 0)
        val_ac += tmp_all_ac
        for key, val in tmp_level_ac.items():

            if key in level_ac.keys():
                level_ac[key] += val
            else:
                level_ac[key] = val

        tmp_all_ac2, tmp_level_ac2 = accuracy_score_new(Y_pred, Y_train[val_idx], 1)
        val_ac_loose += tmp_all_ac2
        for key, val in tmp_level_ac2.items():
            if key in level_ac_loose.keys():
                level_ac_loose[key] += val
            else:
                level_ac_loose[key] = val


    # 输出准确率，交叉调优
    print("\n")

    for key, val in level_ac.items():
        print("AC_validation_model  level: {} accracy: {}".format(key, val/folds.n_splits))
    print("AC_validation_model:", val_ac / folds.n_splits)

    for key, val in level_ac_loose.items():
        print("AC_validation_model_loose  level: {} accracy: {}".format(key, val/folds.n_splits))
    print("AC_validation_model_loose:", val_ac_loose / folds.n_splits)

    print("\n")


    # 全量的样本进行迭代训练
    lgb_train = lgb.Dataset(X_train, label=Y_train)
    lgb_test = lgb.Dataset(X_test, label=Y_test)
    clf = lgb.train(params,
                    lgb_train,
                    # valid_sets=[lgb_train, lgb_val],   #这里感觉有问题，修改为lgb_val
                    valid_sets=[lgb_test],
                    verbose_eval=20,
                    early_stopping_rounds=60,
                    num_boost_round=4800)

    # 模型保存
    modelPath = model_data_dir + "\\zh_model.txt"
    # modelPath = os.path.join(model_data_dir, "model.txt")
    print('Saving model...')
    clf.save_model(modelPath, num_iteration=clf.best_iteration)


    # 模型加载
    print("Loading model...")
    # modelPath = model_data_dir + "model.txt"
    clf = lgb.Booster(model_file=modelPath)

    Y_pred_prob_distribute = clf.predict(X_test, num_iteration=clf.best_iteration)
    Y_pred = [list(x).index(max(x)) for x in Y_pred_prob_distribute]
    Y_pred_prob = [list(x) for x in Y_pred_prob_distribute]

    # 模型评估
    region = "zh"
    print(region + " AC_test_model:", accuracy_score_new(Y_pred, Y_test, 0))
    print(region + " AC_test_model_loose:", accuracy_score_new(Y_pred, Y_test, 1))
    print("\n")

    Y_pred_strategy = predictByStrategy(X_test)
    # print(region + " AC_test_strategy:", accuracy_score_new(Y_pred_strategy, Y_test, 0))
    # print(region + " AC_test_strategy_loose:", accuracy_score_new(Y_pred_strategy, Y_test, 1))
    # print("\n")

    Y_pred_combine = predictByModelCombineStrategy(X_test, Y_pred)
    # print(region + " AC_test_combine:", accuracy_score_new(Y_pred_combine, Y_test, 0))
    # print(region + " AC_test_combine_loose:", accuracy_score_new(Y_pred_combine, Y_test, 1))
    # print("\n")

    Y_pred_gt_prob = getGenericTimelinessProbV2(Y_pred, Y_pred_combine, Y_pred_prob)

    # 输出测试样本预测明细
    predict_for_check = data_dir + "\\predict_for_check_without_normalization.xlsx"
    predict_for_check_data = []
    predict_for_check_data.append(
        ["query", "baidu_pt_distribute", "baidu_pt_prob_distribute", "ori_label", "prob", "model_label", "model_prob_distribute", "strategy_label", "combine_label",
         "model_diff", "strategy_diff", "combine_diff", "gt_prob", "web_30", "web_30_auth", "web_90", "web_90_auth", "web_180",
         "web_180_auth", "web_360", "web_360_auth", "web_all", "news_30", "news_90", "news_180", "news_360",
         "news_all", "web_30_ratio", "web_90_ratio", "web_180_ratio", "web_360_ratio", "web_30_auth_ratio",
         "web_90_auth_ratio", "web_180_auth_ratio", "web_360_auth_ratio", "news_30_ratio", "news_90_ratio",
         "news_180_ratio", "news_360_ratio", "web_p75", "web_p50", "news_p75", "news_p50", "web_month_ave_90",
         "web_month_ave_auth_90", "web_month_ave_180", "web_month_ave_auth_180", "web_month_ave_360",
         "web_month_ave_auth_360", "news_month_ave_90", "news_month_ave_180", "news_month_ave_360",
         "news_month_ave_720", "is_longtail", "strongtime_prob", "intent"])
    X_test_list = np.array(X_test_with_query).tolist()
    Y_test_list = np.array(Y_test).tolist()
    for i in range(len(Y_pred)):
        temp_list = X_test_list[i]

        temp_list.insert(3, Y_test_list[i])   # list.insert(index, obj) 在list的idx=2处插入对象，同时后面的元素后移1位

        temp_list.insert(5, Y_pred[i])
        temp_list.insert(6, ",".join([str(pb) for pb in Y_pred_prob[i]]))
        temp_list.insert(7, Y_pred_strategy[i])
        temp_list.insert(8, Y_pred_combine[i])
        temp_list.insert(9, int(math.fabs(Y_test_list[i] - Y_pred[i])))
        temp_list.insert(10, int(math.fabs(Y_test_list[i] - Y_pred_strategy[i])))
        temp_list.insert(11, int(math.fabs(Y_test_list[i] - Y_pred_combine[i])))
        temp_list.insert(12, Y_pred_gt_prob[i])
        predict_for_check_data.append(temp_list)
    save_data_xlsx(predict_for_check, "sheet1", predict_for_check_data)


    # 导出特征重要性
    importance = clf.feature_importance()
    names = clf.feature_name()
    featurePath = model_data_dir + "\\feature_importance_v3.txt"

    argsort = np.argsort(importance)

    with open(featurePath, 'w+') as file:
        for idx in argsort[::-1]:
            string = names[idx] + ', ' + str(importance[idx]) + '\n'
            file.write(string)

    # 直接画图显示各个特征的重要性
    # plt.figure(figsize=(12, 6))
    lgb.plot_importance(clf, max_num_features=44)
    # plt.title("Featurertances")
    # plt.show()

    # 画某颗树的图
    lgb.plot_tree(clf, tree_index = 0)
