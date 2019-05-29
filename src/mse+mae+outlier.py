# -*- coding: utf-8 -*-

import time
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
%matplotlib inline
import seaborn as sns
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")


data_path = './data/'
train_data = pd.read_csv(data_path + 'train_dataset.csv')
test_data = pd.read_csv(data_path + 'test_dataset.csv')
sample_sub = pd.read_csv(data_path + 'submit_example.csv')
test_data.columns = train_data.columns[:-1]
print('original trian data shape:', train_data.shape,'\n',
     'original test data shape:', test_data.shape)


# feature engineering
# age fill
train_data.loc[train_data['用户年龄'] == 0, '用户年龄'] = np.nan
test_data.loc[test_data['用户年龄'] == 0, '用户年龄'] = np.nan
data = pd.concat([train_data, test_data])
print('merge train & test data', 'data shape:',data.shape)
print('start feature engineering ----')

# 对长尾部分数据进行截断：
def pre_value(data):
    data['用户年龄'].loc[data['用户年龄']>80] = 80
    data['用户网龄（月）'].loc[data['用户网龄（月）']>270] = 270
    data['缴费用户最近一次缴费金额（元）'].loc[data['缴费用户最近一次缴费金额（元）']>450] = 450
    data['用户近6个月平均消费值（元）'].loc[data['用户近6个月平均消费值（元）']>400] = 400
    data['用户账单当月总费用（元）'].loc[data['用户账单当月总费用（元）']>450] = 450
    data['用户当月账户余额（元）'].loc[data['用户当月账户余额（元）']>1000] = 1000
    data['当月通话交往圈人数'].loc[data['当月通话交往圈人数']>400] = 400
    data['当月网购类应用使用次数'].loc[data['当月网购类应用使用次数']>10000] = 10000
    data['当月物流快递类应用使用次数'].loc[data['当月物流快递类应用使用次数']>50] = 50
    data['当月金融理财类应用使用总次数'].loc[data['当月金融理财类应用使用总次数']>10000] = 10000
    data['当月视频播放类应用使用次数'].loc[data['当月视频播放类应用使用次数']>10000] = 10000
    data['当月飞机类应用使用次数'].loc[data['当月飞机类应用使用次数']>100] = 100
    data['当月火车类应用使用次数'].loc[data['当月火车类应用使用次数']>100] = 100
    data['当月旅游资讯类应用使用次数'].loc[data['当月旅游资讯类应用使用次数']>1000] = 1000
    return data
data = pre_value(data)

# 根据缴费字段的金额是否为整数 是否为0 是否为小数 划分为不同的缴费方式
def produce_offline_feat(item):
    if item == 0:
        return 0
    elif item % 10 == 0:
        return 1
    else:
        return 2
data['缴费方式'] = data['缴费用户最近一次缴费金额（元）'].apply(produce_offline_feat)
data = pd.get_dummies(data, columns=['缴费方式'])

# 对不重要的类别字段进行重新组合
data['是否去过高档商场'] = data['当月是否逛过福州仓山万达'] + data['当月是否到过福州山姆会员店']
data['是否去过高档商场'] = data['是否去过高档商场'].map(lambda x: 1 if x>=1 else 0)

# 针对比较重要的费用字段进行处理
def fee_feat(data):
    data['6个月的平均消费-当月账单'] = data['用户近6个月平均消费值（元）'] - data['用户账单当月总费用（元）']
    data['七个月的平均消费'] = (data['用户近6个月平均消费值（元）'] + data['用户账单当月总费用（元）']) / 2
    return data
data = fee_feat(data)
print('feature engineering over ---- ')



params_mae = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'mae',
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'num_leaves': 31,
    'verbose': -1,
    'max_depth': 5,
    # 'max_depth': 3,
    'lambda_l2': 5, 'lambda_l1': 0
}

params_mse = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': 'mae',
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'num_leaves': 31,
    'verbose': -1,
    'max_depth': 5,
    # 'max_depth': 3,
    'lambda_l2': 5, 'lambda_l1': 0
}

NFOLDS = 5
train_label = train_data['信用分']
def kfold_lgb(params, train_df ,train_label, test_df,num_folds=5, objective='', metrics='',debug= False,
            early_stopping_rounds=250, num_boost_round=10000, verbose_eval=False, categorical_features=None):
    
    lgb_params = params
    train_data = train_df
    train_label = train_label
    test_data = test_df
    cv_pred = 0
    valid_best_l2_all = 0
    train_best_l2_all = 0
    feature_importance_df = pd.DataFrame()
    count = 0
    
    print('Starting lgb training... train.shape:{}'.format(train_data.shape))
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2019)
    kf = kfold.split(train_data, train_label)
    for i, (train_fold, validate) in enumerate(kf):
        print('fold: ',i, ' training')
        X_train, X_validate, label_train, label_validate = \
        train_data.iloc[train_fold, :], train_data.iloc[validate, :], \
        train_label[train_fold], train_label[validate]
        
        dtrain = lgb.Dataset(X_train, label_train)
        dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)
        
        watchlist = [dtrain, dvalid]
        bst = lgb.train(params, dtrain, num_boost_round=num_boost_round, valid_sets=watchlist, verbose_eval=verbose_eval,early_stopping_rounds=early_stopping_rounds)
        cv_pred += bst.predict(test_data, num_iteration=bst.best_iteration)
        valid_best_l2_all += bst.best_score['valid_1']['l1']
        train_best_l2_all += bst.best_score['training']['l1']
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = list(X_train.columns)
        fold_importance_df["importance"] = bst.feature_importance(importance_type='gain', iteration=bst.best_iteration)
        fold_importance_df["fold"] = count + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        count += 1
        
    fold_importance_df['importance'] = fold_importance_df['importance'] / 5   
    valid_best_l2_all /= num_folds
    train_best_l2_all /= num_folds
    cv_pred /= 5

    print('cv train score:', 1/(1+train_best_l2_all))
    print('cv valid score:', 1/(1+valid_best_l2_all))
    
    return 1/(1+train_best_l2_all), 1/(1+valid_best_l2_all), fold_importance_df, cv_pred




NFOLDS = 5
train_label = train_data['信用分']

# rm_feat = ['用户编码','信用分','是否大学生客户','用户实名制是否通过核实']
# train_data_use = new_train_data.drop(rm_feat, axis=1)
# test_data_use = new_test_data.drop(rm_feat, axis=1)
# print(data.shape, new_train_data.shape, new_test_data.shape)
print(train_data_use.shape, test_data_use.shape)
oof_pred_all = 0
cv_pred_all = 0
en_amount = 5

for seed in range(en_amount):
    print(seed,'seed')
    NFOLDS = 5
    kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=seed)
    # kf = kfold.split(train_data, train_label)
    kf = kfold.split(train_data_use.values, train_label.values)

    oof_pred = np.zeros(train_data_use.shape[0]) 
    cv_pred = np.zeros(test_data_use.shape[0])
    valid_best_l2_all = 0

    feature_importance_df = pd.DataFrame()
    count = 0
    for i, (train_fold, validate) in enumerate(kf):
        print('fold: ',i, ' training')
        X_train, X_validate, label_train, label_validate = \
        train_data_use.values[train_fold], train_data_use.values[validate], \
        train_label.values[train_fold], train_label.values[validate]
        dtrain = lgb.Dataset(X_train, label_train)
        dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)
        bst = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=dvalid, verbose_eval=-1,early_stopping_rounds=250)
        
        oof_pred[validate] = bst.predict(X_validate,num_iteration=bst.best_iteration) 
        cv_pred += bst.predict(test_data_use, num_iteration=bst.best_iteration)
        valid_best_l2_all += bst.best_score['valid_0']['l1']

#         fold_importance_df = pd.DataFrame()
#         fold_importance_df["feature"] = list(X_train.columns)
#         fold_importance_df["importance"] = bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
#         fold_importance_df["fold"] = count + 1
#         feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        count += 1

    cv_pred /= NFOLDS
    valid_best_l2_all /= NFOLDS
    print(1/(1+valid_best_l2_all))
    oof_pred_all += oof_pred
    cv_pred_all += cv_pred
    print('---'*5)

oof_pred_all /= en_amount
cv_pred_all /= en_amount


oof_pred2_all2 = 0
cv_pred_all2 = 0
en_amount = 5

for seed in range(en_amount):
    print(seed,'seed')
    NFOLDS = 5
    kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=seed+2018)
    # kf = kfold.split(train_data, train_label)
    kf = kfold.split(train_data_use.values, train_label.values)

    oof_pred2 = np.zeros(train_data_use.shape[0]) 
    cv_pred = np.zeros(test_data_use.shape[0])
    valid_best_l2_all = 0

    feature_importance_df = pd.DataFrame()
    count = 0
    for i, (train_fold, validate) in enumerate(kf):
        print('fold: ',i, ' training')
        X_train, X_validate, label_train, label_validate = \
        train_data_use.values[train_fold], train_data_use.values[validate], \
        train_label.values[train_fold], train_label.values[validate]
        dtrain = lgb.Dataset(X_train, label_train)
        dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)
        bst = lgb.train(params2, dtrain, num_boost_round=10000, valid_sets=dvalid, verbose_eval=-1,early_stopping_rounds=250)
        
        oof_pred2[validate] = bst.predict(X_validate,num_iteration=bst.best_iteration) 
        cv_pred += bst.predict(test_data_use, num_iteration=bst.best_iteration)
        valid_best_l2_all += bst.best_score['valid_0']['l1']

#         fold_importance_df = pd.DataFrame()
#         fold_importance_df["feature"] = list(X_train.columns)
#         fold_importance_df["importance"] = bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
#         fold_importance_df["fold"] = count + 1
#         feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        count += 1
    
    cv_pred /= NFOLDS
    valid_best_l2_all /= NFOLDS
    print(1/(1+valid_best_l2_all))
    oof_pred2_all2 += oof_pred2
    cv_pred_all2 += cv_pred
    print('---'*5)

oof_pred2_all2 /= en_amount
cv_pred_all2 /= en_amount


test_data_sub = test_data[['用户编码']]
test_data_sub['score'] = (cv_pred_all2 + cv_pred_all)/2
test_data_sub.columns = ['id','score']
test_data_sub['score'] = test_data_sub['score']