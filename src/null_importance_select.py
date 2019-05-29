# -*- coding: utf-8 -*-  


'''
首选构造自己的特征群
举例说明
'''

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


data_path = '../data/'
train_data = pd.read_csv(data_path + 'train_dataset.csv')
test_data = pd.read_csv(data_path + 'test_dataset.csv')
sample_sub = pd.read_csv(data_path + 'submit_example.csv')
test_data.columns = train_data.columns[:-1]
data = pd.concat([train_data, test_data])
# --------------------------------
# feature engineering

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
    
data['recharge_way'] = data['缴费用户最近一次缴费金额（元）'].apply(produce_offline_feat)
# test_data['recharge_way'] = test_data['缴费用户最近一次缴费金额（元）'].apply(produce_offline_feat)

# 围绕话费账单构造特征
def fee_feat(train_data):
    #看importance，当月话费 和最近半年平均话费都很高，算一下当月/半年 -->稳定性
    train_data['new_当月费用/6个月平均'] = \
    train_data['用户账单当月总费用（元）']/(train_data['用户近6个月平均消费值（元）'] + 1)

    train_data['new_当月费用-6个月平均'] = \
    train_data['用户账单当月总费用（元）'] - (train_data['用户近6个月平均消费值（元）'])
    
    
    #当月话费/当月账户余额
    train_data['new_当月费用/当月余额'] = \
    train_data['用户账单当月总费用（元）']/(train_data['用户当月账户余额（元）'] + 1)
    return train_data
data = fee_feat(data)

# 对不重要的类别字段进行重新组合
# train_data['is_高档商场'] = train_data['当月是否逛过福州仓山万达'] + train_data['当月是否到过福州山姆会员店']
# train_data['is_高档商场'] = train_data['is_高档商场'].map(lambda x: 1 if x>=1 else 0)

data['is_高档商场'] = data['当月是否逛过福州仓山万达'] + data['当月是否到过福州山姆会员店']
data['is_高档商场'] = data['is_高档商场'].map(lambda x: 1 if x>=1 else 0)

data['is_all_高档商场'] = data['当月是否逛过福州仓山万达'] * data['当月是否到过福州山姆会员店']
data['is_all_高档商场'] = data['当月是否逛过福州仓山万达'] * data['当月是否到过福州山姆会员店']

train_data = data[:train_data.shape[0]]
test_data = data[train_data.shape[0]:]



# 开始进行null importance select 方法
'''
总体思路是观察特征在标签被打乱后是否呈现出比较大的差异性
'''
def get_feature_importances(X_train, y_train, shuffle=True, seed=None):
    if shuffle:
    	# 这里是否打乱标签的顺序 
        y_train = pd.DataFrame(y_train, columns=['信用分']).copy().sample(frac=1.0)

    if isinstance(y_train, pd.DataFrame):
        X_train_lgb = lgb.Dataset(X_train.values, y_train.values.reshape(-1), free_raw_data=False, silent=True)
    else:
        X_train_lgb = lgb.Dataset(X_train.values, y_train.reshape(-1), free_raw_data=False, silent=True)

    lgb_params = {

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
    'lambda_l2': 5, 'lambda_l1': 0

    }

    lgb_regressor = lgb.train(params=lgb_params, train_set=X_train_lgb, num_boost_round=1000)
    imp_df = pd.DataFrame()
    imp_df['feature'] = list(X_train.columns)
    imp_df['importance_gain'] = lgb_regressor.feature_importance(importance_type='gain')
    imp_df['importance_split'] = lgb_regressor.feature_importance(importance_type='split')
    return imp_df

# 未打乱标签时 得到的特征重要性排序
actual_imp_df = get_feature_importances(train_data_use, train_label.values, shuffle=False)



# 将标签进行无序打算80次 计算特征重要性
import time

null_imp_df = pd.DataFrame()
nb_runs = 80

start = time.time()
dsp = ''

for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(train_data_use, train_label.values, shuffle=True)
    imp_df['run'] = i + 1
    
    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    
    # Erase previous message
    for l in range(len(dsp)):
        print('\b', end='', flush=True)
        
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = '\nDone with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='', flush=True)



# actual_imp_df
# null_imp_df

feature_scores = []
for _f in actual_imp_df['feature'].unique():
    # null imp values
    f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    # actual imp values
    f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
    # gain score
    gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
    
    # null imp split
    f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    # actual imp split
    f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
    # split score
    split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
    # result
    feature_scores.append((_f, split_score, gain_score))

# 得到真正标签和打乱标签的相对关系
# 打乱标签80次分布的75分位点 和  真正标签的特征重要性 
scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])


# 设定不同的阈值 进行交并补获得特征差异性
features_selected_1 = scores_df.loc[scores_df['split_score'] > 0.00]['feature'].tolist()
features_selected_2 = scores_df.loc[scores_df['gain_score'] > 1.00]['feature'].tolist()
features_selected_3 = list(set(features_selected_1).union(set(features_selected_2)))
features_selected_4 = list(set(features_selected_1).intersection(set(features_selected_2)))
features_selected = list(set(features_selected_4) - set(features_selected_3))
intersect_features_num = len(features_selected_3)
print(features_selected_1,'\n',features_selected_2,'\n',features_selected_3,'\n',features_selected_4)