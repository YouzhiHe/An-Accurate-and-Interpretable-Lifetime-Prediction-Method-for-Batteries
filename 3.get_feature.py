# 分析数据：从battery1开始
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import pickle
from pylab import mpl

# matplotlib没有中文字体，动态解决
plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

battery_all = pickle.load(open(r'.\Data\batch_all.pkl', 'rb'))
# print(battery_all.keys())

numBattery = len(battery_all.keys())
# numFeature = 15
numFeature = 14

# 将所有电池的特征存放在字典中
dict_feature = np.zeros((numBattery, numFeature))
y_cycle_life = np.zeros((numBattery, 1))

# 遍历所有电池
k = 0

for key, value in battery_all.items():

    # 注意python与matlab的不同, Matlab从1开始, Python从0开始

    # Capacity features
    # Initial capacity
    dict_feature[k][0] = battery_all[key]['summary']['QD'][1]
    # Max change in capacity
    dict_feature[k][1] = max(battery_all[key]['summary']['QD'][0:99]) - battery_all[key]['summary']['QD'][1]
    # capacity at cycle 100
    dict_feature[k][2] = battery_all[key]['summary']['QD'][99]

    # Linear fit of Q v. N
    # regress1 = pd.ols(y=battery_all[key]['summary']['QD'][1:99], x=range(1, 99))
    # Slope 斜率
    # dict_feature[k][3] = regress1.beta[0]
    # Intercept 截距
    # dict_feature[k][4] = regress1.beta[1]

    model = LinearRegression()
    x = []
    for i in range(1, 99):
        x.append(i)
    x = np.array(x).reshape((-1, 1))
    y = battery_all[key]['summary']['QD'][1:99]

    model.fit(x, y)
    dict_feature[k][3] = model.coef_
    dict_feature[k][4] = model.intercept_

    # Linear fit of Q v. N, only last 10 cycles
    # regress2 = pd.ols(y=battery_all[key]['summary']['QD'][90:99],x=range(90,99))
    # # Slope 斜率
    # dict_feature[k][5] = regress2.beta[0]
    # # Intercept 截距
    # dict_feature[k][6] = regress2.beta[1]
    x = []
    for i in range(90, 99):
        x.append(i)
    x = np.array(x).reshape((-1, 1))
    y = battery_all[key]['summary']['QD'][90:99]
    # y = y.reshape(-1, 1)
    model.fit(x, y)
    dict_feature[k][5] = model.coef_
    dict_feature[k][6] = model.intercept_

    # Capacity(Q)特征
    QDiff = battery_all[key]['cycles']['99']['Qdlin'] - battery_all[key]['cycles']['9']['Qdlin']
    dict_feature[k][7] = math.log10(abs(min(QDiff)))
    dict_feature[k][8] = math.log10(abs(np.mean(QDiff)))
    dict_feature[k][9] = math.log10(abs(np.var(QDiff)))
    s = pd.Series(QDiff)
    dict_feature[k][10] = math.log10(abs(s.skew()))
    # dict_feature[k][11] = math.log10(abs(s.kurtosis()))
    dict_feature[k][11] = math.log10(abs(QDiff[1]))

    # Peter's proposed features
    # Sum of Qdiff
    dict_feature[k][12] = math.log10(sum(abs(QDiff)))
    # Sum of Qdiff^2
    # numpy有广播(broadcasting)功能
    dict_feature[k][13] = math.log10(sum(QDiff * QDiff))

    # Energy difference
    # 计算数据点间距均匀但不等于 1 的向量的积分。
    # matlab代码：E10 = trapz(batch(i).cycles(10).Qdlin,batch(i).Vdlin);

    # 有问题？？？
    # 暂时仅使用14个特征
    # E10 = np.trapz(battery_all[key]['cycles']['9']['Qdlin'], battery_all[key]['Vdlin'])
    # E100 = np.trapz(battery_all[key]['cycles']['99']['Qdlin'], battery_all[key]['Vdlin'])
    # dict_feature[k][14] = math.log10(E10 - E100)

    # 充电策略是否也应该当做一个特征？？？
    y_cycle_life[k][0] = battery_all[key]['cycle_life']
    k = k + 1

print("获得的特征：\n", dict_feature)
print("对应的循环寿命：\n", y_cycle_life)

# 把numpy存储为pickle文件
# 先整个合并存储吧
dataset_after_feature_engineering = np.hstack((dict_feature, y_cycle_life))

with open(r'dataset_after_feature_engineering.pkl', 'wb') as fp:
    pickle.dump(dataset_after_feature_engineering, fp)