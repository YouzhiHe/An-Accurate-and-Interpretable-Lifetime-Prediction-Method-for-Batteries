import numpy as np
import pandas as pd
import math

import pickle

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import ElasticNetCV
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn import svm

import sklearn.model_selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import pickle

import shap
import pylab
# import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# mpl.use('TkAgg')

import warnings

warnings.filterwarnings("ignore")

# NatureEnergy's paper的训练、测试、二级测试划分
# 应该得按照文章中的训练集和测试集划分，这样才有可比性？？？
numBat1 = 41
numBat2 = 43
numBat3 = 40
numBat = numBat1 + numBat2 + numBat3
numFeature = 14
train_ind = np.arange(1, (numBat1 + numBat2 - 1), 2)
test_ind = np.hstack((np.arange(0, (numBat1 + numBat2), 2), 83))
secondary_test_ind = np.arange(numBat - numBat3, numBat)

# 从文件中读取所有数据
dataset_after_feature_engineering = pickle.load(open(r'dataset_after_feature_engineering.pkl', 'rb'))

print(dataset_after_feature_engineering)

# 三种特征集合：variance, discharge, full
index_variance = [9]
index_discharge = [0, 1, 7, 9, 10]  # 与NatureEnergy有出入, 采用了作者Nature正刊建议的特征集合
index_full = np.arange(14)


# 计算误差
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)) / y_true) * 100


def get_error(y_true, y_pred):
    # 均方根误差（Root Mean Square Error）
    # 平均绝对百分比误差（Mean Absolute Percentage Error）
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_value = mape(y_true, y_pred)
    return rmse, mape_value


# 三个训练或测试集
def get_x_and_y(index_x, index_y):
    t = dataset_after_feature_engineering[index_x, :]
    X = t[:, index_y]
    Y = t[:, 14:]
    return X, Y


index_x = [train_ind, test_ind, secondary_test_ind]
index_y = [index_variance, index_discharge, index_full]
string_x = ["train", "test", "secondary_test"]
string_y = ['variance', 'discharge', 'full']

# 分别用三种特征子集
model_elastic = ElasticNetCV(cv=5, random_state=0)
model_random_forest = RandomForestRegressor(n_estimators=200, random_state=0)
# model_xgboost = XGBRepressor(n_estimators=500, learning_rate=0.05, min_child_weight=5, max_depth=4)
model_xgboost = XGBRegressor(learning_rate=0.1, n_estimators=129, max_depth=5,
                             min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.6,
                             reg_alpha=1000, nthread=4, scale_pos_weight=1, seed=27)
model_catboost = CatBoostRegressor()


def compare_various_model():

    for i in range(len(index_y)):

        X, Y = get_x_and_y(index_x[0], index_y[i])

        # Y = np.log10(Y)

        model_elastic.fit(X, Y)
        model_random_forest.fit(X, Y)
        model_xgboost.fit(X, Y)
        model_catboost.fit(X, Y)

        # 在此处添加预测模型
        model_list = [model_elastic, model_random_forest, model_xgboost, model_catboost]
        model_string = ["elastic net", "random forest", 'xgboost', 'catboost']

        for j in range(len(index_x)):

            X, Y = get_x_and_y(index_x[j], index_y[i])
            # Y = np.log10(Y)

            for k in range(len(model_list)):
                Y_pred = model_list[k].predict(X)
                rmse, mape_value = get_error(Y, Y_pred)
                print(model_string[k], "-> ", string_y[i], " ", string_x[j], "'s RMSE: ", rmse, " ;MAPE: ", mape_value)
    pass


# 针对 Xgboost 进行参数调优争取性能超过 elastic net, 但是暂时还做不到
# 先用TreeSHAP解释一下，看看是什么情况再说吧

def only_xgboost():
    # import pylab

    # 将正则化参数reg_alpha增加到1000
    model_xgboost = XGBRegressor(learning_rate=0.1, n_estimators=129, max_depth=5,
                                 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.6,
                                 reg_alpha=0.1, nthread=4, scale_pos_weight=1, seed=27)

    all_index = np.arange(124)

    # 使用discharge特征集
    X, Y = get_x_and_y(index_x[0], index_y[1])
    model_xgboost.fit(X, Y)

    # train
    Y_pred_train = model_xgboost.predict(X)
    rmse, mape_value = get_error(Y, Y_pred_train)
    print("xgboost discharge train's RMSE: ", rmse, " ; MAPE: ", mape_value)

    # test
    X, Y_test = get_x_and_y(index_x[1], index_y[1])
    Y_pred_test = model_xgboost.predict(X)
    rmse, mape_value = get_error(Y_test, Y_pred_test)
    print("xgboost  discharge test's RMSE: ", rmse, " ; MAPE: ", mape_value)

    # secondary test
    X, Y_secondary = get_x_and_y(index_x[2], index_y[1])
    Y_pred_secondary = model_xgboost.predict(X)
    rmse, mape_value = get_error(Y_secondary, Y_pred_secondary)
    print("xgboost  discharge secondary test's RMSE: ", rmse, " ; MAPE: ", mape_value)

    x = [0, 100, 500, 1000, 2000, 3000]
    y = [0, 100, 500, 1000, 2000, 3000]
    plt.scatter(Y, Y_pred_train, marker='o', )
    plt.scatter(Y_test, Y_pred_test, marker='^')
    plt.scatter(Y_secondary, Y_pred_secondary, marker='s')
    plt.legend(("Train", "Primary test", "Secondary test"))
    plt.plot(x, y)
    plt.xlabel("Observed cycle life")
    plt.ylabel("Predicted cycle life")
    plt.show()

    # X, Y = get_x_and_y(all_index, index_y[1])
    # explainable_ai_with_shapley_value_for_xgboost(model_xgboost, X)


def only_xgboost_new():
    model_xgboost = XGBRegressor()
    all_index = np.arange(124)
    # 使用discharge特征集
    X, Y = get_x_and_y(all_index, index_y[1])
    acc = sklearn.model_selection.cross_val_score(model_xgboost, X, Y, scoring=None, cv=5, n_jobs=1)
    # model_xgboost.fit(X, Y)
    print('交叉验证结果:', acc)

    pass


def xgboost_tuning():
    all_index = np.arange(124)
    # 使用discharge特征集
    X, Y = get_x_and_y(all_index, index_y[1])
    model_xgboost = XGBRegressor()

    # 设定网格搜索的xgboost参数搜索范围，值搜索XGBoost的主要6个参数
    # param_dist = {
    #     'n_estimators': range(100, 1000, 50), 'max_depth': range(2, 14, 2), 'learning_rate': np.linspace(0.01, 2, 10),
    #     'subsample': np.linspace(0.7, 0.9, 20), 'colsample_bytree': np.linspace(0.5, 0.98, 10),
    #     'min_child_weight': range(1, 9, 1)
    # }

    param_dist = {

        'n_estimators': range(1, 1000, 50),
        'max_depth': range(2, 14, 2),
        'learning_rate': np.linspace(0.01, 2, 10),
        'min_child_weight': range(1, 9, 1),
        # 'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
        # 'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],

    }

    # GridSearchCV参数说明，clf1设置训练的学习器
    # param_dist字典类型，放入参数搜索范围
    # scoring = 'neg_log_loss'，精度评价方式设定为“neg_log_loss“
    # n_iter=300，训练300次，数值越大，获得的参数精度越大，但是搜索时间越长
    # n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
    #
    # grid = GridSearchCV(model_xgboost, param_dist, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    # grid.fit(X, Y)

    # RandomizedSearchCV参数说明，clf1设置训练的学习器
    # param_dist字典类型，放入参数搜索范围
    # scoring = 'neg_log_loss'，精度评价方式设定为“neg_log_loss“
    # n_iter=300，训练300次，数值越大，获得的参数精度越大，但是搜索时间越长
    # n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU

    grid = RandomizedSearchCV(model_xgboost, param_dist, cv=5, scoring='neg_root_mean_squared_error', n_iter=300,
                              n_jobs=-1)
    grid.fit(X, Y)

    # 返回最优的训练器
    # print(grid.cv_results_)
    # print("................\n")

    print(grid.best_estimator_)
    # print(grid.best_params_)


def pred_vs_observed_vis(test_y, y_pred, train_y, y_pred_2):

    plt.figure()
    x = [0, 100, 500, 1000, 2000, 3000]
    y = [0, 100, 500, 1000, 2000, 3000]
    plt.scatter(train_y, y_pred_2, marker='o', color='blue')
    plt.scatter(test_y, y_pred, marker='s', color='green')
    # plt.scatter(Y_secondary, Y_pred_secondary, marker='s')
    plt.legend(("Train", "Test"))
    plt.plot(x, y, color='k')
    plt.xlabel("Observed cycle life", fontsize=16)
    plt.ylabel("Predicted cycle life", fontsize=16)

    # 设置坐标刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # 设置坐标轴范围
    plt.xlim((0, 2500))
    plt.ylim((0, 2500))

    plt.show()


def explainable_ai_for_battery_lifetime_prediction():
    # 使用discharge特征集
    all_index = np.arange(124)
    X, Y = get_x_and_y(all_index, index_y[1])

    # 训练集和测试集划分
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(X, Y, test_size=0.3)

    # with open("model_xgboost_2.dat", "rb") as my_profile:
    #     model_xgboost_2 = pickle.load(my_profile)
    #
    # print(model_xgboost_2.best_params_)

    # model_xgboost_2 = RandomizedSearchCV(
    #     XGBRegressor(), param_distributions={'n_estimators': range(1, 1000, 50),
    #                                          'max_depth': range(2, 14, 2),
    #                                          'learning_rate': np.linspace(0.01, 2, 10),
    #                                          'min_child_weight': range(1, 9, 1)},
    #     cv=5, scoring='neg_root_mean_squared_error', n_iter=300, n_jobs=-1)
    # model_xgboost_2.fit(X, Y)
    # print(model_xgboost_2.best_params_)

    model_xgboost = XGBRegressor(n_estimators=251, min_child_weight=3, max_depth=2, learning_rate=0.45222222222222225)
    model_xgboost.fit(X, Y)
    plot_importance(model_xgboost)
    plt.show()
    # print(model_xgboost.feature_importances_)
    # explainable_ai_with_shapley_value_for_xgboost(model_xgboost, X)


def model_comparison():
    # 使用discharge特征集
    all_index = np.arange(124)
    X, Y = get_x_and_y(all_index, index_y[1])
    # dataset = np.hstack((X, Y))

    # 训练集和测试集划分
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(X, Y, test_size=0.3, random_state=2)

    # model_xgboost_2 = XGBRegressor(learning_rate=0.45, max_depth=8, min_child_weight=3,
    #                                n_estimators=101)

    # ............................................................................................

    model_xgboost_2 = RandomizedSearchCV(
        XGBRegressor(), param_distributions={'n_estimators': range(1, 1000, 50),
                                             'max_depth': range(2, 14, 2),
                                             'learning_rate': np.linspace(0.01, 2, 10),
                                             'min_child_weight': range(1, 9, 1)},
        cv=5, scoring='neg_root_mean_squared_error', n_iter=300, n_jobs=-1)
    # model_xgboost_2.fit(train_x, train_y)

    with open("model_xgboost_2.dat", "rb") as my_profile:
        model_xgboost_2 = pickle.load(my_profile)

    # with open("model_xgboost_2.dat", "wb") as my_profile:
    #     pickle.dump(model_xgboost_2, my_profile)

    y_pred = model_xgboost_2.predict(test_x)
    y_pred_2 = model_xgboost_2.predict(train_x)
    pred_vs_observed_vis(test_y, y_pred, train_y, y_pred_2)
    rmse, mape = get_error(test_y, y_pred)
    # plt.figure()
    diff = y_pred - test_y.reshape(1, -1)
    # plt.hist(diff)
    # plt.show()
    sns.distplot(diff, bins=10, kde=False, color='k')
    # 设置坐标刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    print("xgboost's regressor, test set ,RMSE: ", rmse, " ,MAPE: ", mape)
    rmse, mape = get_error(train_y, y_pred_2)
    print("xgboost's regressor, train set ,RMSE: ", rmse, " ,MAPE: ", mape)

    # ............................................................................................

    # model_elastic_2 = GridSearchCV(ElasticNetCV(), param_grid={
    #     'l1_ratio': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}, cv=5)
    # # model_elastic_2.fit(train_x, train_y)
    #
    # with open("model_elastic_net_2", "wb") as my_profile:
    #     pickle.dump(model_elastic_2, my_profile)

    with open("model_elastic_net_2", "rb") as my_profile:
        model_elastic_2 = pickle.load(my_profile)

    y_pred = model_elastic_2.predict(test_x)
    y_pred_2 = model_elastic_2.predict(train_x)
    pred_vs_observed_vis(test_y, y_pred, train_y, y_pred_2)
    rmse, mape = get_error(test_y, y_pred)
    # plt.figure()
    diff = y_pred - test_y.reshape(1, -1)
    # plt.hist(diff)
    # plt.show()
    sns.distplot(diff, bins=10, kde=False, color='k')
    # 设置坐标刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    print("elastic net, test set, RMSE: ", rmse, " ,MAPE: ", mape)
    rmse, mape = get_error(train_y, y_pred_2)
    print("elastic net, train set, RMSE: ", rmse, " ,MAPE: ", mape)

    # ............................................................................................

    # # 自动选择合适的参数
    # svr = GridSearchCV(svm.SVR(), param_grid={"kernel": ("linear", 'rbf'), "C": np.logspace(-3, 3, 7),
    #                                           "gamma": np.logspace(-3, 3, 7)}, cv=5)
    # svr.fit(X, Y)
    # # model_svm = svm.SVR()
    # # model_svm.fit(train_x, train_y)
    # model_svm = svr

    with open("model_svm_2", "rb") as my_profile:
        model_svm = pickle.load(my_profile)

    # with open("model_svm_2", "wb") as my_profile:
    #     pickle.dump(model_svm, my_profile)

    y_pred = model_svm.predict(test_x)
    y_pred_2 = model_svm.predict(train_x)
    pred_vs_observed_vis(test_y, y_pred, train_y, y_pred_2)
    rmse, mape = get_error(test_y, y_pred)
    # plt.figure()
    diff = y_pred - test_y.reshape(1, -1)
    # plt.hist(diff)
    # plt.show()
    sns.distplot(diff, bins=10, kde=False, color='k')
    # 设置坐标刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    print("support vector machine, test set, RMSE: ", rmse, " ,MAPE: ", mape)
    rmse, mape = get_error(train_y, y_pred_2)
    print("support vector machine, train set, RMSE: ", rmse, " ,MAPE: ", mape)

    # ............................................................................................


def explainable_ai_with_shapley_value_for_xgboost(model, X):
    X = pd.DataFrame(X, columns=['2-cycle', 'max-2', 'minimum', 'variance', 'skewness'])
    # X = X.round(2)
    # X保留两位小数

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
    # shap_values = shap.TreeExplainer(model).shap_values(X)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # plt.xlabel(fontsize=16)
    # plt.ylabel(fontsize=16)

    # 设置坐标刻度字体大小
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)

    shap.summary_plot(shap_values, X)
    shap.dependence_plot("2-cycle", shap_values, X)
    shap.dependence_plot("max-2", shap_values, X)
    shap.dependence_plot("minimum", shap_values, X)
    shap.dependence_plot("variance", shap_values, X)
    shap.dependence_plot("skewness", shap_values, X)

    shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X.iloc[:, :])
    shap.dependence_plot(('skewness', 'skewness'), shap_interaction_values, X.iloc[:, :])

    # visualize the single prediction's explanation (use matplotlib=True to avoid Javascript)
    # 选择一些寿命有代表性的（较高或者较低）电池数据来可视化
    index_representative_battery = [1, 2, 41, 42, 44]
    for i in range(len(index_representative_battery)):
        shap.force_plot(explainer.expected_value, shap_values[index_representative_battery[i], :],
                        X.iloc[index_representative_battery[i], :], matplotlib=True)

    # shap.force_plot(explainer.expected_value, shap_values[1, :], X.iloc[5, :], matplotlib=True)
    # shap.force_plot(explainer.expected_value, shap_values[2, :], X.iloc[15, :], matplotlib=True)
    # shap.force_plot(explainer.expected_value, shap_values[41, :], X.iloc[25, :], matplotlib=True)
    # shap.force_plot(explainer.expected_value, shap_values[42, :], X.iloc[35, :], matplotlib=True)


if __name__ == '__main__':

    # dataset = dataset_after_feature_engineering

    # only_xgboost(

    # only_xgboost_new()

    # xgboost_tuning()

    model_comparison()

    # explainable_ai_for_battery_lifetime_prediction()

    # compare_various_model()