'''
Kaggle自行车预测比赛
by Jim 2018.11.15
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit, learning_curve, GridSearchCV, cross_val_score, train_test_split


def generate_feature(data_df):
    # print(data_df[:20])
    # 提取时间特征——hour
    data_df['hour'] = pd.DatetimeIndex(data_df.datetime).hour
    # print(data_df[:20])
    data_df.drop(['datetime', 'holiday', 'workingday', 'atemp',
                  'casual', 'registered'], axis=1, inplace=True)
    # print(data_df[:20])
    data_feature_con = data_df[['temp', 'humidity', 'windspeed', 'hour']]
    data_feature_con = data_feature_con.fillna('NA')  # 谨防缺失值
    x_dict_con = data_feature_con.T.to_dict().values()
    # print(x_dict_con)

    data_feature_cat = data_df[['season', 'weather']]
    data_feature_cat = data_feature_cat.fillna('NA')
    x_dict_cat = data_feature_cat.T.to_dict().values()

    x_vec_con = DictVectorizer(sparse=False).fit_transform(x_dict_con)
    x_vec_cat = DictVectorizer(sparse=False).fit_transform(x_dict_cat)

    x_vec_con = StandardScaler().fit_transform(x_vec_con)
    x_vec_cat = OneHotEncoder(sparse=False).fit_transform(x_vec_cat)

    x_vec = np.concatenate((x_vec_con, x_vec_cat), axis=1)
    # print(x_vec[:30])

    # 取出标签
    y_vec = data_df['count'].astype(float).values
    # print(y_vec[:20], type(y_vec))

    return x_vec, y_vec


def plot_learning_curve(estimator, title, x, y, ylim=None, cv=None, n_jobs=1, train_size=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('训练样本集大小')
    plt.ylabel('R2')
    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, n_jobs=n_jobs,
                                                            train_sizes=train_size, cv=cv)
    # print(train_sizes.shape, train_scores.shape, test_scores.shape)  # (10,) (10, 10) (10, 10)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid(True)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="训练集得分")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="验证集得分")

    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    data = pd.read_csv('./data/train.csv')
    # print(data[:20])
    x, y = generate_feature(data)
    # print(x[:20], x.shape, type(x))
    # print(y[:20], y.shape, type(y))

    # -------------------机器学习算法----------------- #
    # cv = ShuffleSplit(n_splits=3, test_size=.2, train_size=.8, random_state=0)
    # print('Ridge')
    # for train_idx, test_idx in cv.split(x):
    #     svc = Ridge(alpha=1.0, max_iter=1000, solver='saga')
    #     svc.fit(x[train_idx], y[train_idx])
    #     print('train score: {0: .3f}, test score: {1: .3f}'.format(
    #         svc.score(x[train_idx], y[train_idx]), svc.score(x[test_idx], y[test_idx])
    #     ))
    #
    # print('SVR')
    # for train_idx, test_idx in cv.split(x):
    #     svc = SVR(kernel='rbf', C=10, gamma='auto', cache_size=1024)
    #     svc.fit(x[train_idx], y[train_idx])
    #     print('train score: {0: .3f}, test score: {1: .3f}'.format(
    #         svc.score(x[train_idx], y[train_idx]), svc.score(x[test_idx], y[test_idx])
    #     ))
    #
    # print('RandomForestRegressor')
    # for train_idx, test_idx in cv.split(x):
    #     svc = RandomForestRegressor(n_estimators=20, max_features=.5, n_jobs=-1)
    #     svc.fit(x[train_idx], y[train_idx])
    #     print('train score: {0: .3f}, test score: {1: .3f}'.format(
    #         svc.score(x[train_idx], y[train_idx]), svc.score(x[test_idx], y[test_idx])
    #     ))

    # 在挑选特征后相比于寒老师的全特征空间前两个模型都有小幅性能提升，
    # 由于本问题是非线性问题，所以 Ridge 模型应该无法胜任
    # SVR 还没有仔细弄懂，也就没有仔细调参
    # RandomForestRegressor 性能最好，但是过拟合了，需要调参
    # 综合来看，需要细粒度调参，再由 RandomForestRegressor 结果可知，需要更多的特征
    # -------------------机器学习算法----------------- #

    # -------------------基于 RandomForestRegressor 做CV----------------- #
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, train_size=.8, random_state=0)
    # tuned_parameters = [{'n_estimators': [10, 15, 20, 25, 100, 200, 500]}]
    # scores = ['r2']
    # for score in scores:
    #     print(score)
    #     svc = GridSearchCV(RandomForestRegressor(n_jobs=-1), param_grid=tuned_parameters,
    #                        n_jobs=-1, cv=5, return_train_score=False)
    #     svc.fit(x_train, y_train)
    #     print('当前最优模型：', svc.best_estimator_)  # n_estimators=500
    #     print('当前最优模型得分：', svc.best_score_)  # 0.6884305077766761 与寒老师的结果差距很大，需要更多的特征
    #     cv_result = svc.cv_results_
    #     for val_mean, val_std, param in zip(cv_result['mean_test_score'],
    #                                         cv_result['std_test_score'], cv_result['params']):
    #         print('%.3f (+/-%.03f) for %r' % (val_mean, val_std, param))
    # -------------------基于 RandomForestRegressor 做CV----------------- #

    # -------------------学习曲线----------------- #
    # 简单的判定模型的学习状态——欠拟合/过拟合
    cv = ShuffleSplit(n_splits=5, test_size=.2, train_size=.8, random_state=0)
    # svc = RandomForestRegressor(n_estimators=20, n_jobs=-1)  # 妥妥的过拟合啊
    # 改善了过拟合情况，但是模型性能没有提升，需要等多的特征
    svc = RandomForestRegressor(n_estimators=20, n_jobs=-1, max_depth=10, min_samples_split=5)
    title = '学习曲线（RandomForestRegressor(n_estimators=20)）'
    plot_learning_curve(svc, title, x, y, (.0, 1.01), cv=cv, n_jobs=-1)
    # -------------------学习曲线----------------- #
