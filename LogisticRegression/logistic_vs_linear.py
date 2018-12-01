'''
尚学堂讲逻辑回归的例子——参考源码文件：logistic
主要对比了线性回归与逻辑回归对分类问题的解决方案
通过绘图，可以看出，LR能更精确的应对分类问题
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

from sklearn.linear_model import LinearRegression, LogisticRegression


def lr_model(model, x):
    return model.intercept_ + model.coef_ * x


def clf_model(model, x):
    return 1.0 / (1.0 + np.exp(-(model.intercept_ + model.coef_ * x)))


if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    n = 40
    x1 = norm.rvs(loc=2, size=n, scale=2)  # 均值为2，标准差为2的正态分布随机采样
    # print(x1, type(x1), x1.shape)  # <class 'numpy.ndarray'> (40,)
    # print(np.mean(x1), np.std(x1))  # 2.027962447754112 1.398094861776961
    x2 = norm.rvs(loc=8, size=n, scale=3)
    # print(x2, type(x2), x2.shape)  # <class 'numpy.ndarray'> (40,)
    # print(np.mean(x2), np.std(x2))  # 7.459692624902455 3.4299240743361903
    x = np.hstack((x1, x2))
    # print(x, x.shape)
    # print(np.r_[x1, x2])
    y = np.hstack((np.zeros(n), np.ones(n)))
    # print(y, y.shape)  # (80,)

    '''
    # 创建一个 10 * 4 点（point）的图，并设置分辨率为 80
    plt.figure(figsize=(10, 4), dpi=100)
    plt.xlim((-5, 20))
    plt.scatter(x, y, c=y)
    plt.xlabel('特征值')
    plt.ylabel('标签类别')
    plt.grid(True, linestyle='--', color='.1')
    plt.savefig('./picture/logistic_classify.png', bbox_inches="tight")
    plt.show()
    '''

    xs = np.linspace(-5, 15, 10)

    lr = LinearRegression()
    # reshape重新把array变成了80行1列二维数组,符合机器学习多维线性回归格式
    lr.fit(x.reshape(-1, 1), y)

    clf = LogisticRegression(solver='liblinear')
    clf.fit(x.reshape(-1, 1), y)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, c=y)
    plt.plot(x, lr_model(lr, x), 'o', c='blue', label='训练样本')
    plt.plot(xs, lr_model(lr, xs), '*', c='red', label='测试样本')
    plt.legend(loc='best')
    plt.xlabel('特征值')
    plt.ylabel('标签类别')
    plt.grid(True)
    plt.title('线性回归拟合')

    plt.subplot(1, 2, 2)
    plt.scatter(x, y, c=y)
    plt.plot(x, clf_model(lr, x).ravel(), 'o', c='blue', label='训练样本')
    plt.plot(xs, clf_model(lr, xs).ravel(), '*', c='red', label='测试样本')
    plt.legend(loc='best')
    plt.xlabel('特征值')
    plt.ylabel('标签类别')
    plt.grid(True)
    plt.title('逻辑回归拟合')
    plt.tight_layout(pad=0.4, w_pad=0, h_pad=1.0)
    plt.savefig("./picture/logistic_classify2.png", bbox_inches="tight")
    plt.show()
