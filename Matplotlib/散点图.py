'''
散点图——判断变量之间是否存在数量关联趋势，展示离群点（分布规律）
示例：北京3月、10月白天最高气温数据，试找出气温随时间变化的某种规律
'''

__author__ = 'Jim He'

import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    march_temp = [11, 17, 16, 11, 12, 11, 12, 6, 6, 7, 8, 9, 12, 15, 14, 17,
                  18, 21, 16, 17, 20, 14, 15, 15, 15, 19, 21, 22, 22, 22, 23]
    october_temp = [26, 26, 28, 19, 21, 17, 16, 19, 18, 20, 20, 19, 22, 23,
                    17, 20, 21, 20, 22, 15, 11, 15, 5, 13, 17, 10, 11, 13, 12, 13, 6]
    x_3 = [i for i in range(1, 32)]
    x_10 = [i for i in range(51, 82)]

    fig = plt.figure(figsize=(18, 10), dpi=80)
    plt.scatter(x_3, march_temp, c='g', marker='*', label='3月份天气', lw=4)
    plt.scatter(x_10, october_temp, c='r', marker='o', label='10月份天气', lw=4)
    plt.xlabel('3月份                                                                       10月份', fontsize=18)
    plt.ylabel('气温值/℃', fontsize=18)
    plt.title('北京3月份和10月份天气', fontsize=20)
    plt.legend(loc=0, fontsize=16)
    # 设置刻度
    _x_3_tick_labels = ['3月{}日'.format(i) for i in x_3]
    _x_10_tick_labels = ['10月{}日'.format(i-50) for i in x_10]
    plt.xticks(x_3[::3]+x_10[::3], _x_3_tick_labels[::3] + _x_10_tick_labels[::3], fontsize=14, rotation=45)
    plt.yticks(range(min(march_temp + october_temp), max(march_temp + october_temp)+2)[::2], fontsize=14)
    plt.tight_layout(10)
    plt.grid(True, ls='--', alpha=.5)
    plt.savefig('./scatter.png')
    plt.show()
