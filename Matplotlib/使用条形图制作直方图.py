'''
美国2004年人口普查发现有1.24亿人在离家相对较远的地方工作，以下是抽样统计数据
统计后的数据是无法绘制直方图的，因为直方图就是为了展示数据的相关分布情况，而统计
后的数据默认已经提取了相应的分布规律，所以不必使用直方图的了
这里可以使用条形图拟合出类直方图
'''

__author__ = 'Jim He'

import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    interval = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 90]  # 住址与公司时长
    width = [5, 5, 5, 5, 5, 5, 5, 5, 5, 15, 30, 60]  # 组距
    quantity = [836, 2737, 3723, 3926, 3596, 1438, 3273, 642, 824, 613, 215, 47]  # 各组人口数据

    fig = plt.figure(figsize=(14, 8), dpi=80)
    plt.bar(interval, quantity, width=width, align='edge')
    plt.xticks(interval + [interval[-1] + width[-1]], fontsize=14)
    plt.yticks(range(0, max(quantity) + 500)[::500], fontsize=14)
    plt.xlabel('住址-公司时程 分钟', fontsize=16)
    plt.ylabel('人口数量', fontsize=16)
    plt.title('美国2004年人口普查->住址-公司时程人口分布', fontsize=18)
    plt.grid(True, alpha=.2, ls='--', color='r')
    plt.show()

