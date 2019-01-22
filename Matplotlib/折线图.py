'''
折线图基本用法
示例：绘制从10点到12点的每一分钟的气温的折线图
'''

__author__ = "Jim He"

import random
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 虚拟x轴坐标
    x = [i / 2 for i in range(1, 122)]
    # print(x)
    # 虚拟y轴坐标
    y = [random.randint(20, 35) for _ in range(121)]
    # 设置图片大小
    fig = plt.figure(figsize=(16, 8), dpi=80)
    # 画图（默认折线图）
    plt.plot(x, y)
    # 设置x轴和y轴刻度
    x_labels = [
        '10:00', '10:10', '10:20', '10:30', '10:40', '10:50',
        '11:00', '11:10', '11:20', '11:30', '11:40', '11:50', '12:00'
    ]
    # print(x[::10])
    plt.xticks(x[::10], x_labels, fontsize=14, rotation=45)
    plt.yticks(range(min(y), max(y) + 2), fontsize=14)
    # 设置x轴和y轴标签
    plt.xlabel('记录时间', fontsize=16)
    plt.ylabel('气温值', fontsize=16)
    # 设置图片标题
    plt.title('10点-12点气温变化', fontsize=18)
    # 显示图片
    plt.show()
