'''
Matplotlib 基本用法
示例：一天中间隔两小时的气温显示
'''

__author__ = 'Jim He'

import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    temp = [15, 13, 14.5, 17, 20, 25, 26, 26, 24, 22, 18, 15]  # 气温记录值
    # 构建横坐标
    x = range(1, 25, 2)
    # 设置图片大小——通过实例化一个fig实现
    fig = plt.figure(figsize=(10, 8), dpi=80)
    # 画图
    plt.plot(x, temp)
    # 设置x轴和y轴刻度
    time_labels = [
        '00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00',
        '16:00', '18:00', '20:00', '22:00'
    ]
    plt.xticks(x, time_labels, fontsize=14)
    plt.yticks(range(min(temp), max(temp)+2), fontsize=14)
    # 设置x轴和y轴标签
    plt.xlabel('一天中的时间', fontsize=18)
    plt.ylabel('温度', fontsize=18)
    # 保存图片
    # plt.savefig('./test1.svg')
    # 显示
    plt.show()
