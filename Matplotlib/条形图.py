'''
条形图——绘制离散型的数据，能够一眼看出各个数据的大小，比较数据之间的差别（统计）
示例：请直观的展示2017年内地前20名的电影票房的电影
'''

__author__ = 'Jim He'

import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    films = ['战狼2', '速度与激情8', '功夫瑜伽', '西游伏妖篇', '变形金刚5：\n最后的骑士',
             '摔跤吧！爸爸', '加勒比海盗5：\n死无对证', '金刚：骷髅岛', '极限特工：\n终极回归',
             '生化危机6：\n终章', '乘风破浪', '神偷奶爸3', '智取威虎山', '大闹天竺',
             '金刚狼3：\n殊死一战', '蜘蛛侠：\n英雄归来', '悟空传', '银河护卫队2', '情圣', '新木乃伊'
             ]
    money = [56.01, 26.94, 17.53, 16.49, 15.45, 12.96, 11.8, 11.61, 11.28, 11.12, 10.49,
             10.3, 8.75, 7.55, 7.32, 6.99, 6.88, 6.86, 6.58, 6.23
             ]
    '''
    # ----------------垂直显示-------------- #
    fig = plt.figure(figsize=(16, 8), dpi=80)
    plt.bar(range(len(films)), money, width=.4, color='m')
    plt.xticks(range(len(films)), films, rotation=45, fontsize=14)
    plt.yticks(range(0, int(max(money)) + 2)[::4], fontsize=14)
    plt.xlabel('上映电影', fontsize=16)
    plt.ylabel('票房 亿元', fontsize=16)
    plt.title('2017年20部高票房电影', fontsize=18)
    plt.grid(True, alpha=.5, ls='--')
    plt.tight_layout(5)
    plt.show()
    # ----------------垂直显示-------------- #
    '''

    '''
    # ----------------水平显示-------------- #
    films = ['战狼2', '速度与激情8', '功夫瑜伽', '西游伏妖篇', '变形金刚5：最后的骑士',
             '摔跤吧！爸爸', '加勒比海盗5：死无对证', '金刚：骷髅岛', '极限特工：终极回归',
             '生化危机6：终章', '乘风破浪', '神偷奶爸3', '智取威虎山', '大闹天竺',
             '金刚狼3：殊死一战', '蜘蛛侠：英雄归来', '悟空传', '银河护卫队2', '情圣', '新木乃伊'
             ]
    fig = plt.figure(figsize=(16, 10), dpi=80)
    plt.barh(range(len(films)), money[::-1], height=.5, color='orange')
    plt.yticks(range(len(films)), films, fontsize=14)
    plt.xticks(range(0, int(max(money)) + 2)[::4], fontsize=14)
    plt.ylabel('上映电影', fontsize=16)
    plt.xlabel('票房 亿元', fontsize=16)
    plt.title('2017年20部高票房电影', fontsize=18)
    plt.grid(True, alpha=.5, ls='--')
    plt.tight_layout(5)
    plt.show()
    # ----------------水平显示-------------- #
    '''

    # ----------------综合应用-------------- #
    films = ['猩球崛起3：终极之战', '敦刻尔克', '蜘蛛侠：英雄归来', '战狼2']
    money_1 = [15746, 312, 4497, 319]
    money_2 = [12357, 156, 2045, 168]
    money_3 = [2358, 399, 2358, 362]
    bar_width = 0.2
    _x_14 = [i for i in range(len(films))]
    _x_15 = [i + bar_width for i in _x_14]
    _x_16 = [i + bar_width * 2 for i in _x_14]
    fig = plt.figure(figsize=(16, 8), dpi=80)
    plt.bar(_x_14, money_1, width=bar_width, color='r', label='9月14日')
    plt.bar(_x_15, money_2, width=bar_width, color='g', label='9月15日')
    plt.bar(_x_16, money_3, width=bar_width, color='b', label='9月16日')
    plt.xticks(_x_15, films, fontsize=16)
    plt.yticks(fontsize=14)
    plt.ylabel('票房 万元', fontsize=16)
    plt.legend(loc=0, fontsize=16)
    plt.show()
    # ----------------综合应用-------------- #
