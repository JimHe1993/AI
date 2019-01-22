'''
折线图表明数据之间的变化趋势
示例：健身频次
'''

__author__ = 'Jim He'

import matplotlib.pyplot as plt
from matplotlib import font_manager

if __name__ == "__main__":
    # 设置中文字体
    zh_font = font_manager.FontProperties(fname='C:\Windows\Fonts\simhei.ttf')

    x = [year for year in range(11, 31)]  # 年龄
    male = [1, 0, 1, 1, 2, 4, 3, 2, 3, 4, 4, 5, 6, 5, 4, 3, 3, 1, 1, 1]  # 健身频数
    female = [1, 0, 3, 1, 2, 2, 3, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    fig = plt.figure(figsize=(14, 6), dpi=80)
    plt.plot(x, male, label='男性', color='b', lw=4, ls='--', alpha=.5)
    plt.plot(x, female, label='女性', color='m', lw=4, ls='-.', alpha=.8)
    plt.xticks(x, ['{}岁'.format(i) for i in x], rotation=45, fontsize=14, fontproperties=zh_font)
    plt.yticks(range(min(male), max(male) + 2), fontsize=14, fontproperties=zh_font)
    plt.xlabel('年龄', fontsize=16, fontproperties=zh_font)
    plt.ylabel('健身频次 周/次数', fontsize=16, fontproperties=zh_font)
    plt.title('不同年龄段健身频次', fontsize=18, fontproperties=zh_font)
    plt.grid(True, alpha=0.5, ls=':', lw=2)  # 网格配置
    plt.legend(loc=2, prop=zh_font)  # 图例显示配置
    plt.show()

    '''
    还有文本注释，如最大值最小值标记
    图片水印功能
    '''
