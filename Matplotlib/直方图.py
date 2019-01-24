'''
直方图——绘制连续性的数据，展示一组或多组数据的分布状况（统计）
关于数据分组：
如果数据个数在 100 以内，通常分为 5-12 组
组距：每个小组两个端点的距离(自定义)
组数 = 极差 / 组距 = (max(nums) - min(nums)) / bin_width
eg: bin_width = 3, num_bins = (max(nums) - min(nums)) / bin_width
'''

__author__ = 'Jim He'

import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    film_time = [
        131, 98, 125, 131, 124, 139, 131, 117, 128, 108, 135, 138, 131, 102, 107, 114, 119, 128, 121, 142,
        127, 130, 124, 101, 110, 116, 117, 110, 128, 128, 115, 99, 136, 126, 134, 95, 138, 117, 111, 78,
        132, 124, 113, 150, 110, 117, 86, 95, 144, 105, 126, 130, 126, 130, 126, 116, 123, 106, 112, 138,
        123, 86, 101, 99, 136, 123, 117, 119, 105, 137, 123, 128, 125, 104, 109, 134, 132, 134, 156, 106,
        117, 127, 144, 139, 139, 119, 140, 83, 110, 102, 123, 107, 143, 115, 136, 118, 139, 123, 112, 118,
        125, 109, 119, 133, 112, 114, 122, 109, 106, 123, 116, 131, 127, 115, 118, 112, 135, 115, 148, 137,
        116, 103, 144, 83, 123, 111, 110, 111, 100, 154, 136, 100, 118, 119, 133, 134, 106, 129, 126, 110,
        111, 109, 141, 120, 117, 106, 149, 122, 122, 110, 118, 127, 121, 114, 125, 126, 114, 140, 103, 130,
        141, 117, 106, 114, 121, 114, 133, 137, 92, 121, 112, 146, 97, 137, 105, 98, 117, 112, 81, 97,
        139, 113, 134, 106, 144, 110, 137, 137, 111, 104, 117, 100, 111, 101, 110, 105, 129, 137, 112, 120,
        113, 133, 112, 83, 94, 146, 133, 101, 131, 116, 111, 84, 137, 115, 122, 106, 144, 109, 123, 116,
        111, 111, 133, 150, 123, 111, 110, 111, 100, 154, 136, 100, 118, 119, 133, 134, 106, 129, 126, 110,
        111, 109, 141, 120, 117, 106, 149, 122, 122, 110,
    ]
    # print(len(film_time))
    bin_width = 3  # 6
    # print(max(film_time) - min(film_time))
    num_bins = (max(film_time) - min(film_time)) // bin_width

    fig = plt.figure(figsize=(14, 8), dpi=80)
    res = plt.hist(film_time, bins=num_bins, color='#87CEFA')
    _ytick_labels = res[0].astype(int)
    # print(type(_ytick_labels), _ytick_labels.dtype)
    plt.yticks(range(min(_ytick_labels), max(_ytick_labels)+2)[::2], fontsize=14)
    plt.xticks(range(min(film_time), max(film_time)+bin_width, bin_width), fontsize=14)
    plt.xlabel('电影时长 分钟', fontsize=16)
    plt.ylabel('电影数量', fontsize=16)
    plt.title('电影时长分布', fontsize=18)
    plt.grid(True, ls='--', alpha=.05, color='r')

    plt.show()

