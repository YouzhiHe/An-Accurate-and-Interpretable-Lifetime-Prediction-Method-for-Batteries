import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.interpolate import interp1d
import matplotlib

bat_dict = pickle.load(open(r'.\Data\batch_all.pkl', 'rb'))


# 1:

def dataset():
    min_cycle_life = 2000
    max_cycle_life = 1

    for i in bat_dict.keys():
        t = bat_dict[i]['cycle_life']
        if t < min_cycle_life:
            min_cycle_life = t
        if t > max_cycle_life:
            max_cycle_life = t

    # 颜色取100个点
    colors = matplotlib.cm.get_cmap('Wistia')
    col = colors(np.linspace(0, 1, 101))

    for i in bat_dict.keys():
        x = bat_dict[i]['summary']['cycle']
        y = bat_dict[i]['summary']['QD']

        # 尝试绘制光滑曲线
        # x_new = np.linspace(x.min(), x.max(), 1000)
        # func = interp1d(x, y, kind='cubic')
        # y_new = func(x_new)

        # 正则化为0-100内的
        cycle_life = bat_dict[i]['cycle_life']

        max_diff = max_cycle_life - min_cycle_life
        max_diff = max_diff/2

        cur_diff = cycle_life - min_cycle_life

        if cur_diff >= max_diff:
            cur_diff = max_diff
        index = float(cur_diff) / max_diff
        index = index * 100
        index = int(index)

        # cycle_life = float(cycle_life - min_cycle_life) / (max_cycle_life - min_cycle_life)
        # cycle_life = cycle_life * 100
        # cycle_life = cycle_life/2
        # cycle_life = cycle_life + 50
        # cycle_life = int(cycle_life)

        plt.plot(x, y, color=col[index])

        # plt.plot(x, y, color='blue')

    # 设置坐标刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # 设置坐标轴范围
    plt.xlim((0, 1000))
    plt.ylim((0.90, 1.10))

    # 设置坐标轴刻度
    my_x_ticks = np.arange(0, 1001, 200)
    my_y_ticks = np.arange(0.90, 1.11, 0.05)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    plt.xlabel('Cycle number', fontsize=16)
    plt.ylabel('Discharge capacity (Ah)', fontsize=16)

    plt.show()


# 2:

def dis_cap_vs_cycle():
    a = 'b1c9'
    b = 'b2c11'

    print(bat_dict.keys())

    x1 = bat_dict[a]['summary']['cycle']
    y1 = bat_dict[a]['summary']['QD']
    x2 = bat_dict[b]['summary']['cycle']
    y2 = bat_dict[b]['summary']['QD']

    y1[0] = 1.080

    plt.plot(x1, y1, color='blue')
    plt.plot(x2, y2, color='green')

    # 设置坐标刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # 设置坐标轴范围
    plt.xlim((0, 110))
    plt.ylim((1.065, 1.095))

    # 设置坐标轴刻度
    my_x_ticks = np.arange(0, 111, 20)
    my_y_ticks = np.arange(1.065, 1.096, 0.01)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    plt.xlabel('Cycle number', fontsize=16)
    plt.ylabel('Discharge capacity (Ah)', fontsize=16)
    plt.legend(('Cell A', 'Cell B'), fontsize=16)

    plt.show()


# 3:

def dis_cap_vs_vol():
    battery = 'b1c1'

    plt.plot(bat_dict[battery]['cycles']['9']['Qd'], bat_dict[battery]['cycles']['9']['V'], color='blue')
    plt.plot(bat_dict[battery]['cycles']['99']['Qd'], bat_dict[battery]['cycles']['99']['V'], color='green')

    plt.xlabel('Discharge capacity (Ah)', fontsize=16)
    plt.ylabel('Voltage (V)', fontsize=16)

    # 设置坐标刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # 设置坐标轴范围
    plt.xlim((0, 1.2))
    plt.ylim((2.0, 3.6))

    # 设置坐标轴刻度
    my_x_ticks = np.arange(0, 1.21, 0.2)
    my_y_ticks = np.arange(2.0, 3.61, 0.4)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    plt.legend(('Cycle 10', 'Cycle 100'), fontsize=16)
    plt.show()

    pass


# 4:

def vol_vs_Q100_Q10():
    a = 'b1c9'
    b = 'b2c11'

    x1 = bat_dict[a]['cycles']['99']['V']
    x2 = bat_dict[b]['cycles']['99']['V']
    y1 = bat_dict[a]['cycles']['99']['Qd'] - bat_dict[a]['cycles']['9']['Qd'][20:]
    y2 = bat_dict[b]['cycles']['99']['Qd'] - bat_dict[b]['cycles']['9']['Qd'][7:]

    # from scipy.interpolate import splrep
    # # 想画出平滑曲线
    # # x_smooth = np.linspace(x.min(), x.max(), 500)
    # # y_smooth1 = make_interp_spline(x, y1)(x_smooth)
    # # plt.plot(x_smooth, y_smooth1)
    #

    def get_new_x_y(x, y):
        # 尝试绘制光滑曲线
        x_new = np.linspace(x.min(), x.max(), 30)
        func = interp1d(x, y, kind='cubic')
        y_new = func(x_new)
        return x_new, y_new

    # x1, y1 = get_new_x_y(x1, y1)
    # x2, y2 = get_new_x_y(x2, y2)

    plt.plot(y1, x1, color='blue')
    plt.plot(y2, x2, color='green')

    # 设置坐标刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # 设置坐标轴范围
    # plt.xlim((0, 1.2))
    # plt.ylim((2.0, 3.6))

    # 设置坐标轴刻度
    my_x_ticks = np.arange(-0.1, 0.01, 0.02)
    my_y_ticks = np.arange(2.0, 3.61, 0.4)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    plt.xlabel('Q100 - Q10 (Ah)', fontsize=16)
    plt.ylabel('Voltage (V)', fontsize=16)
    plt.legend(('Cell A', 'Cell B'), fontsize=16)

    plt.show()

    pass


if __name__ == '__main__':
    # dataset()
    # dis_cap_vs_cycle()
    dis_cap_vs_vol()
    # vol_vs_Q100_Q10()
    pass