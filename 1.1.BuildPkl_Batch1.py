# 解除警告
# UserWarning: h5py is running against HDF5 1.10.5 when it was built against 1.10.4, this may cause problems
# '{0}.{1}.{2}'.format(*version.hdf5_built_version_tuple)

# import os
# os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'

import h5py
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pickle

# data description： This analysis was originally performed in MATLAB, but here we also provide access information in
# python. In the MATLAB files (.mat), this data is stored in a struct. In the python files (.pkl), this data is
# stored in nested dictionaries.

# The data associated with each battery (cell) can be grouped into one of three categories: descriptors, summary,
# and cycle. Descriptors：for each battery include charging policy, cycle life, barcode and channel. Note that barcode
# and channel are currently not available in the pkl files). Summary data：include information on a per cycle basis,
# # including cycle number, discharge capacity, charge capacity, internal resistance, maximum temperature,
# # average temperature, minimum temperature, and charge time. Cycle data：include information within a cycle,
# # including time, charge capacity, current, voltage, temperature, discharge capacity. We also include derived vectors
# of discharge dQ/dV, linearly interpolated discharge capacity and linearly interpolated temperature.

# The LoadData files show how the data can be loaded and which cells were used for analysis in the paper.



matFilename = './Data/2017-05-12_batchdata_updated_struct_errorcorrect.mat'
f = h5py.File(matFilename)
print(list(f.keys()))
# 有蛮多数据集的，但是只有batch有用
# ['#refs#', '#subsystem#', 'batch', 'batch_date']

batch = f['batch']
print(list(batch.keys()))

# ['Vdlin',
#  'barcode',
#  'channel_id',
#  'cycle_life',
#  'cycles',
#  'policy',
#  'policy_readable',
#  'summary']

# shape[0]表示矩阵的第一个维度：即有多少行，或者多少种充电方式。
num_cells = batch['summary'].shape[0]
# print(batch['summary'].shape)
bat_dict = {}

# batch共有40多块电池的测试数据
# i代表某一块电池的测试数据

for i in range(num_cells):

    # cl代表电池寿命，例如第一颗电池的循环寿命为1190

    cl = f[batch['cycle_life'][i, 0]][()]
    # print("cl: ",cl)
    # cl = f[batch['cycle_life'][i, 0]][()]

    # policy代表电池的充电策略，为一个字符串，例如第一颗为 '3.6C(80%)-3.6C'
    policy = f[batch['policy_readable'][i, 0]][()].tobytes()[::2].decode()
    # policy = f[batch['policy_readable'][i,0]][()].tobytes()[::2].decode()

    # internal resistance
    # 长度为循环寿命的一维向量，大小表现为减小趋势 0.01674
    summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())

    # charge capacity Q代表容量的意思
    # 长度为循环寿命的一维向量，大小表现为减小趋势 1.07104
    summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())

    # discharge capacity
    # 长度为循环寿命的一维向量，大小表现为减小趋势 1.07069
    summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())

    # average temperature
    # 长度为循环寿命的一维向量，大小有波动 31 33 32 都存在
    summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())

    # minimum temperature
    # 长度为循环寿命的一维向量，大小有波动 29 30 31 都存在
    summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())

    # maximum temperature
    # 长度为循环寿命的一维向量，35 左右
    summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())

    # charge time
    # 长度为循环寿命的一维向量，13 左右
    summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())

    # cycle number
    summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())

    # 变成了字典 key-value，value是一个一维数组list
    summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
        summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
               'cycle': summary_CY}

    # cycle对应的是每次循环的细节内容
    # 每次循环 固定时间间隔记录信息 记录点数量不确定
    cycles = f[batch['cycles'][i, 0]]
    cycle_dict = {}

    # cycle_life = cycles['I'].shape[0] - 1
    # 遍历每个电池的每个充电循环（循环寿命）
    for j in range(cycles['I'].shape[0]):

        # 每个循环都是一个数组，但是总共记录了多少个点不是一定的

        #   current
        I = np.hstack(f[cycles['I'][j, 0]][()])
        # if j==5:
        #     print("每个循环记录的点数：",T.shape)

        #   charge capacity
        Qc = np.hstack(f[cycles['Qc'][j, 0]][()])

        #   discharge capacity
        Qd = np.hstack(f[cycles['Qd'][j, 0]][()])

        #    linearly interpolated discharge capacity
        Qdlin = np.hstack(f[cycles['Qdlin'][j, 0]][()])

        #   temperature
        T = np.hstack(f[cycles['T'][j, 0]][()])

        #   linearly interpolated temperature
        Tdlin = np.hstack(f[cycles['Tdlin'][j, 0]][()])

        #   voltage
        V = np.hstack(f[cycles['V'][j, 0]][()])

        #   vectors of discharge dQ/dV
        dQdV = np.hstack(f[cycles['discharge_dQdV'][j, 0]][()])

        #   time
        t = np.hstack(f[cycles['t'][j, 0]][()])

        # 这个cd会被用来提取特征，尤其是 Qdlin, Tdlin, dQdV等
        cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V': V, 'dQdV': dQdV, 't': t}

        # 每次循环所有的记录点也被存入一个字典类型的变量中
        cycle_dict[str(j)] = cd
        # if j == 5:
        #     print("cd: \n",cd)

    # 单个电池的完整信息，包括循环寿命，充电策略，所有充电情况概览，每次充放电循环的详细信息
    cell_dict = {'cycle_life': cl, 'charge_policy': policy, 'summary': summary, 'cycles': cycle_dict}
    # print("cell_dict: \n",cell_dict)

    key = 'b1c' + str(i)
    bat_dict[key] = cell_dict

# print("bat_dict: \n", bat_dict)

# 第43块电池的充放电概览：x轴 循环数，y轴 放电容量值
plt.plot(bat_dict['b1c43']['summary']['cycle'], bat_dict['b1c43']['summary']['QD'])

# 第43块电池的第10次循环详细情况：x轴 放电容量，y轴 电压值
plt.plot(bat_dict['b1c43']['cycles']['10']['Qd'], bat_dict['b1c43']['cycles']['10']['V'])

with open(r'batch1.pkl', 'wb') as fp:
    pickle.dump(bat_dict, fp)
