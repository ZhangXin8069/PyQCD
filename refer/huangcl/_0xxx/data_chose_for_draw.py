import pandas as pd
import numpy as np
import os
from C_pt_load import load_Ratio_data, this_path
from tool import *

#t_sep_list = [8, 9, 10, 11, 12, 13, 14] 
#z_list     = [0, 10]
#n_remove = 3 # 拟合时去掉头尾的点数
#data_name = '4pt_cc'
#data_name = '4pt_jack'

def data_chose(z_,  t_sep_list_,  n_remove_,  data_name_,  only_z = 'none' ):
    # ..............输入..................
    # t_sep_list_ : 用于拟合的 t_sep 列表
    # z_          : 与z=0, 一同拟合的z 值
    # n_remove_    : 头尾一同去掉的点个数

    z_list_ = [0, z_]

    # 读取所有数据并计算 ti_fit, ratio_fit, err_fit
    ratio_all_ = load_Ratio_data(this_path+'/result/Ratio_data/Ratio_data_'+ data_name_ +'.npz')

    
    #ratio_all = symmetric_process_optimized(ratio_all_) # 对数据进行对称化
    #ratio_all = symmetric_process_fit(ratio_all_)       # 对数据进行对称化， 并去掉相同的点。
    ratio_all = ratio_all_ # 不处理

    

    # data:   [z, t_sep, ti_sep, mean, std] 
    ratio_data = ratio_all['data']
    # samples: [samples0, samples1, ..., samples1N]
    ratio_samples = ratio_all['samples']


    # 挑选出符合要求的 z, tsep 数据
    condition1 = np.in1d(ratio_data[:, 0], z_list_ ) & np.in1d(ratio_data[:, 1], t_sep_list_)
    if only_z == 'yes':
        condition1 = np.in1d(ratio_data[:, 0], [z_]) & np.in1d(ratio_data[:, 1], t_sep_list_)


    # 挑选出符合条件的 ti_sep 数据：
    # 对于每个 t_sep，排除 ti_sep 在头尾 n_remove 个点的数据。
    mask = np.zeros(len(ratio_data), dtype=bool)
    for t_sep in t_sep_list_:
        mask |= (ratio_data[:, 1] == t_sep) & (
            (ratio_data[:, 2] < n_remove_) | (ratio_data[:, 2] > t_sep - n_remove_)
        )

    condition2 = ~mask  # 取反，选择不符合 mask 条件的行

    # 组合条件
    condition = condition1 & condition2



    row_indices = np.where(condition)[0]

    z_set               =   ratio_data[row_indices, 0]
    tsep_set            =   ratio_data[row_indices, 1]
    ti_sep_set          =   ratio_data[row_indices, 2] 
    ratio_mean_set      =   ratio_data[row_indices, 3]
    err_set             =   ratio_data[row_indices, 4]

    ratio_samples_fit  =   ratio_samples[row_indices, :]

    return  z_set, tsep_set, ti_sep_set, ratio_mean_set, err_set, ratio_samples_fit, z_list_, t_sep_list_, n_remove_


def dataGPD_chose(z_,  t_sep_list_,  n_remove_,  data_name_,  only_z = 'none' ):
    # ..............输入..................
    # t_sep_list_ : 用于拟合的 t_sep 列表
    # z_          : 与z=0, 一同拟合的z 值
    # n_remove_    : 头尾一同去掉的点个数

    z_list_ = [0, z_]

    # 读取所有数据并计算 ti_fit, ratio_fit, err_fit
    ratio_all_ = load_Ratio_data(this_path+'/result/Ratio_data/GPD/'+ data_name_ +'.npz')

    
    #ratio_all = symmetric_process_optimized(ratio_all_) # 对数据进行对称化
    #ratio_all = symmetric_process_fit(ratio_all_)       # 对数据进行对称化， 并去掉相同的点。
    ratio_all = ratio_all_ # 不处理

    

    # data:   [z, t_sep, ti_sep, mean, std] 
    ratio_data = ratio_all['data']
    # samples: [samples0, samples1, ..., samples1N]
    ratio_samples = ratio_all['samples']


    # 挑选出符合要求的 z, tsep 数据
    condition1 = np.in1d(ratio_data[:, 0], z_list_ ) & np.in1d(ratio_data[:, 1], t_sep_list_)
    if only_z == 'yes':
        condition1 = np.in1d(ratio_data[:, 0], [z_]) & np.in1d(ratio_data[:, 1], t_sep_list_)


    # 挑选出符合条件的 ti_sep 数据：
    # 对于每个 t_sep，排除 ti_sep 在头尾 n_remove 个点的数据。
    mask = np.zeros(len(ratio_data), dtype=bool)
    for t_sep in t_sep_list_:
        mask |= (ratio_data[:, 1] == t_sep) & (
            (ratio_data[:, 2] < n_remove_) | (ratio_data[:, 2] > t_sep - n_remove_)
        )

    condition2 = ~mask  # 取反，选择不符合 mask 条件的行

    # 组合条件
    condition = condition1 & condition2



    row_indices = np.where(condition)[0]

    z_set               =   ratio_data[row_indices, 0]
    tsep_set            =   ratio_data[row_indices, 1]
    ti_sep_set          =   ratio_data[row_indices, 2] 
    ratio_mean_set      =   ratio_data[row_indices, 3]
    err_set             =   ratio_data[row_indices, 4]

    ratio_samples_fit  =   ratio_samples[row_indices, :]

    return  z_set, tsep_set, ti_sep_set, ratio_mean_set, err_set, ratio_samples_fit, z_list_, t_sep_list_, n_remove_


if __name__ == "__main__":
    data_name_ = 'L24x72_dhxmeang1_pz4'
    z_set, tsep_set, ti_sep_set, ratio_mean_set, err_set, ratio_samples_fit, z_list_, t_sep_list_, n_remove_ = data_chose(0,  [7],  0,  data_name_,  only_z = 'yes')
    #print(z_list_)
    print('sample(0-4):tsep7, tins4, z0')
    print( ratio_samples_fit[4, 0])
    print( ratio_samples_fit[4, 1])
    print( ratio_samples_fit[4, 2])
    print( ratio_samples_fit[4, 3])
    print( ratio_samples_fit[4, 4])
   

    data_name_ = 'L24x72_dhxmeang1_pz4'
    z_set, tsep_set, ti_sep_set, ratio_mean_set, err_set, ratio_samples_fit, z_list_, t_sep_list_, n_remove_ = data_chose(3,  [11],  0,  data_name_,  only_z = 'yes')
    #print(z_list_)
    #print(ratio_samples_fit.shape)
    print('sample(2995-2999):tsep11, tins7, z3')
    #print(np.mean(ratio_samples_fit, axis=1))
    print( ratio_samples_fit[7, 2995])
    print( ratio_samples_fit[7, 2996])
    print( ratio_samples_fit[7, 2997])
    print( ratio_samples_fit[7, 2998])
    print( ratio_samples_fit[7, 2999])