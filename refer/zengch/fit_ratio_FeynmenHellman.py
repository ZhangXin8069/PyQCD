import sys
sys.path.append('/public/home/zengch/LQCD/input_file')
sys.path.append('/public/home/zengch/LQCD/tool')

import pandas as pd
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
import time
import matplotlib.pyplot as plt
import os
from tool import *
from data_chose import data_chose
from C_pt_load import this_path

import pdb # 用于调试代码 pdb.set_trace()


file_path =  '/public/home/zengch/LQCD/renorma/result/Ratio_data/FeynmanHellman'
conf = 'L24x72_dhxmean'
Pz = 4
nr_test = 2
t_sep_star = 6
t_sep_end = 12




def fit_ratio_FeynmenHellman(z_fit, nr_,t_set_do, t_set_up):
    file_read = f'{file_path}/Delta_Ratio_data_{conf}_pz{Pz}_nr{nr_}.npy'
    c0_data = np.load(file_read)[z_fit, t_set_do:t_set_up+1,:]
    std_err = np.std(c0_data, axis = 1)

    n_data = np.shape(c0_data)[0]
    n_samples = np.shape(c0_data)[1]

    fit_results_list = []

    c0_in, chi_in = fit_ratio_mean_FeynmenHellman(z_fit, nr_, t_set_do, t_set_up)
  

    for i_sam in range(n_samples):
        c0_data_i = c0_data[:, i_sam]

        def cost_function(c0_th):
            chi2_all = (c0_data_i - c0_th)**2. / std_err **2. 
          
            return np.mean(chi2_all)
        
        m = Minuit(cost_function, c0_th = c0_in)

      
        # 运行拟合
        m.migrad()  # 最小化

        c0  = m.values['c0_th']
        chi2_mean = cost_function(c0)

        
     
            
        # 将拟合结果存储到 DataFrame 中
        fit_results_list.append({
            "sample_i": i_sam,
            "c0": c0,
            "chi2_mean": chi2_mean
        })

    # 保存拟合结果到 CSV 文件
    fit_results = pd.DataFrame(fit_results_list)
    c0_save_path = f'/public/home/zengch/LQCD/renorma/result/ratio_tsep_tisep_FeynmenHellman/{conf}_pz{Pz}/ex{nr_}'
    os.makedirs(c0_save_path, exist_ok=True)
    fit_results.to_csv(f"{c0_save_path}/z{z_fit}_tsep{t_set_do}_{t_set_up}.csv", index=False)
    
    c0_mean = np.mean(fit_results['c0'])
    c0_std  = np.std(fit_results['c0'])
    return c0_mean, c0_std, chi_in

def fit_ratio_mean_FeynmenHellman(z_fit, nr_,  t_set_do, t_set_up):
    file_read = f'{file_path}/Delta_Ratio_data_{conf}_pz{Pz}_nr{nr_}.npy'
    c0_data = np.load(file_read)[z_fit, t_set_do:t_set_up+1,:] # shape = [z, tsep , samples]
    std_err = np.std(c0_data, axis = 1)

    n_data = np.shape(c0_data)[0]
    n_samples = np.shape(c0_data)[1]

    fit_results_list = []

   
    c0_data_mean = np.mean(c0_data, axis = 1)

    def cost_function(c0_th):
        chi2_all = (c0_data_mean - c0_th)**2. / std_err **2. 
        
        return np.mean(chi2_all)
    
    c0_in = 0.58
    m = Minuit(cost_function, c0_th = c0_in)

    
    # 运行拟合
    m.migrad()  # 最小化

    c0  = m.values['c0_th']
    chi2_mean = cost_function(c0)

    print('z=', z_fit)
    print('c0=',  c0)
    print('chi2_mean=', chi2_mean)
    return c0, chi2_mean

def c0_vs_z_FeynmenHellman(nr_, t_sep_do, t_sep_up):

    #t_sep_list = [8, 9, 10, 11, 12, 13, 14] 
    #n_remove =  3 # 拟合时去掉头尾的点数

    res_list = []
    for z_i in range(20):

        c0_z_mean, c0_z_std, chi2mean = fit_ratio_FeynmenHellman(z_i, nr_, t_sep_do, t_sep_up)

        res_list.append({ 'z'       : z_i,
                         'c0_z_mean': c0_z_mean, 
                         'c0_z_std' : c0_z_std,
                         'chi2_mean': chi2mean               
        }
        )

        fit_result  = pd.DataFrame(res_list)
        ratio_fit_path = f'{this_path}/result/ratio_fit_result_FeynmenHellman/{conf}_pz{Pz}/ex{nr_}'
        os.makedirs(ratio_fit_path, exist_ok=True)
        fit_result.to_csv(f'{ratio_fit_path}/c0_vs_z_{t_sep_do}_{t_sep_up}.csv', index=False)
        #fit_result.to_csv(f'ratio_fit_result/a_test'+ data_name +f'_ex{n_remove_}_test{t_sep_list_[0]}_{t_sep_list_[-1]}.csv', index=False)
     
    return 0



#fit_ratio_mean_FeynmenHellman(0, nr_test, t_sep_star, t_sep_end)
#fit_ratio_FeynmenHellman(0, nr_test, t_sep_star, t_sep_end)

c0_vs_z_FeynmenHellman(nr_test, t_sep_star, t_sep_end)