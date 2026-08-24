import sys
sys.path.append('/public/home/zengch/LQCD/input_file')
sys.path.append('/public/home/zengch/LQCD/tool')
sys.path.append('/public/home/zengch/All_TMD_dependence')
import pandas as pd
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
import time
import matplotlib.pyplot as plt
import os
from tool import *
from fit_zr import th_ZR
from hB_data import a_len_set, fm_to_GeV, Nl_set, load_hB_data
from hB_data_FeynmenHellman import load_hB_data_FeynmenHellman
from constant import CA, CF, gammaE, pi, A_s
from scipy.interpolate import interp1d
import pdb # 用于调试代码 pdb.set_trace()
import warnings

this_path = '/public/home/zengch/LQCD/renorma'

ZR_use = 'ZR_a5' # 选择重整化参数
Pz_test = 4         # 选择动量
'''
conf =  'L32x96'
# pz =         0  1  2   3   4   5   6
exn =         [4,  0,  3,  3,  3,  3,  3]
tsep_star =   [8,  0,  8,  8,  8,  8,  8]   
tsep_end  =   [18, 0,  14, 14, 14 ,12, 12]

conf =  'L32x64'
# pz =         0  1  2   3   4   5   6
exn =         [3,  0,  3,  3,  3,  3,  3]
tsep_star =   [7,  0,  7,  7,  7,  7,  7]   
tsep_end  =   [16, 0,  14, 14, 12 ,12, 12]
'''

conf =  'L24x72'
#conf =  'L32x64_C32P23'
# pz =         0   1   2   3   4   5   6
exn =         [3,  0,  2,  2,  2,  2,  2]
tsep_star =   [6,  0,  8,  8,  8,  6,  6]   
tsep_end  =   [16, 0,  12, 12, 12 ,10, 8]
'''
conf =  'L48x144'
# pz =         0  1  2   3   4   5   6
exn =         [5,  0,  4,  4,  4,  4,  4]
tsep_star =   [12,  0,  12,  12,  12,  12,  12]   
tsep_end  =   [20, 0,  18, 18, 16 ,16, 14]
'''
data_pz0 = load_hB_data(f'L24x72_pz0', exn[0], tsep_star[0], tsep_end[0])
hB_pz0   = data_pz0['hB_o'][:20, :]
z_fm_pz0 = data_pz0['z_o'][:20]

data_pz2 = load_hB_data_FeynmenHellman(f'{conf}_dhxmean_pz2', exn[2], tsep_star[2], tsep_end[2])
hB_pz2   = data_pz2['hB_o']
z_fm_pz2 = data_pz2['z_o']

data_pz3 = load_hB_data_FeynmenHellman(f'{conf}_dhxmean_pz3', exn[3], tsep_star[3], tsep_end[3])
hB_pz3   = data_pz3['hB_o']
z_fm_pz3 = data_pz3['z_o']

data_pz4 = load_hB_data_FeynmenHellman(f'{conf}_dhxmean_pz4', exn[4], tsep_star[4], tsep_end[4])
hB_pz4   = data_pz4['hB_o']
z_fm_pz4 = data_pz4['z_o']

'''
data_pz5 = load_hB_data(f'{conf}_pz5', exn[5], tsep_star[6], tsep_end[5])
hB_pz5   = data_pz5['hB_o']
z_fm_pz5 = data_pz5['z_o']


data_pz6 = load_hB_data(f'{conf}_dhxmean_pz6', exn[6], tsep_star[6], tsep_end[6])
hB_pz6   = data_pz6['hB_o']
z_fm_pz6 = data_pz6['z_o']
'''


#z_set  = {0:z_fm_pz0,  2:z_fm_pz2,  3:z_fm_pz3,  4:z_fm_pz4,  5:z_fm_pz5,  6:z_fm_pz6}
z_set  = {0:z_fm_pz0,  2:z_fm_pz2,  3:z_fm_pz3,  4:z_fm_pz4}
if not all(np.array_equal(vec, z_set[0]) for vec in z_set.values()):
    raise ValueError("Not all vectors in z_set are identical!")


#hR_pz = {0:hB_pz0, 2:hB_pz2,  3:hB_pz3,  4:hB_pz4,  5:hB_pz5, 6:hB_pz6}
hR_pz = {0:hB_pz0, 2:hB_pz2,  3:hB_pz3,  4:hB_pz4}

# 输入拟合号的 ZR 
par_fit = pd.read_csv(f'{this_path}/result/ZR_fit_result/{ZR_use}.csv')

k_fit  = par_fit['k'].values
d_fit  = par_fit['d'].values
m0_fit = par_fit['m0'].values
lambda_QCD_fit = par_fit['lambda_QCD'].values
f_columns = [f'f{i}' for i in range(1, 3)]  # 创建f1到f2的列名列表
f_set_fit = par_fit[f_columns].values.T

mu_test = 2.
ZR_fit = th_ZR(z_set[0][:, None], a_len_set[conf], mu_test, k_fit[None ,:], d_fit[None ,:], m0_fit[None ,:], lambda_QCD_fit[None ,:], f_set_fit)


# 输入拟合好的大 hR(lambda) 部分
#par_fit = pd.read_csv('hR_lambda_fit_result/res_pz2_cc_bin5.csv')

lamb_do = 7
lamb_up = 9
lamb_mean = (lamb_do + lamb_up)/ 2.
hR_lamb_name = f'{conf}'
par_fit = pd.read_csv(f'{this_path}/result/hR_lambda_fit_result/{hR_lamb_name}_FeynmenHellman_pz{Pz_test}_lambda{lamb_do}_{lamb_up}_n20.csv')
l1_fit = par_fit['l1'].values
a1_fit = par_fit['a1'].values
lambda0_fit = par_fit['lambda0'].values






def hR_z_Pz(z_, Pz_ , hR_pz, hR_0, conf):

    zs = 0.3

    a_len = a_len_set[conf]
    Nl    = Nl_set[conf]

    Pz_GeV    = Pz_ * 2.* pi / (Nl * a_len) 
    z_GeV     = z_ / fm_to_GeV

    lambda_ = z_GeV * Pz_GeV

    res  =  np.zeros_like(ZR_fit, dtype=float)
    mask = ( z_ < zs)

   

    res[mask]  = hR_pz[mask] / hR_0[mask]

    eta_s =   ZR_fit[~mask][0] / hR_0[~mask][0]  
    res[~mask] = ( hR_pz[~mask] / ZR_fit[~mask] ) * eta_s

    #pdb.set_trace()
   
    return lambda_ , res

def hR_lamb_fit_data(lambda_, Pz_):
    lamb, hR_lamb = hR_z_Pz(z_set[0], Pz_ , hR_pz[Pz_], hR_pz[0], conf)
    interpolation_function = interp1d(lamb, hR_lamb, kind='linear', axis=0)
    
    res =  interpolation_function(lambda_)
    #pdb.set_trace()
    return res 

def hR_lambda_fit_form(lambda_, l1_, a1_, lambda0_):
    res = l1_ * lambda_ **(-a1_) * np.exp(-lambda_/lambda0_)
    #print(res)
    #pdb.set_trace()
    return res

def fit_hR_lambda(par_ini):
    pz_fit = Pz_test
    lab_do = 7
    lab_up = 9
    nbin   = 20
    lambda_fit = np.linspace(lab_do, lab_up, nbin)
    #lambda_fit = np.array([4.3196899, 4.71238898, 5.10508806, 5.49778714, 5.89048623])
    hR_data = hR_lamb_fit_data(lambda_fit, pz_fit)

    #lambda_fit = cc_lamb[11:16]
    #hR_data =    cc_input[11:16]

    #pdb.set_trace()


    hR_data_mean = np.mean(hR_data, axis=1)
    hR_data_std  = np.std( hR_data, axis=1)

    c_inv = np.linalg.inv(np.diag(hR_data_std**2.))

    #c_inv = covariance_matrix_inv(hR_data,'boot')

    #cov_matrix = covariance_matrix(hR_data,'boot')
    #visualize_matrix_num(c_inv @ cov_matrix)


    data_num = len(lambda_fit)

    
    fit_results_list = []

    id_for_fit = range(hR_data.shape[1])
    
    
    for i in id_for_fit:
        #hR_data_i = hR_data[:, i]
        #print(hR_data_i)

        #pdb.set_trace() 

        def cost_function(par):
            l1, a1, lambda0 = par 
            hR_data_i = hR_data[:, i]
       
            hR_th = hR_lambda_fit_form(lambda_fit, l1, a1, lambda0)

            

            del_hR = (hR_th - hR_data_i )
            chi2 = del_hR.T @ c_inv @ del_hR

            mean_chi2 = chi2 / data_num
           
           
            return mean_chi2


        
        par_name = ('l1',   'a1',   'lambda0')
        
        
        m = Minuit(cost_function, par_ini, name=par_name)

        m.limits["lambda0"] = (0.0, 100) 
        #m.limits["l1"] = (-50, 50)  
        m.limits["a1"] = (-100, 100)  
        #m.fixed[  'k']= True
        
        m.migrad()

        l1_fit, a1_fit, lambda0_fit = m.values
        chi_all_mean = cost_function(m.values)

        fit_results_list.append({
            "sample_i": i,
            "l1": l1_fit,
            "a1": a1_fit,
            "lambda0": lambda0_fit,
            "chi2" : chi_all_mean
        })

        # 保存拟合结果到 CSV 文件
        fit_results = pd.DataFrame(fit_results_list)
        fit_results.to_csv(f"{this_path}/result/hR_lambda_fit_result/{conf}_FeynmenHellman_pz{pz_fit}_lambda{lab_do}_{lab_up}_n{nbin}.csv", index=False)
    
    return m.values

def hR_lambda(lambda_, Pz_):

    lamb, hR_lamb = hR_z_Pz(z_set[0], Pz_ , hR_pz[Pz_], hR_pz[0], conf)

    res = np.zeros( (len(lambda_), np.shape(hR_lamb)[1])  ) 
    interpolation_function = interp1d(lamb, hR_lamb, kind='linear', axis=0)
    
    
    lambda_s = lamb_mean
    #lambda_s = 8
    mask = ( lambda_ < lambda_s)
    res[mask]  = interpolation_function(lambda_[mask])
    res[~mask] = hR_lambda_fit_form(lambda_[~mask], l1_fit[:, None], a1_fit[:, None], lambda0_fit[:, None]).T

    #pdb.set_trace()

    #pdb.set_trace()

    
    return res

def hR_x(x_, Pz_): #傅里叶变换后的 hR_x，输入x_必须为向量
    inte_down  = 0
    inte_up    = 72.
    bin_num    = 3200

    lambda_bin = np.linspace(inte_down, inte_up, bin_num)

    

    delta_bin  = (inte_up - inte_down) / bin_num 

    #pdb.set_trace()
    #广播操作设置
    hR_broadcast     = hR_lambda(lambda_bin, Pz_)[:, :, None]
    x_broadcast      = x_[None, None, :]
    lambda_broadcast = lambda_bin[:, None, None]

    #pdb.set_trace()

    hR_x   =  2.* delta_bin * hR_broadcast * np.cos( x_broadcast  * lambda_broadcast)/ 2. / pi
    hR_x   = np.sum(hR_x, axis=0).T

    np.savez(f'{this_path}/result/hR_x/{hR_lamb_name}_FeynmenHellman_pz{Pz_}'  , x = x_ , hRx = hR_x)
   
    return hR_x

if __name__ == "__main__":

    


    # lamda 外推拟合
    #par_ini = [1, 0, 1]
    #par_ini = [0.46191917633411583,-8.287029492467427,0.3552916269882499]
    #fit_hR_lambda(par_ini)
    
    
    #.....画landau外推后的结果
    
    '''
    lambda_test = np.linspace(0, 18, 100)

    res = hR_lambda(lambda_test, Pz_test)

    hbmean = np.mean(res, axis=1)
    hbstd  = np.std(res, axis=1)

    print(lambda_test)
    print(hbmean)
    print(hbstd)
    '''    
    # .....画 quasi - pdf 
    
    x_set = np.linspace(-2, 2, 600)
    res = hR_x(x_set, Pz_test)

    hbmean = np.mean(res, axis=1)
    hbstd  = np.std(res, axis=1)

    print(x_set)
    print(hbmean)
    print(hbstd)
    

 
    
