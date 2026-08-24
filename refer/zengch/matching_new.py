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
from constant import CA, CF, gammaE, pi, A_s, inte
from scipy.interpolate import interp1d
import pdb # 用于调试代码 pdb.set_trace()
#from sympy import Si
from scipy.special import sici

this_path = '/public/home/zengch/LQCD/renorma'
conf      = 'L32x64_C32P23'
conf      = 'L24x72'
pz        = 2
data_name = f'{conf}mean_pz{pz}'
x_hRx = np.load(f'{this_path}/result/hR_x/{data_name}.npz')

def Si(x_):
    return sici(x_)[0]

def hR_tilde(y_num, y_inf): # x_ 必须为向量
  
   
    yy = np.linspace(-y_inf, y_inf, y_num)

    x_set   = x_hRx['x']

    #pdb.set_trace()
    hRx_set = x_hRx['hRx']
   
    interpolation_function = interp1d(x_set, hRx_set, kind='linear', axis=0)
    res =  interpolation_function(yy)
    
    #hist_draw(res[1])
    return yy, res

def hR_PDF_old(x_):  #  m:(np.abs(x_[j])*Zs*a*mom_val)/GeVfm,   r:mu/(np.abs(x_[j])*mom_val)
    
    hR_0 = hR_tilde( x_ )
    
    mu_ = 2.
    type_ = conf
    Pz_   = pz

    #xi_set_, y_,  par_, 
    
    alpha_s = A_s(mu_) * 4. * pi
    a_len = a_len_set[type_]
    Nl    = Nl_set[type_]

    Pz_GeV    = Pz_ * 2.* pi / (Nl * a_len)

    lambda_s = 0.3 / fm_to_GeV * Pz_GeV
  
    def g_1(xi_): # xi < 0 

        res = -2.*(1. - xi_ + xi_**2.) **2. / (1. - xi_)  * np.log(xi_/(xi_-1.))  -  (11.-28.* xi_ + 18. * xi_ **2. - 12. * xi_**3.)/ (6.*(1.- xi_))
        return res

    def g_2(xi_): # 0 < xi <1
        y_ = x_ / xi_
        res = 2. * (1.- xi_ +xi_ **2.)**2. / (1.- xi_)    *  (-np.log(mu_**2./ (4. * y_ **2. * Pz_GeV**2.)) + np.log( xi_ *(1. - xi_))) 
        res = res - (15.- 56. * xi_ +102. * xi_ **2. -96. * xi_**3. + 48.* xi_ ** 4.) / ( 6. * ( 1. - xi_))
        return res
    
    def g_3(xi_): # xi > 1
        return 2. * (1. - xi_ +xi_**2)**2/(1-xi_) * np.log(xi_/(xi_-1.))  +  (11.-28.* xi_+18. * xi_** 2.- 12.* xi_**3.)/ (6.*(1.- xi_))
    
    def g_0(xi_):
        y_ = x_ / xi_
        res = 5./6. * ( -1./np.abs(1. - xi_)  + 2.* Si(((1. - xi_)*np.abs(y_) * lambda_s )) /(pi*(1-xi_)) )
        
        return res

    def int0(xi_):
        res = g_0(xi_)[:,None] * ( hR_tilde( x_/ xi_) / xi_[:,None] - hR_tilde(x_))
        return res * alpha_s * CA / 2. / pi 
    
    def int1(xi_):
        res  = g_1(xi_)[:,None] * ( hR_tilde( x_/ xi_) / xi_[:,None] -  hR_tilde(x_))
        return res * alpha_s * CA / 2. / pi
    
    def int2(xi_):
        res = g_2(xi_)[:,None] * ( hR_tilde( x_/ xi_) / xi_[:,None] - hR_tilde( x_ ))
        return res * alpha_s * CA / 2. / pi 
    
    def int3(xi_):
        res = g_3(xi_)[:,None] * ( hR_tilde( x_/ xi_) / xi_[:,None] - hR_tilde( x_ ))
        return res * alpha_s * CA / 2. / pi 
    


    
    int_max = 100

    N_bin    = 9999
    down_lim = -int_max
    up_lim   =  int_max
    epsilon  = 0.00001

    #x_int = 0.0001
    x_int = abs(x_ )

    xi_int0_m   = np.linspace(down_lim, -x_int, N_bin)
    xi_int0_p   = np.linspace(x_int, up_lim, N_bin)
  
    hR_int0_m   = int0(xi_int0_m)
    hR_int0_p   = int0(xi_int0_p)

    #pdb.set_trace()
        
    hR_1_0      = -np.trapz(hR_int0_m, x=xi_int0_m, axis=0) + np.trapz(hR_int0_p, x=xi_int0_p, axis=0)

    xi_int1     = np.linspace(down_lim, -x_int, N_bin)
    hR_int1     = int1(xi_int1)
    hR_1_1      = np.trapz(hR_int1, x=xi_int1, axis=0)

    xi_int2     = np.linspace(x_int, 1. - epsilon, N_bin)
    hR_int2     = int2(xi_int2)
    hR_1_2      = np.trapz(hR_int2, x=xi_int2, axis=0)

    xi_int3     = np.linspace(1. + epsilon, up_lim, N_bin)
    hR_int3     = int3(xi_int3)
    hR_1_3      = np.trapz(hR_int3, x=xi_int3, axis=0)

    hR_PDF = hR_0 - hR_1_0  + hR_1_1 - hR_1_2 - hR_1_3
    
    return np.mean(hR_PDF), np.std(hR_PDF)

def hR_PDF_old2_p(x_):  #x_ > 0 不可广播    m:(np.abs(x_[j])*Zs*a*mom_val)/GeVfm,   r:mu/(np.abs(x_[j])*mom_val)
    
    hR_0 = hR_tilde( x_ )
    
    mu_ = 2.
    type_ = conf
    Pz_   = pz

    #xi_set_, y_,  par_, 
    
    alpha_s = A_s(mu_) * 4. * pi
    a_len = a_len_set[type_]
    Nl    = Nl_set[type_]

    Pz_GeV    = Pz_ * 2.* pi / (Nl * a_len)

    lambda_s = 0.3 / fm_to_GeV * Pz_GeV
  
    def g_1(xi_): # xi < 0 

        res = -2.*(1. - xi_ + xi_**2.) **2. / (1. - xi_)  * np.log(xi_/(xi_-1.))  -  (11.-28.* xi_ + 18. * xi_ **2. - 12. * xi_**3.)/ (6.*(1.- xi_))
        return res

    def g_2(xi_): # 0 < xi <1
        y_ = x_ / xi_
        res = 2. * (1.- xi_ +xi_ **2.)**2. / (1.- xi_)    *  (-np.log(mu_**2./ (4. * y_ **2. * Pz_GeV**2.)) + np.log( xi_ *(1. - xi_))) 
        res = res - (15.- 56. * xi_ +102. * xi_ **2. -96. * xi_**3. + 48.* xi_ ** 4.) / ( 6. * ( 1. - xi_))
        return res
    
    def g_3(xi_): # xi > 1
        return 2. * (1. - xi_ +xi_**2)**2/(1-xi_) * np.log(xi_/(xi_-1.))  +  (11.-28.* xi_+18. * xi_** 2.- 12.* xi_**3.)/ (6.*(1.- xi_))
    
    def g_0(xi_):
        y_ = x_ / xi_
        return 5./6. * ( -1./np.abs(1. - xi_)  + 2.* Si(((1. - xi_)*np.abs(y_) * lambda_s )) /(pi*(1-xi_)) )

    def int0(y_):

        xi_ = x_ / y_

        #pdb.set_trace()

        res = g_0(xi_)[:,None] * ( y_[:, None] * hR_tilde( y_)  - x_ * hR_tilde( x_ )) / y_[:, None] ** 2.
        
        return res * alpha_s * CA / 2. / pi 
    
    def int1(y_):
        xi_ = x_  / y_

        res  = g_1(xi_)[:,None] * ( y_[:, None] * hR_tilde( y_)  - x_ * hR_tilde( x_ )) / y_[:, None] ** 2.
        return res * alpha_s * CA / 2. / pi
    
    def int2(y_):

        
        xi_ = x_  / y_

        res = g_2(xi_)[:,None] * ( y_[:, None] * hR_tilde( y_)  - x_ * hR_tilde( x_ )) / y_[:, None] ** 2.
        
        return res * alpha_s * CA / 2. / pi 
    
    def int3(y_):

        xi_ = x_  / y_

        res = g_3(xi_)[:,None] * ( y_[:, None] * hR_tilde( y_)  - x_ * hR_tilde( x_ )) / y_[:, None] ** 2.
        
        return res * alpha_s * CA / 2. / pi 
    


    
  
    N_bin    = 9999
    epsilon  = 0.00001
    alpha    = 0.001

    #x_int = 0.0001
    x_int = x_

    xi_int0_m   = np.linspace(-1. + epsilon , - alpha, N_bin)
    xi_int0_p   = np.linspace(alpha, 1. - epsilon, N_bin)
    hR_int0_m   = int0(xi_int0_m)
    hR_int0_p   = int0(xi_int0_p)

    #pdb.set_trace()
        
    hR_1_0      = -np.trapz(hR_int0_m, x=xi_int0_m, axis=0) + np.trapz(hR_int0_p, x=xi_int0_p, axis=0)

    xi_int1     = np.linspace(-1. + epsilon, -alpha, N_bin)    #   (-1 , 0)
    hR_int1     = int1(xi_int1)
    hR_1_1      = np.trapz(hR_int1, x=xi_int1, axis=0)

    xi_int2     = np.linspace(x_int + epsilon , 1. - epsilon, N_bin)  #   (x , 1)
    hR_int2     = int2(xi_int2)
    hR_1_2      = np.trapz(hR_int2, x=xi_int2, axis=0)

    xi_int3     = np.linspace(alpha, x_int - epsilon, N_bin)  #   (0 , x)
    hR_int3     = int3(xi_int3)
    hR_1_3      = np.trapz(hR_int3, x=xi_int3, axis=0)

    hR_PDF = hR_0 - hR_1_0  + hR_1_1 - hR_1_2 - hR_1_3
    
    return np.mean(hR_PDF), np.std(hR_PDF)

def hR_PDF_old2_m(x_):  #x_ < 0 不可广播    m:(np.abs(x_[j])*Zs*a*mom_val)/GeVfm,   r:mu/(np.abs(x_[j])*mom_val)
    
    hR_0 = hR_tilde( x_ )
    
    mu_ = 2.
    type_ = conf
    Pz_   = pz

    #xi_set_, y_,  par_, 
    
    alpha_s = A_s(mu_) * 4. * pi
    a_len = a_len_set[type_]
    Nl    = Nl_set[type_]

    Pz_GeV    = Pz_ * 2.* pi / (Nl * a_len)

    lambda_s = 0.3 / fm_to_GeV * Pz_GeV
  
    def g_1(xi_): # xi < 0 

        res = -2.*(1. - xi_ + xi_**2.) **2. / (1. - xi_)  * np.log(xi_/(xi_-1.))  -  (11.-28.* xi_ + 18. * xi_ **2. - 12. * xi_**3.)/ (6.*(1.- xi_))
        return res

    def g_2(xi_): # 0 < xi <1
        y_ = x_ / xi_
        res = 2. * (1.- xi_ +xi_ **2.)**2. / (1.- xi_)    *  (-np.log(mu_**2./ (4. * y_ **2. * Pz_GeV**2.)) + np.log( xi_ *(1. - xi_))) 
        res = res - (15.- 56. * xi_ +102. * xi_ **2. -96. * xi_**3. + 48.* xi_ ** 4.) / ( 6. * ( 1. - xi_))
        return res
    
    def g_3(xi_): # xi > 1
        return 2. * (1. - xi_ +xi_**2)**2/(1-xi_) * np.log(xi_/(xi_-1.))  +  (11.-28.* xi_+18. * xi_** 2.- 12.* xi_**3.)/ (6.*(1.- xi_))
    
    def g_0(xi_):
        y_ = x_ / xi_
        return 5./6. * ( -1./np.abs(1. - xi_)  + 2.* Si(((1. - xi_)*np.abs(y_) * lambda_s )) /(pi*(1-xi_)) )

    def int0(y_):

        xi_ = x_ / y_

        #pdb.set_trace()

        res = g_0(xi_)[:,None] * ( y_[:, None] * hR_tilde( y_)  - x_ * hR_tilde( x_ )) / y_[:, None] ** 2.
        
        return res * alpha_s * CA / 2. / pi 
    
    def int1(y_):
        xi_ = x_  / y_

        res  = g_1(xi_)[:,None] * ( y_[:, None] * hR_tilde( y_)  - x_ * hR_tilde( x_ )) / y_[:, None] ** 2.
        return res * alpha_s * CA / 2. / pi
    
    def int2(y_):

        
        xi_ = x_  / y_

        res = g_2(xi_)[:,None] * ( y_[:, None] * hR_tilde( y_)  - x_ * hR_tilde( x_ )) / y_[:, None] ** 2.
        
        return res * alpha_s * CA / 2. / pi 
    
    def int3(y_):

        xi_ = x_  / y_

        res = g_3(xi_)[:,None] * ( y_[:, None] * hR_tilde( y_)  - x_ * hR_tilde( x_ )) / y_[:, None] ** 2.
        
        return res * alpha_s * CA / 2. / pi 
    


    
  
    N_bin    = 9999
    epsilon  = 0.00001
    alpha    = 0.001

    #x_int = 0.0001
    x_int = x_ 

    xi_int0_m   = np.linspace(1. - epsilon ,  alpha, N_bin)
    xi_int0_p   = np.linspace(-alpha, -1. + epsilon, N_bin)
    hR_int0_m   = int0(xi_int0_m)
    hR_int0_p   = int0(xi_int0_p)

    #pdb.set_trace()
        
    hR_1_0      = -np.trapz(hR_int0_m, x=xi_int0_m, axis=0) + np.trapz(hR_int0_p, x=xi_int0_p, axis=0)

    xi_int1     = np.linspace(1. - epsilon, alpha, N_bin)    #   (1 , 0)
    hR_int1     = int1(xi_int1)
    hR_1_1      = np.trapz(hR_int1, x=xi_int1, axis=0)

    xi_int2     = np.linspace(x_int - epsilon , -1. + epsilon, N_bin)  #   (x , 1)
    hR_int2     = int2(xi_int2)
    hR_1_2      = np.trapz(hR_int2, x=xi_int2, axis=0)

    xi_int3     = np.linspace(-alpha, x_int + epsilon, N_bin)  #   (0 , x)
    hR_int3     = int3(xi_int3)
    hR_1_3      = np.trapz(hR_int3, x=xi_int3, axis=0)

    hR_PDF = hR_0 - hR_1_0  + hR_1_1 - hR_1_2 - hR_1_3
    
    return np.mean(hR_PDF), np.std(hR_PDF)

def hR_PDF(xx):  #    m:(np.abs(x_[j])*Zs*a*mom_val)/GeVfm,   r:mu/(np.abs(x_[j])*mom_val)
    y_inte_for_hR_tilde = 2. # y 积分上限
    
    
    
    n_len = len(xx)
    y_num = n_len
    y_inf = y_inte_for_hR_tilde - 0.0001 
    yy, hR_0 = hR_tilde(y_num, y_inf) 

   
    dy_ = yy[1]-yy[0]
    #pdb.set_trace()
    
    mu_ = 2.
    type_ = conf
    Pz_   = pz

    #xi_set_, y_,  par_, 
    
    alpha_s = A_s(mu_) * 4. * pi
    a_len = a_len_set[type_]
    Nl    = Nl_set[type_]

    Pz_GeV    = Pz_ * 2.* pi / (Nl * a_len)

    lambda_s = 0.3 / fm_to_GeV * Pz_GeV

    
    if np.any(yy == 0):
        print("警告: y中包含零元素，将导致g_xy除以零错误")

    

    x_col = xx[:, None]
    y_col = yy[None, :]

    x_matr = np.tile(x_col, (1, n_len))  # 水平复制 n 次，得到 n×n 矩阵
    y_matr = np.tile(y_col, (n_len, 1))

    cxi = x_matr / y_matr

    def g_1(xi_, y_): # xi < 0 
        res = -2.*(1. - xi_ + xi_**2.) **2. / (1. - xi_)  * np.log(xi_/(xi_-1.))  -  (11.-28.* xi_ + 18. * xi_ **2. - 12. * xi_**3.)/ (6.*(1.- xi_))
        return res

    def g_2(xi_, y_): # 0 < xi <1
        res = 2. * (1.- xi_ +xi_ **2.)**2. / (1.- xi_)    *  (-np.log(mu_**2./ (4. * y_ **2. * Pz_GeV**2.)) + np.log( xi_ *(1. - xi_))) 
        res = res - (15.- 56. * xi_ +102. * xi_ **2. -96. * xi_**3. + 48.* xi_ ** 4.) / ( 6. * ( 1. - xi_))
        return res
    
    def g_3(xi_, y_): # xi > 1
        return 2. * (1. - xi_ +xi_**2)**2/(1-xi_) * np.log(xi_/(xi_-1.))  +  (11.-28.* xi_+18. * xi_** 2.- 12.* xi_**3.)/ (6.*(1.- xi_))
    
    def g_0(xi_, y_):
       
        return 5./6. * ( -1./np.abs(1. - xi_)  + 2.* Si(((1. - xi_)*np.abs(y_) * lambda_s )) /(pi*(1-xi_)) )

    
    mask_cxi_1 = (cxi < 0)
    mask_cxi_2 = (0 < cxi ) & (cxi < 1)
    mask_cxi_3 = (cxi > 1)

    g_xy = np.zeros_like(cxi)
    

    g_xy[mask_cxi_1] = -g_0(cxi[mask_cxi_1] ,  y_matr[mask_cxi_1] ) - g_1( cxi[mask_cxi_1] ,  y_matr[mask_cxi_1] ) 
    g_xy[mask_cxi_2] =  g_0(cxi[mask_cxi_2] ,  y_matr[mask_cxi_2] ) + g_2( cxi[mask_cxi_2] ,  y_matr[mask_cxi_2] ) 
    g_xy[mask_cxi_3] =  g_0(cxi[mask_cxi_3] ,  y_matr[mask_cxi_3] ) + g_3( cxi[mask_cxi_3] ,  y_matr[mask_cxi_3] ) 


    g_ij = g_xy
    Z_ij = np.zeros_like(g_ij)
    delta_ij = np.identity(n_len)

    dy_matr = np.diag(yy/abs(xx)) * dy_


    C_alp_LO   = dy_matr * alpha_s * CA / 2. / pi 
    M_alp_LO   = g_ij / y_matr - delta_ij * y_matr *  np.sum( g_ij / (y_matr **2.) , axis = 1 )
    z_ij = delta_ij + C_alp_LO @ M_alp_LO

    hR_PDF = np.linalg.inv(z_ij) @ hR_0 

    np.savez(f'{this_path}/result/hR_PDF/{data_name}_test_y2',x=xx, hR_PDF = hR_PDF, hR_tilde = hR_0)
    #pdb.set_trace()
    
    return hR_PDF

def hR_PDF_save():
    x_test = np.linspace(0.001, 0.99, 100)
    # 打开文件准备写入（'w' 模式会覆盖旧文件，用 'a' 可追加）
    with open(f'{this_path}/result/hR_PDF/{data_name}_new.txt', 'w') as f:
        # 写入表头
        header = "x, meanPDF, stdPDF, meanPDF_tilde, stdPDF_tilde\n"
        print(header.strip())  # 打印到屏幕（去掉末尾的 \n）
        f.write(header)        # 写入文件

        for x_ti in x_test:
            PDF_tilde = hR_tilde(x_ti)
            mean, std = hR_PDF_old(x_ti)
            
            # 格式化输出行
            line = f"{x_ti}, {mean}, {std}, {np.mean(PDF_tilde)}, {np.std(PDF_tilde)}\n"
            
            print(line.strip())  # 打印到屏幕
            f.write(line)       # 写入文件

    print("结果已保存到 results.txt")
    return 0


#par_ini = np.array([1., 1., 1., 0])

x_test = np.linspace(-1 + 0.0001, 1 -0.0001, 398)

#dx=0.02
#x_test = np.arange(-1.99,1.99+0.005,dx)
#pdb.set_trace()  
hR_PDF(x_test)
#print(a)

'''
a = hR_PDF_old2_p(0.1)
print(a)
a = hR_PDF_old2_m(-0.1)
print(a)
a = hR_PDF_old(0.1)
print(a)
a = hR_PDF_old(-0.1)
print(a)
'''

#hR_PDF_save()

#x_test = 0
#hR_tilde(x_test)



# 定义 x 和 y 数据
#x = np.linspace(0, 1, 100)  # 积分区间
#y = np.random.rand(5, 100)   # 5 条曲线，每条 100 个点

# 对每条曲线积分（沿 axis=1）
#integral = np.trapz(y, x=x, axis=1)

#pdb.set_trace()
#print(integral.shape)  # (5,)，每条曲线的积分结果