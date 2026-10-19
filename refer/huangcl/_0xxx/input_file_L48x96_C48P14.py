from ctypes import *
import os
import struct
import numpy as np
#import h5py as h5
import pandas as pd
import subprocess
import re
#import lsqfit
import gvar as gv
import random
import sys
import matplotlib
import fileinput
matplotlib.use('AGG') 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import MultipleLocator
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
import math 
from matplotlib.backends.backend_pdf import PdfPages
from timeit import default_timer as timer
#from mpi4py import MPI
import glob




Nt=96

a0=0.105

npz=10  #change！

z_max  = 20
Nz_3pt = z_max
N_tsep_3pt = 31

moms = 4

input_name = f'L48x96_C48P14_moms{moms}_pz{npz}'

ope_z_dir="/public/group/imp/zengch/LQCD/gluon_operator/output/L48x96_C48P14/zdir/*/"
ope_z=ope_z_dir+"ops_dz*_conf*.npz"

ope_x_dir="/public/group/imp/zengch/LQCD/gluon_operator/output/L48x96_C48P14/xdir/*/"
ope_x=ope_x_dir+"ops_dz*_conf*.npz"

ope_y_dir="/public/group/imp/zengch/LQCD/gluon_operator/output/L48x96_C48P14/ydir/*/"
ope_y=ope_y_dir+"ops_dz*_conf*.npz"

ope_list=[ope_x,ope_y,ope_z]
#ope_list=[ope_z]
#ope_dir_list=[ope_x_dir,ope_y_dir,ope_z_dir]



twopt_z_dir=f"/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2825_ms-0.2310_L48x96/momsmear{moms}z/*/"     #change!

twopt_z=twopt_z_dir+"twopt_slice_pp_Px0Py0Pz"+ str(npz) +f"_eginphase{moms}_Cg5g4_nopol_ss_conf*.npy" 


twopt_x_dir=f"/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2825_ms-0.2310_L48x96/momsmear{moms}x/*/"     #change!

twopt_x=twopt_x_dir+"twopt_slice_pp_Px"+ str(npz) +f"Py0Pz0_eginphase{moms}_Cg5g4_nopol_ss_conf*.npy"   


twopt_y_dir=f"/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2825_ms-0.2310_L48x96/momsmear{moms}y/*/"     #change!

twopt_y=twopt_y_dir+"twopt_slice_pp_Px0Py"+ str(npz) +f"Pz0_eginphase{moms}_Cg5g4_nopol_ss_conf*.npy"





twopt_list=[twopt_x,twopt_y,twopt_z]
#twopt_list=[twopt_z]


#t_sep=np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
#t_sep=np.arange(5,30)   #change! larger than fit_type range!
#ndt=t_sep.shape[0]

#tsep_ini=t_sep[0]
#max_dt=t_sep.max()

Resam_type=1   #0 corresponds to jackknife while 1 corresponds to bootstrap    #change!

if(Resam_type==1):
    Nconf_n=3000

#c0=1
#e0=0


#fit_type=['data_2pt','tsep6_re','tsep8_re','tsep10_re','tsep12_re','tsep14_re','tsep16_re','tsep6_im','tsep8_im','tsep10_im','tsep12_im','tsep14_im','tsep16_im']
