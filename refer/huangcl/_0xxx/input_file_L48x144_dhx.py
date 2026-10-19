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




Nt=144

a0=0.0519

npz=3  #change！

delta_z=20   #change!
z_max = 20
Nz_3pt = z_max
N_tsep_3pt = 31

input_name = f'L48x144_dhx_pz{npz}'

ope_z_dir="/public/group/imp/zengch/LQCD/gluon_operator/output/L48x144/zdir/*/"
ope_z=ope_z_dir+"ops_dz"+str(delta_z)+"_conf*.npz"

ope_x_dir="/public/group/imp/zengch/LQCD/gluon_operator/output/L48x144/xdir/*/"
ope_x=ope_x_dir+"ops_dz"+str(delta_z)+"_conf*.npz"

ope_y_dir="/public/group/imp/zengch/LQCD/gluon_operator/output/L48x144/ydir/*/"
ope_y=ope_y_dir+"ops_dz"+str(delta_z)+"_conf*.npz"


ope_list=[ope_x,ope_y,ope_z]
#ope_dir_list=[ope_x_dir,ope_y_dir,ope_z_dir]

twopt_z_dir="/public/group/lqcd/donghx/2pt_Result/beta6.72_mu-0.1850_ms-0.1700_L48x144/momsmear2z/*/"     #change!

twopt_z=twopt_z_dir+"twopt_slice_pp_Px0Py0Pz"+ str(npz) +"_eginphase2_Cg5g4_nopol_ss_conf*.npy" 

twopt_x_dir="/public/group/lqcd/donghx/2pt_Result/beta6.72_mu-0.1850_ms-0.1700_L48x144/momsmear2x/*/"     #change!

twopt_x=twopt_x_dir+"twopt_slice_pp_Px"+ str(npz) +"Py0Pz0_eginphase2_Cg5g4_nopol_ss_conf*.npy"  

twopt_y_dir="/public/group/lqcd/donghx/2pt_Result/beta6.72_mu-0.1850_ms-0.1700_L48x144/momsmear2y/*/"     #change!

twopt_y=twopt_y_dir+"twopt_slice_pp_Px0Py"+ str(npz) +"Pz0_eginphase2_Cg5g4_nopol_ss_conf*.npy" 


 

twopt_list=[twopt_x,twopt_y,twopt_z]




#twopt_dir_list=[twopt_x_dir,twopt_y_dir,twopt_z_dir]


#t_sep=np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
#t_sep=np.arange(10,41)

#ndt=t_sep.shape[0]

#tsep_ini=t_sep[0]
#max_dt=t_sep.max()

Resam_type=1   #0 corresponds to jackknife while 1 corresponds to bootstrap    #change!

if(Resam_type==1):
    Nconf_n=3000

#c0=1
#e0=0


#fit_type=['data_2pt','tsep10_re','tsep12_re','tsep14_re','tsep16_re','tsep18_re','tsep20_re','tsep22_re','tsep24_re','tsep10_im','tsep12_im','tsep14_im','tsep16_im','tsep18_im','tsep20_im','tsep22_im','tsep24_im']
