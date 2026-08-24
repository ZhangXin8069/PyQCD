import math
import matplotlib.pyplot as plt
import glob
from timeit import default_timer as timer
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import MultipleLocator
from ctypes import *
import os
import struct
import numpy as np
# import h5py as h5
import pandas as pd
import subprocess
import re
# import lsqfit
import gvar as gv
import random
import sys
import matplotlib
import fileinput
matplotlib.use('AGG')
# 从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
# from mpi4py import MPI


Nt = 72

a0 = 0.105

npz = 4  # change！

delta_z = 24  # change!

z_max = 24
Nz_3pt = z_max
N_tsep_3pt = 31


input_name = f'L24x72_dhxmeang1_pz{npz}'
# input_name = 'L24x72_dhx'


ope_z_dir = "/public/group/imp/zengch/LQCD/gluon_operator/output/L24x72/zdir/*/"
ope_z = ope_z_dir+"ops_dz"+str(delta_z)+"_conf*.npz"

ope_x_dir = "/public/group/imp/zengch/LQCD/gluon_operator/output/L24x72/xdir/*/"
ope_x = ope_x_dir+"ops_dz"+str(delta_z)+"_conf*.npz"

ope_y_dir = "/public/group/imp/zengch/LQCD/gluon_operator/output/L24x72/ydir/*/"
ope_y = ope_y_dir+"ops_dz"+str(delta_z)+"_conf*.npz"


ope_list = [ope_x, ope_y, ope_z]
# ope_dir_list=[ope_z_dir,ope_y_dir,ope_z_dir]


twopt_z_dir = "/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2770_ms-0.2400_L24x72/momsmear2z/*/"  # change!

twopt_z = twopt_z_dir+"twopt_slice_pp_Px0Py0Pz" + \
    str(npz) + "_eginphase2_Cg5g4_nopol_ss_conf*.npy"

twopt_x_dir = "/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2770_ms-0.2400_L24x72/momsmear2x/*/"  # change!

twopt_x = twopt_x_dir+"twopt_slice_pp_Px" + \
    str(npz) + "Py0Pz0_eginphase2_Cg5g4_nopol_ss_conf*.npy"

twopt_y_dir = "/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2770_ms-0.2400_L24x72/momsmear2y/*/"  # change!

twopt_y = twopt_y_dir+"twopt_slice_pp_Px0Py" + \
    str(npz) + "Pz0_eginphase2_Cg5g4_nopol_ss_conf*.npy"


twopt_mz_dir = "/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2770_ms-0.2400_L24x72/momsmear-2z/*/"  # change!

twopt_mz = twopt_mz_dir+"twopt_slice_pp_Px0Py0Pz" + \
    str(-npz) + "_eginphase-2_Cg5g4_nopol_ss_conf*.npy"

twopt_mx_dir = "/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2770_ms-0.2400_L24x72/momsmear-2x/*/"  # change!

twopt_mx = twopt_mx_dir+"twopt_slice_pp_Px" + \
    str(-npz) + "Py0Pz0_eginphase-2_Cg5g4_nopol_ss_conf*.npy"

twopt_my_dir = "/public/group/lqcd/donghx/2pt_Result/beta6.20_mu-0.2770_ms-0.2400_L24x72/momsmear-2y/*/"  # change!

twopt_my = twopt_my_dir+"twopt_slice_pp_Px0Py" + \
    str(-npz) + "Pz0_eginphase-2_Cg5g4_nopol_ss_conf*.npy"


twopt_list = [
    twopt_x, twopt_y, twopt_z,
    twopt_mx, twopt_my, twopt_mz
]


# t_sep=np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
# t_sep=np.arange(4,30)   #change! larger than fit_type range!
# ndt=t_sep.shape[0]

# tsep_ini=t_sep[0]
# max_dt=t_sep.max()

Resam_type = 1  # 0 corresponds to jackknife while 1 corresponds to bootstrap    #change!

if (Resam_type == 1):
    Nconf_n = 3000

# c0=1
# e0=0


# fit_type=['data_2pt','tsep6_re','tsep8_re','tsep10_re','tsep12_re','tsep14_re','tsep16_re','tsep18_re','tsep20_re','tsep6_im','tsep8_im','tsep10_im','tsep12_im','tsep14_im','tsep16_im','tsep18_im','tsep20_im']
