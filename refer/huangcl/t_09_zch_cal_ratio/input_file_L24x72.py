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




Nt=72

a0=0.105

npz=0  #change！

delta_z=30   #change!

z_max  = 25
Nz_3pt = z_max
N_tsep_3pt = 31


#input_name = 'L24x72'
input_name = f'L24x72_pz{npz}'


ope_z_dir="/public/home/chenc/gluon_unpo/generating_data/operator/unfixed_operator_sum/beta6.20_mu-0.2770_ms-0.2400_L24x72/sum/"
ope_z=ope_z_dir+"ops_sum_dz"+str(delta_z)+"_conf*.npz"

ope_x_dir="/public/home/chenc/gluon_unpo/generating_data/operator/unfixed_operator_sum_xdir/beta6.20_mu-0.2770_ms-0.2400_L24x72/sum/"
ope_x=ope_x_dir+"ops_sum_dz"+str(delta_z)+"_conf*.npz"

ope_y_dir="/public/home/chenc/gluon_unpo/generating_data/operator/unfixed_operator_sum_ydir/beta6.20_mu-0.2770_ms-0.2400_L24x72/sum/"
ope_y=ope_y_dir+"ops_sum_dz"+str(delta_z)+"_conf*.npz"




ope_list=[ope_x,ope_y,ope_z]
#ope_dir_list=[ope_z_dir,ope_y_dir,ope_z_dir]

twopt_z_dir="/public/home/chenc/gluon_unpo/generating_data/unfixed_2pt_sum/proton/beta6.20_mu-0.2770_ms-0.2400_L24x72_cg5gt/sum/"     #change!

twopt_z=twopt_z_dir+"N_2pt_pp_Px0Py0Pz"+str(npz)+".conf*.npz" 

twopt_x_dir="/public/home/chenc/gluon_unpo/generating_data/unfixed_2pt_sum/proton/beta6.20_mu-0.2770_ms-0.2400_L24x72_cg5gt/sum/"     #change!

twopt_x=twopt_x_dir+"N_2pt_pp_Px"+str(npz)+"Py0Pz0.conf*.npz"  

twopt_y_dir="/public/home/chenc/gluon_unpo/generating_data/unfixed_2pt_sum/proton/beta6.20_mu-0.2770_ms-0.2400_L24x72_cg5gt/sum/"     #change!

twopt_y=twopt_y_dir+"N_2pt_pp_Px0Py"+str(npz)+"Pz0.conf*.npz" 




twopt_list=[twopt_x,twopt_y,twopt_z]
#twopt_dir_list=[twopt_z_dir,twopt_y_dir,twopt_z_dir]


#t_sep=np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
#t_sep=np.arange(4,30)   #change! larger than fit_type range!
#ndt=t_sep.shape[0]

#tsep_ini=t_sep[0]
#max_dt=t_sep.max()

Resam_type=1   #0 corresponds to jackknife while 1 corresponds to bootstrap    #change!

if(Resam_type==1):
    Nconf_n=3000

#c0=1
#e0=0


#fit_type=['data_2pt','tsep6_re','tsep8_re','tsep10_re','tsep12_re','tsep14_re','tsep16_re','tsep18_re','tsep20_re','tsep6_im','tsep8_im','tsep10_im','tsep12_im','tsep14_im','tsep16_im','tsep18_im','tsep20_im']
