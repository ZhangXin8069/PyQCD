import numpy as np
import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction import *

def jcknf_sample(data, axis:int = -3):
    n = data.shape[axis]
    data_sum = np.sum(data, axis = axis, keepdims = True)
    jcknf_sample = (data_sum - data)/(n-1)

    return jcknf_sample

Nt = 32
link_max = 10

Nev_src = 50
Nev_link = 200

conf_list = [x for x in range(10000, 15000, 100) if x not in [14800]]

corr_2pt = np.zeros((5, len(conf_list), Nt, Nt), dtype = complex)
corr_3pt_con = np.zeros((5, len(conf_list), 2 * link_max + 1, Nt, Nt), dtype = complex)
bubble = np.zeros((5, len(conf_list), 2 * link_max + 1, Nt), dtype = complex)
t_sep = 8

for conf_indx, conf_id in enumerate(conf_list):

    corr_2pt[0, conf_indx] = np.load(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_2pt_gamma0505_u_e50_s[0]_[0]_[0]_BI_n100_stout_smear_20_0.12_dul_vector_True_src{Nev_src}.npy')
    for link_indx in range(-link_max, link_max + 1, 1):
        corr_3pt_con[0, conf_indx, link_indx + link_max] = np.load(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_3pt_gamma050405_tseq{t_sep}_link_indx{link_indx}_u_e50_s[0]_[0]_[0]_BI_n100_stout_smear_20_0.12_dul_vector_True_src{Nev_src}_curr{Nev_link}.npy')
        bubble[0, conf_indx, link_indx + link_max] = np.load(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_bubble_gamma04_link_indx{link_indx}_u_e50_s[0]_[0]_[0]_BI_n100_stout_smear_20_0.12_dul_vector_True_curr{Nev_link}.npy')

    corr_2pt[1, conf_indx] = np.load(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_2pt_gamma0505_u_e50_s[0]_[0]_[0]_BI_n150_stout_smear_20_0.12_dul_vector_False_src{Nev_src}.npy')
    for link_indx in range(-link_max, link_max + 1, 1):
        corr_3pt_con[1, conf_indx, link_indx + link_max] = np.load(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_3pt_gamma050405_tseq{t_sep}_link_indx{link_indx}_u_e50_s[0]_[0]_[0]_BI_n150_stout_smear_20_0.12_dul_vector_False_src{Nev_src}_curr{Nev_link}.npy')
        bubble[1, conf_indx, link_indx + link_max] = np.load(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_bubble_gamma04_link_indx{link_indx}_u_e50_s[0]_[0]_[0]_BI_n150_stout_smear_20_0.12_dul_vector_False_curr{Nev_link}.npy')


    corr_2pt[2, conf_indx] = np.load(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_2pt_gamma0505_u_e50_s[50, 200]_[10, 10]_[5, 5]_BI_n60_stout_smear_20_0.12_dul_vector_True_src{Nev_src}.npy')
    for link_indx in range(-link_max, link_max + 1, 1):
        corr_3pt_con[2, conf_indx, link_indx + link_max] = np.load(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_3pt_gamma050405_tseq{t_sep}_link_indx{link_indx}_u_e50_s[50, 200]_[10, 10]_[5, 5]_BI_n60_stout_smear_20_0.12_dul_vector_True_src{Nev_src}_curr{Nev_link}.npy')
        bubble[2, conf_indx, link_indx + link_max] = np.load(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_bubble_gamma04_link_indx{link_indx}_u_e50_s[50, 200]_[10, 10]_[5, 5]_BI_n60_stout_smear_20_0.12_dul_vector_True_curr{Nev_link}.npy')

    corr_2pt[3, conf_indx] = np.load(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_2pt_gamma0505_u_e50_s[50, 200]_[10, 10]_[5, 5]_BI_n100_stout_smear_20_0.12_dul_vector_True_src{Nev_src}.npy')
    for link_indx in range(-link_max, link_max + 1, 1):
        corr_3pt_con[3, conf_indx, link_indx + link_max] = np.load(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_3pt_gamma050405_tseq{t_sep}_link_indx{link_indx}_u_e50_s[50, 200]_[10, 10]_[5, 5]_BI_n100_stout_smear_20_0.12_dul_vector_True_src{Nev_src}_curr{Nev_link}.npy')
        bubble[3, conf_indx, link_indx + link_max] = np.load(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_bubble_gamma04_link_indx{link_indx}_u_e50_s[50, 200]_[10, 10]_[5, 5]_BI_n100_stout_smear_20_0.12_dul_vector_True_curr{Nev_link}.npy')
        
    corr_2pt[4, conf_indx] = np.load(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_2pt_gamma0505_u_e50_s[50, 200]_[10, 10]_[5, 5]_BI_n100_stout_smear_20_0.12_dul_vector_True_src{Nev_src}.npy')
    for link_indx in range(-link_max, link_max + 1, 1):
        corr_3pt_con[4, conf_indx, link_indx + link_max] = np.load(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_3pt_gamma050405_tseq{t_sep}_link_indx{link_indx}_u_e50_s[50, 200]_[10, 10]_[5, 5]_BI_n100_stout_smear_20_0.12_dul_vector_True_src{Nev_src}_curr{Nev_link}_2.npy')
        bubble[4, conf_indx, link_indx + link_max] = np.load(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_bubble_gamma04_link_indx{link_indx}_u_e50_s[50, 200]_[10, 10]_[5, 5]_BI_n100_stout_smear_20_0.12_dul_vector_True_curr{Nev_link}_2.npy')
        
        
corr_2pt_sample = jcknf_sample(corr_2pt, axis = -3)
corr_3pt_sample = jcknf_sample(corr_3pt_con, axis = -4)
ratio = np.zeros_like(corr_3pt_sample)

for t in range(Nt):
    ratio[..., t, :] = corr_3pt_sample[..., t, :]/corr_2pt_sample[..., t, (t+t_sep)%Nt].reshape(5, len(conf_list), 1, 1)

# print(ratio[0, 0, 10, 0, 3:6])
ratio_loop_tsrc = loop_tsrc(ratio, indx = [-2, -1])/Nt

ratio_mean = np.real(np.mean(ratio_loop_tsrc, axis = -4))
ratio_err = np.real(np.std(ratio_loop_tsrc, axis = -4) * np.sqrt(len(conf_list) - 1))
import gvar
print(ratio_mean.shape)

for link_indx, link in enumerate(range(-link_max, link_max+1, 1)):
    # print(link, gvar.gvar(ratio_mean[0, link_indx, 0, 3:6], ratio_err[0, link_indx, 0, 3:6]))
    # print(link, gvar.gvar(ratio_mean[1, link_indx, 0, 3:6], ratio_err[1, link_indx, 0, 3:6]))
    # print(link, gvar.gvar(ratio_mean[2, link_indx, 0, 3:6], ratio_err[2, link_indx, 0, 3:6]))
    # print(link, gvar.gvar(ratio_mean[3, link_indx, 0, 3:6], ratio_err[3, link_indx, 0, 3:6]))
    # print(link, gvar.gvar(ratio_mean[4, link_indx, 0, 3:6], ratio_err[4, link_indx, 0, 3:6]))
    print(link, [f'{ratio_mean[0, link_indx, 0, x]:.8f}({ratio_err[0, link_indx, 0, x]:.8f})' for x in range(3, 6)])
    print(link, [f'{ratio_mean[1, link_indx, 0, x]:.8f}({ratio_err[1, link_indx, 0, x]:.8f})' for x in range(3, 6)])
    print(link, [f'{ratio_mean[2, link_indx, 0, x]:.8f}({ratio_err[2, link_indx, 0, x]:.8f})' for x in range(3, 6)])
    print(link, [f'{ratio_mean[3, link_indx, 0, x]:.8f}({ratio_err[3, link_indx, 0, x]:.8f})' for x in range(3, 6)])
    print(link, [f'{ratio_mean[4, link_indx, 0, x]:.8f}({ratio_err[4, link_indx, 0, x]:.8f})' for x in range(3, 6)])
    print('*************************************************')

# corr_2pt_loop_tsrc = loop_tsrc(corr_2pt, indx = [-2, -1])
# corr_3pt_loop_tsrc = loop_tsrc(corr_3pt_con, indx = [-2, -1])

# print(corr_2pt_loop_tsrc[0, 8])
# print(corr_3pt_loop_tsrc[link_max-2:link_max+3, 0, :8])