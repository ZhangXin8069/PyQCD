import numpy as np

A = np.load('/public/home/sush/distillation/gammaisigmai/result/24x72/Px0Py0Pz0/ENV_650/conf10000/corr_ud_3pt_gamma010105_tseq12_link_indx0_u_conserved_False_light_connect_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_stout_smear_20_0.12_src50_sink650_curr650_compensation.npy')
print(np.sum(A))