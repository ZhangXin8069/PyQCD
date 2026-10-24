import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction import *

from opt_einsum import contract, contract_path

import time

from math import prod
from typing import List
import numpy as np

set_backend('numpy')
backend = get_backend()

lattice_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 24]

# param = creat_param(
#     sink_operators = [['d^d', 'gamma_sink', 'u', 1]],
#     source_operators = [['u^d', 'gamma_source', 'd', -1]]
# )

# param_data = param.data()

conf_id = sys.argv[1]
exact_eigen = 50
N_eigen = [100, 200, 300, 500, 800]
N_sum = [20, 20, 20, 20, 20]
N_extract = [10, 10, 10, 10, 10]
noise_eigen = 500

exact_eigen_name = 50
N_eigen_name = [100, 200, 300, 500, 800]
N_sum_name = [20, 20, 20, 20, 20]
N_extract_name = [10, 10, 10, 10, 10]
noise_eigen_name = 500

Nev_max = 650
t_sep = 18

# corr = distillation_func(
#     conf_id = conf_id,
#     param = param_data,
    
#     lattice_size = lattice_size, # Nx, Ny, Nz, Nt 
#     grid_size = grid_size, 
    
#     backend = backend.__name__
# )

mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)

comm = getMPIComm()
rank = getMPIRank()
size = getMPISize()

Nx, Ny, Nz, Nt = lattice_size
Gt = Nt//size

V_list = [x for x in range(Nx**3)][rank*(Nx**3//size):(rank+1)*(Nx**3//size)]
V_max = Nx**3

eigen = corr_eigvecs(Nx = Nx)
Mom_link = creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True)
mu_nu_indx, mu_nu_gamma, mu_indx, mu_gamma =  PFF_Mom_to_gamma_new(Mom_link[:], allow_t = False)

mu_nu_gamma = backend.asarray(mu_nu_gamma)
mu_gamma = backend.asarray(mu_gamma)
phase_link = backend.empty((len(Mom_link), Nx, Nx, Nx, Nc), dtype = complex)

corr_3pt_matrix = backend.zeros((len(Mom_link), len(mu_nu_indx[0]), Nt, Nt), dtype = complex)
corr_2pt_mu_matrix = backend.zeros((len(Mom_link), len(mu_indx[0]), Nt, Nt), dtype = complex)
bubble_light_matrix = backend.zeros((len(Mom_link), len(mu_indx[0]), Nt), dtype = complex)
bubble_strange_matrix = backend.zeros((len(Mom_link), len(mu_indx[0]), Nt), dtype = complex)

for i in range(len(Mom_link)):
    phase_link[i] = backend.asarray(eigen.phase_exp_2pt(Mom = Mom_link[i]))

phase_link = phase_link.reshape(len(Mom_link), Nx * Nx * Nx, Nc)[:, V_list]

import time

g5 = backend.asarray(gamma(5))

omega1 = eigen.create_omega_accelerate(
    exact = exact_eigen,
    N_eigen = N_eigen,
    N_sum = N_sum,
    N_extract = N_extract,
    noise = noise_eigen,
    dim = 1
)

omega2 = eigen.create_omega_accelerate(
    exact = exact_eigen,
    N_eigen = N_eigen,
    N_sum = N_sum,
    N_extract = N_extract,
    noise = noise_eigen,
    dim = 2
)

V = Nx ** 3
vectors = np.zeros((Gt, Nev_max, Nx, Nx, Nx, Nc), dtype = complex)
# disconnect
for t_src in range(Gt):
    vectors[t_src] = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/{conf_id}_t{(t_src + rank*Gt):03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')

vectors_tmp = np.zeros((Nt, Nev_max, len(V_list), Nc), dtype = complex)

for rank_indx in range(size):
    vectors_tmp[rank_indx * Gt:(rank_indx + 1)*Gt] = get_mpi_data(data = vectors.reshape(Gt, Nev_max, V_max, Nc), mdtype = 'Scatter', axis = 2, root = rank_indx)

del vectors

for t_src in range(Nt):
    if rank == 0:
        st_load_time = time.perf_counter()
        
        prop_bubble_light = backend.empty((1, Nx, Nx, Nx, Ns, Ns, Nc, Nev_max), dtype = complex)
        prop_bubble_strange = backend.empty((1, Nx, Nx, Nx, Ns, Ns, Nc, Nev_max), dtype = complex)
        
        try:
            prop_bubble_light[:] = np.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{t_src:03d}_tsink{t_src:03d}_gen_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')
            prop_bubble_strange[:] = np.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/strange/{conf_id}/t{t_src:03d}_tsink{t_src:03d}_gen_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')
            
            prop_bubble_light = prop_bubble_light.reshape(1, V_max, Ns, Ns, Nc, Nev_max)
            prop_bubble_strange = prop_bubble_strange.reshape(1, V_max, Ns, Ns, Nc, Nev_max)
            
        except:
            comm.Abort(1)
            
    else:
        prop_bubble_light = None
        prop_bubble_strange = None
        
    if rank == 0:
        print(f'load prop use time {(time.perf_counter() - st_load_time):.3f} s of tsrc {t_src}')
        
    prop_bubble_light = get_mpi_data(data = prop_bubble_light, mdtype = 'Scatter', axis = 1)
    prop_bubble_strange = get_mpi_data(data = prop_bubble_strange, mdtype = 'Scatter', axis = 1)
    
    st_cal_time = time.perf_counter()
    
    bubble_light_matrix[:, :, t_src] = contract('Vijcn,MVc,MGji,nVc,n->MG', prop_bubble_light[0], phase_link, mu_gamma, vectors_tmp[t_src].conj(), omega1, optimize = 'optimal')
    bubble_strange_matrix[:, :, t_src] = contract('Vijcn,MVc,MGji,nVc,n->MG', prop_bubble_strange[0], phase_link, mu_gamma, vectors_tmp[t_src].conj(), omega1, optimize = 'optimal')

    if rank == 0:
        print(f'cal bubble use time {(time.perf_counter() - st_cal_time):.3f} s of tsrc {t_src}')
        
    del prop_bubble_light, prop_bubble_strange

    # st_mpi_time = time.perf_counter()
    
    # # prop = np.memmap(
    # #     f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{t_src:03d}_gen_e50_stout_smear_20_0.12.mmap', 
    # #     dtype=complex, mode='r', shape=(24 * 24 * 24, 3, 72, 4, 4, 50)
    # #     )
    
    # prop = np.memmap(
    #     f'/beegfs/group-bak/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{t_src:03d}_gen_e50_stout_smear_20_0.12.mmap', 
    #     dtype=complex, mode='r', shape=(24 * 24 * 24, 3, 72, 4, 4, 50)
    #     )
    
    # t_sink_list = [x%Nt for x in range(t_src, t_src + 40, 1)]
    
    # prop_load = backend.zeros((len(V_list), 3, 72, 4, 4, 50), dtype = complex)
    
    # prop_load[:, :, t_sink_list] = prop[rank*(Nx**3//size):(rank+1)*(Nx**3//size), :, t_sink_list, ...]
    
    # if rank == 0:
    #     print(f'mpidata and lexico prop use time {(time.perf_counter() - st_mpi_time):.3f} s of tsrc {t_src}')

    # st_cal_time = time.perf_counter()
    # for t_sink in t_sink_list:
    #     prop_tsink = prop_load[:, :, t_sink]
        
    #     prop_sep_tsink = contract('ab,...bcf,cd->...daf', g5, prop_tsink.conj(), g5)
    #     corr_2pt_mu_matrix[:, :, t_src, t_sink] = corr_2pt_mu_matrix[:, :, t_src, t_sink] + contract('Vcijn,Vckln,MVc,MGli,jk->MG', prop_tsink, prop_sep_tsink, phase_link, mu_gamma, g5, optimize = 'greedy')
        
    # if rank == 0:
    #     print(f'cal 2pt use time {(time.perf_counter() - st_cal_time):.3f} s of tsrc {t_src}')
        
    # if rank == 0:
    #     peram_sep = backend.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{((t_sep + t_src)%Nt):03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')
    # else:
    #     peram_sep = None
    
    # peram_sep = get_mpi_data(data = peram_sep, mdtype = 'TScatter', root = 0, axis = 0)
    
    # peram_sep = seq_peram(peram_sep)
    
    # for Mom_indx in range(len(Mom_link)):
    #     st_all_cal_time = time.perf_counter()
        
    #     if rank == size - 1:
    #         peram_one_Mom = backend.zeros((Nt, 4, 4, Nev_max, 50), dtype = complex)

    #     else:
    #         peram_one_Mom = None
        
    #     for t in t_sink_list:
    #         peram = contract('Vcabn,mVc,Vc->abmn', prop_load[:, :, t], vectors_tmp[t].conj(), phase_link[Mom_indx])
    #         peram_one_Mom_temp = get_mpi_data(data = peram, mdtype = 'Gather', root = size - 1)
    #         if rank == size - 1:
    #             peram_one_Mom[t] = peram_one_Mom_temp.sum(0)
            
    #     peram_one_Mom = get_mpi_data(data = peram_one_Mom, mdtype = 'TScatter', root = size - 1, axis = -1)
    #     peram_one_Mom_sep_tsink = seq_peram(peram = peram_one_Mom[(t_sep + t_src)%Nt])
            
    #     for t_curr in t_sink_list:
    #         if rank == t_curr%size:
    #             peram_sep_curr = peram_sep[t_curr//size]

    #         else:
    #             peram_sep_curr = None
                
    #         peram_sep_curr = get_mpi_data(data = peram_sep_curr, mdtype = 'Bcast', root = t_curr%size)
                
    #         if t_curr != (t_src + t_sep)%Nt:
    #             corr_3pt_matrix[Mom_indx, :, t_src, t_curr] = (
    #                 corr_3pt_matrix[Mom_indx, :, t_src, t_curr] + 
    #                 contract('ijvn,klnm,opmv,Gpi,Glo,jk,v,m->G', peram_one_Mom[t_curr], peram_one_Mom_sep_tsink, peram_sep_curr, mu_nu_gamma[Mom_indx, :, 0], mu_nu_gamma[Mom_indx, :, 1], g5, omega1, omega1, optimize = 'greedy')
    #                 )

    #         else:
    #             corr_3pt_matrix[Mom_indx, :, t_src, t_curr] = (
    #                 corr_3pt_matrix[Mom_indx, :, t_src, t_curr] + 
    #                 contract('mv,ijvn,klnm,opmv,Gpi,Glo,jk->G', omega2, peram_one_Mom[t_curr], peram_one_Mom_sep_tsink, peram_sep_curr, mu_nu_gamma[Mom_indx, :, 0], mu_nu_gamma[Mom_indx, :, 1], g5, optimize = 'greedy')
    #                 )

    #     if rank == 0:
    #         print(f'cal 3pt use time {(time.perf_counter() - st_all_cal_time):.3f} s of tsrc {t_src} Mom_indx {Mom_indx}')

corr_2pt_mu_matrix = get_mpi_data(data = corr_2pt_mu_matrix, mdtype = 'Gather')
corr_3pt_matrix = get_mpi_data(data = corr_3pt_matrix, mdtype = 'Gather')

bubble_light_matrix = get_mpi_data(data = bubble_light_matrix, mdtype = 'Gather', root = 0)
bubble_strange_matrix = get_mpi_data(data = bubble_strange_matrix, mdtype = 'Gather', root = 0)

if rank == 0:
    corr_2pt_mu_matrix = corr_2pt_mu_matrix.sum(0)
    corr_3pt_matrix = corr_3pt_matrix.sum(0)
    
    bubble_light_matrix = bubble_light_matrix.sum(0)
    bubble_strange_matrix = bubble_strange_matrix.sum(0)
    
    print(corr_2pt_mu_matrix.shape)
    print(corr_3pt_matrix.shape)
    
    for Mom_indx, Mom_list in enumerate(Mom_link[:]):
        corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_gen/{Nx}x{Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_50/conf{conf_id}'
        import pathlib

        path = pathlib.Path(corr_save_path)

        if path.exists():
            print('save_path:',corr_save_path)
        
        else:
            path.mkdir(parents = True, exist_ok = True)
            print('mkdir_save_path:',corr_save_path)

        for mu_indx_1, mu in enumerate(mu_indx[Mom_indx]):
            # backend.save(f'{corr_save_path}/corr_2pt_gamma{(mu):02d}_gen_stout_smear_20_0.12_src50.npy', corr_2pt_mu_matrix[Mom_indx, mu_indx_1])
            backend.save(f'{corr_save_path}/corr_bubble_light_gamma{(mu):02d}_gen_stout_smear_20_0.12_src650.npy', bubble_light_matrix[Mom_indx, mu_indx_1])
            backend.save(f'{corr_save_path}/corr_bubble_strange_gamma{(mu):02d}_gen_stout_smear_20_0.12_src650.npy', bubble_strange_matrix[Mom_indx, mu_indx_1])
            backend.save(f'{corr_save_path}/corr_bubble_u-s_gamma{(mu):02d}_gen_stout_smear_20_0.12_src650.npy', bubble_light_matrix[Mom_indx, mu_indx_1] - bubble_strange_matrix[Mom_indx, mu_indx_1])
            
        # for gamma_indx, gamma_list in enumerate(mu_nu_indx[Mom_indx]):
        #     backend.save(f'{corr_save_path}/corr_3pt_gamma0{gamma_list[1]}0{gamma_list[0]}05_tsep{t_sep}_gen_stout_smear_20_0.12_src50_sink{Nev_max}_curr{Nev_max}_compensation.npy', corr_3pt_matrix[Mom_indx, gamma_indx])