import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from function_contraction import *

from opt_einsum import contract

import sys
import time

lattice_size = [12, 12, 12, 32]
grid_size = [1, 1, 1, 1]

param = creat_param(
    sink_operators = [['d^d', 'gamma_sink', 'u', 1]],
    source_operators = [['u^d', 'gamma_source', 'd', -1]]
)

param_data = param.data()

set_backend('cupy')
backend = get_backend()

conf_id = sys.argv[1]

corr = distillation_func(
    conf_id = conf_id,
    param = param_data,
    
    lattice_size = lattice_size, # Nx, Ny, Nz, Nt 
    grid_size = grid_size, 
    
    backend = backend.__name__
)

rank = getMPIRank()
comm = getMPIComm()
size = getMPISize()

gauge_path = f'/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_'
link_max = 0
gauge_link = False
Nev_src = 100

exact_eigen = 100
N_eigen = [0]
N_sum = [0]
N_extract = [0]
noise_eigen = 5084

exact_eigen_name = 100
N_eigen_name = [0]
N_sum_name = [0]
N_extract_name = [0]
noise_eigen_name = 5084

Nev_max = 5184
Nev_curr_list = [x for x in range(200, 900, 100)] + [5184]

Mom_link = creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)
mu_nu_indx, mu_nu_gamma, mu_indx, mu_gamma = PFF_Mom_to_gamma(Mom_link, Pion_Mom = False)

Mom_sink = [[0, 0, 0]]

sink, VdV_link = \
    corr.sink_link_V1(
        eigen_name = [
                f'_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy'
        ],
        eigen_save_path = [
                f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L12x32/{conf_id}/{conf_id}_t'
                # f'/beegfs/group-bak/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L12x32/{conf_id}/{conf_id}_t'
                
            ],
        exact_eigen = exact_eigen,
        N_eigen = N_eigen,
        N_sum = N_sum,
        N_extract = N_extract,
        noise_eigen = noise_eigen,
        t_range = range(0, 32, 8),
        Nev_src = Nev_src,
        link_max = link_max,
        link_dir = 'Z',
        Mom = Mom_sink,
        Mom_link = Mom_link,
        gauge = gauge_link,
        ttype = ':03d',
        add_weight = False,
        read_type = 'numpy', # pickle
    )

import numpy
import time

corr_2pt_matrix = backend.zeros((len(Nev_curr_list), len(Mom_link), 2, corr.Nt, corr.Nt), dtype = complex)
bubble_matrix = backend.zeros((len(Nev_curr_list), len(Mom_link), 2, corr.Nt), dtype = complex)

for t_src_indx, t_src in enumerate(range(0, 8, 8)):
    st_all_time = time.perf_counter()
    
    for Nev in range(0, 5184, 5184//32):
        peram_Nev = \
            corr.peram_V1(
                peram_name=f'_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12_source{Nev}.npy',
                peram_save_path = f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L12x32/light/{conf_id}/t',
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = 5184//32,
                read_type = 'numpy',
                t_source = f'{t_src:03d}',
                scatter = False
                # ttype = '03d'
            )
            
        if Nev == 0:
            peram_u_2pt = peram_Nev[..., :100]
            peram_d_2pt = seq_peram(peram_u_2pt)
            peram = peram_Nev[t_src]

        else:
            peram = numpy.append(peram, peram_Nev[t_src], axis = -1)
    
    for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
        weight_2 = corr.eigen.create_omega(
            exact = 100, 
            N_eigen = [0],
            N_sum = [0],
            N_extract = [0],
            noise = Nev_curr - 100
        )
        
        bubble_matrix[Nev_curr_indx, :, :, t_src] = contract('Mcd,MGab,badc->MG', *[backend.asarray(x) for x in [VdV_link[t_src, :, 0, :Nev_curr, :Nev_curr] * weight_2, mu_gamma, peram[..., :Nev_curr, :Nev_curr]]])
    
        for t_sink in range(0, corr.Nt, 8):
            corr_2pt_matrix[Nev_curr_indx, :, :, t_src, t_sink] = contract(
                'manb,cedf,bd,Mfn,ac,MFem->MF', 
                *[backend.asarray(x) for x in [peram_u_2pt[t_sink, ..., :Nev_curr, :], peram_d_2pt[t_sink, ..., :Nev_curr], sink[t_src, 0].conj().T, VdV_link[t_sink, :, 0, :Nev_curr, :Nev_curr] * weight_2, gamma(5), mu_gamma]], 
                optimize = 'greedy'
                )



for Mom_indx, Mom_list in enumerate(Mom_link[:]):
    for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
        for mu_indx_1, mu in enumerate(mu_indx[Mom_indx]):
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'all_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_20_0.12_curr{Nev_curr}',
                corr_matrix = bubble_matrix[Nev_curr_indx, Mom_indx, mu_indx_1],
                Cpt = 'bubble',
            )
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_20_0.12_src{Nev_src}_sink{Nev_curr}',
                corr_matrix = corr_2pt_matrix[Nev_curr_indx, Mom_indx, mu_indx_1],
                Cpt = '2pt_all_t',
            )