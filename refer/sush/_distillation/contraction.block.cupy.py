import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from function_contraction_new_2 import *

import numpy
from opt_einsum import contract

from pyquda_utils import io

import time

set_backend('cupy')
backend = get_backend()

grid_size = [1, 1, 1, 1]

lattice_size = [32, 32, 32, 96]

exact_eigen = 100
N_eigen = [100, 200, 400, 600, 800]
N_sum = [20, 20, 20, 20, 20]
N_extract = [10, 10, 10, 10, 10]
noise_eigen = 200

exact_eigen_name = 100
N_eigen_name = [100, 200, 400, 600, 800]
N_sum_name = [20, 20, 20, 20, 20]
N_extract_name = [10, 10, 10, 10, 10]
noise_eigen_name = 400
conf_type = "beta6.41_mu-0.2295_ms-0.2050_L32x96"
Nbin = 1
N_stout = 20

t_sep_list = [12, 16, 20]

Nev_src_list = [100]

Nev_curr_list = [100, 400]
# Nev_max = max(Nev_curr_list + [Nev_src])
Nev_max = max(Nev_src_list + Nev_curr_list )

link_max = 0
Mom_sink = [[0, 0, 0]]
Mom_link = creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)
mu_nu_indx, mu_nu_gamma, mu_indx, mu_gamma =  PFF_Mom_to_gamma(Mom_link[:], Pion_Mom = False)

gauge_path = f'/public/group/lqcd/configurations/CLOVER/{conf_type}/{conf_type}_cfg_'

conf_id = sys.argv[1]

if backend.__name__ == 'numpy':
    from pyquda import initGrid
    initGrid(grid_size = grid_size)
    
    if link_max != 0:
        gauge = io.readQIOGauge(f'{gauge_path}{str(conf_id)}.lime')
        gauge_link = (gauge.lexico())
        gauge_link = stout_smear_ndarray(_U = gauge_link, nstep = 20, rho = 0.12)
        del gauge
    
    else:
        gauge_link = False    

elif backend.__name__ == 'cupy':
    from pyquda import init
    init(resource_path=".cache", grid_size = grid_size, latt_size = lattice_size)
    
    if link_max != 0:
        gauge = io.readQIOGauge(f'{gauge_path}{str(conf_id)}.lime')
        gauge.smearSTOUT(20, 0.12, 3)
        gauge_link = (gauge.lexico())
        
        del gauge
    
    else:
        gauge_link = False

# Mom_link = [[1, 1, 0], [1, 0, 1], [0, 1, 1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1], [-1, -1, 0], [-1, 0, -1], [0, -1, -1], [1, -1, 0], [1, 0, -1], [0, 1, -1]]
param = creat_param()
param.sink_operators([['d^d', 'gamma_sink', 'u', 1]])
param.source_operators([['u^d', 'gamma_source', 'd', -1]])
param.curr_operators(['u^d', 'gamma_curr', 'u'])

corr = distillation_func(
    conf_id = conf_id,
    param = param.data(),

    lattice_size = lattice_size, # Nx, Ny, Nz, Nt 
    
    Nev_src = max(Nev_src_list),

    # gauge = gauge_link,
    conserved = False,
    backend = backend.__name__
)


sink, VdV_link = \
    corr.sink_link_V1(
        eigen_name = [
                f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy'
        ],
        eigen_save_path = [
                f'/nexdata/project/lqcd/sush/eigensystem/{conf_type}/{conf_id}/{conf_id}_t'
                # f'/beegfs/group-bak/lqcd/sush/eigensystem/{conf_type}/{conf_id}/{conf_id}_t'
                
            ],
        exact_eigen = exact_eigen,
        N_eigen = N_eigen,
        N_sum = N_sum,
        N_extract = N_extract,
        noise_eigen = noise_eigen,
        Nev_src = max(Nev_src_list),
        link_max = link_max,
        link_dir = 'Z',
        Mom = Mom_sink,
        Mom_link = Mom_link,
        bin = [Nbin, 0, 0, 0],
        bin_type = 'no_bin',
        ttype = ':03d',
        read_type = 'numpy', # pickle
        add_weight = True
    )

param.gamma_projection((gamma(0) + gamma(4))/2)
param.source(sink[:, 0].conj().transpose(0, 2, 1))

param.gamma_source(gamma(5))


N_operator = len(param.param['sink_operators']) * len(param.param['source_operators'])

corr_2pt_matrix = backend.zeros((len(Nev_src_list), N_operator, corr.Nt, corr.Gt), dtype = complex)

corr_2pt_mu_matrix = backend.zeros((len(Nev_src_list), len(Nev_curr_list), N_operator, len(Mom_link), 2, corr.Nt, corr.Gt), dtype = complex)
corr_3pt_matrix = backend.zeros((len(Nev_src_list), len(Nev_curr_list), len(t_sep_list), N_operator, len(Mom_link), 2, 2 * link_max + 1, corr.Nt, corr.Gt), dtype = complex)
# corr_3pt_matrix_dis = backend.zeros((len(Nev_curr_list), len(t_sep_list), 2, len(Mom_link), 2, 2 * link_max + 1, corr.Nt, corr.Gt), dtype = complex)

# corr_3pt_matrix_light_compensation = backend.zeros((len(Nev_curr_list), len(t_sep_list), 2, len(Mom_link), 2, 2 * link_max + 1, corr.Nt), dtype = complex)
# corr_3pt_matrix_strange_compensation = backend.zeros((len(Nev_curr_list), len(t_sep_list), 1, len(Mom_link), 2, 2 * link_max + 1, corr.Nt), dtype = complex)

bubble_mu_light = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2 * link_max + 1, 2, corr.Nt), dtype = complex)
bubble_mu_strange = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2 * link_max + 1, 2, corr.Nt), dtype = complex)

for t_src in range(0, corr.Nt, 1):
    param.t_src(t_src)
    peram_u = \
        corr.peram_V1(
            peram_name=[[f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy']],
            peram_save_path = [[f'/nexdata/project/lqcd/sush/perambulators/{conf_type}/light/{conf_id}/t']],            
            use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
            Nev_src = Nev_max,
            read_type = 'numpy',
            t_load_type = 'slice',
            t = f'{t_src:03d}',
            # ttype = '03d'
        )[0]
        
    param.peram(peram_u, sep = False, quark = 'u', curr = False, curr_id = 0, use_2pt = True)
    param.peram(peram_u, sep = True , quark = 'd', curr = False, curr_id = 0, use_2pt = True)
    param.peram(peram_u, sep = False, quark = 'u', curr = False, curr_id = 0, use_2pt = False)
    param.peram(peram_u, sep = False, quark = 'u', curr = True , curr_id = 0, use_2pt = False)
    
    peram_s = \
        corr.peram_V1(
            peram_name=[[f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy']],
            peram_save_path = [[f'/nexdata/project/lqcd/sush/perambulators/{conf_type}/strange/{conf_id}/t']],            
            use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
            Nev_src = Nev_max,
            read_type = 'numpy',
            t_load_type = 'slice',
            t = f'{t_src:03d}',
            # ttype = '03d'
        )[0]
        
    param.peram(peram_s, sep = False , quark = 's', curr = True, curr_id = 0, use_2pt = False)

    param.gamma_sink(gamma(5))
    
    param.sink(sink[:, 0])
    for Nev_src_indx, Nev_src in enumerate(Nev_src_list):
        param.Nev_sink(Nev_src)
        param.Nev_src(Nev_src)
        
        corr_2pt_matrix[Nev_src_indx, ..., t_src, :] = (
            corr.corr_all(
                param = param.data(),
                Cpt = '2pt',
                t_range = range(0, 60, 1),
                contraction_same_indx = [[-1], [-1], [-1], [-1], [-1], [-1]]
            )
        )

    for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
        param.Nev_sink(Nev_curr)
        param.Nev_curr(Nev_curr, curr_id = 0)

        param.gamma_sink(mu_gamma)
        
        param.VdV_curr(VdV_link[..., :Nev_curr, :Nev_curr], curr_id = 0)
        param.sink(VdV_link[:, :, 0, :Nev_curr, :Nev_curr])
        for Nev_src_indx, Nev_src in enumerate(Nev_src_list):
            param.Nev_src(Nev_src)
            
            corr_2pt_mu_matrix[Nev_src_indx, Nev_curr_indx, ..., t_src, :] = (
                corr.corr_all(
                    param = param.data(),
                    Cpt = '2pt',
                    t_range = range(0, 60, 1),
                    contraction_same_indx = [[-1], [-1], [0], [0, 1], [-1], [-1]]
                )
            )

        param.gamma_curr(mu_gamma)
        
        bubble_mu_light[Nev_curr_indx, ..., t_src] = (
            corr.corr_all(
                param = param.data(),
                Cpt = 'bubble',
                contraction_same_indx = [[-1], [-1], [-1], [-1], [0], [0]],
            )
        )
        
        param.curr_operators(['s^d', 'gamma_curr', 's'])
        
        bubble_mu_strange[Nev_curr_indx, ..., t_src] = (
            corr.corr_all(
                param = param.data(),
                Cpt = 'bubble',
                contraction_same_indx = [[-1], [-1], [-1], [-1], [0], [0]],
            )
        )
        
        param.curr_operators(['u^d', 'gamma_curr', 'u'])
        
    # param.sink(VdV_link[:, :, 0].conj().transpose(0, 1, 3, 2))
    param.gamma_sink(mu_nu_gamma[:, :, 1])
    param.gamma_curr(mu_nu_gamma[:, :, 0])

    for t_sep_indx, t_sep in enumerate(t_sep_list):

        param.t_sep(t_sep)
        peram_u_sep = \
            corr.peram_V1(
                peram_name=[[f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy']],
                peram_save_path = [[f'/nexdata/project/lqcd/sush/perambulators/{conf_type}/light/{conf_id}/t']],
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = Nev_max,
                read_type = backend.__name__,
                t_load_type = 'slice',
                t = f'{((t_sep + t_src)%corr.Nt):03d}'
            )[0]
    
        param.peram(peram_u_sep, sep = True , quark = 'u', curr = True, curr_id = 0, use_2pt = False)

        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            param.VdV_curr(VdV_link[..., :Nev_curr, :Nev_curr], curr_id = 0)
            param.sink(VdV_link[:, :, 0, :Nev_curr, :Nev_curr].conj().transpose(0, 1, 3, 2))
            
            param.Nev_sink(Nev_curr)
            param.Nev_curr(Nev_curr)
            for Nev_src_indx, Nev_src in enumerate(Nev_src_list):
                param.Nev_src(Nev_src)
                
                corr_3pt_matrix[Nev_src_indx, Nev_curr_indx, t_sep_indx, ..., t_src, :] = (
                    corr.corr_all(
                        param = param.data(),
                        Cpt = '3pt',
                        t_range = range(0, 60, 1),
                        contraction_same_indx = [[-1], [-1], [0], [0, 1], [0], [0, 1]],
                    )
                )
for Nev_src_indx, Nev_src in enumerate(Nev_src_list):
    corr.write(
        corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}', 
        corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_src}',
        corr_matrix = corr_2pt_matrix[Nev_src_indx, 0],
        Cpt = '2pt',
    )
    
    corr.write(
        corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}', 
        corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_src}',
        corr_matrix = corr_2pt_matrix[Nev_src_indx, 0],
        Cpt = '2pt_all_t',
    )
    
for Mom_indx, Mom_list in enumerate(Mom_link[:]):
    for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
        for mu_indx_1, mu in enumerate(mu_indx[Mom_indx]):
            for Nev_src_indx, Nev_src in enumerate(Nev_src_list):
                corr.write(
                    corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                    corr_name = f'gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}',
                    corr_matrix = corr_2pt_mu_matrix[Nev_src_indx, Nev_curr_indx, 0, Mom_indx, mu_indx_1],
                    Cpt = '2pt_all_t',
                )
            
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'light_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr{Nev_curr}',
                corr_matrix = bubble_mu_light[Nev_curr_indx, 0, Mom_indx, 0, mu_indx_1],
                Cpt = 'bubble',
            )
            
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'strange_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr{Nev_curr}',
                corr_matrix = bubble_mu_strange[Nev_curr_indx, 0, Mom_indx, 0, mu_indx_1],
                Cpt = 'bubble',
            )
            
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'all_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr{Nev_curr}',
                corr_matrix = bubble_mu_light[Nev_curr_indx, 0, Mom_indx, 0, mu_indx_1] - bubble_mu_strange[Nev_curr_indx, 0, Mom_indx, 0, mu_indx_1],
                Cpt = 'bubble',
            )
            
        for t_sep_indx, t_sep in enumerate(t_sep_list):
            for gamma_indx, gamma_list in enumerate(mu_nu_indx[Mom_indx]):
                for Nev_src_indx, Nev_src in enumerate(Nev_src_list):
                    corr.write(
                        t_sep = t_sep,
                        corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                        corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}', # linkZ, conserved
                        corr_matrix = corr_3pt_matrix[Nev_src_indx, Nev_curr_indx, t_sep_indx, 0, Mom_indx, gamma_indx, :], 
                        Cpt = '3pt',
                        gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                        conserved = corr.conserved,
                        link_quark = 'u',
                    )
                    
                    corr.write(
                        t_sep = t_sep,
                        corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                        corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}', # linkZ, conserved
                        corr_matrix = corr_3pt_matrix[Nev_src_indx, Nev_curr_indx, t_sep_indx, 0, Mom_indx, gamma_indx, :], 
                        Cpt = '3pt_all_t',
                        gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                        conserved = corr.conserved,
                        link_quark = 'u',
                    )
                    
if corr.rank == 0:
    print('complete')
