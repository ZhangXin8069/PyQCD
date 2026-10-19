import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from function_contraction import *

import numpy
from opt_einsum import contract

from pyquda_utils import io

import time
backend_name = 'numpy'

set_backend(backend_name)
backend = get_backend()

grid_size = [1, 1, 1, 24]

lattice_size = [24, 24, 24, 72]

exact_eigen = 50
N_eigen = [100, 200, 300, 500, 800]
N_sum = [20, 20, 20, 20, 20]
N_extract = [10, 10, 10, 10, 10]
noise_eigen = 250

exact_eigen_name = 50
N_eigen_name = [100, 200, 300, 500, 800]
N_sum_name = [20, 20, 20, 20, 20]
N_extract_name = [10, 10, 10, 10, 10]
noise_eigen_name = 250

Nbin = 1
N_stout = 20

t_sep_list = [6, 9, 12]

Nev_src = 100

Nev_curr_list = [50, 400]
Nev_max = Nev_curr_list[-1]
link_max = 0
Mom_sink = [[0, 0, 0]]
Mom_link = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
mu_nu_indx, mu_nu_gamma, mu_indx, mu_gamma =  PFF_Mom_to_gamma(Mom_link[:], Pion_Mom = True)

gauge_path = f'/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_'

conf_id = sys.argv[1]

if backend_name == 'numpy':
    from pyquda import initGrid
    initGrid(grid_size = grid_size)
    
    if link_max != 0:
        gauge = io.readQIOGauge(f'{gauge_path}{str(conf_id)}.lime')
        gauge_link = (gauge.lexico())
        gauge_link = stout_smear_ndarray(_U = gauge_link, nstep = 20, rho = 0.12)
        del gauge
    
    else:
        gauge_link = False    

elif backend_name == 'cupy':
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

corr = distillation_func(
    conf_id = conf_id,

    quark_type = 'ud',
    link_quark = 'u',

    lattice_size = lattice_size, # Nx, Ny, Nz, Nt 
    
    Nev_src = Nev_src,

    gauge = gauge_link,
    conserved = False,
    backend = backend_name
)

sink, VdV_link = \
    corr.sink_link_V1(
        eigen_name = [
                f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen}_V1_stout_smear_20_0.12.npy'
        ],
        eigen_save_path = [
                f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/{conf_id}_t'
                # f'/beegfs/group-bak/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/{conf_id}_t'
                
            ],
        exact_eigen = exact_eigen,
        N_eigen = N_eigen,
        N_sum = N_sum,
        N_extract = N_extract,
        noise_eigen = noise_eigen,
        Nev_src = Nev_src,
        link_max = link_max,
        link_dir = 'Z',
        Mom = Mom_sink,
        Mom_link = Mom_link,
        bin = [Nbin, 0, 0, 0],
        bin_type = 'no_bin',
        ttype = ':03d',
        read_type = 'numpy', # pickle
        add_weight = False      
    )
    
param = gamma_dic()

param['sink_operators'] = [['d', 'sink', 'u']]
param['source_operators'] = [['u', 'source', 'd']]

param['gamma_source'] = gamma(5)


# param['gamma_forword'] = (gamma(0) + gamma(4))/2

N_operator = len(param['sink_operators']) * len(param['source_operators'])

corr_2pt_matrix = backend.zeros((N_operator, corr.Nt, corr.Gt), dtype = complex)
corr_2pt_mu_matrix = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2, corr.Nt, corr.Gt), dtype = complex)
corr_3pt_matrix = backend.zeros((len(Nev_curr_list), len(t_sep_list), N_operator, len(Mom_link), 2, 2 * link_max + 1, corr.Nt, corr.Gt), dtype = complex)
bubble_mu_light = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2 * link_max + 1, 2, corr.Nt), dtype = complex)
bubble_mu_strange = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2 * link_max + 1, 2, corr.Nt), dtype = complex)

param['source'] = sink[:, 0].conj().transpose(0, 2, 1)
param['VdV_curr'] = VdV_link[:, :]
param['Nev_src'] = Nev_src

for t_src in range(0, corr.Nt, 1):
    param['t_src'] = t_src
    param['peram_u'] = param['peram_d_sep'] = \
        corr.peram_V1(
            peram_name=[[f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen}_V1_stout_smear_20_0.12.npy']],
            peram_save_path = [[f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t']],            
            use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
            Nev_src = Nev_max,
            read_type = backend.__name__,
            t_load_type = 'slice',
            t = f'{t_src:03d}',
            # ttype = '03d'
        )[0]
    
    param['peram_s'] = \
        corr.peram_V1(
            peram_name=[[f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen}_V1_stout_smear_20_0.12.npy']],
            peram_save_path = [[f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/strange/{conf_id}/t']],            
            use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
            Nev_src = Nev_max,
            read_type = backend.__name__,
            t_load_type = 'slice',
            t = f'{t_src:03d}',
            # ttype = '03d'
        )[0]
    
    param['Nev_sink'] = Nev_src
    param['gamma_sink'] = gamma(5)
    
    param['sink'] = sink[:, 0]
    
    corr_2pt_matrix[..., t_src, :] = (
        corr.corr_all(
            param = param,
            Cpt = '2pt',
            t_range = range(0, 36, 1),
            contraction_same_indx = [[-1], [-1], [-1], [-1], [-1], [-1]]
        )
    )
    
    for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
        param['Nev_sink'] = Nev_curr
        param['Nev_curr'] = Nev_curr

        param['gamma_sink'] = mu_gamma
        param['sink'] = VdV_link[:, :, 0]

        corr_2pt_mu_matrix[Nev_curr_indx, ..., t_src, :] = (
            corr.corr_all(
                param = param,
                Cpt = '2pt',
                t_range = range(0, 36, 1),
                contraction_same_indx = [[-1], [-1], [0], [0, 1], [-1], [-1]]
            )
        )

        param['gamma_curr'] = mu_gamma
        
        bubble_mu_light[Nev_curr_indx, ..., t_src] = (
            corr.corr_all(
                param = param,
                Cpt = 'bubble',
                link_quark = 'u',
                contraction_same_indx = [[-1], [-1], [-1], [-1], [0], [0]],
            )
        )
        bubble_mu_strange[Nev_curr_indx, ..., t_src] = (
            corr.corr_all(
                param = param,
                Cpt = 'bubble',
                link_quark = 's',
                contraction_same_indx = [[-1], [-1], [-1], [-1], [0], [0]],
            )
        )
        
    param['sink'] = VdV_link[:, :, 0].conj().transpose(0, 1, 3, 2)
    param['gamma_sink'] = mu_nu_gamma[:, 0, :, 1]
    param['gamma_curr'] = mu_nu_gamma[:, 0, :, 0]

    for t_sep_indx, t_sep in enumerate(t_sep_list):

        param['t_sep'] = t_sep
        param['peram_u_sep'] = \
            corr.peram_V1(
                peram_name=[[f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen}_V1_stout_smear_20_0.12.npy']],
                peram_save_path = [[f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t']],
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = Nev_max,
                read_type = backend_name,
                t_load_type = 'slice',
                t = f'{((t_sep + t_src)%corr.Nt):03d}'
            )[0]
        
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            param['Nev_sink'] = Nev_curr
            param['Nev_curr'] = Nev_curr

            corr_3pt_matrix[Nev_curr_indx, t_sep_indx, ..., t_src, :] = (
                corr.corr_all(
                    param = param,
                    Cpt = '3pt',
                    t_range = range(0, 36, 1),
                    link_quark = 'u',
                    contraction_same_indx = [[-1], [-1], [0], [0, 1], [0], [0, 1]],
                )
            )
            
for Mom_indx, Mom_list in enumerate(Mom_link[:]):
    for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
        corr.write(
            corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
            corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_src}',
            corr_matrix = corr_2pt_matrix[0],
            Cpt = '2pt',
        )
        
        for mu_indx_1, mu in enumerate(mu_indx[Mom_indx]):
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}',
                corr_matrix = corr_2pt_mu_matrix[Nev_curr_indx, 0, Mom_indx, mu_indx_1],
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
            for gamma_indx, gamma_list in enumerate(mu_nu_indx[Mom_indx, 0]):
                corr.write(
                    t_sep = t_sep,
                    corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                    corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}', # linkZ, conserved
                    corr_matrix = corr_3pt_matrix[Nev_curr_indx, t_sep_indx, 0, Mom_indx, gamma_indx, :], 
                    Cpt = '3pt',
                    gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                    conserved = corr.conserved,
                    link_quark = 'u',
                )

if corr.rank == 0:
    print('complete')
