import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from function_contraction_test import *

import numpy
from opt_einsum import contract

from pyquda_utils import io

import time
backend_name = 'cupy'

set_backend(backend_name)
backend = get_backend()

grid_size = [1, 1, 1, 12]

lattice_size = [48, 48, 48, 96]

exact_eigen = 100
N_eigen = [0]
N_sum = [0]
N_extract = [0]
noise_eigen = 200

exact_eigen_name = 100
N_eigen_name = [0]
N_sum_name = [0]
N_extract_name = [0]
noise_eigen_name = 200

Mom_link = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]

Nbin = 1
N_stout = 20

t_sep_list = [16]

Nev_src = 100

Nev_curr_list = [100, 300]
Nev_max = max(Nev_curr_list)

link_max = 0

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
        eigen_save_path = [
            f'/public/home/huzc/work/eigen_Field/output/beta6.41_mu-0.2295_ms-0.2050_L48x96/EigenField_100_StoutSmear_20_0.12/{conf_id}/eigField_t',
            f'/public/home/huzc/work/randomV/output/beta6.41_mu-0.2295_ms-0.2050_L48x96/randV_eig100_rand200_StoutSmear_20_0.12_GaussianDIS/{conf_id}/randomVector_t'
        ],
        
        eigen_name = [
            f'_cfg{conf_id}_Ne100.pkl',
            f'_cfg{conf_id}_Neig100_Nrv200.pkl'
        ],
        
        exact_eigen = exact_eigen,
        N_eigen = N_eigen,
        N_sum = N_sum,
        N_extract = N_extract,
        noise_eigen = noise_eigen,
        Nev_src = Nev_src,
        link_max = link_max,
        link_dir = 'Z',
        Mom = [[0, 0, 0]],
        Mom_link = Mom_link,
        bin = [Nbin, 0, 0, 0],
        bin_type = 'no_bin',
        ttype = ':03d',
        read_type = 'numpy', # pickle
        add_weight = True
    )

peram = {}
corr_matrix = {}
source = sink.conj()

mu_nu_indx, mu_nu_gamma, mu_indx, mu_gamma =  PFF_Mom_to_gamma(Mom_link)

corr_2pt_matrix = backend.zeros((1, corr.Nt, corr.Gt), dtype = complex)

# bubble_2pt_matrix_shape = create_result_shape(contraction_same_indx = [[0], [0]], dims = [VdV_link.ndim - 3, mu_gamma - 2], shapes = [VdV_link.shape[1:-2], mu_gamma.shape[:-2]], Nt = corr.Nt, Cpt = 'bubble')

corr_2pt_mu_matrix = backend.zeros((1, len(Nev_curr_list), len(Mom_link), 2*link_max + 1, len(mu_indx[0]), corr.Nt, corr.Gt), dtype = complex)

corr_bubble_light_mu_matrix = backend.zeros((1, len(Nev_curr_list), len(Mom_link), 2*link_max + 1, len(mu_indx[0]), corr.Nt), dtype = complex)
corr_bubble_strange_mu_matrix = backend.zeros((1, len(Nev_curr_list), len(Mom_link), 2*link_max + 1, len(mu_indx[0]), corr.Nt), dtype = complex)

# corr_3pt_matrix_shape = create_result_shape(contraction_same_indx = [[0], [0, 1], [0], [0, 1]], dims = [VdV_link.ndim - 3, mu_nu_gamma - 3, VdV_link.ndim - 3, mu_nu_gamma - 3], shapes = [VdV_link.shape[1:-2], mu_gamma.shape[:-3], VdV_link.shape[1:-2], mu_gamma.shape[:-3]], Nt = corr.Nt, Cpt = 'bubble')

corr_3pt_matrix = backend.zeros((1, len(t_sep_list), len(Nev_curr_list), len(Mom_link), 2*link_max + 1, len(mu_indx[0]), 2*link_max + 1, corr.Nt, corr.Gt), dtype = complex)

for t_src in range(corr.Nt):
    peram['src'] = \
        corr.peram_V1(
            peram_name=[[f'_Neig100_Nrv200_Nsnk300xNsrc300.pkl']],
            peram_save_path = [
                f'/public/home/huzc/data/new_data/IPV_peram/output/beta6.41_mu-0.2295_ms-0.2050_L48x96/IPV_peram_Neig100_Nrv200_StoutSmear_20_0.12_Precise_1e-08_qm-0.2038_GaussianDIS/{conf_id}/IPV_perams_cfg{conf_id}_tsrc'
                ],
            use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
            Nev_src = Nev_max,
            read_type = backend_name,
            t_load_type = 'slice',
            t = f'{t_src}',
            # ttype = '03d'
        )
        
    for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
        corr_matrix.update(
            corr.corr_all(
                t_src = t_src,
                sink = VdV_link,
                source = source[0],
                peram = peram,
                Cpt = ['bubble'],
                sink_src_gamma = [mu_gamma, gamma(5)],
                Nev_sink_src = [Nev_curr, Nev_src],
                link_max = 0,
                VdV_link = VdV_link,
                Nev_curr = Nev_curr,
                link_quark = 'u',
                curr_gamma = mu_gamma,
                contraction_same_indx = [[0], [0], [0], [0]]
            )
        )
        
        corr_bubble_strange_mu_matrix[:, Nev_curr_indx, ..., t_src] = corr_matrix['bubble']
        
    peram['src'] = \
        corr.peram_V1(
            peram_name=[[f'_Neig100_Nrv200_Nsnk300xNsrc300.pkl']],
            peram_save_path = [
                f'/public/home/huzc/data/new_data/IPV_peram/output/beta6.41_mu-0.2295_ms-0.2050_L48x96/IPV_peram_Neig100_Nrv200_StoutSmear_20_0.12_Precise_3e-06_qm-0.2295_GaussianDIS/{conf_id}/IPV_perams_cfg{conf_id}_tsrc'
                ],
            use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
            Nev_src = Nev_max,
            read_type = backend_name,
            t_load_type = 'slice',
            t = f'{t_src}',
            # ttype = '03d'
        )
        
    corr_matrix.update(
        corr.corr_all(
            t_src = t_src,
            sink = sink[0],
            source = source[0],
            peram = peram,
            Cpt = ['2pt'],
            sink_src_gamma = [gamma(5), gamma(5)],
            Nev_sink_src = [Nev_src, Nev_src],
            contraction_same_indx = [[0], [0], [0], [0]]
        )
    )
    corr_2pt_matrix[:, t_src, :] = corr_matrix['2pt']
    
    for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
        corr_matrix.update(
            corr.corr_all(
                t_src = t_src,
                sink = VdV_link,
                source = source[0],
                peram = peram,
                Cpt = ['bubble', '2pt'],
                sink_src_gamma = [mu_gamma, gamma(5)],
                Nev_sink_src = [Nev_curr, Nev_src],
                link_max = 0,
                VdV_link = VdV_link,
                Nev_curr = Nev_curr,
                link_quark = 'u',
                curr_gamma = mu_gamma,
                contraction_same_indx = [[0], [0], [0], [0]]
            )
        )
        
        corr_bubble_light_mu_matrix[:, Nev_curr_indx, ..., t_src] = corr_matrix['bubble']
        corr_2pt_mu_matrix[:, Nev_curr_indx, ..., t_src, :] = corr_matrix['2pt']

    for t_sep_indx, t_sep in enumerate(t_sep_list):
        peram['sep'] = \
            corr.peram_V1(
            peram_name=[[f'_Neig100_Nrv200_Nsnk300xNsrc300.pkl']],
            peram_save_path = [f'/public/home/huzc/data/new_data/IPV_peram/output/beta6.41_mu-0.2295_ms-0.2050_L48x96/IPV_peram_Neig100_Nrv200_StoutSmear_20_0.12_Precise_3e-06_qm-0.2295_GaussianDIS/{conf_id}/IPV_perams_cfg{conf_id}_tsrc'],
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = Nev_max,
                read_type = backend_name,
                t_load_type = 'slice',
                # t = f'{((t_sep + t_src)%corr.Nt)}'
                t = f'{((t_sep + t_src)%corr.Nt)}'
            )
            
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            corr_matrix.update(
                corr.corr_all(
                    t_src = t_src,
                    sink = VdV_link.conj().transpose(0, 1, 2, 4, 3),
                    source = source[0],
                    peram=peram,
                    Cpt = ['3pt'],
                    sink_src_gamma = [mu_nu_gamma[..., 0, :, :], gamma(5)],
                    Nev_sink_src = [Nev_curr, Nev_src],
                    link_max = 0,
                    VdV_link = VdV_link,
                    t_sep = t_sep,
                    Nev_curr = Nev_curr,
                    link_quark = 'u',
                    curr_gamma = mu_nu_gamma[..., 1, :, :],
                    contraction_same_indx = [[0], [0, 1], [0], [0, 1]]
                )
            )
            corr_3pt_matrix[:, t_sep_indx, Nev_curr_indx, ..., t_src, :] = corr_matrix['3pt']

for Mom_indx, Mom_list in enumerate(Mom_link):
    for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
        corr.write(
            corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_link[Mom_indx][2]}Py{Mom_link[Mom_indx][1]}Pz{Mom_link[Mom_indx][0]}/ENV_{Nev_curr}/conf{conf_id}', 
            corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_blending',
            corr_matrix = corr_2pt_matrix[:],
            Cpt = '2pt',
        )
        
        for mu_indx_1, mu in enumerate(mu_indx[Mom_indx]):
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_link[Mom_indx][2]}Py{Mom_link[Mom_indx][1]}Pz{Mom_link[Mom_indx][0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_blending',
                corr_matrix = corr_2pt_mu_matrix[:, Nev_curr_indx, Mom_indx, 0, mu_indx_1],
                Cpt = '2pt_all_t',
            )

            corr.write(
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_link[Mom_indx][2]}Py{Mom_link[Mom_indx][1]}Pz{Mom_link[Mom_indx][0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'light_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_blending',
                corr_matrix = corr_bubble_light_mu_matrix[:, Nev_curr_indx, Mom_indx, 0, mu_indx_1],
                Cpt = 'bubble',
            )
            
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_link[Mom_indx][2]}Py{Mom_link[Mom_indx][1]}Pz{Mom_link[Mom_indx][0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'strange_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_blending',
                corr_matrix = corr_bubble_strange_mu_matrix[:, Nev_curr_indx, Mom_indx, 0, mu_indx_1],
                Cpt = 'bubble',
            )
            
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_link[Mom_indx][2]}Py{Mom_link[Mom_indx][1]}Pz{Mom_link[Mom_indx][0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'all_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_blending',
                corr_matrix = corr_bubble_light_mu_matrix[:, Nev_curr_indx, Mom_indx, 0, mu_indx_1] - corr_bubble_strange_mu_matrix[:, Nev_curr_indx, Mom_indx, 0, mu_indx_1],
                Cpt = 'bubble',
            )
            
        for t_sep_indx, t_sep in enumerate(t_sep_list):
            for gamma_indx, gamma_list in enumerate(mu_nu_indx[Mom_indx]):
                corr.write(
                    t_sep = t_sep,
                    corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_link[Mom_indx][2]}Py{Mom_link[Mom_indx][1]}Pz{Mom_link[Mom_indx][0]}/ENV_{Nev_curr}/conf{conf_id}', 
                    corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_blending', # linkZ, conserved
                    corr_matrix = corr_3pt_matrix[:, t_sep_indx, Nev_curr_indx, Mom_indx, 0, gamma_indx, :], 
                    Cpt = '3pt',
                    gamma = f'0{gamma_list[0]}0{gamma_list[1]}05',
                    conserved = corr.conserved,
                    link_quark = ['u'],
                )

print('complete')
