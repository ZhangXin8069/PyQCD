import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from function_contraction import *

import numpy
from opt_einsum import contract

from pyquda_utils import io

import time
backend_name = 'cupy'

set_backend(backend_name)
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

Nbin = 1
N_stout = 20

t_sep_list = [12]

Nev_src = 100

Nev_curr_list = [100, 400]
Nev_max = Nev_curr_list[-1]

link_max = 0

Mom_sink = [[0, 0, 0]]
Mom_link = [[0, 0, 0]]

gauge_path = f'/public/group/lqcd/configurations/CLOVER/beta6.41_mu-0.2295_ms-0.2050_L32x96/beta6.41_mu-0.2295_ms-0.2050_L32x96_cfg_'

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
                f'_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy'
        ],
        eigen_save_path = [
                f'/nexdata/project/lqcd/sush/eigensystem/beta6.41_mu-0.2295_ms-0.2050_L32x96/{conf_id}/{conf_id}_t'
                # f'/beegfs/group-bak/lqcd/sush/eigensystem/beta6.41_mu-0.2295_ms-0.2050_L32x96/{conf_id}/{conf_id}_t'
                
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
        add_weight = True
    )
    
param = gamma_dic()

param['sink_operators'] = [['d', 'sink', 'u']]
param['source_operators'] = [['u', 'source', 'd']]

param['gamma_source'] = gamma(5)
param['gamma_sink'] = gamma(5)

param['gamma_curr'] = gamma(4)

# param['gamma_forword'] = (gamma(0) + gamma(4))/2

N_operator = len(param['sink_operators']) * len(param['source_operators'])

corr_2pt_matrix = backend.zeros((N_operator, corr.Nt, corr.Gt), dtype = complex)
corr_3pt_matrix = backend.zeros((len(Nev_curr_list), len(t_sep_list), N_operator, len(Mom_link), 2 * link_max + 1, corr.Nt, corr.Gt), dtype = complex)

param['sink'] = sink[:, 0]
param['source'] = sink[:, 0].conj().transpose(0, 2, 1)

param['VdV_curr'] = VdV_link[:, :]

param['Nev_src'] = Nev_src
param['Nev_sink'] = Nev_src

for t_src in range(0, corr.Nt, 1):
    param['t_src'] = t_src
    param['peram_u'] = param['peram_d_sep'] = \
        corr.peram_V1(
            peram_name=[[f'_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy']],
            peram_save_path = [[f'/nexdata/project/lqcd/sush/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/light/{conf_id}/t']],            
            use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
            Nev_src = Nev_max,
            read_type = backend.__name__,
            t_load_type = 'slice',
            t = f'{t_src:03d}',
            # ttype = '03d'
        )[0]
    
    corr_2pt_matrix[..., t_src, :] = (
        corr.corr_all(
            param = param,
            Cpt = '2pt',
            t_range = range(0, 36, 1),
            contraction_same_indx = [[-1], [-1], [-1], [-1], [-1], [-1]]
        )
    )

    for t_sep_indx, t_sep in enumerate(t_sep_list):

        param['t_sep'] = t_sep
        param['peram_u_sep'] = \
            corr.peram_V1(
                peram_name=[[f'_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy']],
                peram_save_path = [[f'/nexdata/project/lqcd/sush/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/light/{conf_id}/t']],
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = Nev_max,
                read_type = backend_name,
                t_load_type = 'slice',
                t = f'{((t_sep + t_src)%corr.Nt):03d}'
            )[0]
        
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
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
        
        for t_sep_indx, t_sep in enumerate(t_sep_list):
            corr.write(
                t_sep = t_sep,
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}', # linkZ, conserved
                corr_matrix = corr_3pt_matrix[Nev_curr_indx, t_sep_indx, 0, Mom_indx, :], 
                Cpt = '3pt',
                gamma = f'050405',
                conserved = corr.conserved,
                link_quark = 'u',
            )

if corr.rank == 0:
    print('complete')
