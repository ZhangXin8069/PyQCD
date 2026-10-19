import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from function_contraction_new_2 import *

import numpy
from opt_einsum import contract

from pyquda_utils import io

import time

set_backend('numpy')
backend = get_backend()

grid_size = [1, 1, 1, 12]

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

t_sep_list = [12]

Nev_src = 100

Nev_curr_list = [0]
# Nev_max = max(Nev_curr_list + [Nev_src])
Nev_max = 400

link_max = 0
Mom_sink = [x[::-1] for x in [[0 ,0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5]]]
# Mom_sink = creat_mom_list(Mom = [[0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5]], fix_Q2 = True)
Mom_link = [[0, 0, 0]]
# mu_nu_indx, mu_nu_gamma, mu_indx, mu_gamma =  PFF_Mom_to_gamma(Mom_link[:], Pion_Mom = False)

gauge_path = f'/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_'

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
param.curr_operators([])

corr = distillation_func(
    conf_id = conf_id,
    param = param.data(),

    lattice_size = lattice_size, # Nx, Ny, Nz, Nt 
    
    Nev_src = Nev_src,

    gauge = gauge_link,
    conserved = False,
    backend = backend.__name__
)


sink, VdV_link = \
    corr.sink_link_V1(
        eigen_name = [
                f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy'
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

param.gamma_projection((gamma(0) + gamma(4))/2)
param.source(sink[:].conj().transpose(0, 1, 3, 2))
param.Nev_src(Nev_src)
param.gamma_source(gamma(5))


N_operator = len(param.param['sink_operators']) * len(param.param['source_operators'])

corr_2pt_matrix = backend.zeros((N_operator, len(Mom_sink), corr.Nt, corr.Gt), dtype = complex)


for t_src in range(0, corr.Nt, 1):
    param.t_src(t_src)
    peram_u = \
        corr.peram_V1(
            peram_name=[[f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy']],
            peram_save_path = [[f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t']],            
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
    

    param.gamma_sink(gamma(5))
    param.Nev_sink(Nev_src)
    param.sink(sink[:])
    
    corr_2pt_matrix[..., t_src, :] = (
        corr.corr_all(
            param = param.data(),
            Cpt = '2pt',
            t_range = range(0, 36, 1),
            contraction_same_indx = [[0], [-1], [0], [-1], [-1], [-1]]
        )
    )


for Mom_indx, Mom_list in enumerate(Mom_sink[:]):
    corr.write(
        corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_2pt/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_100/conf{conf_id}', 
        corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_src}',
        corr_matrix = corr_2pt_matrix[0, Mom_indx],
        Cpt = '2pt',
    )

if corr.rank == 0:
    print('complete')
