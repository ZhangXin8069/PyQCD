import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from function_contraction_new_2 import *

from pyquda_utils import io

import time

grid_size = [1, 1, 1, 1]

backend_name = 'cupy'

set_backend(backend_name)
backend = get_backend()


lattice_size = [12, 12, 12, 32]

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

Nbin = 1
N_stout = 20

t_sep_list = [8]

Nev_src = 100

Nev_curr_list = [x for x in range(100, 900, 40)] + [5184]

Nev_max = max(Nev_curr_list + [Nev_src])

link_max = 0

gauge_path = f'/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L12x32/beta6.20_mu-0.2770_ms-0.2400_L12x32_cfg_'

conf_id = sys.argv[1]

st_smear = time.time()
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
ed_smear = time.time()

print(f'smear use time {(ed_smear - st_smear):.3f}')

param = creat_param()
param.sink_operators([['u', 'u', 'gamma_sink', 'd']])
param.source_operators([['u^d', 'gamma_source', 'd^d', 'u^d']])
param.curr_operators(['u^d', 'gamma_curr', 'u'])

Mom_link = [[0, 0, 0]]

corr = distillation_func(
    conf_id = conf_id,
    param = param.data(),

    lattice_size = lattice_size, # Nx, Ny, Nz, Nt 
    
    Nev_src = Nev_src,

    conserved = False,
    backend = backend.__name__
)

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
        Nev_src = Nev_src,
        link_max = link_max,
        link_dir = 'Z',
        Mom = Mom_link,
        Mom_link = Mom_link,
        ttype = ':03d',
        read_type = 'numpy', # pickle
        add_weight = False
    )

param.source(sink[:, 0].conj().transpose(0, 3, 2, 1))
param.sink(sink[:, 0])

param.gamma_projection((gamma(0) + gamma(4))/2)

param.Nev_src(Nev_src)
param.Nev_sink(Nev_src)

param.gamma_source(gamma(7))
param.gamma_sink(gamma(7))

import numpy as np
gamma_curr = np.zeros((9, 4, 4), dtype = complex)
gamma_projection = np.zeros((9, 4, 4), dtype = complex)
gamma_name = []
# scalar
gamma_curr[0] = gamma(0)
gamma_projection[0] = (gamma(0) + gamma(4))/2
gamma_name += ['scalar']
# pseudo-scalar
gamma_curr[1] = gamma(5)
gamma_projection[1] = (gamma(0) + gamma(4))/2
gamma_name += ['pseudo-scalar']

# vector
gamma_curr[2] = gamma(4)
gamma_projection[2] = (gamma(0) + gamma(4))/2
gamma_name += ['vector']

# axial-vector
gamma_curr[3] = gamma(5)@gamma(1)
gamma_projection[3] = (gamma(0) + gamma(4))@gamma(5)@gamma(1)/2
gamma_name += ['axial-vector-1']

gamma_curr[4] = gamma(5)@gamma(2)
gamma_projection[4] = (gamma(0) + gamma(4))@gamma(5)@gamma(2)/2
gamma_name += ['axial-vector-2']

gamma_curr[5] = gamma(5)@gamma(3)
gamma_projection[5] = (gamma(0) + gamma(4))@gamma(5)@gamma(3)/2
gamma_name += ['axial-vector-3']

# tensor
gamma_curr[6] = gamma(2)@gamma(3)
gamma_projection[6] = (gamma(0) + gamma(4))@gamma(5)@gamma(1)/2
gamma_name += ['tensor-1']

gamma_curr[7] = gamma(1)@gamma(3)
gamma_projection[7] = (gamma(0) + gamma(4))@gamma(5)@gamma(2)/2
gamma_name += ['tensor-2']

gamma_curr[8] = gamma(1)@gamma(2)
gamma_projection[8] = (gamma(0) + gamma(4))@gamma(5)@gamma(3)/2
gamma_name += ['tensor-3']

print(gamma_name)

corr_2pt_matrix = backend.zeros((2, len(gamma_projection), corr.Nt, corr.Gt), dtype = complex)
corr_3pt_u_matrix = backend.zeros((1, len(Nev_curr_list), len(t_sep_list), 4, len(gamma_projection), corr.Nt, corr.Gt), dtype = complex)

corr_3pt_d_matrix = backend.zeros((1, len(Nev_curr_list), len(t_sep_list), 2, len(gamma_projection), corr.Nt, corr.Gt), dtype = complex)

from opt_einsum import contract
import random
for t_src in range(0, corr.Nt, 8):
    param.t_src(t_src)
    
    peram_u = \
        corr.peram_V1(
            peram_name=[[f'_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12_source0.npy']],
            peram_save_path = [[f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L12x32/light/{conf_id}/t']],          
            use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
            Nev_src = Nev_src,
            read_type = 'numpy',
            t_load_type = 'slice',
            t = f'{t_src:03d}',
            # ttype = '03d'
        )[0]
    param.peram(peram_u, sep = False, quark = 'u', curr = False, curr_id = 0, use_2pt = False)
    param.peram(peram_u, sep = False, quark = 'u', curr = False, curr_id = 0, use_2pt = True)
    param.peram(peram_u, sep = False, quark = 'd', curr = False, curr_id = 0, use_2pt = True)
    corr_2pt = corr.corr_all(
        param = param.data(),
        Cpt = '2pt',
        t_range = range(0, 16, 1),
        contraction_same_indx = [[-1], [-1], [-1], [-1], [-1], [-1]],
        keep_projection = True,
    )
    corr_2pt_matrix[..., t_src, :] = contract('Nabt,Pba->NPt', corr_2pt, backend.asarray(gamma_projection))
    
    for t_sep_indx, t_sep in enumerate(t_sep_list):
        param.t_sep(t_sep)
        peram_u_curr = \
            corr.peram_V1(
                peram_name=[[f'_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12_source0.npy']],
                peram_save_path = [[f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L12x32/light/{conf_id}/t']],          
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = Nev_src,
                read_type = 'numpy',
                t_load_type = 'slice',
                t = f'{((t_src + t_sep)%corr.Nt):03d}',
                # ttype = '03d'
            )[0]
            
        
        param.gamma_curr(gamma_curr)
        
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            if Nev_curr >= 100:
                random_vector = [x for x in range(100)] + random.sample([x for x in range(100, Nev_max)], k = Nev_curr - 100)
            
            else:
                random_vector = [x for x in range(Nev_curr)]
                
            param.peram(peram_u[..., random_vector, :], sep = False, quark = 'u', curr = False, curr_id = 0, use_2pt = False)
            param.peram(peram_u[..., random_vector, :], sep = False, quark = 'u', curr = False, curr_id = 0, use_2pt = True)
            
            param.peram(peram_u[..., random_vector, :], sep = False, quark = 'd', curr = False, curr_id = 0, use_2pt = False)
            param.peram(peram_u[..., random_vector, :], sep = False, quark = 'd', curr = False, curr_id = 0, use_2pt = True)
            
            param.peram(peram_u_curr[..., random_vector, :], sep = True, quark = 'u', curr = True, curr_id = 0, use_2pt = False)
            param.peram(peram_u_curr[..., random_vector, :], sep = True, quark = 'd', curr = True, curr_id = 0, use_2pt = False)
            
            if Nev_curr >= 100:
                weight_matrix_2 = corr.eigen.create_omega(
                    exact = 100,
                    N_eigen = [0],
                    N_sum = [0],
                    N_extract = [0],
                    noise = Nev_curr - 100,
                )

            else:
                weight_matrix_2 = 1.0

            param.Nev_curr(Nev_curr, curr_id = 0)
            VdV_link_slicer = ArraySlicer(VdV_link)
            param.VdV_curr(VdV_link_slicer.slice(dims = [1, 2, 3, 4], indices = [0, 0, random_vector, random_vector])[:, 0, 0] * weight_matrix_2)
            
            # np.set_printoptions(linewidth = np.inf, threshold = np.inf)
            # print((VdV_link_slicer.slice(dims = [1, 2, 3, 4], indices = [0, 0, random_vector, random_vector])[:, 0, 0] * weight_matrix_2)[0, -10:, -10:])

            param.curr_operators(['u^d', 'gamma_curr', 'u'])
            
            # corr_3pt_u_matrix[0, Nev_curr_indx, t_sep_indx, :, t_src]
            corr_3pt = corr.corr_all(
                param = param.data(),
                Cpt = '3pt',
                t_range = range(0, 10, 1),
                contraction_same_indx = [[-1], [-1], [-1], [-1], [-1], [-1]],
                keep_projection = True,
                
            )
            # print(corr_3pt.shape)
            
            corr_3pt_u_matrix[0, Nev_curr_indx, t_sep_indx, :, :, t_src] = contract('NPabt,Pba->NPt', corr_3pt, backend.asarray(gamma_projection))
            
            param.curr_operators(['d^d', 'gamma_curr', 'd'])
            
            # corr_3pt_u_matrix[0, Nev_curr_indx, t_sep_indx, :, t_src]
            corr_3pt = corr.corr_all(
                param = param.data(),
                Cpt = '3pt',
                t_range = range(0, 10, 1),
                contraction_same_indx = [[-1], [-1], [-1], [-1], [-1], [-1]],
                keep_projection = True,    
            )
            
            corr_3pt_d_matrix[0, Nev_curr_indx, t_sep_indx, :, :, t_src] = contract('NPabt,Pba->NPt', corr_3pt, backend.asarray(gamma_projection))
            
for _gamma_name_indx, _gamma_name in enumerate(gamma_name):
    corr.write(
        corr_save_path = f'/public/home/sush/distillation/proton/result/{corr.Nx}x{corr.Nt}/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}', 
        corr_name = f'{_gamma_name}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_src}_random',
        corr_matrix = corr_2pt_matrix[:, _gamma_name_indx].sum(0),
        Cpt = '2pt',
    )

    corr.write(
        corr_save_path = f'/public/home/sush/distillation/proton/result/{corr.Nx}x{corr.Nt}/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}', 
        corr_name = f'{_gamma_name}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_src}_random',
        corr_matrix = corr_2pt_matrix[:, _gamma_name_indx].sum(0),
        Cpt = '2pt_all_t',
    )

    for t_sep_indx, t_sep in enumerate(t_sep_list):
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/proton/result/{corr.Nx}x{corr.Nt}/Px0Py0Pz0/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'{_gamma_name}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr{Nev_curr}_random',
                corr_matrix = corr_3pt_u_matrix[:, Nev_curr_indx, t_sep_indx, :, _gamma_name_indx].sum(1),
                Cpt = '3pt',
                t_sep = t_sep,
                gamma = f'0707',
                link_quark = 'u'
            )
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/proton/result/{corr.Nx}x{corr.Nt}/Px0Py0Pz0/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'{_gamma_name}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr{Nev_curr}_random',
                corr_matrix = corr_3pt_d_matrix[:, Nev_curr_indx, t_sep_indx, :, _gamma_name_indx].sum(1),
                Cpt = '3pt',
                t_sep = t_sep,
                gamma = f'0707',
                link_quark = 'd'
            )