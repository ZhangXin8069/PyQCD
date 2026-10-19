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

Nev_curr_list = [x for x in range(100, 5184, 500)] + [5184]
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
param.sink_operators([['u', 'u', 'gamma_sink', 'd', 1]])
param.source_operators([['u^d', 'gamma_source', 'd^d', 'u^d', -1]])
param.curr_operators(['u^d', 'gamma_curr', 'u'])

Mom_link = [[0, 0, 0]]

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


param.source(sink[:, 0].conj())
param.sink(sink[:, 0])

param.Nev_src(Nev_src)
param.Nev_sink(Nev_src)

param.gamma_source(gamma(7))
param.gamma_sink(gamma(7))

corr_2pt_matrix = backend.zeros((2, corr.Nt, corr.Gt), dtype = complex)
# corr_2pt_matrix_gamma = backend.zeros((3, 2, corr.Nt, corr.Gt), dtype = complex)
# corr_bubble_matrix = backend.zeros((3, corr.Nt), dtype = complex)
corr_3pt_matrix = backend.zeros((1, 3, len(Nev_curr_list), len(t_sep_list), 4, corr.Nt, corr.Gt), dtype = complex)
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

    param.gamma_projection((gamma(0) + gamma(4))/2)
    
    corr_2pt_matrix[:, t_src] = corr.corr_all(
        param = param.data(),
        Cpt = '2pt',
        t_range = range(0, 15, 1),
        contraction_same_indx = [[-1], [-1], [-1], [-1], [-1], [-1]],
    )
    
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
            
        param.peram(peram_u_curr, sep = True, quark = 'u', curr = True, curr_id = 0, use_2pt = False)
        
        for _gamma in range(1):
            param.gamma_projection((gamma(0) + gamma(4))@gamma(5)@gamma(_gamma + 1)/2)
            param.gamma_curr(gamma(5)@gamma(_gamma + 1))
            
            for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
                weight_matrix_2 = corr.eigen.create_omega(
                    exact = 100,
                    N_eigen = [0],
                    N_sum = [0],
                    N_extract = [0],
                    noise = Nev_curr - 100,
                )
                param.Nev_curr(Nev_curr, curr_id = 0)
                param.VdV_curr(VdV_link[:, 0, 0, :Nev_curr, :Nev_curr] * weight_matrix_2)

                corr_3pt_matrix[0, _gamma, Nev_curr_indx, t_sep_indx, :, t_src] = corr.corr_all(
                    param = param.data(),
                    Cpt = '3pt',
                    t_range = range(0, 9, 1),
                    contraction_same_indx = [[-1], [-1], [-1], [-1], [-1], [-1]],
                )
                
        # corr_2pt_matrix_gamma[:, _gamma, t_src] = corr.corr_all(
        #     param = param.data(),
        #     Cpt = '2pt',
        #     t_range = range(0, 15, 1),
        #     contraction_same_indx = [[-1], [-1], [-1], [-1], [-1], [-1]],
        # )
        # param.gamma_curr(gamma(5)@gamma(_gamma + 1))
        # corr_bubble_matrix[_gamma, t_src] = corr.corr_all(
        #     param = param.data(),
        #     Cpt = 'bubble',
        #     t_range = range(0, 1, 1),
        #     contraction_same_indx = [[-1], [-1], [-1], [-1], [-1], [-1]],
        # )[0]
        
corr.write(
    corr_save_path = f'/public/home/sush/distillation/proton/result/{corr.Nx}x{corr.Nt}/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}', 
    corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_src}_block',
    corr_matrix = corr_2pt_matrix.sum(0),
    Cpt = '2pt',
)

for t_sep_indx, t_sep in enumerate(t_sep_list):
    for _gamma in range(1):
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/proton/result/{corr.Nx}x{corr.Nt}/Px0Py0Pz0/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr400_block',
                corr_matrix = corr_3pt_matrix[:, _gamma, Nev_curr_indx, t_sep_indx, :].sum(1),
                Cpt = '3pt',
                t_sep = t_sep,
                gamma = f'050{_gamma+1}',
                link_quark = 'u'
            )
            
    # corr.write(
    #     corr_save_path = f'/public/home/sush/distillation/proton/result/{corr.Nx}x{corr.Nt}/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}', 
    #     corr_name = f'gamma{(_gamma+1):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr400_block',
    #     corr_matrix = corr_bubble_matrix[_gamma],
    #     Cpt = 'bubble',
    # )

    # corr.write(
    #     corr_save_path = f'/public/home/sush/distillation/proton/result/{corr.Nx}x{corr.Nt}/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}', 
    #     corr_name = f'gamma{(_gamma+1):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_src}_block',
    #     corr_matrix = corr_2pt_matrix_gamma.sum(0)[_gamma],
    #     Cpt = '2pt_all_t',
    # )