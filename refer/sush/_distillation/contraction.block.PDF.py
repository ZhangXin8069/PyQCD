import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from function_contraction_new_2 import *

from pyquda_utils import io

import time

grid_size = [1, 1, 1, 24]

backend_name = 'numpy'

set_backend(backend_name)
backend = get_backend()


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

t_sep_list = [x for x in range(12, 13)]

Nev_curr_list = [100, 400]
Nev_max = Nev_curr_list[-1]

link_max = 10

gauge_path = f'/public/group/lqcd/configurations/CLOVER/beta6.41_mu-0.2295_ms-0.2050_L32x96/beta6.41_mu-0.2295_ms-0.2050_L32x96_cfg_'

conf_id = sys.argv[1]

# print(gauge_link.shape)
Nev_src = 100

param = creat_param()
param.sink_operators([['u', 'u', 'gamma_sink', 'd', 1]])
param.source_operators([['u^d', 'gamma_source', 'd^d', 'u^d', -1]])
param.curr_operators(['s^d', 'gamma_curr', 's'])

corr = distillation_func(
    conf_id = conf_id,
    param = param.data(),

    lattice_size = lattice_size, # Nx, Ny, Nz, Nt 
    
    Nev_src = Nev_src,
    
    conserved = False,
    backend = backend.__name__
)
st_smear = time.time()
if backend_name == 'numpy':
    from pyquda import initGrid
    initGrid(grid_size = grid_size)
    
    if link_max != 0:
        gauge = io.readQIOGauge(f'{gauge_path}{str(conf_id)}.lime')
        gauge_link = (gauge.lexico())
        gauge_link = stout_smear_ndarray(gauge = gauge_link, nstep = N_stout, rho = 0.12)
        gauge_link = corr.get_mpi_data(data = gauge_link, mdtype = 'Transport', axis = 1)
        
        del gauge
    
    else:
        gauge_link = False    

elif backend_name == 'cupy':
    from pyquda import init
    init(resource_path=".cache", grid_size = grid_size, latt_size = lattice_size)
    
    if link_max != 0:
        gauge = io.readQIOGauge(f'{gauge_path}{str(conf_id)}.lime')
        gauge.smearSTOUT(N_stout, 0.12, 3)
        gauge_link = (gauge.lexico())
        gauge_link = corr.get_mpi_data(data = gauge_link, mdtype = 'Transport', axis = 1)
        
        del gauge
    
    else:
        gauge_link = False
        
ed_smear = time.time()
print(f'smear use time {(ed_smear - st_smear):.3f}')

    
Mom_link = [[0, 0, 0]]
link_dir = ['X', 'Y', 'Z']
for _link_dir in link_dir:
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
            link_dir = _link_dir,
            gauge = gauge_link,
            Mom = [[0, 0, 1]],
            Mom_link = Mom_link,
            ttype = ':03d',
            read_type = 'numpy' # pickle
        )

    param.source(sink[:, 0].conj())
    param.sink(sink[:, 0])
    param.VdV_curr(VdV_link[:, 0])

    param.Nev_src(Nev_src)
    param.Nev_sink(Nev_src)

    param.gamma_source(gamma(7))
    param.gamma_sink(gamma(7))
    param.gamma_curr(gamma_curr = gamma(4))
    
    param.gamma_projection((gamma(0) + gamma(4))/2)

    corr_bubble_matrix = backend.zeros((1, 2 * link_max + 1, len(Nev_curr_list), corr.Nt), dtype = complex)
    corr_2pt_matrix = backend.zeros((2, corr.Nt, corr.Gt), dtype = complex)
        
    for t_src in range(0, corr.Nt, 1):
        param.t_src(t_src)
        
        if _link_dir == 'Z':
            peram_u = \
                corr.peram_V1(
                    peram_name=[[f'_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy']],
                    peram_save_path = [[f'/nexdata/project/lqcd/sush/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/light/{conf_id}/t']],          
                    use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                    Nev_src = Nev_src,
                    read_type = backend.__name__,
                    t_load_type = 'slice',
                    t = f'{t_src:03d}',
                    # ttype = '03d'
                )[0]
            
            param.peram(peram_u, sep = False, quark = 'u', curr = False, curr_id = 0, use_2pt = True)
            param.peram(peram_u, sep = False, quark = 'd', curr = False, curr_id = 0, use_2pt = True)

            corr_2pt_matrix[:, t_src] = corr.corr_all(
                param = param.data(),
                Cpt = '2pt',
                t_range = range(0, 15, 1),
                contraction_same_indx = [[-1], [-1], [-1], [-1], [-1], [-1]],
            )
            
        peram_s = \
            corr.peram_V1(
                peram_name=[[f'_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy']],
                peram_save_path = [[f'/nexdata/project/lqcd/sush/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/strange/{conf_id}/t']],          
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = Nev_max,
                read_type = backend.__name__,
                t_load_type = 'slice',
                t = f'{t_src:03d}',
                # ttype = '03d'
            )[0]
            
        param.peram(peram_s, sep = False, quark = 's', curr = True, curr_id = 0, use_2pt = False)
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            param.Nev_curr(Nev_curr = Nev_curr)
            corr_bubble_matrix[..., Nev_curr_indx, t_src] = corr.corr_all(
                param = param.data(),
                Cpt = 'bubble',
                t_range = range(0, 1, 1),
                contraction_same_indx = [[-1], [-1], [-1], [-1], [-1], [-1]],
            )[0]
            
            
    if _link_dir == 'Z':
        corr.write(
            corr_save_path = f'/public/home/sush/distillation/proton/result/{corr.Nx}x{corr.Nt}/Px0Py0Pz1/ENV_{Nev_src}/conf{conf_id}', 
            corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_src}_block',
            corr_matrix = corr_2pt_matrix.sum(0),
            Cpt = '2pt',
        )
        
        corr.write(
            corr_save_path = f'/public/home/sush/distillation/proton/result/{corr.Nx}x{corr.Nt}/Px0Py0Pz1/ENV_{Nev_src}/conf{conf_id}', 
            corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_src}_block',
            corr_matrix = corr_2pt_matrix.sum(0),
            Cpt = '2pt_all_t',
        )
        
    for link_indx, link_len in enumerate([x for x in range(-10, 11, 1)]):
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
        
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/proton/result/{corr.Nx}x{corr.Nt}/Px0Py0Pz0/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'link_dir_{_link_dir}_link{link_len}_gamma{4:02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_20_0.12_curr{Nev_curr}_block',
                corr_matrix = corr_bubble_matrix[0, link_indx, Nev_curr_indx],
                Cpt = 'bubble',
            )
        