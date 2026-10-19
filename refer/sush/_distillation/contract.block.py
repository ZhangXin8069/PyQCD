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

lattice_size = [24, 24, 24, 72]

exact_eigen = 100
N_eigen = [50, 100, 150, 200]
N_sum = [10, 10, 10, 10]
N_extract = [5, 5, 5, 5]
noise_eigen = 100

exact_eigen_name = 100
N_eigen_name = [50, 100, 150, 200]
N_sum_name = [10, 10, 10, 10]
N_extract_name = [5, 5, 5, 5]
noise_eigen_name = 100

Nbin = 1
N_stout = 20

t_sep_list = [x for x in range(15, 16)]

Nev_src = 100
Nev_curr_list = [exact_eigen + sum(N_sum) + noise_eigen]
Nev_max = Nev_curr_list[-1]
link_max = 0

gauge_path = f'/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_'

conf_id = sys.argv[1]
st_smear = time.time()
if backend_name == 'numpy':
    from pyquda import initGrid
    initGrid(grid_size = grid_size)
    gauge = io.readQIOGauge(f'{gauge_path}{str(conf_id)}.lime')
    gauge_link = (gauge.lexico())
    gauge_link = stout_smear_ndarray(_U = gauge_link, nstep = 20, rho = 0.12)
    
elif backend_name == 'cupy':
    from pyquda import init
    init(resource_path=".cache", grid_size = grid_size, latt_size = lattice_size)
    gauge = io.readQIOGauge(f'{gauge_path}{str(conf_id)}.lime')
    gauge.smearSTOUT(20, 0.12, 3)
    gauge_link = (gauge.lexico())
    
del gauge
ed_smear = time.time()
print(f'smear use time {(ed_smear - st_smear):.3f}')

# Mom_link = [[1, 1, 0], [1, 0, 1], [0, 1, 1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1], [-1, -1, 0], [-1, 0, -1], [0, -1, -1], [1, -1, 0], [1, 0, -1], [0, 1, -1]]

Mom_link = [[1, 0, 0]]
#, [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]
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
                f'/beegfs/group-bak/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/{conf_id}_t'
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
        add_weight = False
    )

source = sink[0].conj().transpose(0, 2, 1)

t_curr_r_list = [x for x in range(8, 15, 1)]

gamma_list_matrix = PFF_Mom_to_gamma(Mom = Mom_link)

corr_3pt_matrix = backend.zeros((1, len(Nev_curr_list), len(Mom_link), len(t_curr_r_list), len(t_sep_list), 2 * link_max + 1, corr.Nt, corr.Gt), dtype = backend.complex128)
corr_2pt_matrix = backend.zeros((1, corr.Nt, corr.Gt), dtype = backend.complex128)
peram = {}
VdV_link_weight = VdV_link.copy()

VdV_link_weight *= corr.eigen.create_omega(
    exact = exact_eigen,
    N_eigen = N_eigen,
    N_sum = N_sum,
    N_extract = N_extract,
    noise = noise_eigen,
    conserved = False,
    Nbin = 1,
    dim = 2
)

for t_src in range(0, corr.Nt, 1):

    peram['src'] = \
        corr.peram_V1(
            peram_name=[[f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen}_V1_stout_smear_20_0.12.npy']],
            peram_save_path = [f'/beegfs/group-bak/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t'],
            use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
            Nev_src = Nev_src,
            read_type = backend_name,
            t_load_type = 'slice',
            t = f'{t_src:03d}',
            # ttype = '03d'
        )
        
    corr_2pt_matrix[0, t_src, :] = \
        corr.corr_all(
            t_src = t_src,
            sink = sink[0], 
            source = sink[0].conj(),
            peram = peram,
            Nev_sink_src = [Nev_src, Nev_src],
            Cpt = '2pt',
            sink_src_gamma = [gamma(5), gamma(5)],
        )
    
    for t_sep_indx, t_sep in enumerate(t_sep_list):
        
        peram['sep'] = \
            corr.peram_V1(
            peram_name=[[f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen}_V1_stout_smear_20_0.12.npy']],
            peram_save_path = [f'/beegfs/group-bak/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t'],
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = Nev_max,
                read_type = backend_name,
                t_load_type = 'slice',
                # t = f'{((t_sep + t_src)%corr.Nt)}'
                t = f'{((t_sep + t_src)%corr.Nt):03d}'
            )
        for t_curr_r_indx, t_curr_r in enumerate(t_curr_r_list):
            peram['cr_cl'] = \
            corr.peram_V1(
            peram_name=[[f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen}_V1_stout_smear_20_0.12.npy']],
            peram_save_path = [f'/beegfs/group-bak/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t'],
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = Nev_max,
                read_type = backend_name,
                t_load_type = 'slice',
                # t = f'{((t_sep + t_src)%corr.Nt)}'
                t = f'{((t_curr_r + t_src)%corr.Nt):03d}'
            )

            for Mom_indx, gamma_list in enumerate(gamma_list_matrix):
                for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
                    corr_3pt_matrix[0, Nev_curr_indx, Mom_indx, t_curr_r_indx, t_sep_indx, :, t_src] = (
                        corr.corr_all(
                            t_src = t_src,
                            sink = sink[0],
                            source = source,
                            peram = peram,
                            Nev_sink_src = [Nev_src, Nev_src],
                            Cpt = '4pt',
                            t_sep = t_sep,
                            t_curr_r = t_curr_r,
                            NEV_curr = Nev_curr, 
                            VdV_link = VdV_link_weight[Mom_indx],
                            link_quark = 'u',
                            curr_gamma = [gamma(4), gamma(4)],
                            sink_src_gamma = [gamma(5), gamma(5)],
                        )
                    )

for Mom_indx, gamma_list in enumerate(gamma_list_matrix):

    for t_sep_indx in range(len(t_sep_list)):

        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            for t_curr_r_indx, t_curr_r in enumerate(t_curr_r_list):
                corr.write(
                    t_sep = t_sep_list[t_sep_indx],
                    corr_save_path = f'/public/home/sush/distillation/4pt/result/{corr.Nx}x{corr.Nt}/Px{Mom_link[Mom_indx][2]}Py{Mom_link[Mom_indx][1]}Pz{Mom_link[Mom_indx][0]}/ENV_{Nev_curr}/conf{conf_id}', 
                    
                    corr_2pt_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_block',
                    corr_2pt_matrix = corr_2pt_matrix,

                    corr_3pt_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_block_t_curr_r{t_curr_r}', # linkZ, conserved
                    corr_3pt_matrix = corr_3pt_matrix[:, Nev_curr_indx, Mom_indx, t_curr_r_indx, t_sep_indx], 
                    
                    gamma = f'0{5}0{4}0405',
                    conserved = corr.conserved,
                    link_quark = ['u'],
                    # source_to_curr = True
                    )

print('complete')
