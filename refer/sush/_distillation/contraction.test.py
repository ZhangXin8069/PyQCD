import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from function_contraction import *

from pyquda_utils import io

import time
import numpy 
from opt_einsum import contract

set_backend('numpy')
backend = get_backend()

grid_size = [1, 1, 1, 1]

lattice_size = [24, 24, 24, 72]

exact_eigen_name = 100
N_eigen_name = [50, 100, 150, 200]
N_sum_name = [10, 10, 10, 10]
N_extract_name = [5, 5, 5, 5]
noise_eigen_name = 100

exact_eigen = 100
N_eigen = [50, 100, 150, 200]
N_sum = [10, 10, 10, 10]
N_extract = [5, 5, 5, 5]
noise_eigen = 0

Nbin = 1
N_stout = 20

t_sep_list = [x for x in range(10, 11)]

Nev_src = 100
Nev_curr = exact_eigen + sum(N_sum) + noise_eigen

link_max = 0

gamma_loop = [1, 2]

gauge_path = f'/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_'

conf_id = sys.argv[1]
st_smear = time.time()
if backend.__name__ == 'numpy':
    from pyquda import initGrid
    initGrid(grid_size = grid_size, latt_size = lattice_size)
    gauge = io.readQIOGauge(f'{gauge_path}{str(conf_id)}.lime')
    gauge_link = (gauge.lexico())
    gauge_link = stout_smear_ndarray(_U = gauge_link, nstep = 20, rho = 0.12)
    
elif backend.__name__ == 'cupy':
    from pyquda import init
    init(resource_path=".cache", grid_size = grid_size, latt_size = lattice_size)
    gauge = io.readQIOGauge(f'{gauge_path}{str(conf_id)}.lime')
    gauge.smearSTOUT(20, 0.12, 3)
    gauge_link = (gauge.lexico())
    
del gauge
ed_smear = time.time()

print(f'smear use time {(ed_smear - st_smear):.3f}')

corr = distillation_func(
    conf_id = conf_id,

    quark_type = 'ud',
    link_quark = 'u',

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
        Mom_link = [[1, 0, 0]],
        bin = [Nbin, 0, 0, 0],
        bin_type = 'no_bin',
        ttype = ':03d',
        read_type = 'numpy', # pickle
        add_weight = False
    )

_weights_2 = corr.eigen.create_omega(
    exact = exact_eigen,
    N_eigen = N_eigen,
    N_sum = N_sum,
    N_extract = N_extract,
    noise = noise_eigen,
    conserved = False,
    Nbin = 1,
    dim = 2
)

_weights = corr.eigen.create_omega(
    exact = exact_eigen,
    N_eigen = N_eigen,
    N_sum = N_sum,
    N_extract = N_extract,
    noise = noise_eigen,
    conserved = False,
    Nbin = 1,
    dim = 4
)[corr.rank*(sum(corr.use_eigen)//corr.size):(corr.rank + 1)*(sum(corr.use_eigen)//corr.size)]

corr_3pt_matrix = backend.zeros((1, len(gamma_loop), len(gamma_loop), len(t_sep_list), 2 * link_max + 1, corr.Nt, corr.Gt), dtype = backend.complex128)
corr_2pt_matrix = backend.zeros((1, corr.Nt, corr.Gt), dtype = backend.complex128)
peram = {}
for t_src in range(0, corr.Nt, 1):
    peram['src'] = \
        corr.peram_V1(
            peram_name=[[f'_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy']],
            peram_save_path = [f'/beegfs/group-bak/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t'],
            use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
            Nev_src = Nev_src,
            read_type = backend.__name__,
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
            peram_name=[[f'_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy']],
            peram_save_path = [f'/beegfs/group-bak/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t'],
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = Nev_curr,
                read_type = backend.__name__,
                t_load_type = 'slice',
                # t = f'{((t_sep + t_src)%corr.Nt)}'
                t = f'{((t_sep + t_src)%corr.Nt):03d}'
            )
            
        for g1_indx, g1 in enumerate(gamma_loop):
            for g2_indx, g2 in enumerate(gamma_loop):
                corr_3pt_matrix[0, g1_indx, g2_indx, t_sep_indx, :, t_src] = (
                    corr.corr_all(
                        t_src = t_src,
                        sink = VdV_link[0, :, 0].conj().transpose(0, 2, 1) * _weights_2,
                        # sink = sink[0],  
                        source = sink[0].conj(),
                        peram = peram,
                        Nev_sink_src = [Nev_curr, Nev_src],
                        Cpt = '3pt',
                        t_sep = t_sep,
                        NEV_curr = Nev_curr, 
                        VdV_link = VdV_link[0] * _weights_2,
                        link_quark = 'u',
                        curr_gamma = gamma(g1),
                        sink_src_gamma = [gamma(g2), gamma(5)],
                    )
                )
                
                st_time = time.time()
                
                VdV_link_tsep = numpy.ascontiguousarray(VdV_link[0, (t_sep + t_src)%corr.Gt, :])
                corr.comm.Bcast(VdV_link_tsep, root = (t_sep + t_src)%corr.Nt // corr.Gt)
                
                VdV_link_tsep_conj = VdV_link_tsep[0].conj().transpose(1, 0)
                
                curr = 0                
                for i in range(_weights.shape[0]):
                    ev_st = (sum(corr.use_eigen)//corr.size)*corr.rank

                    VdV_link_compensation = contract(
                        'Lab,cd,abcd->Labcd',
                        backend.asarray(VdV_link_tsep[:, (ev_st + i):(ev_st + i + 1), :]),
                        backend.asarray(VdV_link_tsep_conj),
                        backend.asarray(_weights[i:(i+1)])
                    )

                    curr += corr.curr(
                            Forward = peram['src'][0, (t_sep + t_src)%corr.Nt],
                            peram_sep = peram['sep'][0, (t_sep + t_src)%corr.Nt, ..., (ev_st + i):(ev_st + i + 1), :],
                            VdV_link = VdV_link_compensation,
                            curr_gamma = backend.asarray(gamma(g1)),
                            Ctype = 'two_curr'
                    )
                
                if corr.rank == (t_sep + t_src)%corr.Nt // corr.Gt:
                    curr_compensation_gather = numpy.ascontiguousarray(numpy.empty(([corr.size] + list(curr.shape)), dtype = complex))
                
                else:
                    curr_compensation_gather = None
                
                
                if backend.__name__ == 'cupy':
                    curr = curr.get()
                
                curr = numpy.ascontiguousarray(curr)
                
                corr.comm.Gather(curr, curr_compensation_gather, root = (t_sep + t_src)%corr.Nt // corr.Gt)
                
                del curr

                if corr.rank == (t_sep + t_src)%corr.Nt // corr.Gt:
                    curr_compensation_gather = curr_compensation_gather.sum(0)

                    corr_3pt_matrix_compensation = contract(
                        "Lcflj,ca,ab,belj,ed,df->L",
                        backend.asarray(curr_compensation_gather),
                        backend.asarray(gamma(g2)),
                        backend.asarray(gamma(5)),
                        peram['src'][0, (t_sep + t_src)%corr.Nt, :, :, 0:Nev_curr, 0:Nev_src].conj(),
                        backend.asarray(gamma(5)),
                        backend.asarray(gamma(5)),
                    )
                    
                    corr_3pt_matrix[0, g1_indx, g2_indx, t_sep_indx, :, t_src, (t_sep + t_src)%corr.Gt] = backend.asarray(corr_3pt_matrix_compensation)
                    print(f'compensation used time {(time.time() - st_time):.3f} s')

                corr.comm.Barrier()

for g1_indx, g1 in enumerate(gamma_loop):
    for g2_indx, g2 in enumerate(gamma_loop):
        for t_sep_indx in range(len(t_sep_list)):
            corr.write(
                t_sep = t_sep_list[t_sep_indx],
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result/{corr.Nx}x{corr.Nt}/Px0Py0Pz1/ENV_{Nev_curr}/conf{conf_id}', 
                
                corr_2pt_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_block',
                corr_2pt_matrix = corr_2pt_matrix,

                corr_3pt_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_block', # linkZ, conserved
                corr_3pt_matrix = corr_3pt_matrix[:, g1_indx, g2_indx, t_sep_indx], 
                
                gamma = f'0{g2}0{g1}05',
                conserved = corr.conserved,
                link_quark = ['u'],
                # source_to_curr = True
                )


print('complete')
