import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction import *

from opt_einsum import contract, contract_path

import time

set_backend('numpy')
backend = get_backend()

lattice_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 24]

param = creat_param(
    sink_operators = [['d^d', 'gamma_sink', 'u', 1]],
    source_operators = [['u^d', 'gamma_source', 'd', -1]]
)

param_data = param.data()
# param_data['type1_uu'] = []


conf_id = sys.argv[1]

corr = distillation_func(
    conf_id = conf_id,
    param = param_data,
    
    lattice_size = lattice_size, # Nx, Ny, Nz, Nt 
    grid_size = grid_size, 
    
    backend = backend.__name__
)

rank = getMPIRank()
comm = getMPIComm()
size = getMPISize()

exact_eigen = 50
N_eigen = [100, 200, 300, 500, 800]
N_sum = [20, 20, 20, 20, 20]
N_extract = [10, 10, 10, 10, 10]
noise_eigen = 500

exact_eigen_name = 50
N_eigen_name = [100, 200, 300, 500, 800]
N_sum_name = [20, 20, 20, 20, 20]
N_extract_name = [10, 10, 10, 10, 10]
noise_eigen_name = 500

L = 24**3*3 - exact_eigen - sum(N_eigen)

_weight = [L/noise_eigen, (L-1)/(noise_eigen-1), (L-2)/(noise_eigen-2), (L-3)/(noise_eigen-3)]

Nbin = 1
N_stout = 20

t_sep_list = [12]

Nev_src = 50

Nev_curr_list = [650]
Nev_max = max(Nev_curr_list + [Nev_src])

link_max = 0
Mom_sink = [[0, 0, 0]]
Mom_link = [[0, 0, 0]]

# mu_nu_indx, mu_nu_gamma, mu_indx, mu_gamma =  PFF_Mom_to_gamma_new(Mom_link[:], allow_t = False)

mu_indx = [[4, 5]]
mu_gamma = tran_indx_to_gamma(mu_indx)

mu_nu_indx = [[[4, 5]]]
mu_nu_gamma = tran_indx_to_gamma(mu_nu_indx)

mu_nu_gamma = backend.asarray(mu_nu_gamma)
mu_gamma = backend.asarray(mu_gamma)

gauge_path = f'/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_'

conf_id = sys.argv[1]

weight_2 = corr.eigen.create_omega_accelerate(
    exact = exact_eigen,
    N_eigen = N_eigen,
    N_sum = N_sum,
    N_extract = N_extract,
    noise = noise_eigen,
    dim = 2
)

eigen = corr_eigvecs(Nx = corr.Nx, backend = backend.__name__)

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
        ttype = ':03d',
        read_type = 'numpy', # pickle
        add_weight = True
    )

param_data['curr_operators'] = ['u^d', 'gamma_curr_mu', 'u']

N_operator = len(param_data['sink_operators']) * len(param_data['source_operators'])

corr_3pt_matrix = backend.zeros((len(Nev_curr_list), len(t_sep_list), N_operator, len(Mom_link), 1, corr.Nt, corr.Gt), dtype = complex)

for t_src in range(0, corr.Nt, 1):
    source = backend.asarray(get_mpi_data(sink[t_src//size, 0, :Nev_src, :Nev_src].conj().transpose(1, 0), mdtype = 'Bcast', root = t_src%size, axis = 0))
    
    peram_u = \
        corr.peram_V1(
            peram_name=f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy',
            peram_save_path = f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t',            
            use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
            Nev_src = Nev_max,
            read_type = 'numpy',
            t_source = f'{t_src:03d}',
            # ttype = '03d'
        )
    
    peram_d = seq_peram(peram_u[..., :Nev_src])

    t_curr_0_list, _, t_curr_0_list_indx = corr.get_mpi_tlist(t = [x%corr.Nt for x in range(t_src, t_src + 36, 1)], gtype = 'Scatter')

    for t_sep_indx, t_sep in enumerate(t_sep_list):
        peram_u_sep = \
            corr.peram_V1(
                peram_name=f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy',
                peram_save_path = f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t',
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = Nev_max,
                read_type = 'numpy',
                t_source = f'{((t_sep + t_src)%corr.Nt):03d}',
                seq = True
            )

        peram_d_2pt = backend.asarray(get_mpi_data(peram_d[((t_sep + t_src)%corr.Nt)//size], mdtype = 'Bcast', root = ((t_sep + t_src)%corr.Nt)%size))
        sink_3pt = backend.asarray(get_mpi_data(VdV_link[((t_sep + t_src)%corr.Nt)//size, :, 0].conj().transpose(0, 2, 1), mdtype = 'Bcast', root = ((t_sep + t_src)%corr.Nt)%size))
        
        time_3pt = time.perf_counter()
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            for t_sink_list_indx, t_sink_indx in enumerate(t_curr_0_list_indx):
                t_sink = t_curr_0_list[t_sink_list_indx]
                Nev_indx = get_Nev(lst = [x for x in range(Nev_curr)], group_sizes = [Nev_curr], M = 50, allow_partial = True)
                
                for Nev_list in Nev_indx:
                    corr_3pt_matrix[Nev_curr_indx, t_sep_indx, 0, ..., t_src, t_sink_indx] += contract(
                        'manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG', 
                        backend.asarray(peram_u[t_sink_indx, :, :, :Nev_curr, :Nev_src]), backend.asarray(peram_u_sep[t_sink_indx, :, :, :Nev_src, :Nev_curr][..., Nev_list]), backend.asarray(peram_d_2pt[..., :Nev_src, :Nev_src]), 
                        backend.asarray(source), backend.asarray(VdV_link[t_sink_indx, :, 0, :Nev_curr, :Nev_curr][:, Nev_list, :]), backend.asarray(sink_3pt[:, :Nev_src, :Nev_src]),
                        backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1])
                        )

        comm.Barrier()
        if rank == 0:
            print(f"complete cal 3pt of tsrc {t_src} tsep {t_sep} use time {(time.perf_counter() - time_3pt):.3f} s")

for Mom_indx, Mom_list in enumerate(Mom_link[:]):
    for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):            
        for t_sep_indx, t_sep in enumerate(t_sep_list):
            for gamma_indx, gamma_list in enumerate(mu_nu_indx[Mom_indx]):
                corr.write(
                    t_sep = t_sep,
                    corr_save_path = f'/public/home/sush/distillation/pion/result_two_curr/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                    corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}', # linkZ, conserved
                    corr_matrix = corr_3pt_matrix[Nev_curr_indx, t_sep_indx, 0, Mom_indx, gamma_indx, :].reshape(1, corr.Nt, -1), 
                    Cpt = '3pt',
                    gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                    conserved = False,
                    link_quark = 'u',
                )
                
                corr.write(
                    t_sep = t_sep,
                    corr_save_path = f'/public/home/sush/distillation/pion/result_two_curr/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                    corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}', # linkZ, conserved
                    corr_matrix = corr_3pt_matrix[Nev_curr_indx, t_sep_indx, 0, Mom_indx, gamma_indx, :].reshape(1, corr.Nt, -1), 
                    Cpt = '3pt_all_t',
                    gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                    conserved = False,
                    link_quark = 'u',
                )