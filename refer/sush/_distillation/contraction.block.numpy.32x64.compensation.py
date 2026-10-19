import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction import *

from opt_einsum import contract, contract_path

import time

set_backend('numpy')
backend = get_backend()

lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 16]

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

exact_eigen = 100
N_eigen = [100, 200, 400, 700, 1100]
N_sum = [20, 20, 20, 20, 20]
N_extract = [10, 10, 10, 10, 10]
noise_eigen = 200

exact_eigen_name = 100
N_eigen_name = [100, 200, 400, 700, 1100]
N_sum_name = [20, 20, 20, 20, 20]
N_extract_name = [10, 10, 10, 10, 10]
noise_eigen_name = 200

L = corr.Nx**3 * Nc - exact_eigen - sum(N_eigen)

_weight = [L/noise_eigen, (L-1)/(noise_eigen-1), (L-2)/(noise_eigen-2), (L-3)/(noise_eigen-3)]

t_sep_list = [20]

Nev_src = 100

Nev_curr_list = [400]
Nev_max = max(Nev_curr_list + [Nev_src])

link_max = 0
Mom_sink = [[0, 0, 0]]
Mom_link = creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)
mu_nu_indx, mu_nu_gamma, mu_indx, mu_gamma =  PFF_Mom_to_gamma_new(Mom_link[:], allow_t = False)

mu_nu_gamma = backend.asarray(mu_nu_gamma)
mu_gamma = backend.asarray(mu_gamma)

gauge_path = f'/public/group/lqcd/configurations/CLOVER/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64_cfg_'

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
                f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_hpy.npy'
        ],
        eigen_save_path = [
                f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t'
                # f'/beegfs/group-bak/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t'
                
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

# corr_2pt_matrix = backend.zeros((N_operator, corr.Nt, corr.Gt), dtype = complex)

# corr_2pt_mu_matrix = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2, corr.Nt, corr.Gt), dtype = complex)

# corr_3pt_matrix = backend.zeros((len(Nev_curr_list), len(t_sep_list), N_operator, len(Mom_link), 2, corr.Nt, corr.Gt), dtype = complex)
corr_3pt_matrix_dis_compensation = backend.zeros((len(Nev_curr_list), len(t_sep_list), 2, len(Mom_link), 2, corr.Nt, corr.Gt), dtype = complex)
# corr_3pt_matrix_con_compensation = backend.zeros((len(Nev_curr_list), len(t_sep_list), 1, len(Mom_link), 2, corr.Nt, corr.Gt), dtype = complex)

# corr_3pt_matrix_light_compensation = backend.zeros((len(Nev_curr_list), len(t_sep_list), 2, len(Mom_link), 2, corr.Nt), dtype = complex)
# corr_3pt_matrix_strange_compensation = backend.zeros((len(Nev_curr_list), len(t_sep_list), 1, len(Mom_link), 2, corr.Nt), dtype = complex)

# bubble_mu_light = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2, corr.Nt), dtype = complex)
# bubble_mu_strange = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2, corr.Nt), dtype = complex)



for t_src in range(0, corr.Nt, 1):
    source = backend.asarray(get_mpi_data(sink[t_src//size, 0].conj().transpose(1, 0), mdtype = 'Bcast', root = t_src%size, axis = 0))
    
    peram_u = \
        corr.peram_V1(
            peram_name=f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_hpy.npy',
            peram_save_path = f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t',            
            use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
            Nev_src = Nev_max,
            read_type = 'numpy',
            t_source = f'{t_src:03d}',
            # ttype = '03d'
        )
    
    peram_d = seq_peram(peram_u[..., :Nev_src])
    
    # peram_s = \
    #     corr.peram_V1(
    #         peram_name=f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_hpy.npy',
    #         peram_save_path = f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/strange/{conf_id}/t',            
    #         use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
    #         Nev_src = Nev_max,
    #         read_type = 'numpy',
    #         t_source = f'{t_src:03d}',
    #         # ttype = '03d'
    #     )

    # t_curr_0_list, _, t_curr_0_list_indx = corr.get_mpi_tlist(t = [x%corr.Nt for x in range(t_src, t_src + max(t_sep_list) * 2 + 10, 1)], gtype = 'Scatter')

    # time_2pt = time.perf_counter()
    # for t_sink_list_indx, t_sink_indx in enumerate(t_curr_0_list_indx):
    #     t_sink = t_curr_0_list[t_sink_list_indx]
    #     corr_2pt_matrix[0, t_src, t_sink_indx] = contract(
    #         'manb,cedf,bd,fn,ac,em->', 
    #         backend.asarray(peram_u[t_sink_indx, ..., :Nev_src, :Nev_src]), backend.asarray(peram_d[t_sink_indx, ..., :Nev_src, :Nev_src]), 
    #         backend.asarray(source), backend.asarray(sink[t_sink_indx, 0]), 
    #         backend.asarray(gamma(5)), backend.asarray(gamma(5))
    #         )
        
    # comm.Barrier()
    # if rank == 0:
    #     print(f"complete cal 2pt of tsrc {t_src} use time {(time.perf_counter() - time_2pt):.3f} s")
    
    # time_2pt_mu = time.perf_counter()
    # for t_sink_list_indx, t_sink_indx in enumerate(t_curr_0_list_indx):
    #     for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
    #         corr_2pt_mu_matrix[Nev_curr_indx, 0, :, :, t_src, t_sink_indx] = contract(
    #             'manb,cedf,bd,Mfn,ac,MFem->MF', 
    #             backend.asarray(peram_u[t_sink_indx, ..., :Nev_curr, :Nev_src]), backend.asarray(peram_d[t_sink_indx, ..., :Nev_src, :Nev_curr]), 
    #             backend.asarray(source), backend.asarray(VdV_link[t_sink_indx, :, 0, :Nev_curr, :Nev_curr]), 
    #             backend.asarray(gamma(5)), backend.asarray(mu_gamma)
    #             )
            
    # for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
    #     bubble_mu_light[Nev_curr_indx, 0, :, :, t_src] = contract('menf,Mfn,MKem->MK', backend.asarray(peram_u[t_src//size, ..., :Nev_curr, :Nev_curr]), backend.asarray(VdV_link[t_src//size, :, 0, :Nev_curr, :Nev_curr]), backend.asarray(mu_gamma))
    #     bubble_mu_strange[Nev_curr_indx, 0, :, :, t_src] = contract('menf,Mfn,MKem->MK', backend.asarray(peram_s[t_src//size, ..., :Nev_curr, :Nev_curr]), backend.asarray(VdV_link[t_src//size, :, 0, :Nev_curr, :Nev_curr]), backend.asarray(mu_gamma))
    
    # comm.Barrier()
    # if rank == 0:
    #     print(f"complete cal 2pt_mu and bubble of tsrc {t_src} use time {(time.perf_counter() - time_2pt_mu):.3f} s")

    for t_sep_indx, t_sep in enumerate(t_sep_list):
        peram_u_sep = \
            corr.peram_V1(
                peram_name=f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_hpy.npy',
                peram_save_path = f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t',
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = Nev_max,
                read_type = 'numpy',
                t_source = f'{((t_sep + t_src - 1)%corr.Nt):03d}',
                seq = True
            )

        peram_s_sep = \
            corr.peram_V1(
                peram_name=f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_hpy.npy',
                peram_save_path = f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/strange/{conf_id}/t',            
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = Nev_max,
                read_type = 'numpy',
                t_source = f'{((t_sep + t_src - 1)%corr.Nt):03d}',
                scatter = False,
                seq = False,
            )
            
        peram_d_2pt = backend.asarray(get_mpi_data(peram_d[((t_sep + t_src)%corr.Nt)//size], mdtype = 'Bcast', root = ((t_sep + t_src)%corr.Nt)%size))
        sink_3pt = backend.asarray(get_mpi_data(VdV_link[((t_sep + t_src)%corr.Nt)//size, :, 0].conj().transpose(0, 2, 1), mdtype = 'Bcast', root = ((t_sep + t_src)%corr.Nt)%size))
        
        # time_3pt = time.perf_counter()
        # for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
        #     Nev_indx = get_Nev(lst = [x for x in range(Nev_curr)], group_sizes = [Nev_curr], M = 50, allow_partial = True)
        #     for t_sink_list_indx, t_sink_indx in enumerate(t_curr_0_list_indx):
        #         t_sink = t_curr_0_list[t_sink_list_indx]
        #         for Nev_list in Nev_indx:
        #             corr_3pt_matrix[Nev_curr_indx, t_sep_indx, 0, ..., t_src, t_sink_indx] += contract(
        #                 'manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG', 
        #                 backend.asarray(peram_u[t_sink_indx, :, :, :Nev_curr, :Nev_src]), backend.asarray(peram_u_sep[t_sink_indx, :, :, :Nev_curr, :Nev_curr][..., Nev_list]), backend.asarray(peram_d_2pt[..., :Nev_src, :Nev_curr]), 
        #                 backend.asarray(source), backend.asarray(VdV_link[t_sink_indx, :, 0, :Nev_curr, :Nev_curr][:, Nev_list, :]), backend.asarray(sink_3pt[:, :Nev_curr, :Nev_curr]),
        #                 backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1])
        #                 )

        # comm.Barrier()
        # if rank == 0:
        #     print(f"complete cal 3pt of tsrc {t_src} tsep {t_sep} use time {(time.perf_counter() - time_3pt):.3f} s")
    

            
        if rank == 0:
            peram_s_sep_diff_tsep = peram_s_sep[((t_sep + t_src - 1)%corr.Nt)]
            
        else:
            peram_s_sep_diff_tsep = None
            
        time_3pt_compensation_mpi_data = time.perf_counter()
        time_3pt = time.perf_counter()
        peram_s_sep_diff_tsep = backend.asarray(get_mpi_data(peram_s_sep_diff_tsep, mdtype = 'Bcast', root = 0))
        sink_3pt_diff_tsep = backend.asarray(sink_3pt / weight_2)
        VdV_curr_tsep_1 = backend.asarray(get_mpi_data(VdV_link[((t_sep + t_src - 1)%corr.Nt)//size, :, 0], mdtype = 'Bcast', root = ((t_sep + t_src - 1)%corr.Nt)%size) / weight_2)
        peram_u_sep_diff_tsep = backend.asarray(get_mpi_data(peram_u_sep[((t_sep + t_src - 1)%corr.Nt)//size], mdtype = 'Bcast', root = ((t_sep + t_src - 1)%corr.Nt)%size))
        peram_u_compensation_diff_tsep = backend.asarray(get_mpi_data(peram_u[((t_sep + t_src)%corr.Nt)//size], mdtype = 'Bcast', root = ((t_sep + t_src)%corr.Nt)%size))
        
        comm.Barrier()
        if rank == 0:
            print(f"complete get mpi data tsrc {t_src} tsep {t_sep} use time {(time.perf_counter() - time_3pt_compensation_mpi_data):.3f} s")

        N_part1 = 200
        N_part2 = 200
        
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            if Nev_curr>exact_eigen:
                st_part1 = time.perf_counter()
                # part1 A, A, A, A,
                Nev_compensation_indx = get_Nev(lst = [x for x in range(N_part1)], group_sizes = [100, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], M = 1)[rank::size]
                
                A1 = 0
                B1 = 0
                C1 = 0
                
                for Nev_weight in Nev_compensation_indx:
                    weight_A4 = eigen.create_omega_accelerate(
                        exact = exact_eigen,
                        N_eigen = N_eigen,
                        N_sum = N_sum,
                        N_extract = N_extract,
                        noise = 0,
                        fixed_first_pos = Nev_weight,
                        dim = 4
                    )
                    
                    # A1 += contract(
                    #     'fnph,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    #     backend.asarray(weight_A4), 
                    #     backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, :N_part1, :][..., Nev_weight]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    #     backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :, :N_part1][:, Nev_weight, :]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, :N_part1]),
                    #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    #     optimize = 'greedy'
                    #     )
                    
                    B1 += contract(
                        'fnph,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                        backend.asarray(weight_A4), 
                        backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[..., :N_part1, :][..., Nev_weight]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                        backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :, :N_part1][:, Nev_weight, :]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, :N_part1]),
                        backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                        optimize = 'greedy'
                        )
                    
                    C1 += contract(
                        'fnph,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                        backend.asarray(weight_A4), 
                        backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep_diff_tsep[..., :N_part1, :][..., Nev_weight]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                        backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :, :N_part1][:, Nev_weight, :]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, :N_part1]),
                        backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                        optimize = 'greedy'
                        )
                    
                # comm.Barrier()
                if rank == 0:
                    print(f'part1 use time {time.perf_counter() - st_part1}')
                
                # part2 A, A, A, B
                st_part2 = time.perf_counter()
                weight_A3 = eigen.create_omega_accelerate(
                    exact = exact_eigen,
                    N_eigen = N_eigen,
                    N_sum = N_sum,
                    N_extract = N_extract,
                    noise = 0,
                    dim = 3
                )[rank::size]
                
                weight_B1 = eigen.create_omega_accelerate(
                    exact = exact_eigen,
                    N_eigen = N_eigen,
                    N_sum = N_sum,
                    N_extract = N_extract,
                    noise = N_part2,
                    dim = 1
                )[N_part1:]
                
                # A21 = contract(
                #     'fnp,h,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A3), backend.asarray(weight_B1), 
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:, :N_part1][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, N_part1:]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A22 = contract(
                #     'nph,f,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A3), backend.asarray(weight_B1), 
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src][..., rank::size, :]), backend.asarray(peram_u_sep_diff_tsep[:, :, :N_part1, N_part1:]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, :N_part1][..., rank::size]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A23 = contract(
                #     'phf,n,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A3), backend.asarray(weight_B1), 
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., :N_part1][..., rank::size]), 
                #     backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, N_part1:]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, :N_part1][:, rank::size, :]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A24 = contract(
                #     'hfn,p,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A3), backend.asarray(weight_B1), 
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, :N_part1, :N_part1][..., rank::size, :]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, :N_part1]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, :N_part1][..., rank::size]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                B21 = contract(
                    'fnp,h,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, :N_part1, :N_part1][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B22 = contract(
                    'nph,f,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, :N_part1, N_part1:][:, :, rank::size, :]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, :N_part1][:, :, rank::size]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B23 = contract(
                    'phf,n,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:, :N_part1]), backend.asarray(peram_d_2pt[..., :N_part1][..., rank::size]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, N_part1:]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, :N_part1][:, rank::size, :]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B24 = contract(
                    'hfn,p,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src][..., rank::size, :]), backend.asarray(peram_u_sep_diff_tsep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, :N_part1]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, :N_part1][:, :, rank::size]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C21 = contract(
                    'fnp,h,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep_diff_tsep[:, :, :N_part1, :N_part1][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C22 = contract(
                    'nph,f,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep_diff_tsep[:, :, :N_part1, N_part1:][:, :, rank::size, :]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, :N_part1][:, :, rank::size]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C23 = contract(
                    'phf,n,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:, :N_part1]), backend.asarray(peram_d_2pt[..., :N_part1][..., rank::size]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, N_part1:]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, :N_part1][:, rank::size, :]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C24 = contract(
                    'hfn,p,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src][..., rank::size, :]), backend.asarray(peram_s_sep_diff_tsep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, :N_part1]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, :N_part1][:, :, rank::size]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                # A2 = A21 + A22 + A23 + A24
                B2 = B21 + B22 + B23 + B24
                C2 = C21 + C22 + C23 + C24
                
                # comm.Barrier()
                if rank == 0:
                    print(f'part2 use time {time.perf_counter() - st_part2}')
                
                # part3 A, A, B, B
                st_part3 = time.perf_counter()
                weight_A2 = eigen.create_omega_accelerate(
                    exact = exact_eigen,
                    N_eigen = N_eigen,
                    N_sum = N_sum,
                    N_extract = N_extract,
                    noise = 0,
                    dim = 2
                )
                
                weight_B2 = eigen.create_omega_accelerate(
                    exact = exact_eigen,
                    N_eigen = N_eigen,
                    N_sum = N_sum,
                    N_extract = N_extract,
                    noise = N_part2,
                    dim = 2
                )[N_part1:, N_part1:][rank::size]
                
                # A31 = contract(
                #     'fn,ph,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:][..., rank::size]), 
                #     backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, :N_part1]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, N_part1:][:, rank::size, :]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A32 = contract(
                #     'fp,nh,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src][:, :, rank::size, :]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:, :N_part1]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, N_part1:][..., rank::size]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, N_part1:]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A33 = contract(
                #     'fh,pn,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:][..., rank::size]), 
                #     backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, N_part1:]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, :N_part1][:, rank::size, :]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A34 = contract(
                #     'np,fh,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, N_part1:]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A35 = contract(
                #     'nh,fp,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A36 = contract(
                #     'ph,fn,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                B31 = contract(
                    'fn,ph,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:][..., rank::size]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, :N_part1]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, N_part1:][:, rank::size, :]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B32 = contract(
                    'fp,nh,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:, :N_part1][..., rank::size, :]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B33 = contract(
                    'fh,np,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:, :N_part1][..., rank::size, :]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B34 = contract(
                    'np,fh,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B35 = contract(
                    'nh,fp,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B36 = contract(
                    'ph,fn,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C31 = contract(
                    'fn,ph,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep_diff_tsep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:][..., rank::size]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, :N_part1]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, N_part1:][:, rank::size, :]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C32 = contract(
                    'fp,nh,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:, :N_part1][..., rank::size, :]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C33 = contract(
                    'fh,np,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:, :N_part1][..., rank::size, :]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C34 = contract(
                    'np,fh,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep_diff_tsep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C35 = contract(
                    'nh,fp,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep_diff_tsep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C36 = contract(
                    'ph,fn,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                # A3 = A31 + A32 + A33 + A34 + A35 + A36
                B3 = B31 + B32 + B33 + B34 + B35 + B36
                C3 = C31 + C32 + C33 + C34 + C35 + C36
                
                # comm.Barrier()
                if rank == 0:
                    print(f'part3 use time {time.perf_counter() - st_part3}')

                # part4 A, B, B, B
                st_part4 = time.perf_counter()
                weight_A1 = eigen.create_omega_accelerate(
                    exact = exact_eigen,
                    N_eigen = N_eigen,
                    N_sum = N_sum,
                    N_extract = N_extract,
                    noise = 0,
                    dim = 1
                )
                
                weight_B3 = eigen.create_omega_accelerate(
                    exact = exact_eigen,
                    N_eigen = N_eigen,
                    N_sum = N_sum,
                    N_extract = N_extract,
                    noise = N_part2,
                    dim = 3
                )[N_part1:, N_part1:, N_part1:][rank::size]
                
                # A41 = contract(
                #     'fnp,h,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_B3), backend.asarray(weight_A1), 
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A42 = contract(
                #     'nph,f,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_B3), backend.asarray(weight_A1), 
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src][:, :, rank::size, :]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, N_part1:]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A43 = contract(
                #     'phf,n,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_B3), backend.asarray(weight_A1), 
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:, N_part1:]), backend.asarray(peram_d_2pt[..., N_part1:][:, :, :, rank::size]), 
                #     backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, :N_part1]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, N_part1:][:, rank::size, :]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A44 = contract(
                #     'hfn,p,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_B3), backend.asarray(weight_A1), 
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:, N_part1:][:, :, rank::size, :]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, N_part1:]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1:, N_part1:][:, :, rank::size]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                B41 = contract(
                    'fnp,h,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:, N_part1:][:, :, :, rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B42 = contract(
                    'nph,f,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:, :N_part1][:, :, rank::size, :]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B43 = contract(
                    'phf,n,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep_diff_tsep[:, :, :N_part1, N_part1:]), backend.asarray(peram_d_2pt[..., N_part1:][:, :, :, rank::size]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, :N_part1]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, N_part1:][:, rank::size, :]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B44 = contract(
                    'hfn,p,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src][:, :, rank::size, :]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:, N_part1:]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, N_part1:]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, N_part1:][:, :, rank::size]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C41 = contract(
                    'fnp,h,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:, N_part1:][:, :, :, rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C42 = contract(
                    'nph,f,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:, :N_part1][:, :, rank::size, :]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C43 = contract(
                    'phf,n,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep_diff_tsep[:, :, :N_part1, N_part1:]), backend.asarray(peram_d_2pt[..., N_part1:][:, :, :, rank::size]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, :N_part1]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:, N_part1:][:, rank::size, :]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C44 = contract(
                    'hfn,p,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:, :Nev_src][:, :, rank::size, :]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:, N_part1:]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr_tsep_1[:, N_part1:, N_part1:]), backend.asarray(sink_3pt_diff_tsep[:, :N_part1, N_part1:][:, :, rank::size]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                # A4 = A41 + A42 + A43 + A44
                B4 = B41 + B42 + B43 + B44
                C4 = C41 + C42 + C43 + C44
                
                # comm.Barrier()
                if rank == 0:
                    print(f'part4 use time {time.perf_counter() - st_part4}')
                
                # part5 B, B, B, B
                st_part5 = time.perf_counter()
                # A_ijkl = contract(
                #     'manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijkk = contract(
                #     'manb,gepf,codp,bd,Mfn,Mpp,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijkk += contract(
                #     'manb,genf,codp,bd,Mfn,Mpn,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijkk += contract(
                #     'manb,gehn,codp,bd,Mnn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijkk += contract(
                #     'manb,gehh,codp,bd,Mhn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                # A_ijkk += contract(
                #     'manb,gehf,codn,bd,Mfn,Mnh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijkk += contract(
                #     'manb,gehf,codf,bd,Mfn,Mfh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_iijj = contract(
                #     'manb,gehn,codh,bd,Mnn,Mhh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                # A_iijj += contract(
                #     'manb,genf,codf,bd,Mfn,Mfn,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                # A_iijj += contract(
                #     'manb,geff,codn,bd,Mfn,Mnf,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijjj = contract(
                #     'manb,genf,codn,bd,Mfn,Mnn,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijjj += contract(
                #     'manb,gehh,codh,bd,Mhn,Mhh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijjj += contract(
                #     'manb,genn,codp,bd,Mnn,Mpn,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijjj += contract(
                #     'manb,gehn,codn,bd,Mnn,Mnh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_iiii = contract(
                #     'manb,genn,codn,bd,Mnn,Mnn,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                
                B_ijkl = contract(
                    'gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B_ijkk = contract(
                    'gahb,menf,codh,bd,Mfn,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B_ijkk += contract(
                    'gahb,mehf,codp,bd,Mfh,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_ijkk += contract(
                    'gahb,menn,codp,bd,Mnn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_ijkk += contract(
                    'gahb,menh,codp,bd,Mhn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_ijkk += contract(
                    'gahb,menf,codn,bd,Mfn,Mnh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_ijkk += contract(
                    'gahb,menf,codf,bd,Mfn,Mfh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                
                B_iijj = contract(
                    'gahb,menn,codh,bd,Mnn,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_iijj += contract(
                    'gahb,mehf,codf,bd,Mfh,Mfh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_iijj += contract(
                    'gahb,menh,codn,bd,Mhn,Mnh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B_ijjj = contract(
                    'gahb,mehf,codh,bd,Mfh,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_ijjj += contract(
                    'gahb,menh,codh,bd,Mhn,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_ijjj += contract(
                    'gahb,mehh,codp,bd,Mhh,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_ijjj += contract(
                    'gahb,meff,codf,bd,Mff,Mfh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B_iiii = contract(
                    'gahb,mehh,codh,bd,Mhh,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C_ijkl = contract(
                    'gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C_ijkk = contract(
                    'gahb,menf,codh,bd,Mfn,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C_ijkk += contract(
                    'gahb,mehf,codp,bd,Mfh,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_ijkk += contract(
                    'gahb,menn,codp,bd,Mnn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_ijkk += contract(
                    'gahb,menh,codp,bd,Mhn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_ijkk += contract(
                    'gahb,menf,codn,bd,Mfn,Mnh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_ijkk += contract(
                    'gahb,menf,codf,bd,Mfn,Mfh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                
                C_iijj = contract(
                    'gahb,menn,codh,bd,Mnn,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_iijj += contract(
                    'gahb,mehf,codf,bd,Mfh,Mfh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_iijj += contract(
                    'gahb,menh,codn,bd,Mhn,Mnh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C_ijjj = contract(
                    'gahb,mehf,codh,bd,Mfh,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_ijjj += contract(
                    'gahb,menh,codh,bd,Mhn,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_ijjj += contract(
                    'gahb,mehh,codp,bd,Mhh,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_ijjj += contract(
                    'gahb,meff,codf,bd,Mff,Mfh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C_iiii = contract(
                    'gahb,mehh,codh,bd,Mhh,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation_diff_tsep[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep_diff_tsep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr_tsep_1[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt_diff_tsep[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                # comm.Barrier()
                if rank == 0:
                    print(f'part5 use time {time.perf_counter() - st_part5}')
                
                # A5 = (
                #     _weight[0] * A_iiii + 
                #     _weight[0] * _weight[1] * (A_ijjj - 4*A_iiii + A_iijj - 3*A_iiii) + 
                #     _weight[0] * _weight[1] * _weight[2] * (A_ijkk - 3 * (A_ijjj - 4*A_iiii) - 2 * (A_iijj - 3*A_iiii) - 6 * A_iiii) +
                #     _weight[0] * _weight[1] * _weight[2] * _weight[3] * (A_ijkl - A_iiii - (A_ijjj - 4*A_iiii + A_iijj - 3*A_iiii) - (A_ijkk - 3 * (A_ijjj - 4*A_iiii) - 2 * (A_iijj - 3*A_iiii) - 6 * A_iiii))
                # )
                
                B5 = (
                    _weight[0] * B_iiii + 
                    _weight[0] * _weight[1] * (B_ijjj - 4*B_iiii + B_iijj - 3*B_iiii) + 
                    _weight[0] * _weight[1] * _weight[2] * (B_ijkk - 3 * (B_ijjj - 4*B_iiii) - 2 * (B_iijj - 3*B_iiii) - 6 * B_iiii) +
                    _weight[0] * _weight[1] * _weight[2] * _weight[3] * (B_ijkl - B_iiii - (B_ijjj - 4*B_iiii + B_iijj - 3*B_iiii) - (B_ijkk - 3 * (B_ijjj - 4*B_iiii) - 2 * (B_iijj - 3*B_iiii) - 6 * B_iiii))
                )
                
                C5 = (
                    _weight[0] * C_iiii + 
                    _weight[0] * _weight[1] * (C_ijjj - 4*C_iiii + C_iijj - 3*C_iiii) + 
                    _weight[0] * _weight[1] * _weight[2] * (C_ijkk - 3 * (C_ijjj - 4*C_iiii) - 2 * (C_iijj - 3*C_iiii) - 6 * C_iiii) +
                    _weight[0] * _weight[1] * _weight[2] * _weight[3] * (C_ijkl - C_iiii - (C_ijjj - 4*C_iiii + C_iijj - 3*C_iiii) - (C_ijkk - 3 * (C_ijjj - 4*C_iiii) - 2 * (C_iijj - 3*C_iiii) - 6 * C_iiii))
                )
                
                # compensation_light_con_data = A1 + A2 + A3 + A4 + A5
                compensation_light_dis_data = B1 + B2 + B3 + B4 + B5
                compensation_strange_dis_data = C1 + C2 + C3 + C4 + C5
                
                comm.Barrier()
                
                # compensation_light_con_data = get_mpi_data(data = compensation_light_con_data, mdtype = 'Gather', root = ((t_src + t_sep - 1)%corr.Nt)%corr.size)
                compensation_light_dis_data = get_mpi_data(data = compensation_light_dis_data, mdtype = 'Gather', root = ((t_src + t_sep - 1)%corr.Nt)%corr.size)
                compensation_strange_dis_data = get_mpi_data(data = compensation_strange_dis_data, mdtype = 'Gather', root = ((t_src + t_sep - 1)%corr.Nt)%corr.size)
                    
                if corr.rank == ((t_src + t_sep - 1)%corr.Nt)%corr.size:
                    # print(compensation_light_con_data.shape)
                    # corr_3pt_matrix_con_compensation[-1, t_sep_indx, 0, ..., t_src, ((t_src + t_sep - 1)%corr.Nt)//corr.size] = compensation_light_con_data.sum(0)
                    corr_3pt_matrix_dis_compensation[-1, t_sep_indx, 0, ..., t_src, ((t_src + t_sep - 1)%corr.Nt)//corr.size] = compensation_light_dis_data.sum(0)
                    corr_3pt_matrix_dis_compensation[-1, t_sep_indx, 1, ..., t_src, ((t_src + t_sep - 1)%corr.Nt)//corr.size] = compensation_strange_dis_data.sum(0)

        peram_u_sep = \
            corr.peram_V1(
                peram_name=f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_hpy.npy',
                peram_save_path = f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t',
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = Nev_max,
                read_type = 'numpy',
                t_source = f'{((t_sep + t_src)%corr.Nt):03d}',
                seq = True
            )
            
        peram_s_sep = \
            corr.peram_V1(
                peram_name=f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_hpy.npy',
                peram_save_path = f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/strange/{conf_id}/t',            
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = Nev_max,
                read_type = 'numpy',
                t_source = f'{((t_sep + t_src)%corr.Nt):03d}',
                scatter = False,
                seq = False,
            )
            
        if rank == 0:
            peram_s_sep = peram_s_sep[(t_sep + t_src)%corr.Nt]
            
        time_3pt_compensation_mpi_data = time.perf_counter()
        time_3pt = time.perf_counter()
        peram_s_sep = backend.asarray(get_mpi_data(peram_s_sep, mdtype = 'Bcast'))
        sink_3pt = backend.asarray(sink_3pt / weight_2)
        VdV_curr = backend.asarray(sink_3pt.conj().transpose(0, 2, 1))
        peram_u_sep = backend.asarray(get_mpi_data(peram_u_sep[((t_sep + t_src)%corr.Nt)//size], mdtype = 'Bcast', root = ((t_sep + t_src)%corr.Nt)%size))
        peram_u_compensation = backend.asarray(get_mpi_data(peram_u[((t_sep + t_src)%corr.Nt)//size], mdtype = 'Bcast', root = ((t_sep + t_src)%corr.Nt)%size))
        
        comm.Barrier()
        if rank == 0:
            print(f"complete get mpi data tsrc {t_src} tsep {t_sep} use time {(time.perf_counter() - time_3pt_compensation_mpi_data):.3f} s")
        
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            if Nev_curr>exact_eigen:
                st_part1 = time.perf_counter()
                # part1 A, A, A, A,
                A1 = 0
                B1 = 0
                C1 = 0
                
                for Nev_weight in Nev_compensation_indx:
                    weight_A4 = eigen.create_omega_accelerate(
                        exact = exact_eigen,
                        N_eigen = N_eigen,
                        N_sum = N_sum,
                        N_extract = N_extract,
                        noise = 0,
                        fixed_first_pos = Nev_weight,
                        dim = 4
                    )
                    
                    # A1 += contract(
                    #     'fnph,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    #     backend.asarray(weight_A4), 
                    #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, :][..., Nev_weight]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    #     backend.asarray(source), backend.asarray(VdV_curr[:, :, :N_part1][:, Nev_weight, :]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                    #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    #     optimize = 'greedy'
                    #     )
                    
                    B1 += contract(
                        'fnph,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                        backend.asarray(weight_A4), 
                        backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[..., :N_part1, :][..., Nev_weight]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                        backend.asarray(source), backend.asarray(VdV_curr[:, :, :N_part1][:, Nev_weight, :]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                        backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                        optimize = 'greedy'
                        )
                    
                    C1 += contract(
                        'fnph,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                        backend.asarray(weight_A4), 
                        backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep[..., :N_part1, :][..., Nev_weight]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                        backend.asarray(source), backend.asarray(VdV_curr[:, :, :N_part1][:, Nev_weight, :]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                        backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                        optimize = 'greedy'
                        )
                    
                # comm.Barrier()
                if rank == 0:
                    print(f'part1 use time {time.perf_counter() - st_part1}')
                
                # part2 A, A, A, B
                st_part2 = time.perf_counter()
                weight_A3 = eigen.create_omega_accelerate(
                    exact = exact_eigen,
                    N_eigen = N_eigen,
                    N_sum = N_sum,
                    N_extract = N_extract,
                    noise = 0,
                    dim = 3
                )[rank::size]
                
                weight_B1 = eigen.create_omega_accelerate(
                    exact = exact_eigen,
                    N_eigen = N_eigen,
                    N_sum = N_sum,
                    N_extract = N_extract,
                    noise = N_part2,
                    dim = 1
                )[N_part1:]
                
                # A21 = contract(
                #     'fnp,h,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A3), backend.asarray(weight_B1), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, :N_part1][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A22 = contract(
                #     'nph,f,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A3), backend.asarray(weight_B1), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src][..., rank::size, :]), backend.asarray(peram_u_sep[:, :, :N_part1, N_part1:]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][..., rank::size]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A23 = contract(
                #     'phf,n,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A3), backend.asarray(weight_B1), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., :N_part1][..., rank::size]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:]), backend.asarray(sink_3pt[:, :N_part1, :N_part1][:, rank::size, :]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A24 = contract(
                #     'hfn,p,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A3), backend.asarray(weight_B1), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, :N_part1][..., rank::size, :]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, :N_part1][..., rank::size]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                B21 = contract(
                    'fnp,h,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, :N_part1][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B22 = contract(
                    'nph,f,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, N_part1:][:, :, rank::size, :]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][:, :, rank::size]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B23 = contract(
                    'phf,n,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, :N_part1]), backend.asarray(peram_d_2pt[..., :N_part1][..., rank::size]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:]), backend.asarray(sink_3pt[:, :N_part1, :N_part1][:, rank::size, :]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B24 = contract(
                    'hfn,p,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src][..., rank::size, :]), backend.asarray(peram_u_sep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, :N_part1][:, :, rank::size]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C21 = contract(
                    'fnp,h,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep[:, :, :N_part1, :N_part1][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C22 = contract(
                    'nph,f,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep[:, :, :N_part1, N_part1:][:, :, rank::size, :]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][:, :, rank::size]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C23 = contract(
                    'phf,n,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep[:, :, N_part1:, :N_part1]), backend.asarray(peram_d_2pt[..., :N_part1][..., rank::size]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:]), backend.asarray(sink_3pt[:, :N_part1, :N_part1][:, rank::size, :]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C24 = contract(
                    'hfn,p,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src][..., rank::size, :]), backend.asarray(peram_s_sep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, :N_part1][:, :, rank::size]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                # A2 = A21 + A22 + A23 + A24
                B2 = B21 + B22 + B23 + B24
                C2 = C21 + C22 + C23 + C24
                
                # comm.Barrier()
                if rank == 0:
                    print(f'part2 use time {time.perf_counter() - st_part2}')
                
                # part3 A, A, B, B
                st_part3 = time.perf_counter()
                weight_A2 = eigen.create_omega_accelerate(
                    exact = exact_eigen,
                    N_eigen = N_eigen,
                    N_sum = N_sum,
                    N_extract = N_extract,
                    noise = 0,
                    dim = 2
                )
                
                weight_B2 = eigen.create_omega_accelerate(
                    exact = exact_eigen,
                    N_eigen = N_eigen,
                    N_sum = N_sum,
                    N_extract = N_extract,
                    noise = N_part2,
                    dim = 2
                )[N_part1:, N_part1:][rank::size]
                
                # A31 = contract(
                #     'fn,ph,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:][..., rank::size]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, N_part1:][:, rank::size, :]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A32 = contract(
                #     'fp,nh,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src][:, :, rank::size, :]), backend.asarray(peram_u_sep[:, :, N_part1:, :N_part1]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:][..., rank::size]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A33 = contract(
                #     'fh,pn,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:][..., rank::size]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:]), backend.asarray(sink_3pt[:, N_part1:, :N_part1][:, rank::size, :]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A34 = contract(
                #     'np,fh,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A35 = contract(
                #     'nh,fp,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, N_part1:, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A36 = contract(
                #     'ph,fn,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                B31 = contract(
                    'fn,ph,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:][..., rank::size]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, N_part1:][:, rank::size, :]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B32 = contract(
                    'fp,nh,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, :N_part1][..., rank::size, :]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B33 = contract(
                    'fh,np,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, :N_part1][..., rank::size, :]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt[:, N_part1:, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B34 = contract(
                    'np,fh,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B35 = contract(
                    'nh,fp,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, N_part1:, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B36 = contract(
                    'ph,fn,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C31 = contract(
                    'fn,ph,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:][..., rank::size]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, N_part1:][:, rank::size, :]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C32 = contract(
                    'fp,nh,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep[:, :, N_part1:, :N_part1][..., rank::size, :]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C33 = contract(
                    'fh,np,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep[:, :, N_part1:, :N_part1][..., rank::size, :]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt[:, N_part1:, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C34 = contract(
                    'np,fh,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C35 = contract(
                    'nh,fp,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, N_part1:, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C36 = contract(
                    'ph,fn,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep[:, :, N_part1:, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                # A3 = A31 + A32 + A33 + A34 + A35 + A36
                B3 = B31 + B32 + B33 + B34 + B35 + B36
                C3 = C31 + C32 + C33 + C34 + C35 + C36
                
                # comm.Barrier()
                if rank == 0:
                    print(f'part3 use time {time.perf_counter() - st_part3}')

                # part4 A, B, B, B
                st_part4 = time.perf_counter()
                weight_A1 = eigen.create_omega_accelerate(
                    exact = exact_eigen,
                    N_eigen = N_eigen,
                    N_sum = N_sum,
                    N_extract = N_extract,
                    noise = 0,
                    dim = 1
                )
                
                weight_B3 = eigen.create_omega_accelerate(
                    exact = exact_eigen,
                    N_eigen = N_eigen,
                    N_sum = N_sum,
                    N_extract = N_extract,
                    noise = N_part2,
                    dim = 3
                )[N_part1:, N_part1:, N_part1:][rank::size]
                
                # A41 = contract(
                #     'fnp,h,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_B3), backend.asarray(weight_A1), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt[:, N_part1:, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A42 = contract(
                #     'nph,f,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_B3), backend.asarray(weight_A1), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src][:, :, rank::size, :]), backend.asarray(peram_u_sep[:, :, N_part1:, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt[:, N_part1:, N_part1:]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A43 = contract(
                #     'phf,n,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_B3), backend.asarray(weight_A1), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, N_part1:]), backend.asarray(peram_d_2pt[..., N_part1:][:, :, :, rank::size]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, N_part1:][:, rank::size, :]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A44 = contract(
                #     'hfn,p,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_B3), backend.asarray(weight_A1), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, N_part1:][:, :, rank::size, :]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:]), backend.asarray(sink_3pt[:, :N_part1:, N_part1:][:, :, rank::size]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                B41 = contract(
                    'fnp,h,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, N_part1:][:, :, :, rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt[:, N_part1:, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B42 = contract(
                    'nph,f,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, :N_part1][:, :, rank::size, :]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt[:, N_part1:, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B43 = contract(
                    'phf,n,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, N_part1:]), backend.asarray(peram_d_2pt[..., N_part1:][:, :, :, rank::size]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, N_part1:][:, rank::size, :]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B44 = contract(
                    'hfn,p,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src][:, :, rank::size, :]), backend.asarray(peram_u_sep[:, :, N_part1:, N_part1:]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:]), backend.asarray(sink_3pt[:, :N_part1, N_part1:][:, :, rank::size]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C41 = contract(
                    'fnp,h,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep[:, :, N_part1:, N_part1:][:, :, :, rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt[:, N_part1:, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C42 = contract(
                    'nph,f,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep[:, :, N_part1:, :N_part1][:, :, rank::size, :]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt[:, N_part1:, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C43 = contract(
                    'phf,n,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep[:, :, :N_part1, N_part1:]), backend.asarray(peram_d_2pt[..., N_part1:][:, :, :, rank::size]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, N_part1:][:, rank::size, :]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C44 = contract(
                    'hfn,p,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src][:, :, rank::size, :]), backend.asarray(peram_s_sep[:, :, N_part1:, N_part1:]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:]), backend.asarray(sink_3pt[:, :N_part1, N_part1:][:, :, rank::size]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                # A4 = A41 + A42 + A43 + A44
                B4 = B41 + B42 + B43 + B44
                C4 = C41 + C42 + C43 + C44
                
                # comm.Barrier()
                if rank == 0:
                    print(f'part4 use time {time.perf_counter() - st_part4}')
                
                # part5 B, B, B, B
                st_part5 = time.perf_counter()
                # A_ijkl = contract(
                #     'manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijkk = contract(
                #     'manb,gepf,codp,bd,Mfn,Mpp,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijkk += contract(
                #     'manb,genf,codp,bd,Mfn,Mpn,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijkk += contract(
                #     'manb,gehn,codp,bd,Mnn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijkk += contract(
                #     'manb,gehh,codp,bd,Mhn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                # A_ijkk += contract(
                #     'manb,gehf,codn,bd,Mfn,Mnh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijkk += contract(
                #     'manb,gehf,codf,bd,Mfn,Mfh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_iijj = contract(
                #     'manb,gehn,codh,bd,Mnn,Mhh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                # A_iijj += contract(
                #     'manb,genf,codf,bd,Mfn,Mfn,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                # A_iijj += contract(
                #     'manb,geff,codn,bd,Mfn,Mnf,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijjj = contract(
                #     'manb,genf,codn,bd,Mfn,Mnn,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijjj += contract(
                #     'manb,gehh,codh,bd,Mhn,Mhh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijjj += contract(
                #     'manb,genn,codp,bd,Mnn,Mpn,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_ijjj += contract(
                #     'manb,gehn,codn,bd,Mnn,Mnh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                # A_iiii = contract(
                #     'manb,genn,codn,bd,Mnn,Mnn,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = 'greedy'
                #     )
                
                
                B_ijkl = contract(
                    'gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B_ijkk = contract(
                    'gahb,menf,codh,bd,Mfn,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B_ijkk += contract(
                    'gahb,mehf,codp,bd,Mfh,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_ijkk += contract(
                    'gahb,menn,codp,bd,Mnn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_ijkk += contract(
                    'gahb,menh,codp,bd,Mhn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_ijkk += contract(
                    'gahb,menf,codn,bd,Mfn,Mnh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_ijkk += contract(
                    'gahb,menf,codf,bd,Mfn,Mfh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                
                B_iijj = contract(
                    'gahb,menn,codh,bd,Mnn,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_iijj += contract(
                    'gahb,mehf,codf,bd,Mfh,Mfh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_iijj += contract(
                    'gahb,menh,codn,bd,Mhn,Mnh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B_ijjj = contract(
                    'gahb,mehf,codh,bd,Mfh,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_ijjj += contract(
                    'gahb,menh,codh,bd,Mhn,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_ijjj += contract(
                    'gahb,mehh,codp,bd,Mhh,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                B_ijjj += contract(
                    'gahb,meff,codf,bd,Mff,Mfh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                B_iiii = contract(
                    'gahb,mehh,codh,bd,Mhh,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C_ijkl = contract(
                    'gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C_ijkk = contract(
                    'gahb,menf,codh,bd,Mfn,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C_ijkk += contract(
                    'gahb,mehf,codp,bd,Mfh,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_ijkk += contract(
                    'gahb,menn,codp,bd,Mnn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_ijkk += contract(
                    'gahb,menh,codp,bd,Mhn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_ijkk += contract(
                    'gahb,menf,codn,bd,Mfn,Mnh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_ijkk += contract(
                    'gahb,menf,codf,bd,Mfn,Mfh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                
                C_iijj = contract(
                    'gahb,menn,codh,bd,Mnn,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_iijj += contract(
                    'gahb,mehf,codf,bd,Mfh,Mfh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_iijj += contract(
                    'gahb,menh,codn,bd,Mhn,Mnh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C_ijjj = contract(
                    'gahb,mehf,codh,bd,Mfh,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_ijjj += contract(
                    'gahb,menh,codh,bd,Mhn,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_ijjj += contract(
                    'gahb,mehh,codp,bd,Mhh,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                C_ijjj += contract(
                    'gahb,meff,codf,bd,Mff,Mfh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                C_iiii = contract(
                    'gahb,mehh,codh,bd,Mhh,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = 'greedy'
                    )
                
                # comm.Barrier()
                if rank == 0:
                    print(f'part5 use time {time.perf_counter() - st_part5}')
                
                # A5 = (
                #     _weight[0] * A_iiii + 
                #     _weight[0] * _weight[1] * (A_ijjj - 4*A_iiii + A_iijj - 3*A_iiii) + 
                #     _weight[0] * _weight[1] * _weight[2] * (A_ijkk - 3 * (A_ijjj - 4*A_iiii) - 2 * (A_iijj - 3*A_iiii) - 6 * A_iiii) +
                #     _weight[0] * _weight[1] * _weight[2] * _weight[3] * (A_ijkl - A_iiii - (A_ijjj - 4*A_iiii + A_iijj - 3*A_iiii) - (A_ijkk - 3 * (A_ijjj - 4*A_iiii) - 2 * (A_iijj - 3*A_iiii) - 6 * A_iiii))
                # )
                
                B5 = (
                    _weight[0] * B_iiii + 
                    _weight[0] * _weight[1] * (B_ijjj - 4*B_iiii + B_iijj - 3*B_iiii) + 
                    _weight[0] * _weight[1] * _weight[2] * (B_ijkk - 3 * (B_ijjj - 4*B_iiii) - 2 * (B_iijj - 3*B_iiii) - 6 * B_iiii) +
                    _weight[0] * _weight[1] * _weight[2] * _weight[3] * (B_ijkl - B_iiii - (B_ijjj - 4*B_iiii + B_iijj - 3*B_iiii) - (B_ijkk - 3 * (B_ijjj - 4*B_iiii) - 2 * (B_iijj - 3*B_iiii) - 6 * B_iiii))
                )
                
                C5 = (
                    _weight[0] * C_iiii + 
                    _weight[0] * _weight[1] * (C_ijjj - 4*C_iiii + C_iijj - 3*C_iiii) + 
                    _weight[0] * _weight[1] * _weight[2] * (C_ijkk - 3 * (C_ijjj - 4*C_iiii) - 2 * (C_iijj - 3*C_iiii) - 6 * C_iiii) +
                    _weight[0] * _weight[1] * _weight[2] * _weight[3] * (C_ijkl - C_iiii - (C_ijjj - 4*C_iiii + C_iijj - 3*C_iiii) - (C_ijkk - 3 * (C_ijjj - 4*C_iiii) - 2 * (C_iijj - 3*C_iiii) - 6 * C_iiii))
                )
                
                # compensation_light_con_data = A1 + A2 + A3 + A4 + A5
                compensation_light_dis_data = B1 + B2 + B3 + B4 + B5
                compensation_strange_dis_data = C1 + C2 + C3 + C4 + C5
                
                comm.Barrier()
                
                # compensation_light_con_data = get_mpi_data(data = compensation_light_con_data, mdtype = 'Gather', root = ((t_src + t_sep)%corr.Nt)%corr.size)
                compensation_light_dis_data = get_mpi_data(data = compensation_light_dis_data, mdtype = 'Gather', root = ((t_src + t_sep)%corr.Nt)%corr.size)
                compensation_strange_dis_data = get_mpi_data(data = compensation_strange_dis_data, mdtype = 'Gather', root = ((t_src + t_sep)%corr.Nt)%corr.size)
                    
                if corr.rank == ((t_src + t_sep)%corr.Nt)%corr.size:
                    # print(compensation_light_con_data.shape)
        
                    # corr_3pt_matrix_con_compensation[-1, t_sep_indx, 0, ..., t_src, ((t_src + t_sep)%corr.Nt)//corr.size] = compensation_light_con_data.sum(0)
                    corr_3pt_matrix_dis_compensation[-1, t_sep_indx, 0, ..., t_src, ((t_src + t_sep)%corr.Nt)//corr.size] = compensation_light_dis_data.sum(0)
                    corr_3pt_matrix_dis_compensation[-1, t_sep_indx, 1, ..., t_src, ((t_src + t_sep)%corr.Nt)//corr.size] = compensation_strange_dis_data.sum(0)

for Mom_indx, Mom_list in enumerate(Mom_link[:]):
    for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
        # corr.write(
        #     corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
        #     corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_hpy_src{Nev_src}_sink{Nev_src}',
        #     corr_matrix = corr_2pt_matrix[0],
        #     Cpt = '2pt',
        # )
        
        # corr.write(
        #     corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
        #     corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_hpy_src{Nev_src}_sink{Nev_src}',
        #     corr_matrix = corr_2pt_matrix[0],
        #     Cpt = '2pt_all_t',
        # )
        
        # for mu_indx_1, mu in enumerate(mu_indx[Mom_indx]):
        #     corr.write(
        #         corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
        #         corr_name = f'gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_hpy_src{Nev_src}_sink{Nev_curr}',
        #         corr_matrix = corr_2pt_mu_matrix[Nev_curr_indx, 0, Mom_indx, mu_indx_1],
        #         Cpt = '2pt_all_t',
        #     )
            
        #     corr.write(
        #         corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
        #         corr_name = f'light_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_hpy_curr{Nev_curr}',
        #         corr_matrix = (bubble_mu_light[Nev_curr_indx, 0, Mom_indx, mu_indx_1]),
        #         Cpt = 'bubble',
        #     )
            
        #     corr.write(
        #         corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
        #         corr_name = f'strange_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_hpy_curr{Nev_curr}',
        #         corr_matrix = (bubble_mu_strange[Nev_curr_indx, 0, Mom_indx, mu_indx_1]),
        #         Cpt = 'bubble',
        #     )
            
        #     corr.write(
        #         corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
        #         corr_name = f'all_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_hpy_curr{Nev_curr}',
        #         corr_matrix = (bubble_mu_light[Nev_curr_indx, 0, Mom_indx, mu_indx_1] - bubble_mu_strange[Nev_curr_indx, 0, Mom_indx, mu_indx_1]),
        #         Cpt = 'bubble',
        #     )
            
        for t_sep_indx, t_sep in enumerate(t_sep_list):
            for gamma_indx, gamma_list in enumerate(mu_nu_indx[Mom_indx]):
                # corr.write(
                #     t_sep = t_sep,
                #     corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                #     corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_hpy_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}_compensation', # linkZ, conserved
                #     corr_matrix = corr_3pt_matrix[Nev_curr_indx, t_sep_indx, 0, Mom_indx, gamma_indx, :].reshape(1, corr.Nt, -1), 
                #     Cpt = '3pt',
                #     gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                #     conserved = False,
                #     link_quark = 'u',
                # )

                # corr.write(
                #     t_sep = t_sep,
                #     corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                #     corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_hpy_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}_compensation', # linkZ, conserved
                #     corr_matrix = corr_3pt_matrix[Nev_curr_indx, t_sep_indx, 0, Mom_indx, gamma_indx, :].reshape(1, corr.Nt, -1), 
                #     Cpt = '3pt_all_t',
                #     gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                #     conserved = False,
                #     link_quark = 'u',
                # )

                # corr.write(
                #     t_sep = t_sep,
                #     corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                #     corr_name = f'light_connect_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_hpy_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}_compensation', # linkZ, conserved
                #     corr_matrix = corr_3pt_matrix_con_compensation[Nev_curr_indx, t_sep_indx, 0, Mom_indx, gamma_indx, :].reshape(1, corr.Nt, -1), 
                #     Cpt = '3pt_all_t',
                #     gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                #     conserved = False,
                #     link_quark = 'u',
                # )

                corr.write(
                    t_sep = t_sep,
                    corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                    corr_name = f'light_disconnect_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_hpy_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}_compensation', # linkZ, conserved
                    corr_matrix = corr_3pt_matrix_dis_compensation[Nev_curr_indx, t_sep_indx, 0, Mom_indx, gamma_indx, :].reshape(1, corr.Nt, -1), 
                    Cpt = '3pt_all_t',
                    gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                    conserved = False,
                    link_quark = 'u',
                )

                corr.write(
                    t_sep = t_sep,
                    corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                    corr_name = f'strange_disconnect_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_hpy_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}_compensation', # linkZ, conserved
                    corr_matrix = corr_3pt_matrix_dis_compensation[Nev_curr_indx, t_sep_indx, 1, Mom_indx, gamma_indx, :].reshape(1, corr.Nt, -1), 
                    Cpt = '3pt_all_t',
                    gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                    conserved = False,
                    link_quark = 'u',
                )

if corr.rank == 0:
    print('complete')
