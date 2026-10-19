import os
import sys

# test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, '/public/home/sush/distillation/')

from lqcddb import *

# from opt_einsum import contract, contract_path

import time

set_backend('numpy')
backend = get_backend()

lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 16]


conf_id = sys.argv[1]

mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)

rank = getMPIRank()
comm = getMPIComm()
size = getMPISize()

exact_eigen = 100
N_eigen = [100, 200, 400, 700, 1100]
N_sum = [20, 20, 20, 20, 20]
N_extract = [10, 10, 10, 10, 10]
noise_eigen = 200

exact_eigen_name = 100
N_eigen_name = [50, 100, 150, 200, 300]
N_sum_name = [20, 20, 20, 20, 20]
N_extract_name = [10, 10, 10, 10, 10]
noise_eigen_name = 200

Nx, Ny, Nz, Nt = lattice_size
Lx, Ly, Lz, Lt = [lattice_size[x]//grid_size[x] for x in range(len(lattice_size))]

Gt = Nt//size

L = Nx**3*Nc - exact_eigen - sum(N_eigen)

_weight = [L/noise_eigen, (L-1)/(noise_eigen-1), (L-2)/(noise_eigen-2), (L-3)/(noise_eigen-3)]

Nbin = 1
N_stout = 20
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
eigen = vertex_creator(Nx = Nx)

weight_2 = eigen.create_omega_accelerate(
    exact = exact_eigen,
    N_eigen = N_eigen,
    N_sum = N_sum,
    N_extract = N_extract,
    noise = noise_eigen,
    dim = 2
)

# eigen = corr_eigvecs(Nx = Nx, backend = backend.__name__)
phase_exp_2pt = backend.zeros((len(Mom_link), Nx, Nx, Nx, Nc), dtype = complex)

for Mom_indx in range(len(Mom_link)):
    phase_exp_2pt[Mom_indx] = eigen.phase_exp_VdV(Mom = Mom_link[Mom_indx])
    
phase_exp_2pt_sink = eigen.phase_exp_VdV(Mom = Mom_sink[0])

t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'TScatter')

VdV_link = backend.zeros((Lt, len(Mom_link), 2*link_max + 1, Nev_max, Nev_max), dtype = complex)
sink = backend.zeros((Lt, 1, Nev_src, Nev_src), dtype = complex)

st_eigen = time.perf_counter()
for t_src_indx, t_src in enumerate(t_rank):
    try:
        eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{t_src:03d}_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_hpy.npy')

    except:
        comm.Abort(1)
        
    for Mom_indx in range(len(Mom_link)):
        VdV_link[t_src_indx, (Mom_indx):((Mom_indx + 1))] = eigen.VdV_sink_t_link(eigvecs = eigvecs[:], link_dir = 'Z', link_max = link_max, phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], gauge_link = False)
        
    sink[t_src_indx] = eigen.Mom_VdV_sink_t(eigvecs = eigvecs[:Nev_src], phase_exp = phase_exp_2pt_sink)

VdV_link = VdV_link * weight_2

if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')

N_operator = 1

corr_2pt_matrix = backend.zeros((N_operator, Nt, Gt), dtype = complex)

corr_2pt_mu_matrix = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2, Nt, Gt), dtype = complex)
corr_3pt_matrix = backend.zeros((len(Nev_curr_list), len(t_sep_list), N_operator, len(Mom_link), 2, Nt, Gt), dtype = complex)

# corr_3pt_matrix_dis_compensation = backend.zeros((1, len(t_sep_list), 2, len(Mom_link), 2, Nt, Gt), dtype = complex)
# corr_3pt_matrix_con_compensation = backend.zeros((1, len(t_sep_list), 1, len(Mom_link), 2, Nt, Gt), dtype = complex)

bubble_mu_light = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2, Nt), dtype = complex)
bubble_mu_strange = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2, Nt), dtype = complex)

for t_src in range(0, Nt, 1):
    source = backend.asarray(get_mpi_data(sink[t_src//size, 0].conj().transpose(1, 0), mdtype = 'Bcast', root = t_src%size, axis = 0))
    try:
        if rank == 0:
            peram_u = backend.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{t_src:03d}_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_hpy.npy')
        else:
            peram_u = None
    except:
        comm.Abort(1)

    peram_u = get_mpi_data(data = peram_u, mdtype = 'TScatter', root = 0, axis = 0)
    peram_d = seq_peram(peram_u[..., :Nev_src])
    
    if rank == 0:
        peram_s = backend.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/strange/{conf_id}/t{t_src:03d}_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_hpy.npy')
    else:
        peram_s = None
    
    peram_s = get_mpi_data(data = peram_s, mdtype = 'TScatter', root = 0, axis = 0)

    t_curr_0_list, _, t_curr_0_list_indx = get_mpi_tlist(Nt = Nt, t = [x%Nt for x in range(t_src, t_src + Nt, 1)], gtype = 'TScatter')
    
    time_2pt_mu = time.perf_counter()
    for t_sink_list_indx, t_sink_indx in enumerate(t_curr_0_list_indx):
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            corr_2pt_mu_matrix[Nev_curr_indx, 0, :, :, t_src, t_sink_indx] = cached_contract(
                'manb,cedf,bd,Mfn,ac,MFem->MF', 
                backend.asarray(peram_u[t_sink_indx, ..., :Nev_curr, :Nev_src]), backend.asarray(peram_d[t_sink_indx, ..., :Nev_src, :Nev_curr]), 
                backend.asarray(source), backend.asarray(VdV_link[t_sink_indx, :, 0, :Nev_curr, :Nev_curr]), 
                backend.asarray(gamma(5)), backend.asarray(mu_gamma), optimize = True
                )
            
    for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
        bubble_mu_light[Nev_curr_indx, 0, :, :, t_src] = cached_contract('menf,Mfn,MKem->MK', backend.asarray(peram_u[t_src//size, ..., :Nev_curr, :Nev_curr]), backend.asarray(VdV_link[t_src//size, :, 0, :Nev_curr, :Nev_curr]), backend.asarray(mu_gamma), optimize = True)
        bubble_mu_strange[Nev_curr_indx, 0, :, :, t_src] = cached_contract('menf,Mfn,MKem->MK', backend.asarray(peram_s[t_src//size, ..., :Nev_curr, :Nev_curr]), backend.asarray(VdV_link[t_src//size, :, 0, :Nev_curr, :Nev_curr]), backend.asarray(mu_gamma), optimize = True)
    
    comm.Barrier()
    if rank == 0:
        print(f"complete cal 2pt_mu and bubble of tsrc {t_src} use time {(time.perf_counter() - time_2pt_mu):.3f} s")

    for t_sep_indx, t_sep in enumerate(t_sep_list):
        if rank == 0:
            peram_u_sep = backend.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{((t_sep + t_src)%Nt):03d}_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_hpy.npy')
        else:
            peram_u_sep = None
        
        peram_u_sep = get_mpi_data(data = peram_u_sep, mdtype = 'TScatter', root = 0, axis = 0)
        peram_u_sep = seq_peram(peram_u_sep)
        
        peram_d_2pt = backend.asarray(get_mpi_data(peram_d[((t_sep + t_src)%Nt)//size], mdtype = 'Bcast', root = ((t_sep + t_src)%Nt)%size))
        sink_3pt = backend.asarray(get_mpi_data(VdV_link[((t_sep + t_src)%Nt)//size, :, 0].conj().transpose(0, 2, 1), mdtype = 'Bcast', root = ((t_sep + t_src)%Nt)%size))
        
        time_3pt = time.perf_counter()
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            for t_sink_list_indx, t_sink_indx in enumerate(t_curr_0_list_indx):
                t_sink = t_curr_0_list[t_sink_list_indx]
                corr_3pt_matrix[Nev_curr_indx, t_sep_indx, 0, ..., t_src, t_sink_indx] = cached_contract(
                    'manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG', 
                    backend.asarray(peram_u[t_sink_indx, :, :, :Nev_curr, :Nev_src]), backend.asarray(peram_u_sep[t_sink_indx, :, :, :Nev_curr, :Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, :Nev_curr]), 
                    backend.asarray(source), backend.asarray(VdV_link[t_sink_indx, :, 0, :Nev_curr, :Nev_curr]), backend.asarray(sink_3pt[:, :Nev_curr, :Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]), optimize = True
                    )

        comm.Barrier()
        if rank == 0:
            print(f"complete cal 3pt of tsrc {t_src} tsep {t_sep} use time {(time.perf_counter() - time_3pt):.3f} s")
    
        peram_s_sep = backend.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/strange/{conf_id}/t{((t_sep + t_src)%Nt):03d}_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_hpy.npy')

        if rank == 0:
            peram_s_sep = peram_s_sep[(t_sep + t_src)%Nt]
            
        time_3pt_compensation_mpi_data = time.perf_counter()
        time_3pt = time.perf_counter()
        # peram_s_sep = backend.asarray(get_mpi_data(peram_s_sep, mdtype = 'Bcast'))
        sink_3pt = backend.asarray(sink_3pt / weight_2)
        VdV_curr = backend.asarray(sink_3pt.conj().transpose(0, 2, 1))
        peram_u_sep = backend.asarray(get_mpi_data(peram_u_sep[((t_sep + t_src)%Nt)//size], mdtype = 'Bcast', root = ((t_sep + t_src)%Nt)%size))
        peram_u_compensation = backend.asarray(get_mpi_data(peram_u[((t_sep + t_src)%Nt)//size], mdtype = 'Bcast', root = ((t_sep + t_src)%Nt)%size))
        
        comm.Barrier()
        if rank == 0:
            print(f"complete get mpi data tsrc {t_src} tsep {t_sep} use time {(time.perf_counter() - time_3pt_compensation_mpi_data):.3f} s")

        N_part1 = 200
        N_part2 = 200
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            if Nev_curr == Nev_max:
                st_part1 = time.perf_counter()
                # part1 A, A, A, A,
                Nev_compensation_indx = [[x] for x in range(rank, N_part1, size)]
                
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
                    
                    A1 += cached_contract(
                        'fnph,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                        backend.asarray(weight_A4), 
                        backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, :][..., Nev_weight]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                        backend.asarray(source), backend.asarray(VdV_curr[:, :, :N_part1][:, Nev_weight, :]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                        backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                        optimize = ['dp', 'greedy', 'auto']
                        )
                    
                    # B1 += cached_contract(
                    #     'fnph,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    #     backend.asarray(weight_A4), 
                    #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[..., :N_part1, :][..., Nev_weight]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    #     backend.asarray(source), backend.asarray(VdV_curr[:, :, :N_part1][:, Nev_weight, :]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                    #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    #     optimize = ['dp', 'greedy', 'auto']
                    #     )
                    
                    # C1 += cached_contract(
                    #     'fnph,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    #     backend.asarray(weight_A4), 
                    #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep[..., :N_part1, :][..., Nev_weight]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    #     backend.asarray(source), backend.asarray(VdV_curr[:, :, :N_part1][:, Nev_weight, :]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                    #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    #     optimize = ['dp', 'greedy', 'auto']
                    #     )
                    
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
                
                A21 = cached_contract(
                    'fnp,h,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, :N_part1][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A22 = cached_contract(
                    'nph,f,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src][..., rank::size, :]), backend.asarray(peram_u_sep[:, :, :N_part1, N_part1:]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][..., rank::size]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A23 = cached_contract(
                    'phf,n,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., :N_part1][..., rank::size]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:]), backend.asarray(sink_3pt[:, :N_part1, :N_part1][:, rank::size, :]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A24 = cached_contract(
                    'hfn,p,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A3), backend.asarray(weight_B1), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, :N_part1][..., rank::size, :]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, :N_part1][..., rank::size]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                # B21 = cached_contract(
                #     'fnp,h,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A3), backend.asarray(weight_B1), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, :N_part1][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # B22 = cached_contract(
                #     'nph,f,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A3), backend.asarray(weight_B1), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, N_part1:][:, :, rank::size, :]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][:, :, rank::size]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # B23 = cached_contract(
                #     'phf,n,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A3), backend.asarray(weight_B1), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, :N_part1]), backend.asarray(peram_d_2pt[..., :N_part1][..., rank::size]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:]), backend.asarray(sink_3pt[:, :N_part1, :N_part1][:, rank::size, :]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # B24 = cached_contract(
                #     'hfn,p,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A3), backend.asarray(weight_B1), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src][..., rank::size, :]), backend.asarray(peram_u_sep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, :N_part1][:, :, rank::size]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C21 = cached_contract(
                #     'fnp,h,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A3), backend.asarray(weight_B1), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep[:, :, :N_part1, :N_part1][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C22 = cached_contract(
                #     'nph,f,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A3), backend.asarray(weight_B1), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep[:, :, :N_part1, N_part1:][:, :, rank::size, :]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][:, :, rank::size]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C23 = cached_contract(
                #     'phf,n,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A3), backend.asarray(weight_B1), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep[:, :, N_part1:, :N_part1]), backend.asarray(peram_d_2pt[..., :N_part1][..., rank::size]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:]), backend.asarray(sink_3pt[:, :N_part1, :N_part1][:, rank::size, :]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C24 = cached_contract(
                #     'hfn,p,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A3), backend.asarray(weight_B1), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src][..., rank::size, :]), backend.asarray(peram_s_sep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, :N_part1][:, :, rank::size]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                A2 = A21 + A22 + A23 + A24
                # B2 = B21 + B22 + B23 + B24
                # C2 = C21 + C22 + C23 + C24
                
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
                
                A31 = cached_contract(
                    'fn,ph,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:][..., rank::size]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, N_part1:][:, rank::size, :]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A32 = cached_contract(
                    'fp,nh,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src][:, :, rank::size, :]), backend.asarray(peram_u_sep[:, :, N_part1:, :N_part1]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:][..., rank::size]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A33 = cached_contract(
                    'fh,pn,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:][..., rank::size]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:]), backend.asarray(sink_3pt[:, N_part1:, :N_part1][:, rank::size, :]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A34 = cached_contract(
                    'np,fh,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A35 = cached_contract(
                    'nh,fp,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, N_part1:, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A36 = cached_contract(
                    'ph,fn,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_A2), backend.asarray(weight_B2), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                # B31 = cached_contract(
                #     'fn,ph,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:][..., rank::size]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, N_part1:][:, rank::size, :]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # B32 = cached_contract(
                #     'fp,nh,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, :N_part1][..., rank::size, :]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # B33 = cached_contract(
                #     'fh,np,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, :N_part1][..., rank::size, :]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt[:, N_part1:, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # B34 = cached_contract(
                #     'np,fh,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # B35 = cached_contract(
                #     'nh,fp,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, N_part1:, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # B36 = cached_contract(
                #     'ph,fn,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C31 = cached_contract(
                #     'fn,ph,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep[:, :, :N_part1, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:][..., rank::size]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, N_part1:][:, rank::size, :]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C32 = cached_contract(
                #     'fp,nh,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep[:, :, N_part1:, :N_part1][..., rank::size, :]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C33 = cached_contract(
                #     'fh,np,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep[:, :, N_part1:, :N_part1][..., rank::size, :]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt[:, N_part1:, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C34 = cached_contract(
                #     'np,fh,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, N_part1:]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C35 = cached_contract(
                #     'nh,fp,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1][:, rank::size, :]), backend.asarray(sink_3pt[:, N_part1:, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C36 = cached_contract(
                #     'ph,fn,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_A2), backend.asarray(weight_B2), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep[:, :, N_part1:, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt[:, :N_part1, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                A3 = A31 + A32 + A33 + A34 + A35 + A36
                # B3 = B31 + B32 + B33 + B34 + B35 + B36
                # C3 = C31 + C32 + C33 + C34 + C35 + C36
                
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

                if rank == 0:
                    weight_B3 = eigen.create_omega_accelerate(
                        exact = exact_eigen,
                        N_eigen = N_eigen,
                        N_sum = N_sum,
                        N_extract = N_extract,
                        noise = N_part2,
                        dim = 3
                    )[N_part1:, N_part1:, N_part1:]

                else:
                    weight_B3 = None
                
                weight_B3 = get_mpi_data(data = weight_B3, mdtype = 'TScatter', axis = 0)
                
                A41 = cached_contract(
                    'fnp,h,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, N_part1:][..., rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt[:, N_part1:, :N_part1]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A42 = cached_contract(
                    'nph,f,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src][:, :, rank::size, :]), backend.asarray(peram_u_sep[:, :, N_part1:, :N_part1]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt[:, N_part1:, N_part1:]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A43 = cached_contract(
                    'phf,n,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, N_part1:]), backend.asarray(peram_d_2pt[..., N_part1:][:, :, :, rank::size]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, N_part1:][:, rank::size, :]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A44 = cached_contract(
                    'hfn,p,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(weight_B3), backend.asarray(weight_A1), 
                    backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, N_part1:][:, :, rank::size, :]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                    backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:]), backend.asarray(sink_3pt[:, :N_part1:, N_part1:][:, :, rank::size]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                # B41 = cached_contract(
                #     'fnp,h,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_B3), backend.asarray(weight_A1), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, N_part1:][:, :, :, rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt[:, N_part1:, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # B42 = cached_contract(
                #     'nph,f,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_B3), backend.asarray(weight_A1), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, N_part1:, :N_part1][:, :, rank::size, :]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt[:, N_part1:, N_part1:]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # B43 = cached_contract(
                #     'phf,n,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_B3), backend.asarray(weight_A1), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_u_sep[:, :, :N_part1, N_part1:]), backend.asarray(peram_d_2pt[..., N_part1:][:, :, :, rank::size]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, N_part1:][:, rank::size, :]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # B44 = cached_contract(
                #     'hfn,p,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_B3), backend.asarray(weight_A1), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src][:, :, rank::size, :]), backend.asarray(peram_u_sep[:, :, N_part1:, N_part1:]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:]), backend.asarray(sink_3pt[:, :N_part1, N_part1:][:, :, rank::size]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C41 = cached_contract(
                #     'fnp,h,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_B3), backend.asarray(weight_A1), 
                #     backend.asarray(peram_u_compensation[:, :, :N_part1, :Nev_src]), backend.asarray(peram_s_sep[:, :, N_part1:, N_part1:][:, :, :, rank::size]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:][:, rank::size, :]), backend.asarray(sink_3pt[:, N_part1:, :N_part1]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C42 = cached_contract(
                #     'nph,f,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_B3), backend.asarray(weight_A1), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep[:, :, N_part1:, :N_part1][:, :, rank::size, :]), backend.asarray(peram_d_2pt[..., N_part1:]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, :N_part1, N_part1:][:, :, rank::size]), backend.asarray(sink_3pt[:, N_part1:, N_part1:]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C43 = cached_contract(
                #     'phf,n,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_B3), backend.asarray(weight_A1), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src]), backend.asarray(peram_s_sep[:, :, :N_part1, N_part1:]), backend.asarray(peram_d_2pt[..., N_part1:][:, :, :, rank::size]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, :N_part1]), backend.asarray(sink_3pt[:, N_part1:, N_part1:][:, rank::size, :]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C44 = cached_contract(
                #     'hfn,p,gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(weight_B3), backend.asarray(weight_A1), 
                #     backend.asarray(peram_u_compensation[:, :, N_part1:, :Nev_src][:, :, rank::size, :]), backend.asarray(peram_s_sep[:, :, N_part1:, N_part1:]), backend.asarray(peram_d_2pt[..., :N_part1]), 
                #     backend.asarray(source), backend.asarray(VdV_curr[:, N_part1:, N_part1:]), backend.asarray(sink_3pt[:, :N_part1, N_part1:][:, :, rank::size]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                A4 = A41 + A42 + A43 + A44
                # B4 = B41 + B42 + B43 + B44
                # C4 = C41 + C42 + C43 + C44
                
                # comm.Barrier()
                if rank == 0:
                    print(f'part4 use time {time.perf_counter() - st_part4}')
                
                # part5 B, B, B, B
                st_part5 = time.perf_counter()
                A_ijkl = cached_contract(
                    'manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A_ijkk = cached_contract(
                    'manb,gepf,codp,bd,Mfn,Mpp,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A_ijkk += cached_contract(
                    'manb,genf,codp,bd,Mfn,Mpn,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A_ijkk += cached_contract(
                    'manb,gehn,codp,bd,Mnn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A_ijkk += cached_contract(
                    'manb,gehh,codp,bd,Mhn,Mph,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                A_ijkk += cached_contract(
                    'manb,gehf,codn,bd,Mfn,Mnh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A_ijkk += cached_contract(
                    'manb,gehf,codf,bd,Mfn,Mfh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A_iijj = cached_contract(
                    'manb,gehn,codh,bd,Mnn,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                A_iijj += cached_contract(
                    'manb,genf,codf,bd,Mfn,Mfn,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                A_iijj += cached_contract(
                    'manb,geff,codn,bd,Mfn,Mnf,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A_ijjj = cached_contract(
                    'manb,genf,codn,bd,Mfn,Mnn,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A_ijjj += cached_contract(
                    'manb,gehh,codh,bd,Mhn,Mhh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A_ijjj += cached_contract(
                    'manb,genn,codp,bd,Mnn,Mpn,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A_ijjj += cached_contract(
                    'manb,gehn,codn,bd,Mnn,Mnh,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                A_iiii = cached_contract(
                    'manb,genn,codn,bd,Mnn,Mnn,ac,MGem,MGog->MG',
                    backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                    backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                    optimize = ['dp', 'greedy', 'auto']
                    )
                
                
                # B_ijkl = cached_contract(
                #     'gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # B_ijkk = cached_contract(
                #     'gahb,menf,codh,bd,Mfn,Mhh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # B_ijkk += cached_contract(
                #     'gahb,mehf,codp,bd,Mfh,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # B_ijkk += cached_contract(
                #     'gahb,menn,codp,bd,Mnn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # B_ijkk += cached_contract(
                #     'gahb,menh,codp,bd,Mhn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # B_ijkk += cached_contract(
                #     'gahb,menf,codn,bd,Mfn,Mnh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # B_ijkk += cached_contract(
                #     'gahb,menf,codf,bd,Mfn,Mfh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                
                # B_iijj = cached_contract(
                #     'gahb,menn,codh,bd,Mnn,Mhh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # B_iijj += cached_contract(
                #     'gahb,mehf,codf,bd,Mfh,Mfh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # B_iijj += cached_contract(
                #     'gahb,menh,codn,bd,Mhn,Mnh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # B_ijjj = cached_contract(
                #     'gahb,mehf,codh,bd,Mfh,Mhh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # B_ijjj += cached_contract(
                #     'gahb,menh,codh,bd,Mhn,Mhh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # B_ijjj += cached_contract(
                #     'gahb,mehh,codp,bd,Mhh,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # B_ijjj += cached_contract(
                #     'gahb,meff,codf,bd,Mff,Mfh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # B_iiii = cached_contract(
                #     'gahb,mehh,codh,bd,Mhh,Mhh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_u_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C_ijkl = cached_contract(
                #     'gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C_ijkk = cached_contract(
                #     'gahb,menf,codh,bd,Mfn,Mhh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C_ijkk += cached_contract(
                #     'gahb,mehf,codp,bd,Mfh,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # C_ijkk += cached_contract(
                #     'gahb,menn,codp,bd,Mnn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # C_ijkk += cached_contract(
                #     'gahb,menh,codp,bd,Mhn,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # C_ijkk += cached_contract(
                #     'gahb,menf,codn,bd,Mfn,Mnh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # C_ijkk += cached_contract(
                #     'gahb,menf,codf,bd,Mfn,Mfh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                
                # C_iijj = cached_contract(
                #     'gahb,menn,codh,bd,Mnn,Mhh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # C_iijj += cached_contract(
                #     'gahb,mehf,codf,bd,Mfh,Mfh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # C_iijj += cached_contract(
                #     'gahb,menh,codn,bd,Mhn,Mnh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C_ijjj = cached_contract(
                #     'gahb,mehf,codh,bd,Mfh,Mhh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # C_ijjj += cached_contract(
                #     'gahb,menh,codh,bd,Mhn,Mhh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # C_ijjj += cached_contract(
                #     'gahb,mehh,codp,bd,Mhh,Mph,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                # C_ijjj += cached_contract(
                #     'gahb,meff,codf,bd,Mff,Mfh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # C_iiii = cached_contract(
                #     'gahb,mehh,codh,bd,Mhh,Mhh,ac,MGem,MGog->MG',
                #     backend.asarray(peram_u_compensation[:, :, N_part1:Nev_curr, :Nev_src][..., rank::size]), backend.asarray(peram_s_sep[:, :, N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, N_part1:Nev_curr]), 
                #     backend.asarray(source[rank::size]), backend.asarray(VdV_curr[..., N_part1:Nev_curr, N_part1:Nev_curr]), backend.asarray(sink_3pt[:, N_part1:Nev_curr, N_part1:Nev_curr]),
                #     backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1]),
                #     optimize = ['dp', 'greedy', 'auto']
                #     )
                
                # comm.Barrier()
                if rank == 0:
                    print(f'part5 use time {time.perf_counter() - st_part5}')
                
                A5 = (
                    _weight[0] * A_iiii + 
                    _weight[0] * _weight[1] * (A_ijjj - 4*A_iiii + A_iijj - 3*A_iiii) + 
                    _weight[0] * _weight[1] * _weight[2] * (A_ijkk - 3 * (A_ijjj - 4*A_iiii) - 2 * (A_iijj - 3*A_iiii) - 6 * A_iiii) +
                    _weight[0] * _weight[1] * _weight[2] * _weight[3] * (A_ijkl - A_iiii - (A_ijjj - 4*A_iiii + A_iijj - 3*A_iiii) - (A_ijkk - 3 * (A_ijjj - 4*A_iiii) - 2 * (A_iijj - 3*A_iiii) - 6 * A_iiii))
                )
                
                # B5 = (
                #     _weight[0] * B_iiii + 
                #     _weight[0] * _weight[1] * (B_ijjj - 4*B_iiii + B_iijj - 3*B_iiii) + 
                #     _weight[0] * _weight[1] * _weight[2] * (B_ijkk - 3 * (B_ijjj - 4*B_iiii) - 2 * (B_iijj - 3*B_iiii) - 6 * B_iiii) +
                #     _weight[0] * _weight[1] * _weight[2] * _weight[3] * (B_ijkl - B_iiii - (B_ijjj - 4*B_iiii + B_iijj - 3*B_iiii) - (B_ijkk - 3 * (B_ijjj - 4*B_iiii) - 2 * (B_iijj - 3*B_iiii) - 6 * B_iiii))
                # )
                
                # C5 = (
                #     _weight[0] * C_iiii + 
                #     _weight[0] * _weight[1] * (C_ijjj - 4*C_iiii + C_iijj - 3*C_iiii) + 
                #     _weight[0] * _weight[1] * _weight[2] * (C_ijkk - 3 * (C_ijjj - 4*C_iiii) - 2 * (C_iijj - 3*C_iiii) - 6 * C_iiii) +
                #     _weight[0] * _weight[1] * _weight[2] * _weight[3] * (C_ijkl - C_iiii - (C_ijjj - 4*C_iiii + C_iijj - 3*C_iiii) - (C_ijkk - 3 * (C_ijjj - 4*C_iiii) - 2 * (C_iijj - 3*C_iiii) - 6 * C_iiii))
                # )
                
                compensation_all_matrix = backend.empty((3, 5, len(Mom_link), 2), dtype = complex)

                compensation_all_matrix[0, 0] = A1; compensation_all_matrix[0, 1] = A2; compensation_all_matrix[0, 2] = A3; compensation_all_matrix[0, 3] = A4; compensation_all_matrix[0, 4] = A5
                # compensation_all_matrix[1, 0] = B1; compensation_all_matrix[1, 1] = B2; compensation_all_matrix[1, 2] = B3; compensation_all_matrix[1, 3] = B4; compensation_all_matrix[1, 4] = B5
                # compensation_all_matrix[2, 0] = C1; compensation_all_matrix[2, 1] = C2; compensation_all_matrix[2, 2] = C3; compensation_all_matrix[2, 3] = C4; compensation_all_matrix[2, 4] = C5
        
                comm.Barrier()
                
                compensation_all_matrix = get_mpi_data(data = compensation_all_matrix, mdtype = 'Gather', root = ((t_src + t_sep)%Nt)%size)
                    
                if rank == ((t_src + t_sep)%Nt)%size:
                    print(compensation_all_matrix.shape)
                    compensation_all_matrix = compensation_all_matrix.sum(0)
                    corr_3pt_matrix[-1, t_sep_indx, 0, ..., t_src, ((t_src + t_sep)%Nt)//size] = compensation_all_matrix[0].sum(0)
                    
                    # corr_3pt_matrix_con_compensation[-1, t_sep_indx, 0, ..., t_src, ((t_src + t_sep)%Nt)//size] = compensation_all_matrix[0].sum(0)
                    # corr_3pt_matrix_dis_compensation[:, -1, t_sep_indx, 0, ..., t_src, ((t_src + t_sep)%Nt)//size] = compensation_all_matrix[1].sum(0)
                    # corr_3pt_matrix_dis_compensation[:, -1, t_sep_indx, 1, ..., t_src, ((t_src + t_sep)%Nt)//size] = compensation_all_matrix[2].sum(0)
                    
corr_3pt_matrix = get_mpi_data(data = corr_3pt_matrix, mdtype = 'TGather', axis = -1, root = 0)
corr_2pt_mu_matrix = get_mpi_data(data = corr_2pt_mu_matrix, mdtype = 'TGather', axis = -1, root = 0)
bubble_mu_light = get_mpi_data(data = bubble_mu_light[..., rank::size], mdtype = 'TGather', root = 0, axis = -1)
bubble_mu_strange = get_mpi_data(data = bubble_mu_strange[..., rank::size], mdtype = 'TGather', root = 0, axis = -1)

if rank == 0: 
    for Mom_indx, Mom_list in enumerate(Mom_link[:]):
        corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{Nx}x{Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_400/conf{conf_id}'
        import pathlib

        path = pathlib.Path(corr_save_path)

        if path.exists():
            print('save_path:',corr_save_path)
        
        else:
            path.mkdir(parents = True, exist_ok = True)
            print('mkdir_save_path:',corr_save_path)

        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            for mu_indx_1, mu in enumerate(mu_indx[Mom_indx]):
                backend.save(f'{corr_save_path}/corr_2pt_gamma{(mu):02d}_stout_smear_20_0.12_src100_blending.npy', corr_2pt_mu_matrix[Nev_curr_indx, 0, Mom_indx, mu_indx_1])
                backend.save(f'{corr_save_path}/corr_bubble_u-s_gamma{(mu):02d}_stout_smear_20_0.12_src400_blending.npy', bubble_mu_light[Nev_curr_indx, 0, Mom_indx, mu_indx_1] - bubble_mu_strange[Nev_curr_indx, 0, Mom_indx, mu_indx_1])
                
            for t_sep_indx, t_sep in enumerate(t_sep_list):
                for gamma_indx, gamma_list in enumerate(mu_nu_indx[Mom_indx]):
                    backend.save(f'{corr_save_path}/corr_3pt_gamma0{gamma_list[1]}0{gamma_list[0]}05_tsep{t_sep}_stout_smear_20_0.12_src100_sink{Nev_max}_curr{Nev_max}_compensation.npy', corr_3pt_matrix[Nev_curr_indx, t_sep_indx, 0, Mom_indx, gamma_indx])
if rank == 0:
    print('complete')
