import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from function_contraction import *

from opt_einsum import contract, contract_path, contract_expression

import time

def get_Nev(lst, group_sizes, M, allow_partial=True):
    """
    增强版本，带有更多控制选项
    
    参数:
        lst: 一维列表
        group_sizes: 分组大小列表
        M: 每个小组的大小
        allow_partial: 是否允许最后一个小组成立（当元素不足M个时）
    
    返回:
        嵌套列表
    """
    if not isinstance(lst, list):
        raise TypeError("lst必须是列表")
    if not isinstance(group_sizes, list):
        raise TypeError("group_sizes必须是列表")
    if M <= 0:
        raise ValueError("M必须大于0")
    
    result = []
    start_idx = 0
    
    for size in group_sizes:
        if start_idx >= len(lst):
            # 如果已经处理完所有元素，添加空列表作为占位符
            if allow_partial:
                result.append([])
            break
        
        # 获取当前大组
        end_idx = min(start_idx + size, len(lst))
        big_group = lst[start_idx:end_idx]
        
        # 将大组按M分成小组
        small_groups = []
        for i in range(0, len(big_group), M):
            small_group = big_group[i:i+M]
            # 只有当allow_partial为True或小组长度等于M时才添加
            if allow_partial or len(small_group) == M:
                small_groups.append(small_group)
            elif small_group:  # 如果allow_partial为False但有剩余元素
                # 可以选择丢弃、合并到前一组或抛出异常
                # 这里选择丢弃
                pass
        
        result.append(small_groups)
        start_idx += size
    
    # 处理剩余元素
    if start_idx < len(lst) and allow_partial:
        remaining = lst[start_idx:]
        small_groups = [remaining[i:i+M] for i in range(0, len(remaining), M)]
        result.append(small_groups)
    
    return [small_group for big_group in result for small_group in big_group]

set_backend('numpy')
backend = get_backend()

lattice_size = [32, 32, 32, 96]
grid_size = [1, 1, 1, 32]

param = creat_param(
    sink_operators = [['d^d', 'gamma_sink', 'u', 1]],
    source_operators = [['u^d', 'gamma_source', 'd', -1]]
)

param_data = param.data()
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
N_eigen = [100, 200, 400, 600, 800]
N_sum = [20, 20, 20, 20, 20]
N_extract = [10, 10, 10, 10, 10]
noise_eigen = 200

exact_eigen_name = 100
N_eigen_name = [100, 200, 400, 600, 800]
N_sum_name = [20, 20, 20, 20, 20]
N_extract_name = [10, 10, 10, 10, 10]
noise_eigen_name = 400

L = 24**3*3 - exact_eigen - sum(N_eigen)

_weight = [L/noise_eigen, (L-1)/(noise_eigen-1), (L-2)/(noise_eigen-2), (L-3)/(noise_eigen-3)]

Nbin = 1
N_stout = 20

t_sep_list = [12]

Nev_src = 100

Nev_curr_list = [400]
Nev_max = max(Nev_curr_list + [Nev_src])

link_max = 0
Mom_sink = [[0, 0, 0]]
# Mom_link = creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)
# mu_nu_indx, mu_nu_gamma, mu_indx, mu_gamma =  PFF_Mom_to_gamma(Mom_link[:], Pion_Mom = False)
Mom_link = [[0, 0, 0]]

mu_nu_indx = [[[1, 2], [2, 1]]]
mu_indx = [[1, 2]]

mu_nu_gamma = tran_indx_to_gamma(backend.asarray(mu_nu_indx))
mu_gamma = tran_indx_to_gamma(backend.asarray(mu_indx))

mu_nu_gamma = backend.asarray(mu_nu_gamma)
mu_gamma = backend.asarray(mu_gamma)

gauge_path = f'/public/group/lqcd/configurations/CLOVER/beta6.41_mu-0.2295_ms-0.2050_L32x96/beta6.41_mu-0.2295_ms-0.2050_L32x96_cfg_'

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
        ttype = ':03d',
        read_type = 'numpy', # pickle
        add_weight = True
    )
    
param_data['curr_operators'] = ['u^d', 'gamma_curr_mu', 'u']

N_operator = len(param_data['sink_operators']) * len(param_data['source_operators'])

corr_2pt_matrix = backend.zeros((N_operator, corr.Nt, corr.Gt), dtype = complex)

corr_2pt_mu_matrix = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2, corr.Nt, corr.Gt), dtype = complex)
corr_3pt_matrix = backend.zeros((len(Nev_curr_list), len(t_sep_list), N_operator, len(Mom_link), 2, corr.Nt, corr.Gt), dtype = complex)
corr_3pt_matrix_dis = backend.zeros((len(Nev_curr_list), len(t_sep_list), 2, len(Mom_link), 2, corr.Nt, corr.Gt), dtype = complex)

corr_3pt_matrix_light_compensation = backend.zeros((len(Nev_curr_list), len(t_sep_list), 2, len(Mom_link), 2, corr.Nt), dtype = complex)
corr_3pt_matrix_strange_compensation = backend.zeros((len(Nev_curr_list), len(t_sep_list), 1, len(Mom_link), 2, corr.Nt), dtype = complex)

bubble_mu_light = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2, corr.Nt), dtype = complex)
bubble_mu_strange = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2, corr.Nt), dtype = complex)

Nev_compensation_indx = get_Nev(lst = [x for x in range(Nev_max)], group_sizes = [100, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 200], M = 7, allow_partial = True)[corr.rank::corr.size]

for t_src in range(0, 1, 1):
    source = backend.asarray(get_mpi_data(sink[t_src//size, 0].conj().transpose(1, 0), mdtype = 'Bcast', root = t_src%size, axis = 0))
    
    peram_u = \
        corr.peram_V1(
            peram_name=f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy',
            peram_save_path = f'/nexdata/project/lqcd/sush/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/light/{conf_id}/t',            
            use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
            Nev_src = Nev_max,
            read_type = 'numpy',
            t_source = f'{t_src:03d}',
            # ttype = '03d'
        )
    
    peram_d = seq_peram(peram_u[..., :Nev_src])
    
    peram_s = \
        corr.peram_V1(
            peram_name=f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy',
            peram_save_path = f'/nexdata/project/lqcd/sush/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/strange/{conf_id}/t',            
            use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
            Nev_src = Nev_max,
            read_type = 'numpy',
            t_source = f'{t_src:03d}',
            # ttype = '03d'
        )

    t_curr_0_list, _, t_curr_0_list_indx = corr.get_mpi_tlist(t = [x%corr.Nt for x in range(t_src, t_src + 36, 1)], gtype = 'Scatter')

    time_2pt = time.perf_counter()
    for t_sink_list_indx, t_sink_indx in enumerate(t_curr_0_list_indx):
        t_sink = t_curr_0_list[t_sink_list_indx]
        corr_2pt_matrix[0, t_src, t_sink_indx] = contract(
            'manb,cedf,bd,fn,ac,em->', 
            backend.asarray(peram_u[t_sink_indx, ..., :Nev_src, :Nev_src]), backend.asarray(peram_d[t_sink_indx, ..., :Nev_src, :Nev_src]), 
            backend.asarray(source), backend.asarray(sink[t_sink_indx, 0]), 
            backend.asarray(gamma(5)), backend.asarray(gamma(5))
            )
        
    if rank == 0:
        print(f"complete cal 2pt of tsrc {t_src} use time {(time.perf_counter() - time_2pt):.3f} s")
    
    time_2pt_mu = time.perf_counter()
    for t_sink_list_indx, t_sink_indx in enumerate(t_curr_0_list_indx):
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            corr_2pt_mu_matrix[Nev_curr_indx, 0, :, :, t_src, t_sink_indx] = contract(
                'manb,cedf,bd,Mfn,ac,MFem->MF', 
                backend.asarray(peram_u[t_sink_indx, ..., :Nev_curr, :Nev_src]), backend.asarray(peram_d[t_sink_indx, ..., :Nev_src, :Nev_curr]), 
                backend.asarray(source), backend.asarray(VdV_link[t_sink_indx, :, 0, :Nev_curr, :Nev_curr]), 
                backend.asarray(gamma(5)), backend.asarray(mu_gamma)
                )
            
            bubble_mu_light[Nev_curr_indx, 0, :, :, t_src] = contract('menf,Mfn,MKem->MK', backend.asarray(peram_u[t_sink_indx, ..., :Nev_curr, :Nev_curr]), backend.asarray(VdV_link[t_sink_indx, :, 0, :Nev_curr, :Nev_curr]), backend.asarray(mu_gamma))
            bubble_mu_strange[Nev_curr_indx, 0, :, :, t_src] = contract('menf,Mfn,MKem->MK', backend.asarray(peram_s[t_sink_indx, ..., :Nev_curr, :Nev_curr]), backend.asarray(VdV_link[t_sink_indx, :, 0, :Nev_curr, :Nev_curr]), backend.asarray(mu_gamma))
    
    if rank == 0:
        print(f"complete cal 2pt_mu and bubble of tsrc {t_src} use time {(time.perf_counter() - time_2pt_mu):.3f} s")

    for t_sep_indx, t_sep in enumerate(t_sep_list):
        peram_u_sep = \
            corr.peram_V1(
                peram_name=f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy',
                peram_save_path = f'/nexdata/project/lqcd/sush/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/light/{conf_id}/t',
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
                
                corr_3pt_matrix[Nev_curr_indx, t_sep_indx, 0, ..., t_src, t_sink_indx] = contract(
                    'manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG', 
                    backend.asarray(peram_u[t_sink_indx, :, :, :Nev_curr, :Nev_src]), backend.asarray(peram_u_sep[t_sink_indx, :, :, :Nev_curr, :Nev_curr]), backend.asarray(peram_d_2pt[..., :Nev_src, :Nev_curr]), 
                    backend.asarray(source), backend.asarray(VdV_link[t_sink_indx, :, 0, :Nev_curr, :Nev_curr]), backend.asarray(sink_3pt[:, :Nev_curr, :Nev_curr]),
                    backend.asarray(gamma(5)), backend.asarray(mu_nu_gamma[:, :, 0]), backend.asarray(mu_nu_gamma[:, :, 1])
                    )
        if rank == 0:
            print(f"complete cal 3pt of tsrc {t_src} tsep {t_sep} use time {(time.perf_counter() - time_3pt):.3f} s")
    
        peram_s_sep = \
            corr.peram_V1(
                peram_name=f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy',
                peram_save_path = f'/nexdata/project/lqcd/sush/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/strange/{conf_id}/t',            
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
        peram_u = backend.asarray(get_mpi_data(peram_u[((t_sep + t_src)%corr.Nt)//size], mdtype = 'Bcast', root = ((t_sep + t_src)%corr.Nt)%size))
        
        if rank == 0:
            print(f"complete get mpi data tsrc {t_src} tsep {t_sep} use time {(time.perf_counter() - time_3pt_compensation_mpi_data):.3f} s")

        N_part1 = 200
        N_part2 = 200
        
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):

            
            Nev_compensation_list_A = get_Nev(lst = [x for x in range(Nev_max)], group_sizes = [Nev_max], M = 20, allow_partial = True)
            Nev_compensation_list_B = get_Nev(lst = [x for x in range(Nev_max)], group_sizes = [Nev_max], M = 40, allow_partial = True)
            
            for Nev_weight in Nev_compensation_indx:
                expr_A = contract_expression(
                    'fnph,mand,gehf,codp,Mfn,Mph,ac,MGem,MGog->MG',
                    (len(Nev_weight), Nev_curr, 20, Nev_curr), (4, 4, Nev_curr, Nev_src), (4, 4, Nev_curr, len(Nev_weight)), (4, 4, Nev_src, 20), (6, len(Nev_weight), Nev_curr), (6, 20, Nev_curr), (4, 4), (6, 2, 4, 4), (6, 2, 4, 4),
                    optimize = 'greedy'
                )
                
                expr_B = contract_expression(
                    'fnph,gahd,menf,codp,Mfn,Mph,ac,MGem,MGog->MG',
                    (len(Nev_weight), Nev_curr, Nev_curr, 40), (4, 4, 40, Nev_src), (4, 4, Nev_curr, len(Nev_weight)), (4, 4, Nev_src, Nev_curr), (6, len(Nev_weight), Nev_curr), (6, Nev_curr, 40), (4, 4), (6, 2, 4, 4), (6, 2, 4, 4),
                    optimize = 'greedy'
                )
                
                st_Nev_time = time.perf_counter()
                weight_4 = eigen.create_omega_accelerate(
                    exact = exact_eigen,
                    N_eigen = N_eigen,
                    N_sum = N_sum,
                    N_extract = N_extract,
                    noise = N_part2,
                    fixed_first_pos = Nev_weight,
                    dim = 4
                )
                
                if rank == 0:
                    print(f'weight time {time.perf_counter() - st_Nev_time:.3f} s')
                    
                
                time_3pt_compensation = time.perf_counter()
                
                # for Nev_list in Nev_compensation_list_A:
                #     # time_3pt_compensation_one = time.perf_counter()
                    
                #     # A = contract(
                #     #     'fnph,mand,gehf,codp,Mfn,Mph,ac,MGem,MGog->MG',
                #     #     weight_4[..., Nev_list, :], 
                #     #     peram_u[:, :, :Nev_curr, :Nev_src], peram_u_sep[:, :, :, Nev_weight], peram_d_2pt[..., :Nev_src, Nev_list], 
                #     #     VdV_curr[:, Nev_weight, :Nev_curr], sink_3pt[:, Nev_list, :Nev_curr],
                #     #     gamma(5), mu_nu_gamma[:, :, 0], mu_nu_gamma[:, :, 1],
                #     #     optimize = 'greedy'
                #     #     )
                #     A = expr_A(
                #         weight_4[..., Nev_list, :], 
                #         peram_u[:, :, :Nev_curr, :Nev_src], peram_u_sep[:, :, :, Nev_weight], peram_d_2pt[..., :Nev_src, Nev_list], 
                #         VdV_curr[:, Nev_weight, :Nev_curr], sink_3pt[:, Nev_list, :Nev_curr],
                #         gamma(5), mu_nu_gamma[:, :, 0], mu_nu_gamma[:, :, 1],
                #         )
                    
                #     # if rank == 0:
                #     #     print(f"{Nev_list} complete cal A 3pt compensation of tsrc {t_src} tsep {t_sep} Nev_weight {Nev_weight} use time {(time.perf_counter() - time_3pt_compensation_one):.3f} s")
                #     corr_3pt_matrix_light_compensation[Nev_curr_indx, t_sep_indx, 0, ..., t_src] = corr_3pt_matrix_light_compensation[Nev_curr_indx, t_sep_indx, 0, ..., t_src] + A

                for Nev_list in Nev_compensation_list_B:
                    # time_3pt_compensation_one = time.perf_counter()
                    # B = contract(
                    #     'fnph,gahd,menf,codp,Mfn,Mph,ac,MGem,MGog->MG',
                    #     weight_4[..., Nev_list], 
                    #     peram_u[:, :, Nev_list, :Nev_src], peram_u_sep[:, :, :, Nev_weight], peram_d_2pt[..., :Nev_src, :Nev_curr], 
                    #     VdV_curr[:, Nev_weight, :Nev_curr], sink_3pt[:, :Nev_curr, Nev_list],
                    #     gamma(5), mu_nu_gamma[:, :, 0], mu_nu_gamma[:, :, 1],
                    #     optimize = 'greedy'
                    #     )
                    B = expr_B(
                        weight_4[..., Nev_list], 
                        peram_u[:, :, Nev_list, :Nev_src], peram_u_sep[:, :, :, Nev_weight], peram_d_2pt[..., :Nev_src, :Nev_curr], 
                        VdV_curr[:, Nev_weight, :Nev_curr], sink_3pt[:, :Nev_curr, Nev_list],
                        gamma(5), mu_nu_gamma[:, :, 0], mu_nu_gamma[:, :, 1],
                        )
                    # if rank == 0:
                    #     print(f"{Nev_list} complete cal C 3pt compensation of tsrc {t_src} tsep {t_sep} Nev_weight {Nev_weight} use time {(time.perf_counter() - time_3pt_compensation_one):.3f} s")

                    # time_3pt_compensation_one = time.perf_counter()
                    # C = contract(
                    #     'fnph,gahd,menf,codp,Mfn,Mph,ac,MGem,MGog->MG',
                    #     weight_4[..., Nev_list], 
                    #     peram_u[:, :, Nev_list, :Nev_src], peram_s_sep[:, :, :, Nev_weight], peram_d_2pt[..., :Nev_src, :Nev_curr], 
                    #     VdV_curr[:, Nev_weight, :Nev_curr], sink_3pt[:, :Nev_curr, Nev_list],
                    #     gamma(5), mu_nu_gamma[:, :, 0], mu_nu_gamma[:, :, 1],
                    #     optimize = 'greedy'
                    #     )
                    C = expr_B(
                        weight_4[..., Nev_list], 
                        peram_u[:, :, Nev_list, :Nev_src], peram_s_sep[:, :, :, Nev_weight], peram_d_2pt[..., :Nev_src, :Nev_curr], 
                        VdV_curr[:, Nev_weight, :Nev_curr], sink_3pt[:, :Nev_curr, Nev_list],
                        gamma(5), mu_nu_gamma[:, :, 0], mu_nu_gamma[:, :, 1],
                        )
                    # if rank == 0:
                    #     print(f"{Nev_list} complete cal C 3pt compensation of tsrc {t_src} tsep {t_sep} Nev_weight {Nev_weight} use time {(time.perf_counter() - time_3pt_compensation_one):.3f} s")
                        
                    
                    corr_3pt_matrix_light_compensation[Nev_curr_indx, t_sep_indx, 1, ..., t_src] = corr_3pt_matrix_light_compensation[Nev_curr_indx, t_sep_indx, 1, ..., t_src] + B
                    corr_3pt_matrix_strange_compensation[Nev_curr_indx, t_sep_indx, 0, ..., t_src] = corr_3pt_matrix_strange_compensation[Nev_curr_indx, t_sep_indx, 0, ..., t_src] + C
                        
                if rank == 0:
                    print(f"complete cal 3pt compensation of tsrc {t_src} tsep {t_sep} Nev_weight {Nev_weight} use time {(time.perf_counter() - time_3pt_compensation):.3f} s")
                    
        compensation_light_data = get_mpi_data(data = corr_3pt_matrix_light_compensation[Nev_curr_indx, t_sep_indx, ..., t_src], mdtype = 'Gather', root = ((t_src + t_sep)%corr.Nt)%corr.size)
        compensation_strange_data = get_mpi_data(data = corr_3pt_matrix_strange_compensation[Nev_curr_indx, t_sep_indx, ..., t_src], mdtype = 'Gather', root = ((t_src + t_sep)%corr.Nt)%corr.size)
            
        if corr.rank == ((t_src + t_sep)%corr.Nt)%corr.size:
            print(compensation_light_data.shape)
            
            corr_3pt_matrix_light_compensation[Nev_curr_indx, t_sep_indx, ..., t_src] = compensation_light_data.sum(0)
            corr_3pt_matrix_strange_compensation[Nev_curr_indx, t_sep_indx, ..., t_src] = compensation_strange_data.sum(0)
            
for t_src in range(0, corr.Nt, 1):
    for t_sep_indx, t_sep in enumerate(t_sep_list):
        if corr.rank == ((t_src + t_sep)%corr.Nt)%corr.size:
            corr_3pt_matrix[:, t_sep_indx, 0, ..., t_src, ((t_src + t_sep)%corr.Nt)//corr.size] = corr_3pt_matrix_light_compensation[:, t_sep_indx, 0, ..., t_src]
            corr_3pt_matrix_dis[:, t_sep_indx, 0, ..., t_src, ((t_src + t_sep)%corr.Nt)//corr.size] = corr_3pt_matrix_light_compensation[:, t_sep_indx, 1, ..., t_src]
            corr_3pt_matrix_dis[:, t_sep_indx, 1, ..., t_src, ((t_src + t_sep)%corr.Nt)//corr.size] = corr_3pt_matrix_strange_compensation[:, t_sep_indx, 0, ..., t_src]

for Mom_indx, Mom_list in enumerate(Mom_link[:]):
    for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
        corr.write(
            corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
            corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_src}',
            corr_matrix = corr_2pt_matrix[0],
            Cpt = '2pt',
        )
        
        corr.write(
            corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
            corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_src}',
            corr_matrix = corr_2pt_matrix[0],
            Cpt = '2pt_all_t',
        )
        
        for mu_indx_1, mu in enumerate(mu_indx[Mom_indx]):
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}',
                corr_matrix = corr_2pt_mu_matrix[Nev_curr_indx, 0, Mom_indx, mu_indx_1],
                Cpt = '2pt_all_t',
            )
            
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'light_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr{Nev_curr}',
                corr_matrix = bubble_mu_light[Nev_curr_indx, 0, Mom_indx, mu_indx_1],
                Cpt = 'bubble',
            )
            
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'strange_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr{Nev_curr}',
                corr_matrix = bubble_mu_strange[Nev_curr_indx, 0, Mom_indx, mu_indx_1],
                Cpt = 'bubble',
            )
            
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'all_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr{Nev_curr}',
                corr_matrix = bubble_mu_light[Nev_curr_indx, 0, Mom_indx, mu_indx_1] - bubble_mu_strange[Nev_curr_indx, 0, Mom_indx, mu_indx_1],
                Cpt = 'bubble',
            )
            
        for t_sep_indx, t_sep in enumerate(t_sep_list):
            for gamma_indx, gamma_list in enumerate(mu_nu_indx[Mom_indx]):
                corr.write(
                    t_sep = t_sep,
                    corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                    corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}_compensation_2', # linkZ, conserved
                    corr_matrix = corr_3pt_matrix[Nev_curr_indx, t_sep_indx, 0, Mom_indx, gamma_indx, :].reshape(1, corr.Nt, -1), 
                    Cpt = '3pt',
                    gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                    conserved = False,
                    link_quark = 'u',
                )
                
                corr.write(
                    t_sep = t_sep,
                    corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                    corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}_compensation_2', # linkZ, conserved
                    corr_matrix = corr_3pt_matrix[Nev_curr_indx, t_sep_indx, 0, Mom_indx, gamma_indx, :].reshape(1, corr.Nt, -1), 
                    Cpt = '3pt_all_t',
                    gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                    conserved = False,
                    link_quark = 'u',
                )
                
                corr.write(
                    t_sep = t_sep,
                    corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                    corr_name = f'light_disconnect_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}_compensation_2', # linkZ, conserved
                    corr_matrix = corr_3pt_matrix_dis[Nev_curr_indx, t_sep_indx, 0, Mom_indx, gamma_indx, :].reshape(1, corr.Nt, -1), 
                    Cpt = '3pt_all_t',
                    gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                    conserved = False,
                    link_quark = 'u',
                )
                
                corr.write(
                    t_sep = t_sep,
                    corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                    corr_name = f'strange_disconnect_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}_compensation_2', # linkZ, conserved
                    corr_matrix = corr_3pt_matrix_dis[Nev_curr_indx, t_sep_indx, 1, Mom_indx, gamma_indx, :].reshape(1, corr.Nt, -1), 
                    Cpt = '3pt_all_t',
                    gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                    conserved = False,
                    link_quark = 'u',
                )

if corr.rank == 0:
    print('complete')
