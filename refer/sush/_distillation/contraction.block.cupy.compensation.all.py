import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from function_contraction_new_2 import *

import numpy
from opt_einsum import contract

from pyquda_utils import io

import time

set_backend('cupy')
backend = get_backend()

grid_size = [1, 1, 1, 1]

lattice_size = [24, 24, 24, 72]

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

Nbin = 1
N_stout = 20

t_sep_list = [12]

Nev_src = 100

Nev_curr_list = [650]
Nev_max = max(Nev_curr_list + [Nev_src])

link_max = 0
Mom_sink = [[0, 0, 0]]
Mom_link = creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)
mu_nu_indx, mu_nu_gamma, mu_indx, mu_gamma =  PFF_Mom_to_gamma(Mom_link[:], Pion_Mom = False)

gauge_path = f'/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_'

conf_id = sys.argv[1]

def nested_group_enhanced(lst, group_sizes, M, allow_partial=True):
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
param.curr_operators(['u^d', 'gamma_curr', 'u'])
corr = distillation_func(
    conf_id = conf_id,
    param = param.data(),

    lattice_size = lattice_size, # Nx, Ny, Nz, Nt 
    
    Nev_src = Nev_src,

    conserved = False,
    backend = backend.__name__
)

eigen = corr_eigvecs(Nx = corr.Nx, backend = backend.__name__)


weight_2 = corr.eigen.create_omega(
    exact = exact_eigen,
    N_eigen = N_eigen,
    N_sum = N_sum,
    N_extract = N_extract,
    noise = noise_eigen,
    dim = 2
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
        add_weight = True
    )

param.gamma_projection((gamma(0) + gamma(4))/2)
param.source(sink[:, 0].conj().transpose(0, 2, 1))
param.Nev_src(Nev_src)
param.gamma_source(gamma(5))
param.VdV_curr(VdV_link[:, :], curr_id = 0)

N_operator = len(param.param['sink_operators']) * len(param.param['source_operators'])

corr_2pt_matrix = backend.zeros((N_operator, corr.Nt, corr.Gt), dtype = complex)

corr_2pt_mu_matrix = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2, corr.Nt, corr.Gt), dtype = complex)
corr_3pt_matrix = backend.zeros((len(Nev_curr_list), len(t_sep_list), N_operator, len(Mom_link), 2, 2 * link_max + 1, corr.Nt, corr.Gt), dtype = complex)
corr_3pt_matrix_dis = backend.zeros((len(Nev_curr_list), len(t_sep_list), 2, len(Mom_link), 2, 2 * link_max + 1, corr.Nt, corr.Gt), dtype = complex)

corr_3pt_matrix_light_compensation = backend.zeros((len(Nev_curr_list), len(t_sep_list), 2, len(Mom_link), 2, 2 * link_max + 1, corr.Nt), dtype = complex)
corr_3pt_matrix_strange_compensation = backend.zeros((len(Nev_curr_list), len(t_sep_list), 1, len(Mom_link), 2, 2 * link_max + 1, corr.Nt), dtype = complex)

bubble_mu_light = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2 * link_max + 1, 2, corr.Nt), dtype = complex)
bubble_mu_strange = backend.zeros((len(Nev_curr_list), N_operator, len(Mom_link), 2 * link_max + 1, 2, corr.Nt), dtype = complex)

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
    
    peram_s = \
        corr.peram_V1(
            peram_name=[[f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy']],
            peram_save_path = [[f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/strange/{conf_id}/t']],            
            use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
            Nev_src = Nev_max,
            read_type = 'numpy',
            t_load_type = 'slice',
            t = f'{t_src:03d}',
            # ttype = '03d'
        )[0]
        
    param.peram(peram_s, sep = False , quark = 's', curr = True, curr_id = 0, use_2pt = False)

    param.gamma_sink(gamma(5))
    param.Nev_sink(Nev_src)
    param.sink(sink[:, 0])
    
    corr_2pt_matrix[..., t_src, :] = (
        corr.corr_all(
            param = param.data(),
            Cpt = '2pt',
            t_range = range(0, 30, 1),
            contraction_same_indx = [[-1], [-1], [-1], [-1], [-1], [-1]]
        )
    )

    for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
        param.Nev_sink(Nev_curr)
        param.Nev_curr(Nev_curr, curr_id = 0)

        param.gamma_sink(mu_gamma)
        param.sink(VdV_link[:, :, 0])
        
        corr_2pt_mu_matrix[Nev_curr_indx, ..., t_src, :] = (
            corr.corr_all(
                param = param.data(),
                Cpt = '2pt',
                t_range = range(0, 30, 1),
                contraction_same_indx = [[-1], [-1], [0], [0, 1], [-1], [-1]]
            )
        )

        param.gamma_curr(mu_gamma)
        
        bubble_mu_light[Nev_curr_indx, ..., t_src] = (
            corr.corr_all(
                param = param.data(),
                Cpt = 'bubble',
                contraction_same_indx = [[-1], [-1], [-1], [-1], [0], [0]],
            )
        )
        
        param.curr_operators(['s^d', 'gamma_curr', 's'])
        
        bubble_mu_strange[Nev_curr_indx, ..., t_src] = (
            corr.corr_all(
                param = param.data(),
                Cpt = 'bubble',
                contraction_same_indx = [[-1], [-1], [-1], [-1], [0], [0]],
            )
        )
        
        param.curr_operators(['u^d', 'gamma_curr', 'u'])
        
    param.gamma_sink(mu_nu_gamma[:, :, 1])
    param.gamma_curr(mu_nu_gamma[:, :, 0])

    for t_sep_indx, t_sep in enumerate(t_sep_list):
        param.t_sep(t_sep)
        peram_u_sep = \
            corr.peram_V1(
                peram_name=[[f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy']],
                peram_save_path = [[f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t']],
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = Nev_max,
                read_type = backend.__name__,
                t_load_type = 'slice',
                t = f'{((t_sep + t_src)%corr.Nt):03d}'
            )[0]
    
        param.peram(peram_u_sep, sep = True , quark = 'u', curr = True, curr_id = 0, use_2pt = False)
        
        peram_s_sep = \
            corr.peram_V1(
                peram_name=[[f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy']],
                peram_save_path = [[f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/strange/{conf_id}/t']],            
                use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
                Nev_src = Nev_max,
                read_type = 'numpy',
                t_load_type = 'slice',
                t = f'{((t_sep + t_src)%corr.Nt):03d}',
                # ttype = '03d'
            )[0]
            
        param.peram(peram_s_sep, sep = True , quark = 's', curr = True, curr_id = 0, use_2pt = False)
        
        # compensation
        param.sink(VdV_link[:, :, 0].conj().transpose(0, 1, 3, 2) / weight_2)
        param.VdV_curr(VdV_link[:, :] / weight_2, curr_id = 0)
            
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            param.Nev_sink(Nev_curr)
            param.Nev_curr(Nev_curr)
            
            Nev_compensation_indx = nested_group_enhanced(lst = [x for x in range(Nev_max)], group_sizes = [50, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, Nev_curr - 150], M = 3, allow_partial = True)[corr.rank::corr.size]

            for Nev_weight in Nev_compensation_indx:
                st_Nev_time = time.perf_counter()

                weight_4 = eigen.create_omega_accelerate(
                    exact = exact_eigen,
                    N_eigen = N_eigen,
                    N_sum = N_sum,
                    N_extract = N_extract,
                    noise = Nev_curr - 150,
                    fixed_first_pos = Nev_weight,
                    dim = 4
                )
                
                print(f'weight time {time.perf_counter() - st_Nev_time:.3f} s')
                
                corr.comm.Barrier()

                param.weight_matrix(weight_matrix = weight_4, Ndim = 4)
                
                param.curr_operators(['u^d', 'gamma_curr', 'u'])
                A = corr.corr_all(
                    param = param.data(),
                    Cpt = '3pt',
                    t_range = range(0, 1, 1),
                    contraction_same_indx = [[-1], [-1], [0], [0, 1], [0], [0, 1]],
                    compensation = True,
                    Nev_compensation = Nev_weight
                )
                
                param.curr_operators(['s^d', 'gamma_curr', 's'])
                B = corr.corr_all(
                    param = param.data(),
                    Cpt = '3pt',
                    t_range = range(0, 1, 1),
                    contraction_same_indx = [[-1], [-1], [0], [0, 1], [0], [0, 1]],
                    compensation = True,
                    Nev_compensation = Nev_weight
                )
                
                if corr.rank == ((t_src + t_sep)%corr.Nt)%corr.size:
                    corr_3pt_matrix_light_compensation[Nev_curr_indx, t_sep_indx, ..., t_src] = corr_3pt_matrix_light_compensation[Nev_curr_indx, t_sep_indx, ..., t_src] + A
                    corr_3pt_matrix_strange_compensation[Nev_curr_indx, t_sep_indx, ..., t_src] = corr_3pt_matrix_strange_compensation[Nev_curr_indx, t_sep_indx, ..., t_src] + B

        param.curr_operators(['u^d', 'gamma_curr', 'u'])
        param.sink(VdV_link[:, :, 0].conj().transpose(0, 1, 3, 2))
        param.VdV_curr(VdV_link[:, :], curr_id = 0)
        for Nev_curr_indx, Nev_curr in enumerate(Nev_curr_list):
            param.Nev_sink(Nev_curr)
            param.Nev_curr(Nev_curr)

            corr_3pt_matrix[Nev_curr_indx, t_sep_indx, ..., t_src, :] = (
                corr.corr_all(
                    param = param.data(),
                    Cpt = '3pt',
                    t_range = range(0, 30, 1),
                    contraction_same_indx = [[-1], [-1], [0], [0, 1], [0], [0, 1]],
                )
            )

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
                corr_matrix = bubble_mu_light[Nev_curr_indx, 0, Mom_indx, 0, mu_indx_1],
                Cpt = 'bubble',
            )
            
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'strange_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr{Nev_curr}',
                corr_matrix = bubble_mu_strange[Nev_curr_indx, 0, Mom_indx, 0, mu_indx_1],
                Cpt = 'bubble',
            )
            
            corr.write(
                corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                corr_name = f'all_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr{Nev_curr}',
                corr_matrix = bubble_mu_light[Nev_curr_indx, 0, Mom_indx, 0, mu_indx_1] - bubble_mu_strange[Nev_curr_indx, 0, Mom_indx, 0, mu_indx_1],
                Cpt = 'bubble',
            )
            
        for t_sep_indx, t_sep in enumerate(t_sep_list):
            for gamma_indx, gamma_list in enumerate(mu_nu_indx[Mom_indx]):
                corr.write(
                    t_sep = t_sep,
                    corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                    corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}_compensation', # linkZ, conserved
                    corr_matrix = corr_3pt_matrix[Nev_curr_indx, t_sep_indx, 0, Mom_indx, gamma_indx, :], 
                    Cpt = '3pt',
                    gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                    conserved = corr.conserved,
                    link_quark = 'u',
                )
                
                corr.write(
                    t_sep = t_sep,
                    corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                    corr_name = f'e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}_compensation', # linkZ, conserved
                    corr_matrix = corr_3pt_matrix[Nev_curr_indx, t_sep_indx, 0, Mom_indx, gamma_indx, :], 
                    Cpt = '3pt_all_t',
                    gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                    conserved = corr.conserved,
                    link_quark = 'u',
                )
                
                corr.write(
                    t_sep = t_sep,
                    corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                    corr_name = f'light_disconnect_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}_compensation', # linkZ, conserved
                    corr_matrix = corr_3pt_matrix_dis[Nev_curr_indx, t_sep_indx, 0, Mom_indx, gamma_indx, :], 
                    Cpt = '3pt_all_t',
                    gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                    conserved = corr.conserved,
                    link_quark = 'u',
                )
                
                corr.write(
                    t_sep = t_sep,
                    corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_curr}/conf{conf_id}', 
                    corr_name = f'strange_disconnect_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{Nev_curr}_curr{Nev_curr}_compensation', # linkZ, conserved
                    corr_matrix = corr_3pt_matrix_dis[Nev_curr_indx, t_sep_indx, 1, Mom_indx, gamma_indx, :], 
                    Cpt = '3pt_all_t',
                    gamma = f'0{gamma_list[1]}0{gamma_list[0]}05',
                    conserved = corr.conserved,
                    link_quark = 'u',
                )

if corr.rank == 0:
    print('complete')
