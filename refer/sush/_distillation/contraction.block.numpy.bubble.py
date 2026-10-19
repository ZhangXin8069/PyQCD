import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from function_contraction import *

from opt_einsum import contract, contract_path

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

lattice_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 12]

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

Nev_src = 100

Nev_curr_list = [650]
Nev_max = max(Nev_curr_list + [Nev_src])

link_max = 0
Mom_sink = [[0, 0, 0]]
# Mom_link = creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)
# mu_nu_indx, mu_nu_gamma, mu_indx, mu_gamma =  PFF_Mom_to_gamma(Mom_link[:], Pion_Mom = False)

Mom_link = [[0, 0, 0]]

mu_nu_indx = [[[1, 2], [2, 1], [1, 2], [2, 1], [1, 2], [2, 1]]]
mu_indx = [[1, 2, 3, 4]]

mu_nu_gamma = tran_indx_to_gamma(backend.asarray(mu_nu_indx))
mu_gamma = tran_indx_to_gamma(backend.asarray(mu_indx))

mu_nu_gamma = backend.asarray(mu_nu_gamma)
mu_gamma = backend.asarray(mu_gamma)

gauge_path = f'/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_'

conf_id = sys.argv[1]



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
        add_weight = False
    )

param_data['curr_operators'] = ['u^d', 'gamma_curr_mu', 'u']

N_operator = len(param_data['sink_operators']) * len(param_data['source_operators'])
corr_2pt_mu_matrix = backend.zeros((2, N_operator, len(Mom_link), 4, corr.Nt, corr.Gt), dtype = complex)

bubble_mu_light = backend.zeros((2, N_operator, len(Mom_link), 4, corr.Nt), dtype = complex)
bubble_mu_strange = backend.zeros((2, N_operator, len(Mom_link), 4, corr.Nt), dtype = complex)



weight_2 = corr.eigen.create_omega_accelerate(
    exact = exact_eigen,
    N_eigen = N_eigen,
    N_sum = [x // 2 for x in N_sum],
    N_extract = [x // 2 for x in N_extract],
    noise = 250,
    dim = 2
)
import random

for t_src in range(0, corr.Nt, 1):
    Nev_mu = [x for x in range(50)] + [10 * y + x + 50 for y in range(10) for x in random.sample(range(10), k = 5)] + [x for x in random.sample(range(150, 650, 1), k = 250)]
    Nev_mu.sort()
    Nev_nu = [x for x in range(50)] + [x for x in range(Nev_max) if x not in Nev_mu]
    
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
    
    peram_s = \
        corr.peram_V1(
            peram_name=f'_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_smear_20_0.12.npy',
            peram_save_path = f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/strange/{conf_id}/t',            
            use_eigen = [exact_eigen, sum(N_sum), noise_eigen],
            Nev_src = Nev_max,
            read_type = 'numpy',
            t_source = f'{t_src:03d}',
            # ttype = '03d'
        )

    t_curr_0_list, _, t_curr_0_list_indx = corr.get_mpi_tlist(t = [x%corr.Nt for x in range(t_src, t_src + 36, 1)], gtype = 'Scatter')
    

    source = backend.asarray(get_mpi_data(sink[t_src//size, 0].conj().transpose(1, 0), mdtype = 'Bcast', root = t_src%size, axis = 0))
    
    time_2pt_mu = time.perf_counter()
    for t_sink_list_indx, t_sink_indx in enumerate(t_curr_0_list_indx):

        corr_2pt_mu_matrix[0, 0, :, :, t_src, t_sink_indx] = contract(
            'manb,cedf,bd,Mfn,ac,MFem->MF', 
            backend.asarray(peram_u[t_sink_indx, :, :, :, :Nev_src][..., Nev_mu, :]), backend.asarray(peram_d[t_sink_indx, :, :, :Nev_src, :][..., Nev_mu]), 
            backend.asarray(source), backend.asarray(ArraySlicer(VdV_link[t_src//size, :, 0, :, :]).slice(dims = [-1, -2], indices = [Nev_mu, Nev_mu])* weight_2) , 
            backend.asarray(gamma(5)), backend.asarray(mu_gamma)
            )
        
        corr_2pt_mu_matrix[1, 0, :, :, t_src, t_sink_indx] = contract(
            'manb,cedf,bd,Mfn,ac,MFem->MF', 
            backend.asarray(peram_u[t_sink_indx, :, :, :, :Nev_src][..., Nev_nu, :]), backend.asarray(peram_d[t_sink_indx, :, :, :Nev_src, :][..., Nev_nu]), 
            backend.asarray(source), backend.asarray(ArraySlicer(VdV_link[t_src//size, :, 0, :, :]).slice(dims = [-1, -2], indices = [Nev_nu, Nev_nu])* weight_2) , 
            backend.asarray(gamma(5)), backend.asarray(mu_gamma)
            )

    bubble_mu_light[0, 0, :, :, t_src] = contract(
        'menf,Mfn,MKem->MK', 
        backend.asarray(ArraySlicer(peram_u[t_src//size, ..., :, :]).slice(dims = [-1, -2], indices = [Nev_mu, Nev_mu])), 
        backend.asarray(ArraySlicer(VdV_link[t_src//size, :, 0, :, :]).slice(dims = [-1, -2], indices = [Nev_mu, Nev_mu])* weight_2) , 
        backend.asarray(mu_gamma)
        )
    
    bubble_mu_strange[0, 0, :, :, t_src] = contract(
        'menf,Mfn,MKem->MK', 
        backend.asarray(ArraySlicer(peram_s[t_src//size, ..., :, :]).slice(dims = [-1, -2], indices = [Nev_mu, Nev_mu])), 
        backend.asarray(ArraySlicer(VdV_link[t_src//size, :, 0, :, :]).slice(dims = [-1, -2], indices = [Nev_mu, Nev_mu])* weight_2) , 
        backend.asarray(mu_gamma)
        )
    
    bubble_mu_light[1, 0, :, :, t_src] = contract(
        'menf,Mfn,MKem->MK', 
        backend.asarray(ArraySlicer(peram_u[t_src//size, ..., :, :]).slice(dims = [-1, -2], indices = [Nev_nu, Nev_nu])), 
        backend.asarray(ArraySlicer(VdV_link[t_src//size, :, 0, :, :]).slice(dims = [-1, -2], indices = [Nev_nu, Nev_nu])* weight_2) , 
        backend.asarray(mu_gamma)
        )
    
    bubble_mu_strange[1, 0, :, :, t_src] = contract(
        'menf,Mfn,MKem->MK', 
        backend.asarray(ArraySlicer(peram_s[t_src//size, ..., :, :]).slice(dims = [-1, -2], indices = [Nev_nu, Nev_nu])), 
        backend.asarray(ArraySlicer(VdV_link[t_src//size, :, 0, :, :]).slice(dims = [-1, -2], indices = [Nev_nu, Nev_nu])* weight_2) , 
        backend.asarray(mu_gamma)
        )
    
    if rank == 0:
        print(f"complete cal 2pt_mu and bubble of tsrc {t_src} use time {(time.perf_counter() - time_2pt_mu):.3f} s")

for Mom_indx, Mom_list in enumerate(Mom_link[:]):
    for mu_indx_1, mu in enumerate(mu_indx[Mom_indx]):
        corr.write(
            corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_bubble/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{350}/conf{conf_id}', 
            corr_name = f'gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{350}_mu',
            corr_matrix = corr_2pt_mu_matrix[0, 0, Mom_indx, mu_indx_1],
            Cpt = '2pt_all_t',
        )
        
        corr.write(
            corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_bubble/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{350}/conf{conf_id}', 
            corr_name = f'light_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr{350}_mu',
            corr_matrix = bubble_mu_light[0, 0, Mom_indx, mu_indx_1],
            Cpt = 'bubble',
        )
        
        corr.write(
            corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_bubble/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{350}/conf{conf_id}', 
            corr_name = f'strange_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr{350}_mu',
            corr_matrix = bubble_mu_strange[0, 0, Mom_indx, mu_indx_1],
            Cpt = 'bubble',
        )
        
        corr.write(
            corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_bubble/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{350}/conf{conf_id}', 
            corr_name = f'all_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr{350}_mu',
            corr_matrix = bubble_mu_light[0, 0, Mom_indx, mu_indx_1] - bubble_mu_strange[0, 0, Mom_indx, mu_indx_1],
            Cpt = 'bubble',
        )
        
        corr.write(
            corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_bubble/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{350}/conf{conf_id}', 
            corr_name = f'gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_src{Nev_src}_sink{350}_nu',
            corr_matrix = corr_2pt_mu_matrix[1, 0, Mom_indx, mu_indx_1],
            Cpt = '2pt_all_t',
        )
        
        corr.write(
            corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_bubble/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{350}/conf{conf_id}', 
            corr_name = f'light_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr{350}_nu',
            corr_matrix = bubble_mu_light[1, 0, Mom_indx, mu_indx_1],
            Cpt = 'bubble',
        )
        
        corr.write(
            corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_bubble/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{350}/conf{conf_id}', 
            corr_name = f'strange_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr{350}_nu',
            corr_matrix = bubble_mu_strange[1, 0, Mom_indx, mu_indx_1],
            Cpt = 'bubble',
        )
        
        corr.write(
            corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_bubble/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{350}/conf{conf_id}', 
            corr_name = f'all_gamma{(mu):02d}_e{exact_eigen_name}_s{N_eigen_name}_{N_sum_name}_{N_extract_name}_BI_n{noise_eigen_name}_stout_smear_{N_stout}_0.12_curr{350}_nu',
            corr_matrix = bubble_mu_light[1, 0, Mom_indx, mu_indx_1] - bubble_mu_strange[1, 0, Mom_indx, mu_indx_1],
            Cpt = 'bubble',
        )