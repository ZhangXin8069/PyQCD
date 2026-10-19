import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction import *

from opt_einsum import contract, contract_path

import time
from typing import Literal

def Readin_gauge(conf_file, lattice_size):
    '''
    用于读取规范场
    '''
    backend = get_backend()
    Nz, Ny, Nx, Nt = lattice_size
    
    f = open("%s" % conf_file, "rb")
    gauge = backend.fromfile(f, dtype=">f8")
    gauge = backend.array(gauge)

    gauge = gauge.reshape(Nt, Nx, Nx, Nx, 4, 3, 3, 2)
    gauge = gauge[..., 0] + gauge[..., 1] * 1j
    f.close()

    return gauge.transpose(4, 0, 1, 2, 3, 5, 6) # 方向，T, Z, Y, X, C, C

def transpose_gauge(gauge_link, link_indx:int, link_dir:Literal["Z", "Y", "X"]):
    '''
    用于构建N格的link
    '''
    backend = get_backend()
    
    if link_dir=='Z':   axis_dir=1
    elif link_dir=='Y':   axis_dir=2
    elif link_dir=='X':   axis_dir=3
    # 选取特定的方向
    _gauge_link = gauge_link[3 - axis_dir]
    
    # 初始化规范场为一个单位阵
    gauge_link_rolled = backend.zeros_like(_gauge_link)
    gauge_link_rolled[:] = backend.identity(3, dtype=complex)
    
    if link_indx < 0:
        for link_indx_2 in range(abs(link_indx)):
            gauge_link_rolled = gauge_link_rolled @ backend.roll(_gauge_link, abs(link_indx)-link_indx_2, axis = axis_dir)

        gauge_link_rolled = gauge_link_rolled.transpose(0, 1, 2, 3, 5, 4).conj()
        
    elif link_indx > 0:
        for link_indx_2 in range(abs(link_indx)):
            gauge_link_rolled = gauge_link_rolled @ backend.roll(_gauge_link, -1* link_indx_2, axis = axis_dir)
            
    return gauge_link_rolled

conf_id = sys.argv[1]

import numpy as np
# 初始化numpy or cupy
set_backend('numpy')
backend = get_backend()

# 格子数据和 mpi 
lattice_size = [24, 24, 24, 72]
# grid size 并不代表将某一个维度进行mpi初始化，仅仅代表初始化多少个进程
grid_size = [24, 1, 1, 1]
mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)

conf_id = sys.argv[1]

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size

t_sep = 10
link_max = 5
Nev_src = 50

# 将空间分给不同的rank进行计算, 这个程序以计算Z为例子
Z_list = [x for x in range(Nz)][rank%Nz:rank%Nz+1]
Y_list = [x for x in range(Ny)][rank//Nz * Ny//(size//Nz):(rank//Nz + 1) * Ny//(size//Nz)]
X_list = [x for x in range(Nx)]

fun_eigen = corr_eigvecs(Nx = Nx, backend = backend)

# sink Mom
Mom_sink = creat_mom_list(Mom = [0, 0, 0], fix_Q2 = True) # P = [0, 0, 0]
phase_exp_3pt = backend.empty((len(Mom_sink), Nx, Nx, Nx), dtype = complex) # e^{-ipx}
for i in range(len(Mom_sink)):
    phase_exp_3pt[i] = backend.asarray(fun_eigen.phase_exp_3pt(Mom = Mom_sink[i]))
    
# 把时间分给不同的rank。
# 第一个输出表示具体的时间进行分配，按照rank::size的方式进行。
# 第二个输出表示 具体在哪一个rank下进行计算
# 第三个输出表示，这个时间对应的index是什么
t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'Scatter')

# 进行VVV的计算
sink_VVV = backend.zeros((len(t_rank), len(Mom_sink), Nev_src, Nev_src, Nev_src), dtype = complex)
st_eigen = time.perf_counter()
for t_src_indx, t_src in enumerate(t_rank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/{conf_id}_t{t_src:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')
    
    for Mom_indx in range(len(Mom_sink)):
        sink_VVV[t_src_indx, Mom_indx] = fun_eigen.Mom_VVV_sink_t_3(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src])
        
if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')

#读取规范场
if rank == 0:
    gauge_link = Readin_gauge(
        f'/nexdata/project/lqcd/sush/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{conf_id}.lime.contents/msg02.rec04.ildg-binary-data', 
        lattice_size = lattice_size
        )

else:
    gauge_link = None

# 将场复制到所有的rank中
gauge_link = get_mpi_data(data = gauge_link, mdtype = 'Bcast', root = 0)

# 定义质子投影算符
# projection = (gamma(0) + gamma(4))/2

corr_2pt_matrix = backend.zeros((2, len(Mom_sink), Nt, Nt//size), dtype = complex)
gamma_curr = backend.asarray([gamma(0), gamma(1), gamma(2), gamma(3), gamma(4), gamma(5), 
                              gamma(5)@gamma(1), gamma(5)@gamma(2), gamma(5)@gamma(3), 
                              gamma(1)@gamma(2)])

projection = backend.asarray([
    (gamma(0) + gamma(4))/2, (gamma(0) + gamma(4))/2, (gamma(0) + gamma(4))/2, (gamma(0) + gamma(4))/2, (gamma(0) + gamma(4))/2, (gamma(0) + gamma(4))/2, 
    (gamma(0) + gamma(4))@gamma(5)@gamma(1)/2, (gamma(0) + gamma(4))@gamma(5)@gamma(2)/2, (gamma(0) + gamma(4))@gamma(5)@gamma(3)/2, 
    (gamma(0) + gamma(4))@gamma(5)@gamma(3)/2])

corr_3pt_matrix_d = backend.zeros((len(gamma_curr), 2, len(Mom_sink), 2 * link_max + 1, Nt, Nt), dtype = complex)
corr_3pt_matrix_u = backend.zeros((len(gamma_curr), 4, len(Mom_sink), 2 * link_max + 1, Nt, Nt), dtype = complex)

for t_src in range(Nt):
    st_mpi_time = time.perf_counter()
    
    # 将source的VVV进行分配，所有rank都一样
    if rank == t_src%size:
        src_VVV = sink_VVV[t_src//size]
        
    else:
        src_VVV = None
        
    src_VVV = get_mpi_data(data = src_VVV, mdtype = 'Bcast', root = t_src%size).conj()

    _, _, t_sink_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + Nt//2, 1), gtype = 'TScatter')

    # 读取传播子
    if rank == 0:
        peram_u_src = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{t_src:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy'
            )[..., :Nev_src, :Nev_src]

    else:
        peram_u_src = None
        
    # peram_u_src = get_mpi_data(data = peram_u_src, mdtype = 'TScatter', root = 0, axis = 0)
    
    # 计算两点函数
    # for t_sink in t_sink_list_indx:
    #     corr_2pt_matrix[0, :, t_src, t_sink] = -1.0 * contract(
    #         'gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M',
    #         peram_u_src[t_sink], peram_u_src[t_sink], peram_u_src[t_sink], 
    #         gamma(7), gamma(7),
    #         src_VVV, sink_VVV[t_sink],
    #         projection
    #     )
        
    #     corr_2pt_matrix[1, :, t_src, t_sink] = 1.0 * contract(
    #         'gahb,menf,ocpd,ac,og,Mbdf,Mnph,em->M',
    #         peram_u_src[t_sink], peram_u_src[t_sink], peram_u_src[t_sink], 
    #         gamma(7), gamma(7),
    #         src_VVV, sink_VVV[t_sink],
    #         projection
    #     )
    if rank == ((t_src + t_sep)%Nt)%size:
        sink_3pt_VVV = sink_VVV[(t_src + t_sep)%Nt//size]
        
    else:
        sink_3pt_VVV = None
        
        
    sink_3pt_VVV = get_mpi_data(data = sink_3pt_VVV, mdtype = 'Bcast', root = ((t_src + t_sep)%Nt)%size)
    # peram_u_src_3pt = get_mpi_data(data = peram_u_src[(t_src + t_sep)%Nt//size], mdtype = 'Bcast', root = ((t_src + t_sep)%Nt)%size)
    
    if rank == 0:
        peram_u_src_3pt = peram_u_src[(t_src + t_sep)%Nt]
        
    else:
        peram_u_src_3pt = None
        
    peram_u_src_3pt = get_mpi_data(data = peram_u_src_3pt, mdtype = 'Bcast', root = 0)
    
    
    t_curr_list = [x%Nt for x in range(t_src + 3, t_src + 8, 1)]
    
    # 读取 general distillation propagator
    prop_src = np.memmap(
        # f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{t_src:03d}_gen_e50_stout_smear_20_0.12.mmap', 
        f'/beegfs/group-bak/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{t_src:03d}_gen_e50_stout_smear_20_0.12.mmap', 
        dtype=complex, mode='r', shape=(Nz, Ny, Nx, Nc, Nt, 4, 4, 50)
        )
    
    prop_src_load = ArraySlicer(prop_src).slice(dims = [0, 1, 2, 4], indices = [Z_list, Y_list, X_list, t_curr_list]).copy()
    prop_sep = np.memmap(
        # f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{(t_src + t_sep)%Nt:03d}_gen_e50_stout_smear_20_0.12.mmap', 
        f'/beegfs/group-bak/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{(t_src + t_sep)%Nt:03d}_gen_e50_stout_smear_20_0.12.mmap', 
        dtype=complex, mode='r', shape=(Nz, Ny, Nx, Nc, Nt, 4, 4, 50)
        )
    
    prop_sep_load = ArraySlicer(prop_sep).slice(dims = [0, 1, 2, 4], indices = [Z_list, Y_list, X_list, t_curr_list]).copy()
    prop_sep_load = contract('ab,...bce,cd->...dae', gamma(5), prop_sep_load.conj(), gamma(5))
    
    if rank == 0:
        print(f'load prop and peram use time {(time.perf_counter() - st_mpi_time):.3f} s of tsrc {t_src}')
    
    for link_indx in [-4, -3, 0, 3, 4]:
        st_cal_link_time = time.perf_counter()
        # 进行流的收缩
        peram_curr = cached_contract(
            'zyxcTabn,zyxgTdef,Gae,Tzyxgc->TGdbfn', 
            backend.roll(prop_src_load, shift = -1 * link_indx, axis = 0), prop_sep_load, 
            gamma_curr, 
            transpose_gauge(ArraySlicer(gauge_link).slice(dims = [1, 2, 3, 4], indices = [t_curr_list, Z_list, Y_list, X_list]), link_indx = link_indx, link_dir = 'X'),
            optimize = ['auto', 'dp', 'greedy']
        )
            
        # 守恒流的计算
        # peram_d_curr_B = contract(
        #     'zyxcTabn,zyxgTdef,ae,Tzyxgc->Tdbfn', backend.roll(prop_src_load, shift = -1, axis = 4), prop_sep_load, gamma(4) - gamma(0), ArraySlicer(gauge_link[3]).slice(dims = [0, 1, 2, 3], indices = [t_curr_list, Z_list, Y_list, X_list])
        #     # 'zyxctabn,zyxCtdef,ae,tzyxcC->tdbfn', backend.roll(prop_src_load, shift = 1 * link_indx, axis = 0), prop_sep_load, gamma(4), transpose_gauge(gauge_link[:, t_curr_list], link_indx = link_indx, link_dir = 'Z')
        # )/2
        
        # peram_d_curr_F = contract(
        #     'zyxcTabn,zyxgTdef,ae,Tzyxgc->Tdbfn', prop_src_load, backend.roll(prop_sep_load, shift = -1, axis = 4), gamma(4) + gamma(0), ArraySlicer(gauge_link[3]).slice(dims = [0, 1, 2, 3], indices = [t_curr_list, Z_list, Y_list, X_list]).conj().transpose(0, 1, 2, 3, 5, 4)
        #     # 'zyxctabn,zyxCtdef,ae,tzyxcC->tdbfn', backend.roll(prop_src_load, shift = 1 * link_indx, axis = 0), prop_sep_load, gamma(4), transpose_gauge(gauge_link[:, t_curr_list], link_indx = link_indx, link_dir = 'Z')
        # )/2
        
        # peram_curr = peram_d_curr_B + peram_d_curr_F
        
        
        for t_curr_indx, t_curr in enumerate(t_curr_list):
            corr_3pt_matrix_d[:, 0, :, link_indx, t_src, t_curr] = 1.0 * cached_contract(
                'Geifj,ckdl,agbh,ce,gi,Mbdf,Mhjl,Gka->GM',
                peram_curr[t_curr_indx], peram_u_src_3pt, peram_u_src_3pt, 
                gamma(7), gamma(7),
                sink_3pt_VVV, src_VVV, 
                projection, optimize = ['auto', 'dp', 'greedy']
            )
            
            corr_3pt_matrix_d[:, 1, :, link_indx, t_src, t_curr] = -1.0 * cached_contract(
                'Geifj,akbl,cgdh,ce,gi,Mbdf,Mhjl,Gka->GM',
                peram_curr[t_curr_indx], peram_u_src_3pt, peram_u_src_3pt, 
                gamma(7), gamma(7),
                sink_3pt_VVV, src_VVV, 
                projection, optimize = ['auto', 'dp', 'greedy']
            )
            
            corr_3pt_matrix_u[:, 0, :, link_indx, t_src, t_curr] = -1.0 * cached_contract(
                'eifj,Gakbl,cgdh,ce,gi,Mbdf,Mhjl,Gka->GM',
                peram_u_src_3pt, peram_curr[t_curr_indx], peram_u_src_3pt, 
                gamma(7), gamma(7),
                sink_3pt_VVV, src_VVV,
                projection, optimize = ['auto', 'dp', 'greedy']
            )
            
            corr_3pt_matrix_u[:, 1, :, link_indx, t_src, t_curr] = +1.0 * cached_contract(
                'eifj,Gagbh,ckdl,ce,gi,Mbdf,Mhjl,Gka->GM',
                peram_u_src_3pt, peram_curr[t_curr_indx], peram_u_src_3pt, 
                gamma(7), gamma(7),
                sink_3pt_VVV, src_VVV,
                projection, optimize = ['auto', 'dp', 'greedy']
            )
            
            corr_3pt_matrix_u[:, 2, :, link_indx, t_src, t_curr] = +1.0 * cached_contract(
                'eifj,Gckdl,agbh,ce,gi,Mbdf,Mhjl,Gka->GM',
                peram_u_src_3pt, peram_curr[t_curr_indx], peram_u_src_3pt, 
                gamma(7), gamma(7),
                sink_3pt_VVV, src_VVV,
                projection, optimize = ['auto', 'dp', 'greedy']
            )
            
            corr_3pt_matrix_u[:, 3, :, link_indx, t_src, t_curr] = -1.0 * cached_contract(
                'eifj,Gcgdh,akbl,ce,gi,Mbdf,Mhjl,Gka->GM',
                peram_u_src_3pt, peram_curr[t_curr_indx], peram_u_src_3pt, 
                gamma(7), gamma(7),
                sink_3pt_VVV, src_VVV,
                projection, optimize = ['auto', 'dp', 'greedy']
            )
            
        if rank == 0:
            print(f'cal 3pt of link {link_indx} use time {(time.perf_counter() - st_cal_link_time):.3f} s of tsrc {t_src}')
            
        
# 将不同rank 计算的结果进行收集，
corr_3pt_matrix_d = get_mpi_data(data = corr_3pt_matrix_d, mdtype = 'Gather', root = 0, axis = 0)
corr_3pt_matrix_u = get_mpi_data(data = corr_3pt_matrix_u, mdtype = 'Gather', root = 0, axis = 0)
# corr_2pt_matrix = get_mpi_data(data = corr_2pt_matrix, mdtype = 'TGather', root = 0, axis = -1)
if rank == 0:
    # 由于3pt按照空间维度进行分配，所以所有rank收集到一起的数据需要求和
    corr_3pt_matrix_d = corr_3pt_matrix_d.sum(0)
    corr_3pt_matrix_u = corr_3pt_matrix_u.sum(0)
    
    # 对不同的收缩图进行求和
    corr_3pt_matrix_d = corr_3pt_matrix_d.sum(1)
    corr_3pt_matrix_u = corr_3pt_matrix_u.sum(1)
    
    # corr_2pt_matrix = corr_2pt_matrix.sum(0)
    # print(corr_3pt_matrix[..., 0, :10])
    
    # 存入数据
    for Mom_indx, Mom_list in enumerate(Mom_sink):
        corr_save_path = f'/public/home/sush/distillation/proton/result_gen/{Nx}x{Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_{Nev_src}/conf{conf_id}'
        import pathlib

        path = pathlib.Path(corr_save_path)

        if path.exists():
            print('save_path:',corr_save_path)
        
        else:
            path.mkdir(parents = True, exist_ok = True)
            print('mkdir_save_path:',corr_save_path)

        # backend.save(f'{corr_save_path}/corr_2pt_gamma0707_stout_smear_20_0.12_src{Nev_src}_sink{Nev_src}.npy', corr_2pt_matrix[Mom_indx])
            
        for link_indx in range(-link_max, link_max + 1, 1):
            backend.save(f'{corr_save_path}/corr_3pt_u_tsep{t_sep}_link_indx{link_indx}X_gen_stout_smear_20_0.12_src{Nev_src}_sink{Nev_src}.npy', corr_3pt_matrix_u[:, Mom_indx, link_indx])
            backend.save(f'{corr_save_path}/corr_3pt_d_tsep{t_sep}_link_indx{link_indx}X_gen_stout_smear_20_0.12_src{Nev_src}_sink{Nev_src}.npy', corr_3pt_matrix_d[:, Mom_indx, link_indx])