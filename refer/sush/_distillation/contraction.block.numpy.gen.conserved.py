import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction import *

from opt_einsum import contract, contract_path

import time
from typing import Literal

def Readin_gauge(conf_file, lattice_size):
    
    backend = get_backend()
    Nz, Ny, Nx, Nt = lattice_size
    
    f = open("%s" % conf_file, "rb")
    gauge = backend.fromfile(f, dtype=">f8")
    gauge = backend.array(gauge)

    gauge = gauge.reshape(Nt, Nx, Nx, Nx, 4, 3, 3, 2)
    gauge = gauge[..., 0] + gauge[..., 1] * 1j
    f.close()

    return gauge

def transpose_gauge(gauge_link, link_indx:int, link_dir:Literal["Z", "Y", "X"]):
    backend = get_backend()
    
    if link_dir=='Z':   axis_dir=1
    elif link_dir=='Y':   axis_dir=2
    elif link_dir=='X':   axis_dir=3
    
    _gauge_link = gauge_link[3 - axis_dir]

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
grid_size = [1, 1, 1, 24]
mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)

conf_id = sys.argv[1]

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size

t_sep = 12
link_max = 0
Nev_src = 50

# 将空间分给不同的rank进行计算
Z_list = [x for x in range(Nz)]
Y_list = [x for x in range(Ny)][rank*(Ny//size):(rank+1)*(Ny//size)]
X_list = [x for x in range(Nx)]

fun_eigen = corr_eigvecs(Nx = Nx, backend = backend)
# sink Mom
Mom_sink = creat_mom_list(Mom = [0, 0, 0], fix_Q2 = True) # P = [0, 0, 0]
phase_exp_2pt = backend.empty((len(Mom_sink), Nx, Nx, Nx, 3), dtype = complex) # e^{-ipx}

for i in range(len(Mom_sink)):
    phase_exp_2pt[i] = backend.asarray(fun_eigen.phase_exp_2pt(Mom = Mom_sink[i]))

t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'Scatter')

sink_VdV = np.zeros((Nt//size, len(Mom_sink), Nev_src, Nev_src), dtype = complex)
st_eigen = time.perf_counter()
for t_src_indx, t_src in enumerate(t_rank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/{conf_id}_t{t_src:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')
    
    sink_VdV[t_src_indx] = fun_eigen.Mom_VdV_sink_t_3(phase_exp = phase_exp_2pt, eigvecs = eigvecs[:Nev_src])
        
if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')

#读取规范场
if rank == 0:
    gauge_link = Readin_gauge(
        # f'/public/home/sush/share_work/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{conf_id}.lime.contents/msg02.rec04.ildg-binary-data', 
        f'/nexdata/project/lqcd/sush/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{conf_id}.stout_smear.1.0.125.lime.contents/msg02.rec04.ildg-binary-data', 
        lattice_size = lattice_size
        )

    #将规范场的维度设置为(Nd, Nt, Nz, Ny, Nx, Nc, Nc)即link方向，时间，Z, Y, X, c, c
    gauge_link = gauge_link.transpose(4, 0, 1, 2, 3, 5, 6)

else:
    gauge_link = None

gauge_link = get_mpi_data(data = gauge_link, mdtype = 'Bcast', root = 0)

###二者等价
# gauge_link = gauge_link[np.ix_([x for x in range(4)], [x for x in range(Nt)], Z_list, Y_list, X_list, [x for x in range(3)], [x for x in range(3)])]
# gauge_link = ArraySlicer(gauge_link).slice(dims = [2, 3, 4], indices = [Z_list, Y_list, X_list])

projection = (gamma(0) + gamma(4))/2

corr_2pt_matrix = backend.zeros((len(Mom_sink), Nt, Nt//size), dtype = complex)
corr_3pt_matrix = backend.zeros((len(Mom_sink), 2 * link_max + 1, Nt, Nt), dtype = complex)

for t_src in range(1, 2, 1):
    st_mpi_time = time.perf_counter()
    src_VdV = get_mpi_data(data = sink_VdV[t_src//size], mdtype = 'Bcast', root = t_src%size).conj()
    _, _, t_sink_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + Nt//2, 1), gtype = 'Scatter')
    
    if rank == 0:
        peram_u_src = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{t_src:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy'
            )[..., :Nev_src, :Nev_src]

    else:
        peram_u_src = None
        
    peram_u_src = get_mpi_data(data = peram_u_src, mdtype = 'TScatter', root = 0, axis = 0)
    peram_u_src_seq = seq_peram(peram_u_src)
    
    for t_sink in t_sink_list_indx:
        corr_2pt_matrix[:, t_src, t_sink] = 1.0 * contract(
            'manb,cedf,ac,em,Mbd,Mfn->M',
            peram_u_src[t_sink], peram_u_src_seq[t_sink],
            gamma(5), gamma(5),
            src_VdV, sink_VdV[t_sink]
        )
        
    sink_3pt_VdV = get_mpi_data(data = sink_VdV[(t_src + t_sep)%Nt//size], mdtype = 'Bcast', root = ((t_src + t_sep)%Nt)%size)
    peram_u_src_seq_3pt = get_mpi_data(data = peram_u_src_seq[(t_src + t_sep)%Nt//size], mdtype = 'Bcast', root = ((t_src + t_sep)%Nt)%size)
    
    t_curr_list = [x%Nt for x in range(t_src, t_src + t_sep + 3, 1)]
    
    prop_src = np.memmap(
        f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{t_src:03d}_gen_e50_stout_smear_20_0.12.mmap', 
        dtype=complex, mode='r', shape=(Nz, Ny, Nx, Nc, Nt, 4, 4, 50)
        )
    
    prop_src_load = ArraySlicer(prop_src).slice(dims = [0, 1, 2, 4], indices = [Z_list, Y_list, X_list, t_curr_list]).copy()

    
    prop_sep = np.memmap(
        f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{(t_src + t_sep)%Nt:03d}_gen_e50_stout_smear_20_0.12.mmap', 
        dtype=complex, mode='r', shape=(Nz, Ny, Nx, Nc, Nt, 4, 4, 50)
        )
    
    prop_sep_load = ArraySlicer(prop_sep).slice(dims = [0, 1, 2, 4], indices = [Z_list, Y_list, X_list, t_curr_list]).copy()
    prop_sep_load = contract('ab,...bce,cd->...dae', gamma(5), prop_sep_load.conj(), gamma(5))
    
    # for link_indx in range(1, 2, 1):
    for link_indx in range(-link_max, link_max + 1, 1):

        # peram_curr = contract(
        #     'zyxcTabn,zyxgTdef,Gae,Tzyxgc->TGdbfn', backend.roll(prop_src_load, shift = -1 * link_indx, axis = 0), prop_sep_load, gamma_curr, transpose_gauge(ArraySlicer(gauge_link).slice(dims = [1, 2, 3, 4], indices = [t_curr_list, Z_list, Y_list, X_list]), link_indx = link_indx, link_dir = 'Y')
        # )
        # peram_curr = contract(
        #     'zyxcTabn,zyxgTdef,ae,Tzyxgc->Tdbfn', backend.roll(prop_src_load, shift = -1 * link_indx, axis = 0), prop_sep_load, gamma(4), transpose_gauge(ArraySlicer(gauge_link).slice(dims = [1, 2, 3, 4], indices = [t_curr_list, Z_list, Y_list, X_list]), link_indx = link_indx, link_dir = 'Z')
        # )
        
        peram_d_curr_B = contract(
            'zyxcTabn,zyxgTdef,ae,Tzyxgc->Tdbfn', backend.roll(prop_src_load, shift = -1, axis = 4), prop_sep_load, gamma(4) - gamma(0), ArraySlicer(gauge_link[3]).slice(dims = [0, 1, 2, 3], indices = [t_curr_list, Z_list, Y_list, X_list])
            # 'zyxctabn,zyxCtdef,ae,tzyxcC->tdbfn', backend.roll(prop_src_load, shift = 1 * link_indx, axis = 0), prop_sep_load, gamma(4), transpose_gauge(gauge_link[:, t_curr_list], link_indx = link_indx, link_dir = 'Z')
        )/2
        
        peram_d_curr_F = contract(
            'zyxcTabn,zyxgTdef,ae,Tzyxgc->Tdbfn', prop_src_load, backend.roll(prop_sep_load, shift = -1, axis = 4), gamma(4) + gamma(0), ArraySlicer(gauge_link[3]).slice(dims = [0, 1, 2, 3], indices = [t_curr_list, Z_list, Y_list, X_list]).conj().transpose(0, 1, 2, 3, 5, 4)
            # 'zyxctabn,zyxCtdef,ae,tzyxcC->tdbfn', backend.roll(prop_src_load, shift = 1 * link_indx, axis = 0), prop_sep_load, gamma(4), transpose_gauge(gauge_link[:, t_curr_list], link_indx = link_indx, link_dir = 'Z')
        )/2
        
        peram_curr = peram_d_curr_B + peram_d_curr_F
        # peram_curr = peram_d_curr_F
        
        for t_curr_indx, t_curr in enumerate(t_curr_list):
            corr_3pt_matrix[:, link_indx, t_src, t_curr] = 1.0 * contract(
            'manb,cedf,ac,em,Mbd,Mfn->M',
            peram_curr[t_curr_indx], peram_u_src_seq_3pt,
            gamma(5), gamma(5),
            src_VdV, sink_3pt_VdV
            )
            
    
    if rank == 0:
        print(f'mpidata and lexico prop use time {(time.perf_counter() - st_mpi_time):.3f} s of tsrc {t_src}')
        

corr_3pt_matrix = get_mpi_data(data = corr_3pt_matrix, mdtype = 'Gather')
corr_2pt_matrix = get_mpi_data(data = corr_2pt_matrix, mdtype = 'TGather', root = 0, axis = -1)

if rank == 0:
    corr_3pt_matrix = corr_3pt_matrix.sum(0)
    
    # corr_3pt_matrix = corr_3pt_matrix.sum(0)
    # corr_2pt_matrix = corr_2pt_matrix.sum(0)
    print(corr_3pt_matrix[0, :, 1, :13])
    print(corr_2pt_matrix[0, 1, :20])
    
#     for Mom_indx, Mom_list in enumerate(Mom_link[:]):
#         corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_gen/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_50/conf{conf_id}'
#         import pathlib

#         path = pathlib.Path(corr_save_path)

#         if path.exists():
#             print('save_path:',corr_save_path)
        
#         else:
#             path.mkdir(parents = True, exist_ok = True)
#             print('mkdir_save_path:',corr_save_path)

#         for gamma_indx, gamma_list in enumerate(mu_nu_indx[Mom_indx]):
#             backend.save(f'{corr_save_path}/corr_3pt_gamma0{gamma_list[1]}0{gamma_list[0]}05_tsep{t_sep}_gen_stout_smear_20_0.12_src50_sink{Nev_max}_curr{Nev_max}_compensation.npy', corr_3pt_matrix[Mom_indx, gamma_indx])