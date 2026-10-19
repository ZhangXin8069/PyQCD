import sys

import time
from typing import Literal
from lqcddb.base import *
from lqcddb.constant import gamma
from lqcddb.contraction import seq_peram
from lqcddb.eigvectors import vertex_creator
from lqcddb.analyse import loop_tsrc
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

conf_id = sys.argv[1]

import numpy as np
# 初始化numpy or cupy
set_backend('numpy')
backend = get_backend()

# 格子数据和 mpi 
lattice_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 12]
mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)

conf_id = sys.argv[1]

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size

t_sep = 8
link_max = 0
Nev_src = 50
Nev_max = 650

# 将空间分给不同的rank进行计算
Z_list = [x for x in range(Nz)][rank*(Ny//size):(rank+1)*(Ny//size)]
Y_list = [x for x in range(Ny)]
X_list = [x for x in range(Nx)]

#读取规范场
if rank == 0:
    gauge_link = Readin_gauge(
        f'/public/home/sush/share_work/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{conf_id}.stout_smear.1.0.125.lime.contents/msg02.rec04.ildg-binary-data', 
        lattice_size = lattice_size
        )

    #将规范场的维度设置为(Nd, Nt, Nz, Ny, Nx, Nc, Nc)即link方向，时间，Z, Y, X, c, c
    gauge_link = gauge_link.transpose(4, 0, 1, 2, 3, 5, 6)
    
else:
    gauge_link = None

gauge_link = get_mpi_data(data = gauge_link, mdtype = 'TScatter', root = 0, axis = 1)

fun_eigen = vertex_creator(Nx = Nx)
# sink Mom
Mom_sink = creat_mom_list(Mom = [0, 0, 0], fix_Q2 = True) # P = [0, 0, 0]
phase_exp_2pt = backend.empty((len(Mom_sink), Nx, Nx, Nx, 3), dtype = complex) # e^{-ipx}

for i in range(len(Mom_sink)):
    phase_exp_2pt[i] = backend.asarray(fun_eigen.phase_exp_2pt(Mom = Mom_sink[i]))

t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'TScatter')

sink_VdV = np.zeros((Nt//size, len(Mom_sink), Nev_src, Nev_src), dtype = complex)
curr_VdV = np.zeros((Nt//size, len(Mom_sink), 2, Nev_max, Nev_max), dtype = complex)

st_eigen = time.perf_counter()
for t_src_indx, t_src in enumerate(t_rank):
    eigvecs_max = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/{conf_id}_t{(t_src + 1) % Nt:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')
    eigvecs_min = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/{conf_id}_t{(t_src) % Nt:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')
    
    sink_VdV[t_src_indx] = fun_eigen.Mom_VdV_sink_t(phase_exp = phase_exp_2pt, eigvecs = eigvecs_min[:Nev_src])
    curr_VdV[t_src_indx] = fun_eigen.VdV_sink_t_link(eigvecs = eigvecs_max, link_dir = 'Z', link_max = link_max, phase_exp = phase_exp_2pt, t = t_src_indx, gauge_link = gauge_link, eigvecs_min = eigvecs_min, conserved = True)
    
curr_VdV = curr_VdV * fun_eigen.create_omega_accelerate(
    exact = 50,
    N_eigen = [100, 200, 300, 500, 800],
    N_sum = [20, 20, 20, 20, 20],
    N_extract = [10, 10, 10, 10, 10],
    noise = 500,
    conserved = True
)

if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')



###二者等价
# gauge_link = gauge_link[np.ix_([x for x in range(4)], [x for x in range(Nt)], Z_list, Y_list, X_list, [x for x in range(3)], [x for x in range(3)])]
# gauge_link = ArraySlicer(gauge_link).slice(dims = [2, 3, 4], indices = [Z_list, Y_list, X_list])

projection = (gamma(0) + gamma(4))/2

corr_2pt_matrix = backend.zeros((len(Mom_sink), Nt, Nt//size), dtype = complex)
corr_3pt_matrix = backend.zeros((len(Mom_sink), 2, Nt, Nt//size), dtype = complex)

for t_src in range(Nt):
    st_mpi_time = time.perf_counter()
    src_VdV = get_mpi_data(data = sink_VdV[t_src//size], mdtype = 'Bcast', root = t_src%size).conj()
    _, _, t_sink_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + Nt//2, 1), gtype = 'Scatter')
    
    if rank == 0:
        peram_u_src = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{t_src:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy'
            )[..., :Nev_src]
        peram_u_src_max = backend.roll(peram_u_src, axis = 0, shift = -1)
    else:
        peram_u_src = None
        peram_u_src_max = None
        
        
    peram_u_src = get_mpi_data(data = peram_u_src, mdtype = 'TScatter', root = 0, axis = 0)
    peram_u_src_max = get_mpi_data(data = peram_u_src_max, mdtype = 'TScatter', root = 0, axis = 0)
    
    peram_u_src_seq = seq_peram(peram_u_src)
    
    for t_sink in t_sink_list_indx:
        corr_2pt_matrix[:, t_src, t_sink] = 1.0 * cached_contract(
            'manb,cedf,ac,em,Mbd,Mfn->M',
            peram_u_src[t_sink, ..., :Nev_src, :Nev_src], peram_u_src_seq[t_sink, ..., :Nev_src, :Nev_src],
            gamma(5), gamma(5),
            src_VdV, sink_VdV[t_sink]
        )
        
    sink_3pt_VdV = get_mpi_data(data = sink_VdV[(t_src + t_sep)%Nt//size], mdtype = 'Bcast', root = ((t_src + t_sep)%Nt)%size)
    peram_u_src_seq_3pt = get_mpi_data(data = peram_u_src_seq[(t_src + t_sep)%Nt//size, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = ((t_src + t_sep)%Nt)%size)
    
    _, _, t_sink_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + t_sep + 2, 1), gtype = 'Scatter')

    
    if rank == 0:
        peram_u_sep = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{(t_src+t_sep)%Nt:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy'
            )[..., :Nev_src]
        
        peram_u_sep = seq_peram(peram_u_sep)
        
        peram_u_sep_max = backend.roll(peram_u_sep, axis = 0, shift = -1)
        
    else:
        peram_u_sep = None
        peram_u_sep_max = None
        
    peram_u_sep     = get_mpi_data(data = peram_u_sep,     mdtype = 'TScatter', root = 0, axis = 0)
    peram_u_sep_max = get_mpi_data(data = peram_u_sep_max, mdtype = 'TScatter', root = 0, axis = 0)
    
    
    for t_curr in t_sink_list_indx:
        corr_3pt_matrix[:, 0, t_src, t_curr] = 1.0 * cached_contract(
            'codp,manb,gehf,ac,em,og,Mbd,fn,Mph->M',
            peram_u_src_seq_3pt, peram_u_src_max[t_curr], peram_u_sep[t_curr],
            gamma(5), gamma(4) - gamma(0), gamma(5),
            src_VdV, curr_VdV[t_curr, 0, 0], sink_3pt_VdV
        )
        corr_3pt_matrix[:, 1, t_src, t_curr] = 1.0 * cached_contract(
            'codp,manb,gehf,ac,em,og,Mbd,fn,Mph->M',
            peram_u_src_seq_3pt, peram_u_src[t_curr], peram_u_sep_max[t_curr],
            gamma(5), gamma(4) + gamma(0), gamma(5),
            src_VdV, curr_VdV[t_curr, 0, 1], sink_3pt_VdV
        )
        
    if rank == 0:
        print(f'mpidata and lexico prop use time {(time.perf_counter() - st_mpi_time):.3f} s of tsrc {t_src}')
        

corr_3pt_matrix = get_mpi_data(data = corr_3pt_matrix, mdtype = 'TGather', root = 0, axis = -1)
corr_2pt_matrix = get_mpi_data(data = corr_2pt_matrix, mdtype = 'TGather', root = 0, axis = -1)

if rank == 0:
    corr_3pt_matrix = corr_3pt_matrix.sum(1)
    
    corr_save_path = f'/public/home/sush/distillation/pion/result/C24P29/Px0Py0Pz0/ENV_50/conf{conf_id}'
    import pathlib

    path = pathlib.Path(corr_save_path)

    if path.exists():
        print('save_path:',corr_save_path)
    
    else:
        path.mkdir(parents = True, exist_ok = True)
        print('mkdir_save_path:',corr_save_path)
        
    backend.save(f'{corr_save_path}/corr_2pt_stout_smear_20_0.12_src50_sink50.npy', corr_2pt_matrix[0])
    backend.save(f'{corr_save_path}/corr_3pt_conserved_tsep{t_sep}_stout_smear_20_0.12_src50_sink50_curr{Nev_max}.npy', corr_3pt_matrix[0])