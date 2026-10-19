import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from lqcddb.base import *
from lqcddb.constant import gamma
from lqcddb.eigvectors import vertex_creator
from lqcddb.contraction import seq_peram
from lqcddb.io import check_dir_path

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

conf_id = sys.argv[1]

import numpy as np
# 初始化numpy or cupy
set_backend('cupy')
backend = get_backend()

# 格子数据和 mpi 
lattice_size = [64, 64, 64, 128]
grid_size = [1, 1, 1, 1]
mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)

conf_id = sys.argv[1]

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size
Nev_src = 300


fun_eigen = vertex_creator(Nx = Nx)
# sink Mom
Mom_sink = creat_mom_list(Mom = [0, 0, 0], fix_Q2 = True) + [[1, 0, 0]] + [[2, 0, 0]] + [[3, 0, 0]]# P = [0, 0, 0]
# Mom_sink = [[1, 0, 0]]# P = [0, 0, 0]

phase_exp_2pt = backend.empty((len(Mom_sink), Nx, Nx, Nx, 3), dtype = complex) # e^{-ipx}

for i in range(len(Mom_sink)):
    phase_exp_2pt[i] = backend.asarray(fun_eigen.phase_exp_2pt(Mom = Mom_sink[i]))

t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'TScatter')

sink_VdV = backend.zeros((Nt//size, len(Mom_sink), Nev_src, Nev_src), dtype = complex)

st_eigen = time.perf_counter()
for t_src_indx, t_src in enumerate(t_rank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.41_mu-0.2334_ms-0.2030_L64x128/{conf_id}/{conf_id}_t{t_src:03d}_e300_stout_smear_10_0.12.npy')
    for i in range(len(Mom_sink)):
        sink_VdV[t_src_indx, i] = fun_eigen.Mom_VdV_sink_t(phase_exp = phase_exp_2pt[i], eigvecs = eigvecs[:Nev_src])

del eigvecs

if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')



###二者等价
# gauge_link = gauge_link[backend.ix_([x for x in range(4)], [x for x in range(Nt)], Z_list, Y_list, X_list, [x for x in range(3)], [x for x in range(3)])]
# gauge_link = ArraySlicer(gauge_link).slice(dims = [2, 3, 4], indices = [Z_list, Y_list, X_list])

projection = (gamma(0) + gamma(4))/2

corr_2pt_matrix = backend.zeros((len(Mom_sink), Nt, Nt//size), dtype = complex)

for t_src in range(Nt):
    st_mpi_time = time.perf_counter()
    src_VdV = get_mpi_data(data = sink_VdV[t_src//size], mdtype = 'Bcast', root = t_src%size).conj().transpose(0, 2, 1)
    _, _, t_sink_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + Nt//2, 1), gtype = 'TScatter')
    
    if rank == 0:
        peram_u_src = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.41_mu-0.2334_ms-0.2030_L64x128/light/{conf_id}/t{t_src:03d}_e300_stout_smear_10_0.12.npy'
            )[..., :Nev_src]

    else:
        peram_u_src = None
        
    peram_u_src = get_mpi_data(data = peram_u_src, mdtype = 'TScatter', root = 0, axis = 0)
    peram_u_src_seq = seq_peram(peram_u_src)
    
    for t_sink in t_sink_list_indx:
        corr_2pt_matrix[:, t_src, t_sink] = 1.0 * cached_contract(
            'manb,cedf,ac,em,Mbd,Mfn->M',
            peram_u_src[t_sink, ..., :Nev_src, :Nev_src], peram_u_src_seq[t_sink, ..., :Nev_src, :Nev_src],
            gamma(5), gamma(5),
            src_VdV, sink_VdV[t_sink]
        )
        
    if rank == 0:
        print(f'cal use time {(time.perf_counter() - st_mpi_time):.3f} s of tsrc {t_src}')

corr_2pt_matrix = get_mpi_data(data = corr_2pt_matrix, mdtype = 'TGather', root = 0, axis = -1)

if rank == 0:
    for Mom_indx, Mom_list in enumerate(Mom_sink[:]):
        corr_save_path = f'/public/home/sush/distillation/pion/result/F64P13/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_300/conf{conf_id}'
        check_dir_path(corr_save_path)
        backend.save(f'{corr_save_path}/corr_2pt_e300_stout_smear_10_0.12.npy', corr_2pt_matrix[Mom_indx])