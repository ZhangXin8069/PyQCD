"""
这是一个mpi计算pion 矢量流的例子
"""
import sys

from lqcddb.base import (
    get_backend, getMPIComm, getMPIRank, getMPISize, get_mpi_data,
    get_mpi_tlist, mpinit, set_backend, creat_mom_list
)
from lqcddb.eigvectors import vertex_creator
from lqcddb.constant import gamma, Nc, Nd
from lqcddb.contraction import seq_peram
from lqcddb.analyse import loop_tsrc
from opt_einsum import contract, contract_path

import time

set_backend('numpy')
backend = get_backend()

lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 12]

def Readin_gauge(conf_file, lattice_size):
    
    Nz, Ny, Nx, Nt = lattice_size
    
    f = open("%s" % conf_file, "rb")
    gauge = backend.fromfile(f, dtype=">f8")
    gauge = backend.array(gauge)

    gauge = gauge.reshape(Nt, Nx, Nx, Nx, 4, 3, 3, 2)
    gauge = gauge[..., 0] + gauge[..., 1] * 1j
    f.close()

    return gauge

conf_id = sys.argv[1]

mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)
rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size
Lx, Ly, Lz, Lt = [lattice_size[x]//grid_size[x] for x in range(len(lattice_size))]

Mom_sink = creat_mom_list(Mom = [0, 0, 0], fix_Q2 = True)# P = [0, 0, 0]
P = [0]

phase_exp = backend.empty((len(Mom_sink), Nx, Nx, Nx, 3), dtype = complex) # e^{-ipx}
fun_eigen = vertex_creator(Nx = Nx)

for i in range(len(Mom_sink)):
    phase_exp[i] = fun_eigen.phase_exp_VdV(Mom = Mom_sink[i])


Nev_src = 100
Nev_link = 650
link_max = 0

t_sep = 8

t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'TScatter')

sink = backend.zeros((Lt, len(Mom_sink), Nev_src, Nev_src), dtype = complex)
# VdV_link = backend.zeros((Lt, 2 * link_max + 1, Nev_link, Nev_link), dtype = complex)

for t_src_indx, t_src in enumerate(t_rank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/{conf_id}_t{t_src:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')

    sink[t_src_indx] = fun_eigen.Mom_VdV_sink_t(phase_exp = phase_exp, eigvecs = eigvecs[:Nev_src])


corr_2pt = backend.zeros((len(Mom_sink), Nt, Lt), dtype = complex)
# corr_3pt_con = backend.zeros((2 * link_max + 1, Nt, Lt), dtype = complex)
# bubble = backend.zeros((2 * link_max + 1, Nt), dtype = complex)

for t_src in range(Nt):
    source = get_mpi_data(data = sink[t_src//size], mdtype = 'Bcast', root = t_src%size).transpose(0, 2, 1).conj()
    t_sink_list_rank, _, t_sink_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + Nt//2, 1), gtype = 'TScatter')
    
    if rank == 0:
        peram_u_src = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{t_src:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy'
            )[..., :Nev_src, :Nev_src]
        # peram_u_src = backend.zeros((Nt, Nd, Nd, 20, 20), dtype = complex)
    else:
        peram_u_src = None
        
    peram_u_src = get_mpi_data(data = peram_u_src, mdtype = 'TScatter', root = 0, axis = 0)
    peram_d_src = seq_peram(peram_u_src)
    
    for t_sink in t_sink_list_indx:
        corr_2pt[:, t_src, t_sink] = contract(
            'manb,cedf,Mbd,Mfn,ac,em->M',
            peram_u_src[t_sink, :, :, :Nev_src, :Nev_src],
            peram_d_src[t_sink, :, :, :Nev_src, :Nev_src],
            source,
            sink[t_sink],  
            backend.asarray(gamma(5)),
            backend.asarray(gamma(5)),
            )
    
from pathlib import Path
corr_2pt = get_mpi_data(data = corr_2pt, mdtype = 'TGather', root = 0, axis = 2)

if rank == 0:
    corr_2pt = loop_tsrc(data = corr_2pt, indx = [1, 2], Boundary_Conditions = 'Periodic')
    
    for i in range(len(Mom_sink)):
        corr_save_path = Path(f'/public/home/sush/distillation/pion/result/C24P29/Px0Py0Pz{P[i]}/ENV_50/conf{conf_id}')
        corr_save_path.mkdir(parents=True, exist_ok=True)
        backend.save(f'{corr_save_path}/corr_ud_2pt_src{Nev_src}.npy', corr_2pt[i])
    
#     for link_indx in range(-link_max, link_max + 1, 1):
#         backend.save(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_3pt_gamma050405_tseq{t_sep}_link_indx{link_indx}_u_e50_s[0]_[0]_[0]_BI_n150_stout_smear_20_0.12_dul_vector_False_src{Nev_src}_curr{Nev_link}.npy', corr_3pt_con[link_indx + link_max])
#         backend.save(f'/public/home/sush/distillation/pion/result/12x32/Px0Py0Pz0/ENV_200/conf{conf_id}/corr_ud_bubble_gamma04_link_indx{link_indx}_u_e50_s[0]_[0]_[0]_BI_n150_stout_smear_20_0.12_dul_vector_False_curr{Nev_link}.npy', bubble[link_indx + link_max])
        
