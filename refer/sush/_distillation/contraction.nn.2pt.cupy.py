import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from lqcddb import *

import time

set_backend('cupy')
backend = get_backend()

lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 1]

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

fun_eigen = vertex_creator(Nx = Nx)

Mom_sink_VDV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))
Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))

Mom_sqrt = backend.asarray([sum([y**2 for y in x]) for x in Mom_sink_VDV])

Mom_len = len(Mom_sink_VDV)

phase_exp_2pt = backend.zeros((Mom_len, Nx, Nx, Nx, Nc), dtype = complex)
# phase_exp_3pt = backend.zeros((Mom_len, Nx, Nx, Nx), dtype = complex)

for Mom_indx in range(Mom_len):
    phase_exp_2pt[Mom_indx] = fun_eigen.phase_exp_2pt(Mom = Mom_sink_VDV[Mom_indx])
    # phase_exp_3pt[Mom_indx] = fun_eigen.phase_exp_3pt(Mom = Mom_sink_VVV[Mom_indx])

Nev_src = 100
Nev_src_list = [100]

t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'Scatter')

import numpy as np

sink_VdV = backend.zeros((Lt, Mom_len, Nev_src, Nev_src), dtype = complex)
# sink_VVV = np.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype = complex)

st_eigen = time.perf_counter()
for t_src_indx, tsrc in enumerate(t_rank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')
    
    for Mom_indx in range(Mom_len):
        sink_VdV[t_src_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t_3(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs[:Nev_src])
        # sink_VVV[t_src_indx, Mom_indx] = fun_eigen.Mom_VVV_sink_t_3(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src]).get()

if rank == 0:
    sink_VVV = np.load(f'/public/home/sush/distillation/0v2b/VVV/{conf_id}_VVV.npy')
    print(sink_VVV.shape)
    
else:
    sink_VVV = None

sink_VVV = get_mpi_data(sink_VVV, mdtype = 'TScatter', axis = 0)
    
if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')

projection = (gamma(0) + gamma(4))/2

corr_2pt_PP = backend.zeros((3, 3, Mom_len, Mom_len, Nt, Lt), dtype = complex)
sink_operator = [
    [('|', 'u', 'u', 'gamma_7', 'd', '|')], 
    [('|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'), ('|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'), ('|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'u', '|')],
    [('|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'), ('|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'), ('|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'u', '|')],
    ]

src_operator = [
    [('|', 'u', 'u', 'gamma_7', 'd', '|')], 
    [('|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'), ('|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'), ('|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'd', '|')],
    [('|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'), ('|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'), ('|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'd', '|')],
    ]

operator_groups = [[
    [[s, t] for s in sink_operator[i] for t in src_operator[j]]
    for j in range(3)
] for i in range(3)]

GR = GammaRegistry()
VR = VRegistry()
PR = PeramRegistry()

GR.register('gamma_7', gamma(7))
GR.register('gamma_5', gamma(5))
st_peram = time.perf_counter()

peram_u = np.zeros((Nt, Nt, 4, 4, Nev_src, Nev_src), dtype=complex)
for tsrc in range(Nt):
    data = np.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')
    peram_u[tsrc] = data[:, :, :, :Nev_src, :Nev_src]

if rank == 0:
    mem_gb = peram_u.nbytes / 1e9
    print(f'Load perams: {(time.perf_counter() - st_peram):.1f}s  ({mem_gb:.1f} GB CPU)')
    
for tsrc in range(Nt):
    source_VdV = get_mpi_data(data = sink_VdV[tsrc//size], mdtype = 'Bcast', root = tsrc%size).transpose(0, 2, 1).conj()
    source_VVV = backend.asarray(get_mpi_data(data = sink_VVV[tsrc//size], mdtype = 'Bcast', root = tsrc%size)).conj()
    t_sink_list_rank, _, t_sink_list_indx = get_mpi_tlist(Nt = Nt, t = range(tsrc, tsrc + 32, 1), gtype = 'TScatter')
    
    st_cal = time.perf_counter()
    VR.register('VVV_0', 'tsrc', source_VVV)
    VR.register('VDV_1', 'tsrc', source_VdV)
    
    PR.register('light', ('tsrc', 'tsrc'), peram_u[tsrc, tsrc])
    
    for tsink in t_sink_list_indx:
        PR.register('light', ('tsink', 'tsrc'), peram_u[tsrc, tsink])
        PR.register('light', ('tsink', 'tsink'), peram_u[tsink, tsink])
        PR.register('light', ('tsrc', 'tsink'), seq_peram(peram_u[tsrc, tsink]))

        for i in range(3):
            for j in range(3):
                VR.register('VVV_0', 'tsink', sink_VVV[tsink])
                VR.register('VDV_1', 'tsink', sink_VdV[tsink])

                dycn = dynamic_contraction(
                    operator_groups[i][j], 
                    peram_registry = PR,
                    v_registry = VR,
                    gamma_registry = GR,
                    Cpt = '2pt',
                    Vindex = [],
                    Gindex = [],
                    use_equivalence = True,
                    ignore_dis = False,    
                    )
                
                # dycn.calculate_all(optimize = ['greedy', 'dp', 'auto'])
        
        
        
    if rank == 0:
        print(f'calculate 2pt of tsrc {tsrc} use time {(time.perf_counter() - st_cal):.3f} s')

corr_2pt_PP = get_mpi_data(data = corr_2pt_PP, mdtype = 'TGather', root = 0, axis = -1)

if rank == 0:
    corr_save_path = f'/public/home/sush/distillation/0v2b/result/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'
    check_dir_path(corr_save_path)
    
    backend.save(f'{corr_save_path}/corr_2pt_PP', corr_2pt_PP)
    # backend.save(f'{corr_save_path}/corr_2pt_NN', corr_2pt_NN)
    