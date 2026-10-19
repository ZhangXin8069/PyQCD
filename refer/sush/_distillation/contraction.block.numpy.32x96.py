import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction import *

from opt_einsum import contract, contract_path

import time

set_backend('numpy')
backend = get_backend()

lattice_size = [32, 32, 32, 96]
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

param = creat_param(
    sink_operators = [['d^d', 'gamma_sink', 'u', 1]],
    source_operators = [['u^d', 'gamma_source', 'd', -1]]
)

corr = distillation_func(
    conf_id = conf_id,
    param = param.data(),
    
    lattice_size = lattice_size, # Nx, Ny, Nz, Nt 
    grid_size = grid_size, 
    
    backend = backend.__name__
)

# mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend)
rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size
Lx, Ly, Lz, Lt = [lattice_size[x]//grid_size[x] for x in range(len(lattice_size))]

fun_eigen = corr_eigvecs(Nx = Nx, backend = backend)
Mom_sink = creat_mom_list(Mom = [0, 0, 1])
phase_exp = backend.zeros((len(Mom_sink), Nx, Nx, Nx, Nc), dtype = complex)

for Mom_indx, Mom in enumerate(Mom_sink):
    phase_exp[Mom_indx] = fun_eigen.phase_exp_2pt(Mom = Mom)


Nev_src = 400
Nev_src_list = [100, 150, 200, 300, 400]

t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'Scatter')

sink = backend.zeros((Lt, len(Mom_sink), Nev_src, Nev_src), dtype = complex)
st_eigen = time.perf_counter()
for t_src_indx, t_src in enumerate(t_rank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.41_mu-0.2295_ms-0.2050_L32x96/{conf_id}/{conf_id}_t{t_src:03d}_e100_s[100, 200, 400, 600, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n400_V1_stout_smear_20_0.12.npy')
    
    for Mom_indx, Mom in enumerate(Mom_sink):
        sink[t_src_indx, Mom_indx] = fun_eigen.Mom_VdV_sink_t_2(phase_exp = phase_exp[Mom_indx], eigvecs = eigvecs[:Nev_src])
    
    # VdV_link[t_src_indx] = fun_eigen.VdV_sink_t_link(
    #     eigvecs = eigvecs,
    #     link_dir = 'Z',
    #     link_max = link_max,
    #     phase_exp = phase_exp,
    #     gauge_link = gauge_link,
    #     t = t_src_indx
    # )
print(f'load and calculate sink use time {(time.perf_counter() - st_eigen):.3f} s')

corr_2pt = backend.zeros((len(Nev_src_list), len(Mom_sink), Nt, Lt), dtype = complex)
for t_src in range(Nt):
    st_cal = time.perf_counter()
    
    source = get_mpi_data(data = sink[t_src//size], mdtype = 'Bcast', root = t_src%size).transpose(0, 2, 1).conj()
    t_sink_list_rank, _, t_sink_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + 50, 1), gtype = 'Scatter')
    
    if rank == 0:
        peram_u_src = backend.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/light/{conf_id}/t{t_src:03d}_e100_s[100, 200, 400, 600, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n400_V1_stout_smear_20_0.12.npy'
            )

    else:
        peram_u_src = None
        
    peram_u_src = get_mpi_data(data = peram_u_src, mdtype = 'TScatter', root = 0, axis = 0)
    peram_d_src = seq_peram(peram_u_src)
    for _Nev_src_indx, _Nev_src in enumerate(Nev_src_list):
        for t_sink in t_sink_list_indx:
            corr_2pt[_Nev_src_indx, :, t_src, t_sink] = contract(
                'manb,cedf,Mbd,Mfn,ac,em->M',
                peram_u_src[t_sink, :, :, :_Nev_src, :_Nev_src],
                peram_d_src[t_sink, :, :, :_Nev_src, :_Nev_src],
                source[:, :_Nev_src, :_Nev_src],
                sink[t_sink, :, :_Nev_src, :_Nev_src],  
                backend.asarray(gamma(5)),
                backend.asarray(gamma(5)),
                )

    if rank == 0:
        print(f'load peram and calculate 2pt of t_src {t_src} use time {(time.perf_counter() - st_cal):.3f} s')

corr_2pt = get_mpi_data(data = corr_2pt, mdtype = 'TGather', root = 0, axis = -1)

if rank == 0:
    
    for Mom_indx, Mom in enumerate(Mom_sink):
        corr_save_path = f'/public/home/sush/distillation/pion/result/32x96/Px{Mom[2]}Py{Mom[1]}Pz{Mom[0]}/ENV_400/conf{conf_id}/'

        import pathlib

        path = pathlib.Path(corr_save_path)

        if path.exists():
            print('save_path:',corr_save_path)
        
        else:
            path.mkdir(parents = True, exist_ok = True)
            print('mkdir_save_path:',corr_save_path)
            
        for _Nev_src_indx, _Nev_src in enumerate(Nev_src_list):
            backend.save(f'{corr_save_path}/corr_ud_2pt_gamma0505_e100_s[100, 200, 400, 600, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n400_V1_stout_smear_20_0.12_src{_Nev_src}.npy', corr_2pt[_Nev_src_indx, Mom_indx])