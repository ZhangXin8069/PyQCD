import sys
import numpy as np
from lqcddb import *
import time

# M0 运行时：固定导入、后端、输入和总计时
st_total = time.perf_counter()
set_backend('cupy')
backend = get_backend()

lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 2]
conf_id = sys.argv[1]

# M1 格点与 MPI：通信器和局部几何
mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size
Lx, Ly, Lz, Lt = [lattice_size[x]//grid_size[x] for x in range(len(lattice_size))]

fun_eigen = vertex_creator(Nx = Nx)

# M2 动量与相位：只构造下方实际使用的动量和相位输入
Mom_sink_VDV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))
# Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))
Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))[::-1]

Mom_len = len(Mom_sink_VDV)

phase_exp_2pt = backend.zeros((Mom_len, Nx, Nx, Nx, Nc), dtype = complex)
phase_exp_3pt = backend.zeros((Mom_len, Nx, Nx, Nx), dtype = complex)

for Mom_indx in range(Mom_len):
    phase_exp_2pt[Mom_indx] = fun_eigen.phase_exp_2pt(Mom = Mom_sink_VDV[Mom_indx])
    phase_exp_3pt[Mom_indx] = fun_eigen.phase_exp_3pt(Mom = Mom_sink_VVV[Mom_indx])

Nev_src = 100
Nev_max = 400
tsep = 20

trank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'TScatter')

# import numpy as np

# M3 动态清单：算符、流、注册表和输出轴
sink_operator = [
    # [[-np.sqrt(1/6), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'], [np.sqrt(1/6), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'], [np.sqrt(2/3), '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|']],
    # [[-np.sqrt(1/6), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'], [np.sqrt(1/6), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'], [np.sqrt(2/3), '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|']],
    [['|', 'd', 'u', 'gamma_7', 'd', '|']]
    ]

source_operator = [
    [['|', 'd^d', 'gamma_5', 'u', '|', '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|']]
    ]

current_operator = [['|', 'u^d', 'gamma_1', 'd', '|']]

Cpt = '3pt'
Projection = True
use_equivalence = False
ignore_dis = False
optimize = 'auto'
# 下方 M4 到 M6 必须准确提供这些动态输入。

# M4 顶点数据：加载本征矢并构造所需的 VDV/VVV 张量
VdV_link = backend.zeros((Lt, Mom_len, Nev_max, Nev_max), dtype = complex)
sink_VVV = backend.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype = complex)

comm.Barrier()
st_eigen = time.perf_counter()
if rank == 0:
    print(f'[conf {conf_id}] VVV/VDV preparation started', flush = True)
for tsrc_indx, tsrc in enumerate(trank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')
    
    for Mom_indx in range(Mom_len):
        # VdV_link[tsrc_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs)
        VdV_link[tsrc_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs)
        sink_VVV[tsrc_indx, Mom_indx] = fun_eigen.Mom_VVV_sink_t(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src])

VdV_link *= fun_eigen.create_omega_accelerate(
    exact = 100,
    N_eigen = [100, 200, 400, 700, 1100],
    N_sum = [20, 20, 20, 20, 20],
    N_extract = [10, 10, 10, 10, 10],
    noise = 200,
)

# VdV_link = VdV_link[..., :Nev_src, :Nev_src]

del phase_exp_2pt, phase_exp_3pt

if backend.__name__ == 'cupy':
    backend.cuda.get_current_stream().synchronize()
comm.Barrier()
if rank == 0:
    print(f'[conf {conf_id}] VVV/VDV preparation completed in {(time.perf_counter() - st_eigen):.3f} s', flush = True)


# M5 peram 数据：加载并分发 M3 计划所需的 perambulator
corr_NJN_GEVP = backend.zeros((len(sink_operator), Nt, Lt, 4, Ns//2, Ns//2, Mom_len, Mom_len), dtype = complex)

comm.Barrier()
st_load = time.perf_counter()
if rank == 0:
    print(f'[conf {conf_id}] peram loading/distribution started', flush = True)
    peram_u = np.zeros((Nt, Nt, Ns, Ns, Nev_max, Nev_src), dtype = complex)
    for tsrc in range(Nt):
        peram_u[tsrc] = np.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
            )[..., :Nev_max, :Nev_src]
        
else:
    peram_u = None

# M6 注册表：只绑定 M3 所需的 gamma/V/peram 名称
GR = GammaRegistry()
VR = VRegistry()
PR = PeramRegistry()

GR.register('gamma_7',  gamma(7))
GR.register('gamma_5',  gamma(5))
GR.register('Projector',  [((gamma(4) + gamma(0))/2.0)[:, :2], ((gamma(4) + gamma(0))/2.0)[:, :2]])

# M7 动态循环：注册所需切片并执行 dynamic_contraction
peram_u = backend.asarray(get_mpi_data(data = peram_u, mdtype = 'TScatter', root = 0, axis = 1))
if backend.__name__ == 'cupy':
    backend.cuda.get_current_stream().synchronize()
comm.Barrier()
if rank == 0:
    print(f'[conf {conf_id}] peram loading/distribution completed in {(time.perf_counter() - st_load):.3f} s', flush = True)

st_main = time.perf_counter()
for tsrc in range(Nt):
    st_cal = time.perf_counter()
    if rank == 0:
        print(f'[conf {conf_id}] 3pt tsrc={tsrc} ({tsrc + 1}/{Nt}) started', flush = True)
    tsink = (tsrc + tsep)%Nt
    
    VVV_src = backend.asarray(get_mpi_data(data = sink_VVV[tsrc//size, :], mdtype = 'Bcast', root = tsrc%size))
    VDV_src = backend.asarray(get_mpi_data(data = VdV_link[tsrc//size, :, :Nev_src, :Nev_src], mdtype = 'Bcast', root = tsrc%size))
    VVV_src = VVV_src.conj()
    VDV_src = VDV_src.conj().transpose(0, 2, 1)
    
    VVV_sink = backend.asarray(get_mpi_data(data = sink_VVV[tsink//size, :], mdtype = 'Bcast', root = tsink%size))
    VDV_sink = backend.asarray(get_mpi_data(data = VdV_link[tsink//size, :, :Nev_src, :Nev_src], mdtype = 'Bcast', root = tsink%size))
    VDV_sink = VDV_sink.conj().transpose(0, 2, 1)
    
    _, _, tsink_indx = get_mpi_tlist(Nt = Nt, t = [x for x in range(tsrc + 5, tsrc + tsep - 3, 1)], gtype = 'TScatter')
    
    peram_u_src_src = get_mpi_data(data = peram_u[tsrc, tsrc//size, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = tsrc%size)
    peram_u_sink_src = get_mpi_data(data = peram_u[tsrc, tsink//size, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = tsink%size)
    peram_u_sink_sink = get_mpi_data(data = peram_u[tsink, tsink//size, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = tsink%size)

    PR.register('light', ('tsrc', 'tsrc' ), backend.asarray(peram_u_src_src))
    PR.register('light', ('tsink', 'tsrc'), backend.asarray(peram_u_sink_src))
    PR.register('light', ('tsrc', 'tsink' ), seq_peram(backend.asarray(peram_u_sink_src)))
    PR.register('light', ('tsink', 'tsink' ), backend.asarray(peram_u_sink_sink))
    
    VR.register('VDV_0', 'tsrc',  VDV_src[:])
    VR.register('VVV_0', 'tsrc',  VVV_src[:])
    
    for tcur0 in tsink_indx:
        st_cal_cur = time.perf_counter()
        peram_u_tcur0_tsrc = backend.asarray(peram_u[tsrc, tcur0])
        peram_u_tcur0_tsink = backend.asarray(peram_u[tsink, tcur0])
        
        PR.register('light', ('tcur0', 'tsrc'), peram_u_tcur0_tsrc)
        PR.register('light', ('tsrc', 'tcur0'), seq_peram(peram_u_tcur0_tsrc))

        PR.register('light', ('tcur0', 'tsink'), peram_u_tcur0_tsink)
        PR.register('light', ('tsink', 'tcur0'), seq_peram(peram_u_tcur0_tsink))
        
        for Mom_indx in range(Mom_len):
            VR.register('VDV_0', 'tcur0',  VdV_link[tcur0, Mom_indx * 1:(Mom_indx + 1)* 1])
            
            for i in range(0, len(sink_operator)):
                # if i == 0:
                #     VR.register('VDV_0', 'tsink',  VDV_sink[0:1])
                #     VR.register('VVV_0', 'tsink',  VVV_sink[Mom_indx * 1:(Mom_indx + 1)* 1])

                # elif i == 1:
                #     VR.register('VDV_0', 'tsink',  VDV_sink[Mom_indx * 1:(Mom_indx + 1)* 1])
                #     VR.register('VVV_0', 'tsink',  VVV_sink[0:1])
                
                # elif i == 2:
                VR.register('VVV_0', 'tsink',  VVV_sink[Mom_indx * 1:(Mom_indx + 1)* 1])

                for j in range(4):
                    GR.register('gamma_1',  gamma(j + 1) @ gamma(5))
                    operator_groups = [[s, t] + current_operator for s in sink_operator[i] for t in source_operator[0]]
                    Vindex = ['N', 'N', 'M', 'M']
                    dc = dynamic_contraction(
                        operator_groups, 
                        peram_registry = PR,
                        v_registry = VR,
                        gamma_registry = GR,
                        Cpt = Cpt,
                        Vindex = Vindex,
                        use_equivalence = use_equivalence,
                        ignore_dis = ignore_dis,
                        Projection = Projection,
                        optimize = optimize,
                        # plot = './test/'
                        )
                
                    corr_NJN_GEVP[i, tsrc, tcur0, j, ..., Mom_indx * 1:(Mom_indx + 1)* 1, :] = dc.calculate_all()

        if rank == 0:
            if backend.__name__ == 'cupy':
                backend.cuda.get_current_stream().synchronize()
            print(f'[conf {conf_id}] rank=0 local tsrc={tsrc} tcur={trank[tcur0]} (local_index={tcur0}) completed in {(time.perf_counter() - st_cal_cur):.3f} s', flush = True)

    if backend.__name__ == 'cupy':
        backend.cuda.get_current_stream().synchronize()
    comm.Barrier()
    if rank == 0:
        print(f'[conf {conf_id}] 3pt tsrc={tsrc} ({tsrc + 1}/{Nt}) completed in {(time.perf_counter() - st_cal):.3f} s', flush = True)

if rank == 0:
    print(f'[conf {conf_id}] main loop completed: {Nt} sources in {(time.perf_counter() - st_main):.3f} s', flush = True)

# M8 收集与保存：收集、源时间求和、保存并打印总耗时
comm.Barrier()
st_gather = time.perf_counter()
if rank == 0:
    print(f'[conf {conf_id}] correlator gather started', flush = True)
corr_NJN_GEVP = get_mpi_data(corr_NJN_GEVP, mdtype = 'TGather', root = 0, axis = 2)
    
if rank == 0:
    print(f'[conf {conf_id}] correlator gather completed in {(time.perf_counter() - st_gather):.3f} s', flush = True)
    corr_save_path = f'/public/home/sush/distillation/0v2b/result_nex/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'
    
    import pathlib
    path = pathlib.Path(corr_save_path)
    
    if path.exists():
        print('save_path:',corr_save_path, flush = True)
    
    else:
        path.mkdir(parents = True, exist_ok = True)
        print('mkdir_save_path:',corr_save_path, flush = True)
        
    st_save = time.perf_counter()
    print(f'[conf {conf_id}] source sum and save started', flush = True)
    backend.save(f'{corr_save_path}/corr_3pt_NJN_tsep{tsep}.npy', loop_tsrc(corr_NJN_GEVP, indx = [1, 2], Boundary_Conditions = 'Antiperiodic', Ctype = '3pt', t_sep = tsep))
    print(f'[conf {conf_id}] saved {corr_save_path}/corr_3pt_NJN_tsep{tsep}.npy; source sum/save time {(time.perf_counter() - st_save):.3f} s', flush = True)
    print(f'[conf {conf_id}] total elapsed time {time.perf_counter() - st_total:.3f} s', flush = True)
