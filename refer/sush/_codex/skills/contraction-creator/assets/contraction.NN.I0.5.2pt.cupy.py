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
Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))
# Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))[::-1]

Mom_len = len(Mom_sink_VDV)

phase_exp_2pt = backend.zeros((Mom_len, Nx, Nx, Nx, Nc), dtype = complex)
phase_exp_3pt = backend.zeros((Mom_len, Nx, Nx, Nx), dtype = complex)

for Mom_indx in range(Mom_len):
    phase_exp_2pt[Mom_indx] = fun_eigen.phase_exp_2pt(Mom = Mom_sink_VDV[Mom_indx])
    phase_exp_3pt[Mom_indx] = fun_eigen.phase_exp_3pt(Mom = Mom_sink_VVV[Mom_indx])

Nev_src = 100
Nev_max = 400
tsep = 24
tgap = 6

Tn = tsep - tgap + 2

trank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'Scatter')

# M3 动态清单：在此固定算符、注册表和输出轴
sink_operator = [
    [['|', 'd', 'u', 'gamma_7', 'd', '|']], 
    [[-np.sqrt(1/6), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'], [np.sqrt(1/6), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'], [np.sqrt(2/3), '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|']],
    [[-np.sqrt(1/6), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'], [np.sqrt(1/6), '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'], [np.sqrt(2/3), '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|']]
    ]

source_operator = [
    [[-1, '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|']], 
    [[-np.sqrt(1/6), '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'], [np.sqrt(1/6), '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'], [np.sqrt(2/3), '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'u', '|']],
    [[-np.sqrt(1/6), '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'], [np.sqrt(1/6), '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'], [np.sqrt(2/3), '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'u', '|']]
    ]

Cpt = '2pt'
Projection = False
use_equivalence = False
ignore_dis = False
optimize = ['greedy', 'dp', 'auto']
dynamic_operator_groups = [
    [s, t]
    for sink_group in sink_operator
    for source_group in source_operator
    for s in sink_group
    for t in source_group
]
# 此计划是变量清单：M4 到 M6 只提供其中的 peram/V/gamma 输入。
dynamic_plan = run_wick_analysis(
    dynamic_operator_groups, Cpt=Cpt,
    use_equivalence=use_equivalence, ignore_dis=ignore_dis, verbose=False)
required_peram = sorted({
    (p[2], tuple(p[4]))
    for entry in dynamic_plan
    for p in entry[2]['peram'][entry[3]]
})
required_v = sorted({
    (v[1], v[3])
    for entry in dynamic_plan
    for v in entry[2]['V']
})
required_gamma = sorted({
    g[1]
    for entry in dynamic_plan
    for g in entry[2]['gamma_pos']
})
# 下方 M4 到 M6 必须准确提供这些动态输入。

corr_NN = backend.zeros((Nt, Lt, Ns, Ns, Mom_len, len(sink_operator), len(source_operator)), dtype = complex)


# M4 顶点数据：加载本征矢并构造所需的 VDV/VVV 张量
VdV_sink = backend.zeros((Lt, Mom_len, Nev_src, Nev_src), dtype = complex)
sink_VVV = np.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype = complex)

comm.Barrier()
st_eigen = time.perf_counter()
for tsrc_indx, tsrc in enumerate(trank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')
    
    for Mom_indx in range(Mom_len):
        # VdV_sink[tsrc_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs)
        VdV_sink[tsrc_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs[:Nev_src])
        sink_VVV[tsrc_indx, Mom_indx] = fun_eigen.Mom_VVV_sink_t(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src]).get()

del phase_exp_2pt, phase_exp_3pt

if backend.__name__ == 'cupy':
    backend.cuda.get_current_stream().synchronize()
comm.Barrier()
if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')

# M5 peram 数据：加载并分发计划所需的 perambulator
comm.Barrier()
st_load = time.perf_counter()
if rank == 0:
    peram_u = np.zeros((Nt, Nt, Ns, Ns, Nev_src, Nev_src), dtype = complex)
    for tsrc in range(Nt):
        peram_u[tsrc] = np.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')[..., :Nev_src, :Nev_src]
else:
    peram_u = None
    
peram_u = get_mpi_data(peram_u, mdtype = 'TScatter', root = 0, axis = 1)
    
comm.Barrier()
if rank == 0:
    print(f'load peram use time {(time.perf_counter() - st_load):.3f} s')

# M6 注册表：只绑定 M3 所需的 gamma/V/peram 名称
GR = GammaRegistry()
VR = VRegistry()
PR = PeramRegistry()

GR.register('gamma_7',  gamma(7))
GR.register('gamma_5',  gamma(5))
GR.register('Projector',  (gamma(4) + gamma(0))/2.0)

# M7 动态循环：绑定当前切片并执行计划
for tsrc in range(Nt):
    st_cal = time.perf_counter()
    tsink_list, _, tsink_indx = get_mpi_tlist(Nt = Nt, t = [x for x in range(tsrc, tsrc + Nt//2, 1)], gtype = 'TScatter')
    
    VVV_src = backend.asarray(get_mpi_data(data = sink_VVV[tsrc//size, :], mdtype = 'Bcast', root = tsrc%size))
    VDV_src = backend.asarray(get_mpi_data(data = VdV_sink[tsrc//size, :, :Nev_src, :Nev_src], mdtype = 'Bcast', root = tsrc%size))
    
    peram_u_src_src = get_mpi_data(data = peram_u[tsrc, tsrc//size, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = tsrc%size)
    PR.register('light', ('tsrc', 'tsrc' ), backend.asarray(peram_u_src_src))
    for i in range(0, 3):
        for j in range(0, 3):
            for _tsink, tsink in enumerate(tsink_indx):
                PR.register('light', ('tsink', 'tsrc'), backend.asarray(peram_u[tsrc, tsink]))
                PR.register('light', ('tsrc', 'tsink' ), seq_peram(backend.asarray(peram_u[tsrc, tsink])))
                PR.register('light', ('tsink', 'tsink' ), backend.asarray(peram_u[tsink_list[_tsink], tsink]))

                if i == 0:
                    VR.register('VVV_0', 'tsink',  backend.asarray(sink_VVV[tsink, :]))

                elif i == 1:
                    VR.register('VDV_0', 'tsink',  VdV_sink[tsink, 0:1])
                    VR.register('VVV_0', 'tsink',  backend.asarray(sink_VVV[tsink, :]))

                elif i == 2:
                    VR.register('VDV_0', 'tsink',  VdV_sink[tsink])
                    VR.register('VVV_0', 'tsink',  backend.asarray(sink_VVV[tsink, 0:1]))

                if j == 0:
                    VR.register('VVV_0', 'tsrc',  backend.asarray(VVV_src).conj())

                elif j == 1:
                    VR.register('VDV_0', 'tsrc',  VDV_src[0:1].transpose(0, 2, 1).conj())
                    VR.register('VVV_0', 'tsrc',  backend.asarray(VVV_src).conj())
                    
                elif j == 2:
                    VR.register('VDV_0', 'tsrc',  VDV_src.transpose(0, 2, 1).conj())
                    VR.register('VVV_0', 'tsrc',  backend.asarray(VVV_src[0:1]).conj())
                
                operator_groups = [[s, t] for s in sink_operator[i] for t in source_operator[j]]
                Vindex = ['M']*2 if (i + 1) * (j + 1) == 1 else ['M']*3 if (i + 1) * (j + 1)<4 else ['M']*4
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
                    optimize = optimize
                    )

                corr_NN[tsrc, tsink, ..., i, j] = dc.calculate_all()
                
    if backend.__name__ == 'cupy':
        backend.cuda.get_current_stream().synchronize()
    comm.Barrier()
    if rank == 0:
        print(f'calculate 2pt of tsrc {tsrc} use time {(time.perf_counter() - st_cal):.3f} s')

# M8 收集与保存：收集、源时间平均、保存并打印总耗时
comm.Barrier()
corr_NN = get_mpi_data(corr_NN, mdtype = 'TGather', root = 0, axis = 1)

if rank == 0:
    corr_save_path = f'/public/home/sush/distillation/0v2b/result_nex/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'
    import pathlib
    path = pathlib.Path(corr_save_path)
    if path.exists():
        print('save_path:',corr_save_path)
    
    else:
        path.mkdir(parents = True, exist_ok = True)
        print('mkdir_save_path:',corr_save_path)
        
    backend.save(f'{corr_save_path}/corr_3pt_NN_2pt_src100.npy', loop_tsrc(corr_NN[:], indx = [0, 1], Boundary_Conditions = 'Antiperiodic'))
    print(f'total elapsed time {time.perf_counter() - st_total:.3f} s')
