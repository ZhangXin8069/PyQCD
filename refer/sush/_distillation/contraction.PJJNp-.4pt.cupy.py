"""Finite-volume Euclidean hadronic five-point function for n pi- -> p.

It computes <0|O_p J_L,mu J_L,nu O_(n pi-)^dagger|0>_L with two bare local
left-handed currents J_L=(V_+-A_+)/2. It is the hadronic QCD input to the
light-Majorana n pi- -> p e- e- amplitude, not the complete amplitude: no
neutrino kernel, electron tensor, current-time integration, low-state
Minkowski restoration, or finite-volume matching is performed here.
Current/source and current/sink contact points are saved but must be excluded
or separately matched in the physical analysis.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "/public/home/sush/distillation/lqcddb_update/src")

from lqcddb.base.backend import get_backend, set_backend
from lqcddb.base.base_functions import cached_contract, creat_mom_list
from lqcddb.base.mpi_init import getMPIComm, getMPIRank, getMPISize, get_mpi_data, get_mpi_tlist, mpinit
from lqcddb.constant.constant import Nc
from lqcddb.constant.gamma_matrix import gamma
from lqcddb.contraction.baroperator import conjugate_operator
from lqcddb.contraction.dynamic import GammaRegistry, PeramRegistry, VRegistry, dynamic_contraction, run_wick_analysis
from lqcddb.eigvectors.vertex import vertex_creator


# M0 运行时
program_start = time.perf_counter()
quick_test = os.environ.get("PJJNP_QUICK_TEST", "0") == "1"
set_backend("cupy")
backend = get_backend()

# M1 格点与 MPI
lattice_size = [32, 32, 32, 64]
mpi_size = getMPISize()
grid_size = [1, 1, 1, mpi_size]

if len(sys.argv) != 2:
    raise ValueError("Usage: python contraction.PJJNp-.4pt.cupy.py CONF_ID")

conf_id = sys.argv[1]
mpinit(grid_size=grid_size, latt_size=lattice_size, backend=backend.__name__)

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size
Lt = Nt // size
Nev_src = 100
Nev_max = 400
tsep = 21
Tn = 3 if quick_test else tsep + 1
vertex_q_chunk = 3
q_chunk = 9

if Nt % size != 0:
    raise ValueError(f"Nt must be divisible by MPI size; Nt={Nt}, size={size}")
if size < 2:
    raise ValueError("At least two MPI ranks are required for concurrent perambulator loading")
if Tn < size:
    raise ValueError(f"Tn must be at least MPI size; Tn={Tn}, size={size}")

ensemble_tag = "beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64"
blending_tag = "e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy"
eigen_root = Path("/nexdata/project/lqcd/sush/eigensystem")
peram_root = Path("/nexdata/project/lqcd/sush/perambulators")
output_root = Path("/nexdata/project/lqcd/sush/result")

fun_eigen = vertex_creator(Nx=Nx)

Mom_sink_VDV = [[0, 0, 0]]
Mom_sink_VDV += sorted(creat_mom_list(Mom=[0, 0, 1], fix_Q2=True))
Mom_sink_VDV += sorted(creat_mom_list(Mom=[0, 1, 1], fix_Q2=True))
Mom_sink_VDV += sorted(creat_mom_list(Mom=[1, 1, 1], fix_Q2=True))
Mom_sink_VVV = list(Mom_sink_VDV)
if quick_test:
    Mom_sink_VDV = [[0, 0, 0]]
    Mom_sink_VVV = [[0, 0, 0]]
Mom_len = len(Mom_sink_VDV)

# 快速一致性测试：只取一个源、一个流时间对和零动量；默认关闭，物理运行不变。
# Nπ 外部态动量接口："zero" 保持默认的零动量收缩；"all" 收缩全部 27 个态。
npi_momentum_mode = "zero"
if npi_momentum_mode not in {"zero", "all"}:
    raise ValueError("npi_momentum_mode must be 'zero' or 'all'")

npi_momentum_indices = [0] if npi_momentum_mode == "zero" else list(range(Mom_len))
npi_momentum_len = len(npi_momentum_indices)

# M2 动量与相位
vertex_start = time.perf_counter()
phase_exp_VDV = backend.zeros((Mom_len, Nx, Ny, Nz, Nc), dtype=complex)
for Mom_index in range(Mom_len):
    phase_exp_VDV[Mom_index] = fun_eigen.phase_exp_VdV(Mom=Mom_sink_VDV[Mom_index])

phase_exp_VVV = backend.zeros((Mom_len, Nx, Ny, Nz), dtype=complex)
for Mom_index in range(Mom_len):
    phase_exp_VVV[Mom_index] = fun_eigen.phase_exp_VVV(Mom=Mom_sink_VVV[Mom_index])
trank, _, _ = get_mpi_tlist(Nt=Nt, t=range(Nt), gtype="TScatter")
VdV_link = backend.zeros((Lt, Mom_len, Nev_max, Nev_max), dtype=complex)
sink_VVV = backend.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype=complex)

eigen_load_time = 0.0
vdv_time = 0.0
vvv_time = 0.0

for local_time_index, absolute_time in enumerate(trank):
    eigen_file = eigen_root / ensemble_tag / conf_id / f"{conf_id}_t{absolute_time:03d}_{blending_tag}.npy"

    step_start = time.perf_counter()
    eigvecs = backend.load(str(eigen_file))
    backend.cuda.get_current_stream().synchronize()
    eigen_load_time += time.perf_counter() - step_start

    step_start = time.perf_counter()
    for q_start in range(0, Mom_len, vertex_q_chunk):
        q_stop = min(q_start + vertex_q_chunk, Mom_len)
        VdV_link[local_time_index, q_start:q_stop] = fun_eigen.Mom_VdV_sink_t(phase_exp=phase_exp_VDV[q_start:q_stop], eigvecs=eigvecs[:Nev_max])
    backend.cuda.get_current_stream().synchronize()
    vdv_time += time.perf_counter() - step_start

    step_start = time.perf_counter()
    sink_VVV[local_time_index] = fun_eigen.Mom_VVV_sink_t(phase_exp=phase_exp_VVV, eigvecs=eigvecs[:Nev_src])
    backend.cuda.get_current_stream().synchronize()
    vvv_time += time.perf_counter() - step_start
    del eigvecs

step_start = time.perf_counter()
omega = fun_eigen.create_omega_accelerate(exact=100, N_eigen=[100, 200, 400, 700, 1100], N_sum=[20, 20, 20, 20, 20], N_extract=[10, 10, 10, 10, 10], noise=200)
VdV_link *= omega[:Nev_max, :Nev_max]
backend.cuda.get_current_stream().synchronize()
omega_time = time.perf_counter() - step_start
vertex_time = time.perf_counter() - vertex_start

del omega, phase_exp_VDV, phase_exp_VVV
backend.get_default_memory_pool().free_all_blocks()

local_vertex_times = np.asarray([eigen_load_time, vdv_time, vvv_time, omega_time, vertex_time])
vertex_times = np.asarray(get_mpi_data(local_vertex_times, mdtype="Allgather")).reshape(size, -1)

if rank == 0:
    print(
        f"TIME vertices eigen_load={vertex_times[:, 0].max():.3f}s VDV={vertex_times[:, 1].max():.3f}s VVV={vertex_times[:, 2].max():.3f}s omega={vertex_times[:, 3].max():.3f}s total={vertex_times[:, 4].max():.3f}s",
        flush=True,
    )

# M3 dynamic_manifest：五点函数具有 (t_sink,t_cur0,t_cur1,t_src) 四个时间标签。
# 固定 t_sink=t_src+tsep 后，两个流时间独立；源时间在 M8 保留而不立即平均。
initial_npi_operator = ["|", "d", "u", "gamma_7", "d", "|", "|", "u^d", "gamma_5", "d", "|"]
source_operator = conjugate_operator(initial_npi_operator)
sink_operator = ["|", "u", "u", "gamma_7", "d", "|"]
current_operator = ["|", "u^d", "gamma_mu", "d", "|", "|", "u^d", "gamma_nu", "d", "|"]
operator_groups = [[sink_operator, source_operator, current_operator]]

Cpt = "4pt"
Vindex = ["N", "K", "K", "M", "M"]
Gindex = ["", "G", "H", "", "", ""]
use_equivalence = False
ignore_dis = False
optimize = ["greedy", "dp", "auto"]

expected_source_operator = [1.0, "|", "d^d", "gamma_5", "u", "|", "|", "u^d", "gamma_7", "d^d", "d^d", "|"]
if source_operator != expected_source_operator:
    raise RuntimeError(f"Unexpected source operator: {source_operator}")

GR = GammaRegistry()
gamma5_backend = backend.asarray(gamma(5))
gamma_vector = backend.asarray([gamma(mu) for mu in (1, 2, 3, 4)])
gamma_axial = backend.asarray([gamma(mu) @ gamma(5) for mu in (1, 2, 3, 4)])
gamma_current = 0.5 * (gamma_vector - gamma_axial)

p_plus = (backend.asarray(gamma(0)) + backend.asarray(gamma(4))) / 2.0
# spin_map = np.sqrt(2.0) * p_plus[:, :2]
# spin_identity_residual = backend.max(backend.abs(spin_map.conj().T @ spin_map - backend.eye(2)))
# spin_projector_residual = backend.max(backend.abs(spin_map @ spin_map.conj().T - p_plus))
# spin_map_residual = max(float(spin_identity_residual.item()), float(spin_projector_residual.item()))
# if spin_map_residual > 1.0e-13:
#     raise RuntimeError(f"Positive-energy spin-map residual={spin_map_residual:.3e}")

GR.register("gamma_7", backend.asarray(gamma(7)))
GR.register("gamma_5", gamma5_backend)
GR.register("gamma_mu", gamma_current)
GR.register("gamma_nu", gamma_current)
GR.register("Projector", {
    # Projector 专用结构：第一维与原自旋收缩，第二维保留。
    # contraction=[] 明确禁止 dynamic 收缩投影矩阵的剩余维度。
    "sink": [p_plus.conj()],
    "source": [p_plus],
    "contraction": ['0-0'],
})

max_local_rows = (Tn + size - 1) // size
# 输出外部动量轴为 (p_sink, q_cur, p_Npi_src)：汇端质子固定零动量，
# 只有源端 Nπ 动量由 npi_momentum_mode 控制。
current_shape = (Tn, Tn, 4, 4, 1, Mom_len, npi_momentum_len)
raw_shape = (Nt,) + current_shape
corr_4pt_raw = backend.zeros(raw_shape, dtype=complex) if rank == 0 else None

peram_read_total = 0.0
peram_distribution_total = 0.0
vertex_distribution_total = 0.0
contraction_total = 0.0
gather_average_total = 0.0

for tsrc in range(Nt):
    source_start = time.perf_counter()
    tsink = (tsrc + tsep) % Nt
    current_times = [(tsrc + dt) % Nt for dt in range(Tn)]
    tcur0_times, _, tcur0_indices = get_mpi_tlist(Nt=Nt, t=current_times, gtype="TScatter")
    tcur0_positions = [dt for dt, absolute_time in enumerate(current_times) if absolute_time % size == rank]
    if [current_times[dt] for dt in tcur0_positions] != tcur0_times:
        raise RuntimeError("TScatter changed the outer-current ordering")

    corr_4pt = backend.zeros((max_local_rows, Tn, 4, 4, 1, Mom_len, npi_momentum_len), dtype=complex)
    corr_4pt_labels = np.full(max_local_rows, -1, dtype=np.int64)
    corr_4pt_labels[: len(tcur0_positions)] = tcur0_positions

    src_owner, src_index = get_mpi_tlist(Nt=Nt, t=tsrc, gtype="find")
    sink_owner, sink_index = get_mpi_tlist(Nt=Nt, t=tsink, gtype="find")
    peram_src_file = peram_root / ensemble_tag / "light" / conf_id / f"t{tsrc:03d}_{blending_tag}.npy"
    peram_sink_file = peram_root / ensemble_tag / "light" / conf_id / f"t{tsink:03d}_{blending_tag}.npy"

    step_start = time.perf_counter()
    if rank == 0:
        peram_u_src_root = backend.load(str(peram_src_file))
        peram_u_src_root = peram_u_src_root[..., :Nev_max, :Nev_src].copy()
    else:
        peram_u_src_root = None
    if rank == 1:
        peram_u_sink_root = backend.load(str(peram_sink_file))
        peram_u_sink_root = peram_u_sink_root[..., :Nev_max, :Nev_src].copy()
    else:
        peram_u_sink_root = None
    backend.cuda.get_current_stream().synchronize()
    comm.Barrier()
    peram_read_time = time.perf_counter() - step_start
    peram_read_total += peram_read_time

    step_start = time.perf_counter()
    source_payload = peram_u_src_root[tsrc] if rank == 0 else None
    sink_payload = peram_u_src_root[tsink] if rank == 0 else None
    peram_src_src = backend.asarray(get_mpi_data(source_payload, mdtype="Bcast", root=0))
    peram_sink_src = backend.asarray(get_mpi_data(sink_payload, mdtype="Bcast", root=0))

    peram_tcur_src_cache = []
    peram_src_tcur_cache = []
    peram_sink_tcur_cache = []
    for current_time in current_times:
        source_payload = peram_u_src_root[current_time] if rank == 0 else None
        sink_payload = peram_u_sink_root[current_time] if rank == 1 else None
        peram_tcur_src = backend.asarray(get_mpi_data(source_payload, mdtype="Bcast", root=0))
        peram_tcur_sink = backend.asarray(get_mpi_data(sink_payload, mdtype="Bcast", root=1))
        peram_tcur_src_cache.append(peram_tcur_src)
        peram_src_tcur_cache.append(cached_contract("ab,...bcef,cd->...dafe", gamma5_backend, peram_tcur_src.conj(), gamma5_backend, optimize="auto"))
        peram_sink_tcur_cache.append(cached_contract("ab,...bcef,cd->...dafe", gamma5_backend, peram_tcur_sink.conj(), gamma5_backend, optimize="auto"))
    backend.cuda.get_current_stream().synchronize()
    peram_distribution_time = time.perf_counter() - step_start
    peram_distribution_total += peram_distribution_time

    del peram_u_src_root, peram_u_sink_root
    backend.get_default_memory_pool().free_all_blocks()

    step_start = time.perf_counter()
    source_payload = VdV_link[src_index, npi_momentum_indices, :Nev_src, :Nev_src] if rank == src_owner else None
    source_VdV = backend.asarray(get_mpi_data(source_payload, mdtype="Bcast", root=src_owner)).transpose(0, 2, 1).conj()
    source_payload = sink_VVV[src_index, npi_momentum_indices] if rank == src_owner else None
    source_VVV = backend.asarray(get_mpi_data(source_payload, mdtype="Bcast", root=src_owner)).conj()
    # N 轴属于汇端质子，固定为零动量；M 轴才是源端 Nπ 动量。
    sink_payload = sink_VVV[sink_index, 0:1] if rank == sink_owner else None
    sink_VVV_t = backend.asarray(get_mpi_data(sink_payload, mdtype="Bcast", root=sink_owner))

    VdV_tcur_cache = []
    for current_time in current_times:
        current_owner, current_index = get_mpi_tlist(Nt=Nt, t=current_time, gtype="find")
        current_payload = VdV_link[current_index] if rank == current_owner else None
        current_VdV = get_mpi_data(current_payload, mdtype="Bcast", root=current_owner)
        VdV_tcur_cache.append(backend.asarray(current_VdV))
    backend.cuda.get_current_stream().synchronize()
    vertex_distribution_time = time.perf_counter() - step_start
    vertex_distribution_total += vertex_distribution_time

    PR = PeramRegistry()
    VR = VRegistry()
    # A single light perambulator implements the exact mu=md, no-QED isospin limit.
    PR.register("light", ("tsrc", "tsrc"), peram_src_src[..., :Nev_src, :Nev_src])
    PR.register("light", ("tsink", "tsrc"), peram_sink_src[..., :Nev_src, :Nev_src])
    VR.register("VDV_0", "tsrc", source_VdV)
    VR.register("VVV_0", "tsrc", source_VVV)
    VR.register("VVV_0", "tsink", sink_VVV_t)

    # M7 动态循环
    contraction_start = time.perf_counter()
    dycn_4pt = None
    for local_row, (tcur0_position, tcur0_index) in enumerate(zip(tcur0_positions, tcur0_indices)):
        PR.register("light", ("tcur0", "tsrc"), peram_tcur_src_cache[tcur0_position])
        PR.register("light", ("tsrc", "tcur0"), peram_src_tcur_cache[tcur0_position])
        PR.register("light", ("tsink", "tcur0"), peram_sink_tcur_cache[tcur0_position])

        for tcur1_position in range(Tn):
            PR.register("light", ("tcur1", "tsrc"), peram_tcur_src_cache[tcur1_position])
            PR.register("light", ("tsrc", "tcur1"), peram_src_tcur_cache[tcur1_position])
            PR.register("light", ("tsink", "tcur1"), peram_sink_tcur_cache[tcur1_position])

            for q_start in range(0, Mom_len, q_chunk):
                q_stop = min(q_start + q_chunk, Mom_len)
                # phase_exp_VdV uses exp(-i q.x); the adjoint second vertex carries -q.
                VR.register("VDV_0", "tcur0", VdV_link[tcur0_index, q_start:q_stop])
                current_VdV = VdV_tcur_cache[tcur1_position][q_start:q_stop].transpose(0, 2, 1).conj()
                VR.register("VDV_0", "tcur1", current_VdV)

                if dycn_4pt is None:
                    dycn_4pt = dynamic_contraction(
                        operator_groups,
                        peram_registry=PR,
                        v_registry=VR,
                        gamma_registry=GR,
                        Cpt=Cpt,
                        Vindex=Vindex,
                        Gindex=Gindex,
                        use_equivalence=use_equivalence,
                        ignore_dis=ignore_dis,
                        Projection=True,
                        optimize=optimize,
                        verbose=True,
                        max_detail=-1,
                    )
                    if len(dycn_4pt) != 36:
                        raise RuntimeError(f"Expected 36 Wick diagrams, got {len(dycn_4pt)}")

                value = dycn_4pt.calculate_all()
                expected_shape = (4, 4, 1, q_stop - q_start, npi_momentum_len)
                if tuple(value.shape) != expected_shape:
                    raise RuntimeError(f"Dynamic shape {value.shape}, expected {expected_shape}")
                corr_4pt[local_row, tcur1_position, ..., q_start:q_stop, :] = value

    backend.cuda.get_current_stream().synchronize()
    contraction_time = time.perf_counter() - contraction_start
    contraction_total += contraction_time

    gather_start = time.perf_counter()
    corr_4pt_gathered = get_mpi_data(corr_4pt, mdtype="TGather", root=0, axis=0)
    corr_4pt_gathered_labels = get_mpi_data(corr_4pt_labels, mdtype="TGather", root=0, axis=0)

    if rank == 0:
        valid_labels = np.asarray(corr_4pt_gathered_labels)
        valid_labels = valid_labels[valid_labels >= 0]
        if len(valid_labels) != Tn or len(set(map(int, valid_labels))) != Tn or sorted(map(int, valid_labels)) != list(range(Tn)):
            raise RuntimeError("TGather labels do not cover every relative current time once")
        # M8：保存每个源时间；反周期 wrap 符号只施加一次。
        boundary_sign = -1.0 if tsrc + tsep >= Nt else 1.0
        for gathered_row, relative_row in enumerate(corr_4pt_gathered_labels):
            if relative_row >= 0:
                corr_4pt_raw[tsrc, int(relative_row)] = boundary_sign * corr_4pt_gathered[gathered_row]
        backend.cuda.get_current_stream().synchronize()

    gather_average_time = time.perf_counter() - gather_start
    gather_average_total += gather_average_time
    source_time = time.perf_counter() - source_start

    local_source_times = np.asarray([peram_read_time, peram_distribution_time, vertex_distribution_time, contraction_time, gather_average_time, source_time])
    source_times = np.asarray(get_mpi_data(local_source_times, mdtype="Allgather")).reshape(size, -1)
    if rank == 0:
        print(
            f"TIME source={tsrc:03d} peram_read={source_times[:, 0].max():.3f}s peram_distribute={source_times[:, 1].max():.3f}s vertex_distribute={source_times[:, 2].max():.3f}s contraction={source_times[:, 3].max():.3f}s gather_average={source_times[:, 4].max():.3f}s total={source_times[:, 5].max():.3f}s",
            flush=True,
        )

    del corr_4pt, corr_4pt_gathered, corr_4pt_gathered_labels, corr_4pt_labels
    del peram_src_src, peram_sink_src
    del peram_tcur_src_cache, peram_src_tcur_cache, peram_sink_tcur_cache
    del source_VdV, source_VVV, sink_VVV_t, VdV_tcur_cache
    del PR, VR, dycn_4pt, value
    backend.get_default_memory_pool().free_all_blocks()

if rank == 0:
    corr_4pt = corr_4pt_raw
    if tuple(corr_4pt.shape) != raw_shape:
        raise RuntimeError(f"Final shape {corr_4pt.shape}, expected {raw_shape}")
    if not bool(backend.all(backend.isfinite(corr_4pt)).item()):
        raise RuntimeError("Final correlator contains non-finite values")

    corr_save_path = output_root / "E32P29" / "Px0Py0Pz0" / f"ENV_{Nev_src}" / f"conf{conf_id}"
    corr_save_path.mkdir(parents=True, exist_ok=True)
    output_suffix = "_quicktest" if quick_test else ""
    output_file = corr_save_path / f"corr_4pt_PJJNpi_hadronic_left_full_tsep{tsep}_npi{npi_momentum_mode}_raw_tsrc_tcur0_tcur1_nevsrc{Nev_src}_nsrc{Nt}_ksrc0{output_suffix}.npy"
    save_start = time.perf_counter()
    backend.save(output_file, corr_4pt)
    save_time = time.perf_counter() - save_start
else:
    save_time = 0.0

comm.Barrier()
local_totals = np.asarray([peram_read_total, peram_distribution_total, vertex_distribution_total, contraction_total, gather_average_total, save_time, time.perf_counter() - program_start])
total_times = np.asarray(get_mpi_data(local_totals, mdtype="Allgather")).reshape(size, -1)

if rank == 0:
    print(
        f"TIME totals vertices={vertex_times[:, 4].max():.3f}s peram_read={total_times[:, 0].max():.3f}s peram_distribute={total_times[:, 1].max():.3f}s vertex_distribute={total_times[:, 2].max():.3f}s contraction={total_times[:, 3].max():.3f}s gather_average={total_times[:, 4].max():.3f}s save={total_times[:, 5].max():.3f}s wall={total_times[:, 6].max():.3f}s",
        flush=True,
    )
    print(f"Saved: {output_file}", flush=True)
    if quick_test:
        print("QUICK_TEST: one source, one tcur0, one tcur1, zero momentum", flush=True)
    print(f"Shape: {corr_4pt.shape}; dtype={corr_4pt.dtype}", flush=True)
