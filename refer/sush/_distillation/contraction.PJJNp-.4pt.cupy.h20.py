"""Single-H20 production correlator <O_p J_L,mu J_L,nu O_(n pi-)^dagger>.

Bare Euclidean J_L,mu = ubar gamma_mu (1-gamma5) d / 2, isospin limit.
Prepare tensors in complex128; final Wick contraction and output: complex64.
All 64 sources, independent current offsets 0..21, all 27 current momenta.
Proton and neutron/pion momenta are zero; the currents carry q and -q.
Keep both P+ projected Dirac spin axes, all Lorentz components and all 36
Wick diagrams. No leptons, neutrino kernel, time integration or spin trace.
"""

# M0 Runtime
import os
import sys
import time
from pathlib import Path

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
sys.dont_write_bytecode = True
sys.path.insert(0, "/public/home/sush/distillation/lqcddb_update/src")

import numpy as np
from lqcddb.base.backend import get_backend, set_backend
from lqcddb.base.base_functions import cached_contract, creat_mom_list
from lqcddb.base.mpi_init import getMPISize, mpinit
from lqcddb.constant.gamma_matrix import gamma
from lqcddb.contraction.baroperator import conjugate_operator
from lqcddb.contraction.dynamic import (
    GammaRegistry, PeramRegistry, VRegistry, dynamic_contraction,
    run_wick_analysis, validate_plan,
)
from lqcddb.eigvectors.vertex import vertex_creator

st_total = time.perf_counter()
if len(sys.argv) != 2:
    raise ValueError("Usage: python contraction.PJJNp-.4pt.cupy.h20.py CONF_ID")
conf_id = sys.argv[1]
set_backend("cupy")
backend = get_backend()

# M1 Lattice and single GPU
lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 1]
if getMPISize() != 1:
    raise ValueError("This production program requires one MPI process / GPU")
mpinit(grid_size=grid_size, latt_size=lattice_size, backend="cupy", device=0)
Nx, Ny, Nz, Nt = lattice_size
Nev_src, Nev_max, tsep = 100, 400, 21
Tn = tsep + 1
vertex_q_chunk = 3
# Batch all 27 current momenta in each direct dynamic contraction. The H20
# has room for the larger temporary tensors; vertex preparation stays chunked.
q_chunk = 27
properties = backend.cuda.runtime.getDeviceProperties(0)
gpu_name = properties["name"]
if isinstance(gpu_name, bytes):
    gpu_name = gpu_name.decode()
if "H20" not in gpu_name:
    raise RuntimeError(f"An H20 is required, got {gpu_name}")

ensemble_tag = "beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64"
blending_tag = "e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy"
data_root = Path("/public/home/sush/distillation/0v2b")
eigen_root = data_root / "eigensystem" / ensemble_tag / conf_id
peram_root = data_root / "perambulators" / ensemble_tag / "light" / conf_id
corr_save_path = data_root / "result_nex" / "E32P29" / "Px0Py0Pz0" / f"ENV_{Nev_src}" / f"conf{conf_id}"

output_file = corr_save_path / (
    f"corr_4pt_PJJNpi_hadronic_left_full_tsep{tsep}_npizero_raw_tsrc_tcur0_tcur1"
    f"_nevsrc{Nev_src}_nsrc{Nt}_ksrc0_complex64.npy"
)
partial_file = output_file.with_name(output_file.name + ".partial")
print(f"GPU {gpu_name}; conf={conf_id}; preparation=complex128; contraction/output=complex64", flush=True)

# M2 Momentum and phases, ordered [pz,py,px], exp(-i q.x)
Mom_sink_VDV = [[0, 0, 0]]
for shell in ([0, 0, 1], [0, 1, 1], [1, 1, 1]):
    Mom_sink_VDV += sorted(creat_mom_list(Mom=shell, fix_Q2=True))
Mom_len = len(Mom_sink_VDV)
fun_eigen = vertex_creator(Nx=Nx)
phase_exp_VDV = backend.stack([fun_eigen.phase_exp_VdV(Mom=q) for q in Mom_sink_VDV])
# Only zero-momentum baryon vertices are used at source and sink.
phase_exp_VVV = fun_eigen.phase_exp_VVV(Mom=[0, 0, 0])[None]

# M3 Dynamic manifest, before loading any eigenvectors or perambulators
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
dynamic_plan = run_wick_analysis(
    operator_groups, Cpt=Cpt, Vindex=Vindex, Gindex=Gindex,
    use_equivalence=use_equivalence, ignore_dis=ignore_dis, verbose=False,
)
required_peram = {(p[2], tuple(p[4])) for e in dynamic_plan for p in e[2]["peram"][e[3]]}
required_v = {(v[1], v[3]) for e in dynamic_plan for v in e[2]["V"]}
required_gamma = {g[1] for e in dynamic_plan for g in e[2]["gamma_pos"]}
# required peram (light): (src,src),(sink,src),(cur0/1,src),(src,cur0/1),(sink,cur0/1).
# required V (per-time names): VVV_0@src/sink; VDV_0@src/cur0/cur1.
# required gamma: gamma_7,gamma_5,gamma_mu,gamma_nu,Projector.
# peram axes: (spin_quark,spin_antiquark,ev_quark,ev_antiquark), 100 modes
# at hadron ends and 400 at current ends; VDV=(q,ev_bar,ev), VVV=(p,ev0,ev1,ev2).
# output axes: (tsrc,dt_mu,dt_nu,spin_F,spin_I,mu,nu,p_F=0,q,k_rel=0).
# Source conjugation coefficient is +1, with reversed pion/neutron factor order.
if len(dynamic_plan) != 36:
    raise RuntimeError(f"Expected all 36 Wick diagrams, got {len(dynamic_plan)}")

# M4 Vertex data: keep all preparation and blending weights in complex128
st_eigen = time.perf_counter()
VdV_link = backend.empty((Nt, Mom_len, Nev_max, Nev_max), dtype=backend.complex128)
sink_VVV = backend.empty((Nt, 1, Nev_src, Nev_src, Nev_src), dtype=backend.complex128)
for absolute_time in range(Nt):
    eigen_file = eigen_root / f"{conf_id}_t{absolute_time:03d}_{blending_tag}.npy"
    eigvecs = backend.asarray(np.load(eigen_file)[:Nev_max], dtype=backend.complex128)
    for q_start in range(0, Mom_len, vertex_q_chunk):
        q_stop = min(q_start + vertex_q_chunk, Mom_len)
        VdV_link[absolute_time, q_start:q_stop] = fun_eigen.Mom_VdV_sink_t(
            phase_exp=phase_exp_VDV[q_start:q_stop], eigvecs=eigvecs,
        )
    sink_VVV[absolute_time] = fun_eigen.Mom_VVV_sink_t(
        phase_exp=phase_exp_VVV, eigvecs=eigvecs[:Nev_src],
    )
    del eigvecs
omega = fun_eigen.create_omega_accelerate(
    exact=100, N_eigen=[100, 200, 400, 700, 1100], N_sum=[20] * 5,
    N_extract=[10] * 5, noise=200,
)
VdV_link *= omega[:Nev_max, :Nev_max]
backend.cuda.get_current_stream().synchronize()
vertex_seconds = time.perf_counter() - st_eigen
del omega, phase_exp_VDV, phase_exp_VVV
backend.get_default_memory_pool().free_all_blocks()
print(f"TIME vertices={vertex_seconds:.3f}s", flush=True)

raw_shape = (Nt, Tn, Tn, 4, 4, 4, 4, 1, Mom_len, 1)
corr_save_path.mkdir(parents=True, exist_ok=True)
# Publish the final filename only after every source has been computed.
corr_4pt_raw = np.lib.format.open_memmap(partial_file, mode="w+", dtype=np.complex64, shape=raw_shape)
peram_total = registry_total = contraction_total = save_total = 0.0

for tsrc in range(Nt):
    st_source = time.perf_counter()
    tsink = (tsrc + tsep) % Nt
    current_times = [(tsrc + dt) % Nt for dt in range(Tn)]

    # M5 Peram data: full input time axes, indexed by GLOBAL absolute time
    st_load = time.perf_counter()
    peram_src = np.load(peram_root / f"t{tsrc:03d}_{blending_tag}.npy")
    peram_sink = np.load(peram_root / f"t{tsink:03d}_{blending_tag}.npy")
    expected_peram_shape = (Nt, 4, 4, Nev_max, Nev_max)
    if peram_src.shape != expected_peram_shape or peram_sink.shape != expected_peram_shape:
        raise ValueError(f"Unexpected perambulator shapes: {peram_src.shape}, {peram_sink.shape}")
    peram_src_src = backend.asarray(np.ascontiguousarray(peram_src[tsrc, :, :, :Nev_src, :Nev_src]), dtype=backend.complex128)
    peram_sink_src = backend.asarray(np.ascontiguousarray(peram_src[tsink, :, :, :Nev_src, :Nev_src]), dtype=backend.complex128)
    gamma5_backend = backend.asarray(gamma(5), dtype=backend.complex128)
    peram_tcur_src_cache, peram_src_tcur_cache, peram_sink_tcur_cache = [], [], []
    for current_time in current_times:
        forward = backend.asarray(np.ascontiguousarray(peram_src[current_time, :, :, :Nev_max, :Nev_src]), dtype=backend.complex128)
        to_sink = backend.asarray(np.ascontiguousarray(peram_sink[current_time, :, :, :Nev_max, :Nev_src]), dtype=backend.complex128)
        peram_tcur_src_cache.append(forward)
        peram_src_tcur_cache.append(cached_contract("ab,bcef,cd->dafe", gamma5_backend, forward.conj(), gamma5_backend))
        peram_sink_tcur_cache.append(cached_contract("ab,bcef,cd->dafe", gamma5_backend, to_sink.conj(), gamma5_backend))
    del peram_src, peram_sink, forward, to_sink
    backend.cuda.get_current_stream().synchronize()
    peram_seconds = time.perf_counter() - st_load
    peram_total += peram_seconds

    # M6 Registries: cast the fully prepared tensors at the contraction boundary
    st_registry = time.perf_counter()
    GR, VR, PR = GammaRegistry(), VRegistry(), PeramRegistry()
    gamma_vector = backend.stack([backend.asarray(gamma(mu), dtype=backend.complex128) for mu in (1, 2, 3, 4)])
    gamma_current = 0.5 * (gamma_vector - gamma_vector @ gamma5_backend)
    p_plus = 0.5 * (backend.asarray(gamma(0), dtype=backend.complex128) + backend.asarray(gamma(4), dtype=backend.complex128))
    GR.register("gamma_7", backend.asarray(gamma(7), dtype=backend.complex64))
    GR.register("gamma_5", gamma5_backend.astype(backend.complex64))
    GR.register("gamma_mu", gamma_current.astype(backend.complex64))
    GR.register("gamma_nu", gamma_current.astype(backend.complex64))
    GR.register("Projector", {
        "sink": [p_plus.conj().astype(backend.complex64)],
        "source": [p_plus.astype(backend.complex64)], "contraction": [],
    })
    peram_tcur_src_cache = [a.astype(backend.complex64) for a in peram_tcur_src_cache]
    peram_src_tcur_cache = [a.astype(backend.complex64) for a in peram_src_tcur_cache]
    peram_sink_tcur_cache = [a.astype(backend.complex64) for a in peram_sink_tcur_cache]
    PR.register("light", ("tsrc", "tsrc"), peram_src_src.astype(backend.complex64))
    PR.register("light", ("tsink", "tsrc"), peram_sink_src.astype(backend.complex64))
    VR.register("VDV_0", "tsrc", VdV_link[tsrc, 0:1, :Nev_src, :Nev_src].transpose(0, 2, 1).conj().astype(backend.complex64))
    VR.register("VVV_0", "tsrc", sink_VVV[tsrc].conj().astype(backend.complex64))
    VR.register("VVV_0", "tsink", sink_VVV[tsink].astype(backend.complex64))
    # Convert each INPUT vertex once per source, before any Wick contraction.
    # Keep both q and -q orientations contiguous; no contraction intermediates
    # or diagram values are cached or reused between calculate_all() calls.
    current_VDV = [VdV_link[t].astype(backend.complex64) for t in current_times]
    current_VDV_reverse = [
        VdV_link[t].transpose(0, 2, 1).conj().astype(backend.complex64, order="C")
        for t in current_times
    ]
    del peram_src_src, peram_sink_src
    backend.cuda.get_current_stream().synchronize()
    registry_seconds = time.perf_counter() - st_registry
    registry_total += registry_seconds

    # M7 Complete current-time plane; independent mu/nu, q/-q, no spin trace
    backend.cuda.get_current_stream().synchronize()
    st_cal = time.perf_counter()
    # The complete per-source result is only about 26 MiB. Transfer it once
    # after the current-time plane, avoiding one synchronization per q block.
    corr_4pt_gpu = backend.empty(raw_shape[1:], dtype=backend.complex64)
    # Modulo two, the three proton fermion ends determine the wrap parity;
    # each two-fermion current is even. Apply this sign once, after all points.
    boundary_sign = -1 if tsrc + tsep >= Nt else 1
    dc = None
    for i, tcur0 in enumerate(current_times):
        PR.register("light", ("tcur0", "tsrc"), peram_tcur_src_cache[i])
        PR.register("light", ("tsrc", "tcur0"), peram_src_tcur_cache[i])
        PR.register("light", ("tsink", "tcur0"), peram_sink_tcur_cache[i])
        for j, tcur1 in enumerate(current_times):
            PR.register("light", ("tcur1", "tsrc"), peram_tcur_src_cache[j])
            PR.register("light", ("tsrc", "tcur1"), peram_src_tcur_cache[j])
            PR.register("light", ("tsink", "tcur1"), peram_sink_tcur_cache[j])
            for q_start in range(0, Mom_len, q_chunk):
                q_stop = min(q_start + q_chunk, Mom_len)
                VR.register("VDV_0", "tcur0", current_VDV[i][q_start:q_stop])
                VR.register("VDV_0", "tcur1", current_VDV_reverse[j][q_start:q_stop])
                if dc is None:
                    missing = validate_plan(dynamic_plan, peram_registry=PR, v_registry=VR, gamma_registry=GR)
                    if missing:
                        raise RuntimeError(f"Missing contraction inputs: {missing}")
                    dc = dynamic_contraction(
                        operator_groups, peram_registry=PR, v_registry=VR, gamma_registry=GR,
                        Cpt=Cpt, Vindex=Vindex, Gindex=Gindex,
                        use_equivalence=use_equivalence, ignore_dis=ignore_dis,
                        Projection=True, optimize=optimize, verbose=False,
                    )
                value = dc.calculate_all()
                expected_shape = (4, 4, 4, 4, 1, q_stop-q_start, 1)
                if value.dtype != np.dtype(np.complex64) or tuple(value.shape) != expected_shape:
                    raise RuntimeError(f"Unexpected contraction output: {value.shape}, {value.dtype}")
                corr_4pt_gpu[i, j, ..., q_start:q_stop, :] = value
    backend.cuda.get_current_stream().synchronize()
    contraction_seconds = time.perf_counter() - st_cal
    contraction_total += contraction_seconds

    # M8 Save every source separately on the raw source axis; no source average
    st_save = time.perf_counter()
    corr_4pt_gpu *= boundary_sign
    corr_4pt = backend.asnumpy(corr_4pt_gpu)
    if not np.isfinite(corr_4pt).all():
        raise RuntimeError(f"Non-finite correlator at source {tsrc}")
    corr_4pt_raw[tsrc] = corr_4pt
    corr_4pt_raw.flush()
    save_seconds = time.perf_counter() - st_save
    save_total += save_seconds
    print(
        f"TIME source={tsrc:03d} peram={peram_seconds:.3f}s contraction={contraction_seconds:.3f}s "
        f"registry={registry_seconds:.3f}s save={save_seconds:.3f}s total={time.perf_counter()-st_source:.3f}s",
        flush=True,
    )
    del corr_4pt, corr_4pt_gpu, current_VDV, current_VDV_reverse
    del peram_tcur_src_cache, peram_src_tcur_cache, peram_sink_tcur_cache
    del PR, VR, GR, dc, value
    # Retain the CuPy allocation pool across sources; freed input/output buffers
    # can be allocated again without repeating cudaMalloc/cudaFree cycles.

corr_4pt_raw.flush()
del corr_4pt_raw
os.replace(partial_file, output_file)
print(
    f"TIME totals vertices={vertex_seconds:.3f}s peram={peram_total:.3f}s "
    f"registry={registry_total:.3f}s contraction={contraction_total:.3f}s "
    f"save={save_total:.3f}s wall={time.perf_counter()-st_total:.3f}s",
    flush=True,
)
print(f"Saved: {output_file}; shape={raw_shape}; dtype=complex64", flush=True)
