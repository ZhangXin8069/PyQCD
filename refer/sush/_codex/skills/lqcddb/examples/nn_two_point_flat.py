#!/usr/bin/env python3
"""Flat NN two-point production template for ``lqcddb_update``.

This is the two-point companion to ``njn_three_point_flat.py``.  It keeps the
same top-level modules and visible data flow; only the operator, registry and
dynamic-contraction part changes because a two-point group has no current.
Copy the sections for another two-point observable and replace the marked
operator/momentum/path rows inline.  No orchestration helper functions are
used.
"""

# ============================================================================
# MODULE 1: imports.
# ============================================================================
import os
import sys
import time
from pathlib import Path

import numpy as np

from lqcddb.base import (
    creat_mom_list,
    get_backend,
    get_mpi_data,
    get_mpi_tlist,
    getMPIRank,
    getMPISize,
    mpinit,
    set_backend,
)
from lqcddb.constant import Nc, Ns, gamma
from lqcddb.contraction import (
    GammaRegistry,
    PeramRegistry,
    VRegistry,
    conjugate_operator,
    dynamic_contraction,
    seq_peram,
)
from lqcddb.eigvectors import vertex_creator
from lqcddb.analyse import loop_tsrc

# ============================================================================
# MODULE 2: runtime parameters, paths, and output contract.
# ============================================================================
if len(sys.argv) < 2:
    raise SystemExit("usage: nn_two_point_flat.py CONF_ID")

conf_id = sys.argv[1]
BACKEND_NAME = os.environ.get("LQCDB_BACKEND", "cupy")
LATTICE_SIZE = [32, 32, 32, 64]
GRID_SIZE = [1, 1, 1, 2]
ENSEMBLE = "E32P29"
CPT = "2pt"
ENSEMBLE_DIR = (
    "beta6.308_mu-0.2510_ms-0.2170_u00.859727_"
    "u0s0.954467_L32x64"
)
EIGEN_ROOT = Path("/nexdata/project/lqcd/sush/eigensystem") / ENSEMBLE_DIR
PERAM_ROOT = Path("/nexdata/project/lqcd/sush/perambulators") / ENSEMBLE_DIR
OUTPUT_ROOT = Path("/public/home/sush/distillation/0v2b/result_nex")

NX, NY, NZ, NT = LATTICE_SIZE
NEV_HADRON = 100
NEV_STORED = 400
MOMENTUM_CHUNK = 1
USE_EQUIVALENCE = False
IGNORE_DISCONNECTED = False
BOUNDARY_CONDITION = "Antiperiodic"
FULL_CORRELATOR_WRAP_PHASE = -1.0  # three baryon lines give odd winding parity
SOURCE_ESTIMATOR = "source_sum"       # no implicit division by NT
OUTPUT_AXES_RAW = (
    "operator_group", "t_src_abs", "t_sink_local",
    "sink_projected_spin", "source_projected_spin", "momentum",
)
OUTPUT_AXES_SAVED = (
    "operator_group", "source_sum_singleton", "t_sink_minus_t_src",
    "sink_projected_spin", "source_projected_spin", "momentum",
)
PERAM_FILE_SUFFIX = (
    "_e100_s[100, 200, 400, 700, 1100]_"
    "[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy"
)

# ============================================================================
# MODULE 3: backend/MPI initialization.
# ============================================================================
set_backend(BACKEND_NAME)
backend = get_backend()
mpinit(grid_size=GRID_SIZE, latt_size=LATTICE_SIZE, backend=backend.__name__)
rank = getMPIRank()
size = getMPISize()
if size != int(np.prod(GRID_SIZE)) or NT % size != 0:
    raise RuntimeError(f"incompatible MPI layout: size={size}, grid={GRID_SIZE}, Nt={NT}")
LT = NT // size
fun_eigen = vertex_creator(Nx=NX)

# ============================================================================
# MODULE 4: momentum table and phases.
# ============================================================================
Mom_sink = (
    [[0, 0, 0]]
    + sorted(creat_mom_list(Mom=[0, 0, 1], fix_Q2=True))
    + sorted(creat_mom_list(Mom=[0, 1, 1], fix_Q2=True))
    + sorted(creat_mom_list(Mom=[1, 1, 1], fix_Q2=True))
)
Mom_len = len(Mom_sink)
if Mom_len != 27:
    raise RuntimeError(f"unexpected NN momentum count {Mom_len}")
if MOMENTUM_CHUNK <= 0 or Mom_len % MOMENTUM_CHUNK != 0:
    raise RuntimeError(f"MOMENTUM_CHUNK={MOMENTUM_CHUNK} must divide Mom_len={Mom_len}")
# The source VVV below is the conjugate of the sink construction; its physical
# momentum is therefore the opposite of the listed sink momentum.
phase_sink = backend.zeros((Mom_len, NX, NX, NX), dtype=complex)
for Mom_indx in range(Mom_len):
    phase_sink[Mom_indx] = fun_eigen.phase_exp_VVV(Mom=Mom_sink[Mom_indx])

# ============================================================================
# MODULE 5: local VVV construction.
# ============================================================================
times_owned, owner_labels, local_time_indices = get_mpi_tlist(
    Nt=NT, t=range(NT), gtype="TScatter"
)
if len(times_owned) != LT or any(owner != rank for owner in owner_labels):
    raise RuntimeError(f"rank {rank}: invalid local time ownership")
sink_VVV_local = backend.zeros(
    (LT, Mom_len, NEV_HADRON, NEV_HADRON, NEV_HADRON), dtype=complex
)
st_vertex = time.perf_counter()
for t_local, t_abs in zip(local_time_indices, times_owned):
    eigen_path = (
        EIGEN_ROOT / conf_id
        / f"{conf_id}_t{t_abs:03d}_e100_s[100, 200, 400, 700, 1100]_"
          "[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy"
    )
    eigvecs = backend.load(eigen_path)
    if eigvecs.ndim != 5 or tuple(eigvecs.shape) != (NEV_STORED, NZ, NY, NX, Nc):
        raise RuntimeError(f"unexpected eigenvector shape at t={t_abs}: {eigvecs.shape}")
    if t_local == local_time_indices[0]:
        exact_rows = eigvecs[:NEV_HADRON].reshape(NEV_HADRON, -1)
        gram_residual = float(backend.max(backend.abs(exact_rows @ exact_rows.conj().T - backend.eye(NEV_HADRON))).item())
        if gram_residual > 1e-9:
            raise RuntimeError(f"t={t_abs}: complex eigenvector Gram residual {gram_residual:.3e}")
    for Mom_indx in range(Mom_len):
        sink_VVV_local[t_local, Mom_indx] = fun_eigen.Mom_VVV_sink_t(
            phase_exp=phase_sink[Mom_indx], eigvecs=eigvecs[:NEV_HADRON]
        )
        if t_local == local_time_indices[0] and Mom_indx == 0:
            vvv = sink_VVV_local[t_local, Mom_indx]
            vvv_residual = max(
                float(backend.max(backend.abs(vvv.swapaxes(-3, -2) + vvv)).item()),
                float(backend.max(backend.abs(vvv.swapaxes(-2, -1) + vvv)).item()),
            )
            if vvv_residual > 1e-9:
                raise RuntimeError(f"t={t_abs}: VVV permutation residual {vvv_residual:.3e}")
if rank == 0:
    print(f"vertex stage: {time.perf_counter() - st_vertex:.3f} s")
del phase_sink

# ============================================================================
# MODULE 6: NN operators, registries, and output axes.
# ============================================================================
SINK_OPERATOR = ["|", "d", "u", "gamma_7", "d", "|"]
INITIAL_OPERATOR = SINK_OPERATOR
SOURCE_OPERATOR = conjugate_operator(INITIAL_OPERATOR)
if SOURCE_OPERATOR[1:] != [
    "|", "u^d", "gamma_7", "d^d", "d^d", "|",
] or complex(SOURCE_OPERATOR[0]) != -1.0:
    raise RuntimeError(f"unexpected NN source conjugate: {SOURCE_OPERATOR}")
operator_tokens = SINK_OPERATOR + SOURCE_OPERATOR
for flavor in ("u", "d", "s", "c", "b", "t"):
    if operator_tokens.count(flavor) != operator_tokens.count(f"{flavor}^d"):
        raise RuntimeError(f"unbalanced {flavor} flavor in NN operator group")
if int(CPT.removesuffix("pt")) - 2 != 0:
    raise RuntimeError(f"{CPT} must not contain a current factor")

GR = GammaRegistry()
GR.register("gamma_7", backend.asarray(gamma(7)))
projector = backend.asarray((gamma(4) + gamma(0)) / 2.0)
spin_map = backend.sqrt(2.0) * projector[:, :2]
spin_map_norm = float(backend.max(backend.abs(spin_map.conj().T @ spin_map - backend.eye(Ns // 2))).item())
spin_projector_residual = float(backend.max(backend.abs(spin_map @ spin_map.conj().T - projector)).item())
if spin_map_norm > 1e-12 or spin_projector_residual > 1e-12:
    raise RuntimeError(f"invalid external-spin map: norm={spin_map_norm}, projector={spin_projector_residual}")
GR.register("Projector", [spin_map.conj(), spin_map])
VR = VRegistry()
PR = PeramRegistry()
REGISTRY_AXIS_CONTRACT = {
    "peram": "(spin_quark,spin_antiquark,ev_quark,ev_antiquark)",
    "VVV": "(momentum,ev_1,ev_2,ev_3)",
    "gamma": "(spin_left,spin_right)",
}

# (operator_group, t_src_abs, t_sink_local, sink_projected_spin,
#  source_projected_spin, momentum)
corr_NN = backend.zeros(
    (1, NT, LT, Ns // 2, Ns // 2, Mom_len), dtype=complex
)

# ============================================================================
# MODULE 7: perambulator staging and TScatter.
# ============================================================================
st_peram = time.perf_counter()
if rank == 0:
    peram_all = backend.zeros(
        (NT, NT, Ns, Ns, NEV_HADRON, NEV_HADRON), dtype=complex
    )
    for t_src_abs in range(NT):
        peram_path = PERAM_ROOT / "light" / conf_id / f"t{t_src_abs:03d}{PERAM_FILE_SUFFIX}"
        raw_peram = backend.load(peram_path)
        expected_shape = (NT, Ns, Ns, NEV_STORED, NEV_STORED)
        if tuple(raw_peram.shape) != expected_shape:
            raise RuntimeError(f"unexpected peram shape at source {t_src_abs}: {raw_peram.shape}")
        peram_all[t_src_abs] = raw_peram[..., :NEV_HADRON, :NEV_HADRON]
    peram_forward_check = peram_all[0, 1]
    peram_reverse_check = peram_all[1, 0]
    peram_reverse_residual = float(backend.max(backend.abs(peram_reverse_check - seq_peram(peram_forward_check))).item())
    peram_reverse_scale = float(backend.max(backend.abs(peram_reverse_check)).item())
    if peram_reverse_residual > 1e-7 * max(1.0, peram_reverse_scale):
        raise RuntimeError(f"physical gamma5-peram residual {peram_reverse_residual:.3e}")
else:
    peram_all = None
peram_all = backend.asarray(get_mpi_data(peram_all, mdtype="TScatter", root=0, axis=1))
if tuple(peram_all.shape) != (NT, LT, Ns, Ns, NEV_HADRON, NEV_HADRON):
    raise RuntimeError(f"unexpected local peram shape {peram_all.shape}")
if rank == 0:
    print(f"perambulator stage: {time.perf_counter() - st_peram:.3f} s")

# ============================================================================
# MODULE 8: source/sink loop and dynamic contraction.
# ============================================================================
dc = None
for t_src_abs in range(NT):
    t_src_owner, t_src_local = get_mpi_tlist(Nt=NT, t=t_src_abs, gtype="find")
    source_vvv_payload = sink_VVV_local[t_src_local] if rank == t_src_owner else None
    source_VVV = backend.asarray(get_mpi_data(source_vvv_payload, mdtype="Bcast", root=t_src_owner)).conj()
    VR.register("VVV_0", "tsrc", source_VVV)

    sink_abs_owned, _, sink_local_indices = get_mpi_tlist(
        Nt=NT, t=range(NT), gtype="TScatter"
    )
    for t_sink_abs, t_sink_local in zip(sink_abs_owned, sink_local_indices):
        sink_VVV = sink_VVV_local[t_sink_local]
        peram_sink_src = peram_all[t_src_abs, t_sink_local]
        PR.register("light", ("tsink", "tsrc"), peram_sink_src)
        VR.register("VVV_0", "tsink", sink_VVV)
        for Mom_indx in range(0, Mom_len, MOMENTUM_CHUNK):
            Mom_slice = slice(Mom_indx, Mom_indx + MOMENTUM_CHUNK)
            VR.register("VVV_0", "tsink", sink_VVV[Mom_slice])
            VR.register("VVV_0", "tsrc", source_VVV[Mom_slice])
            if dc is None:
                dc = dynamic_contraction(
                    [[SINK_OPERATOR, SOURCE_OPERATOR]],
                    peram_registry=PR,
                    v_registry=VR,
                    gamma_registry=GR,
                    Cpt=CPT,
                    Vindex=["M", "M"],
                    use_equivalence=USE_EQUIVALENCE,
                    ignore_dis=IGNORE_DISCONNECTED,
                    Projection=True,
                    optimize=["greedy", "dp", "auto"],
                    verbose=(rank == 0),
                )
                if len(dc.plan) != 2:
                    raise RuntimeError(f"expected 2 NN Wick entries, got {len(dc.plan)}")
            value = dc.calculate_all()
            expected_shape = (Ns // 2, Ns // 2, MOMENTUM_CHUNK)
            if tuple(value.shape) != expected_shape:
                raise RuntimeError(f"dynamic output {value.shape} != {expected_shape}")
            corr_NN[0, t_src_abs, t_sink_local, ..., Mom_slice] = value

# ============================================================================
# MODULE 9: TGather, source translation, and root-only save.
# ============================================================================
corr_NN = get_mpi_data(corr_NN, mdtype="TGather", root=0, axis=2)
if rank == 0:
    corr_NN = loop_tsrc(
        corr_NN,
        indx=[1, 2],
        Boundary_Conditions=BOUNDARY_CONDITION,
        Ctype=CPT,
        boundary_phase=FULL_CORRELATOR_WRAP_PHASE,
    )
    if tuple(corr_NN.shape) != (1, 1, NT, Ns // 2, Ns // 2, Mom_len):
        raise RuntimeError(f"unexpected translated NN shape {corr_NN.shape}")
    if not bool(backend.isfinite(corr_NN).all()):
        raise FloatingPointError("non-finite NN correlator")
    output_dir = OUTPUT_ROOT / ENSEMBLE / "Px0Py0Pz0" / f"ENV_{NEV_HADRON}" / f"conf{conf_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    backend.save(output_dir / "corr_2pt_NN_flat.npy", corr_NN)
    print(f"saved NN source-sum correlator: {output_dir}")
