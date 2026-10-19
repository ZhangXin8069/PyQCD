#!/usr/bin/env python3
"""Flat NJNp three-point production template for ``lqcddb_update``.

This file is intentionally written as a downward-running production program,
not as a ``main -> run -> loader`` call tree.  To build another contraction,
copy the stable modules in order and replace the operator/current and registry
lines marked ``OBSERVABLE-SPECIFIC``.  The numerical contraction is a
Euclidean N-pi -> N one-current example; it is a template, not a claim that
the selected physical current is universally appropriate.

The reference mode keeps the complete source-time perambulator staging visible
for auditability.  After the baseline agrees with a direct small-tensor
calculation, replace only MODULE 7 by source-at-a-time streaming and retain
the same registry keys, axes, and output layout.
"""

# ============================================================================
# MODULE 1: imports -- named imports are deliberate (no top-level wildcard).
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
# MODULE 2: runtime parameters, ensemble paths, and output contract.
# ============================================================================
if len(sys.argv) < 2:
    raise SystemExit("usage: njn_three_point_flat.py CONF_ID")

conf_id = sys.argv[1]
BACKEND_NAME = os.environ.get("LQCDB_BACKEND", "cupy")
LATTICE_SIZE = [32, 32, 32, 64]       # (Lx, Ly, Lz, Nt)
GRID_SIZE = [1, 1, 1, 2]              # one rank/GPU per temporal slice
ENSEMBLE = "E32P29"
CPT = "3pt"
ENSEMBLE_DIR = (
    "beta6.308_mu-0.2510_ms-0.2170_u00.859727_"
    "u0s0.954467_L32x64"
)
EIGEN_ROOT = Path("/nexdata/project/lqcd/sush/eigensystem") / ENSEMBLE_DIR
PERAM_ROOT = Path("/nexdata/project/lqcd/sush/perambulators") / ENSEMBLE_DIR
OUTPUT_ROOT = Path("/public/home/sush/distillation/0v2b/result_nex")

NX, NY, NZ, NT = LATTICE_SIZE
TSEP = 20
NEV_HADRON = 100
NEV_CURRENT = 400
MOMENTUM_CHUNK = 1                 # baseline; benchmark before changing
USE_EQUIVALENCE = False            # enable only after phase-aware proof
IGNORE_DISCONNECTED = False
BOUNDARY_CONDITION = "Antiperiodic"
FULL_CORRELATOR_WRAP_PHASE = -1.0  # odd net fermion winding for this baryonic transition
SOURCE_ESTIMATOR = "source_sum"       # no implicit division by NT
CONTACT_POLICY = "retain only offsets 5..16 for this example"
OUTPUT_AXES_RAW = (
    "operator_group", "t_src_abs", "t_cur_local", "mu",
    "sink_projected_spin", "source_projected_spin", "Mom_N", "Mom_M",
)
OUTPUT_AXES_SAVED = (
    "operator_group", "source_sum_singleton", "t_cur_minus_t_src", "mu",
    "sink_projected_spin", "source_projected_spin", "Mom_N", "Mom_M",
)

# This example intentionally mirrors the reference insertion window.  The
# endpoints and the four nearest slices are an observable choice, not a
# universal default; record any change in the output manifest.
INSERTION_OFFSETS = list(range(5, TSEP - 3))

# The loaded perambulator file has (Nt, Ns, Ns, Nev, Nev) axes.  The source
# and hadron ports below retain 100 vectors while current ports retain 400.
PERAM_FILE_SUFFIX = (
    "_e100_s[100, 200, 400, 700, 1100]_"
    "[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy"
)

# ============================================================================
# MODULE 3: backend/MPI initialization and ownership checks.
# ============================================================================
set_backend(BACKEND_NAME)
backend = get_backend()
mpinit(grid_size=GRID_SIZE, latt_size=LATTICE_SIZE, backend=backend.__name__)

rank = getMPIRank()
size = getMPISize()
if size != int(np.prod(GRID_SIZE)):
    raise RuntimeError(f"MPI size {size} != requested grid {GRID_SIZE}")
if NT % size != 0:
    raise RuntimeError(f"Nt={NT} must be divisible by MPI size={size}")
LT = NT // size

fun_eigen = vertex_creator(Nx=NX)

# ============================================================================
# MODULE 4: complete momentum table and live phase API.
# ============================================================================
Mom_VDV = (
    [[0, 0, 0]]
    + sorted(creat_mom_list(Mom=[0, 0, 1], fix_Q2=True))
    + sorted(creat_mom_list(Mom=[0, 1, 1], fix_Q2=True))
    + sorted(creat_mom_list(Mom=[1, 1, 1], fix_Q2=True))
)
Mom_VVV = (
    [[0, 0, 0]]
    + sorted(creat_mom_list(Mom=[0, 0, 1], fix_Q2=True))[::-1]
    + sorted(creat_mom_list(Mom=[0, 1, 1], fix_Q2=True))[::-1]
    + sorted(creat_mom_list(Mom=[1, 1, 1], fix_Q2=True))[::-1]
)
Mom_len = len(Mom_VDV)
if Mom_len != len(Mom_VVV) or Mom_len != 27:
    raise RuntimeError(f"unexpected momentum table lengths: {Mom_len}, {len(Mom_VVV)}")
if MOMENTUM_CHUNK <= 0 or Mom_len % MOMENTUM_CHUNK != 0:
    raise RuntimeError(f"MOMENTUM_CHUNK={MOMENTUM_CHUNK} must divide Mom_len={Mom_len}")
if not all(tuple(vvv) == tuple(-x for x in vdv) for vdv, vvv in zip(Mom_VDV, Mom_VVV)):
    raise RuntimeError("VVV/VDV momentum routing is not an indexed +/- pair")

# Current lqcddb_update names: phase_exp_VdV is color-replicated and
# phase_exp_VVV is colorless.  Momentum order is (p_z, p_y, p_x).
phase_exp_VdV = backend.zeros((Mom_len, NX, NX, NX, Nc), dtype=complex)
phase_exp_VVV = backend.zeros((Mom_len, NX, NX, NX), dtype=complex)
for Mom_indx in range(Mom_len):
    phase_exp_VdV[Mom_indx] = fun_eigen.phase_exp_VdV(Mom=Mom_VDV[Mom_indx])
    phase_exp_VVV[Mom_indx] = fun_eigen.phase_exp_VVV(Mom=Mom_VVV[Mom_indx])

# ============================================================================
# MODULE 5: local VVV/VDV construction and blending weight.
# ============================================================================
times_owned, owner_labels, local_time_indices = get_mpi_tlist(
    Nt=NT, t=range(NT), gtype="TScatter"
)
if len(times_owned) != LT:
    raise RuntimeError(f"rank {rank}: expected {LT} owned times, got {len(times_owned)}")
if any(owner != rank for owner in owner_labels):
    raise RuntimeError(f"rank {rank}: invalid TScatter owner labels {owner_labels}")
if any(local_time_indices[i] != i for i in range(LT)):
    raise RuntimeError(f"rank {rank}: unexpected local time map {local_time_indices}")

VdV_link = backend.zeros(
    (LT, Mom_len, NEV_CURRENT, NEV_CURRENT), dtype=complex
)
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
    if eigvecs.ndim != 5 or tuple(eigvecs.shape[1:]) != (NZ, NY, NX, Nc):
        raise RuntimeError(f"unexpected eigenvector shape at t={t_abs}: {eigvecs.shape}")
    if eigvecs.shape[0] != NEV_CURRENT:
        raise RuntimeError(
            f"current eigen extent {eigvecs.shape[0]} != manifest {NEV_CURRENT}"
        )
    if t_local == local_time_indices[0]:
        exact_rows = eigvecs[:NEV_HADRON].reshape(NEV_HADRON, -1)
        gram_residual = float(backend.max(backend.abs(exact_rows @ exact_rows.conj().T - backend.eye(NEV_HADRON))).item())
        if gram_residual > 1e-9:
            raise RuntimeError(f"t={t_abs}: complex eigenvector Gram residual {gram_residual:.3e}")
    for Mom_indx in range(Mom_len):
        VdV_link[t_local, Mom_indx:Mom_indx + 1] = fun_eigen.Mom_VdV_sink_t(
            phase_exp=phase_exp_VdV[Mom_indx:Mom_indx + 1], eigvecs=eigvecs
        )
        sink_VVV_local[t_local, Mom_indx] = fun_eigen.Mom_VVV_sink_t(
            phase_exp=phase_exp_VVV[Mom_indx], eigvecs=eigvecs[:NEV_HADRON]
        )
        if t_local == local_time_indices[0] and Mom_indx == 0:
            vvv = sink_VVV_local[t_local, Mom_indx]
            vvv_residual = max(
                float(backend.max(backend.abs(vvv.swapaxes(-3, -2) + vvv)).item()),
                float(backend.max(backend.abs(vvv.swapaxes(-2, -1) + vvv)).item()),
            )
            if vvv_residual > 1e-9:
                raise RuntimeError(f"t={t_abs}: VVV permutation residual {vvv_residual:.3e}")

omega = fun_eigen.create_omega_accelerate(
    exact=100,
    N_eigen=[100, 200, 400, 700, 1100],
    N_sum=[20, 20, 20, 20, 20],
    N_extract=[10, 10, 10, 10, 10],
    noise=200,
)
if tuple(omega.shape) != (NEV_CURRENT, NEV_CURRENT):
    raise RuntimeError(f"unexpected blending weight shape {omega.shape}")
VdV_link *= omega                 # apply Ω exactly once, before registry use
del phase_exp_VdV, phase_exp_VVV, omega
if rank == 0:
    print(f"vertex stage: {time.perf_counter() - st_vertex:.3f} s")

# ============================================================================
# MODULE 6: physical operators, gamma/projector registry, and output axes.
# ============================================================================
SINK_OPERATOR = ["|", "d", "u", "gamma_7", "d", "|"]
INITIAL_OPERATOR = [
    "|", "d", "u", "gamma_7", "d", "|",
    "|", "u^d", "gamma_5", "d", "|",
]
SOURCE_OPERATOR = conjugate_operator(INITIAL_OPERATOR)
if SOURCE_OPERATOR[1:] != [
    "|", "d^d", "gamma_5", "u", "|",
    "|", "u^d", "gamma_7", "d^d", "d^d", "|",
]:
    raise RuntimeError(f"unexpected Nπ Hermitian-conjugate source: {SOURCE_OPERATOR}")
if complex(SOURCE_OPERATOR[0]) != 1.0:
    raise RuntimeError(f"unexpected Nπ source coefficient: {SOURCE_OPERATOR[0]}")

# OBSERVABLE-SPECIFIC: replace this current and its four registered matrices.
# This example uses the charged Euclidean axial current
# A_mu = ubar gamma_mu gamma_5 d, mu=1..4, in explicit component order. A full
# weak left-handed current must be declared separately, not inferred.
CURRENT_OPERATOR = ["|", "u^d", "gamma_A", "d", "|"]
CURRENT_GAMMA = [backend.asarray(gamma(mu) @ gamma(5)) for mu in range(1, 5)]
operator_tokens = SINK_OPERATOR + SOURCE_OPERATOR + CURRENT_OPERATOR
for flavor in ("u", "d", "s", "c", "b", "t"):
    if operator_tokens.count(flavor) != operator_tokens.count(f"{flavor}^d"):
        raise RuntimeError(f"unbalanced {flavor} flavor in NJNp operator group")
if int(CPT.removesuffix("pt")) - 2 != 1:
    raise RuntimeError(f"{CPT} does not match the one explicit current factor")

GR = GammaRegistry()
GR.register("gamma_7", backend.asarray(gamma(7)))
GR.register("gamma_5", backend.asarray(gamma(5)))
projector = backend.asarray((gamma(4) + gamma(0)) / 2.0)
spin_map = backend.sqrt(2.0) * projector[:, :2]
spin_map_norm = float(backend.max(backend.abs(spin_map.conj().T @ spin_map - backend.eye(Ns // 2))).item())
spin_projector_residual = float(backend.max(backend.abs(spin_map @ spin_map.conj().T - projector)).item())
if spin_map_norm > 1e-12 or spin_projector_residual > 1e-12:
    raise RuntimeError(f"invalid external-spin map: norm={spin_map_norm}, projector={spin_projector_residual}")
GR.register("Projector", [spin_map.conj(), spin_map])
for current_component in CURRENT_GAMMA:
    if tuple(current_component.shape) != (Ns, Ns):
        raise RuntimeError(f"current gamma must be ({Ns},{Ns}), got {current_component.shape}")

VR = VRegistry()
PR = PeramRegistry()
REGISTRY_AXIS_CONTRACT = {
    "peram": "(spin_quark,spin_antiquark,ev_quark,ev_antiquark)",
    "VDV": "(momentum,ev_left,ev_right); exact Wick port order is checked in dc.plan",
    "VVV": "(momentum,ev_1,ev_2,ev_3)",
    "gamma": "(spin_left,spin_right); current component is an outer Python loop",
}

# Output: (operator_group, t_src_abs, t_cur_local, mu,
#          sink_projected_spin, source_projected_spin, Mom_N, Mom_M).
# After MODULE 9, axis 1 is a singleton source axis and axis 2 is the
# source-relative insertion axis; the saved array is a source sum.
corr_NJN = backend.zeros(
    (1, NT, LT, len(CURRENT_GAMMA), Ns // 2, Ns // 2, Mom_len, Mom_len),
    dtype=complex,
)

# ============================================================================
# MODULE 7: root perambulator staging and TScatter.
# ============================================================================
# This is the transparent reference staging mode.  For a large production
# run, replace this whole module by source-at-a-time loading, but preserve the
# same four-dimensional peram axes and exact registry keys below.
st_peram = time.perf_counter()
if rank == 0:
    peram_all = backend.zeros(
        (NT, NT, Ns, Ns, NEV_CURRENT, NEV_HADRON), dtype=complex
    )
    for t_src_abs in range(NT):
        peram_path = (
            PERAM_ROOT / "light" / conf_id
            / f"t{t_src_abs:03d}{PERAM_FILE_SUFFIX}"
        )
        raw_peram = backend.load(peram_path)
        expected_shape = (NT, Ns, Ns, NEV_CURRENT, NEV_CURRENT)
        if tuple(raw_peram.shape) != expected_shape:
            raise RuntimeError(
                f"unexpected peram shape at source {t_src_abs}: {raw_peram.shape}"
            )
        peram_all[t_src_abs] = raw_peram[..., :NEV_CURRENT, :NEV_HADRON]
    peram_forward_check = peram_all[0, 1, ..., :NEV_HADRON, :NEV_HADRON]
    peram_reverse_check = peram_all[1, 0, ..., :NEV_HADRON, :NEV_HADRON]
    peram_reverse_residual = float(backend.max(backend.abs(peram_reverse_check - seq_peram(peram_forward_check))).item())
    peram_reverse_scale = float(backend.max(backend.abs(peram_reverse_check)).item())
    if peram_reverse_residual > 1e-7 * max(1.0, peram_reverse_scale):
        raise RuntimeError(f"physical gamma5-peram residual {peram_reverse_residual:.3e}")
else:
    peram_all = None
peram_all = backend.asarray(
    get_mpi_data(data=peram_all, mdtype="TScatter", root=0, axis=1)
)
if tuple(peram_all.shape) != (NT, LT, Ns, Ns, NEV_CURRENT, NEV_HADRON):
    raise RuntimeError(f"unexpected local peram shape: {peram_all.shape}")
if rank == 0:
    print(f"perambulator stage: {time.perf_counter() - st_peram:.3f} s")

# ============================================================================
# MODULE 8: source/time/momentum loop and dynamic contraction.
# ============================================================================
# A single plan is reused after the first complete registry population.  This
# is the optimization-safe form because USE_EQUIVALENCE is false here; if
# equivalence is enabled, rebuild when gamma transpose signatures or topology
# changes.  The visible loop order remains source -> current time -> momentum
# -> current component -> dynamic calculation.
dc = None
for t_src_abs in range(NT):
    t_sink_abs = (t_src_abs + TSEP) % NT
    t_src_owner, t_src_local = get_mpi_tlist(Nt=NT, t=t_src_abs, gtype="find")
    t_sink_owner, t_sink_local = get_mpi_tlist(Nt=NT, t=t_sink_abs, gtype="find")

    # Fixed source/sink vertices are broadcast from their physical owners.
    source_vvv_payload = sink_VVV_local[t_src_local] if rank == t_src_owner else None
    source_vdv_payload = VdV_link[t_src_local, :, :NEV_HADRON, :NEV_HADRON] if rank == t_src_owner else None
    sink_vvv_payload = sink_VVV_local[t_sink_local] if rank == t_sink_owner else None
    source_VVV = backend.asarray(get_mpi_data(source_vvv_payload, mdtype="Bcast", root=t_src_owner)).conj()
    source_VDV = backend.asarray(get_mpi_data(source_vdv_payload, mdtype="Bcast", root=t_src_owner)).conj().transpose(0, 2, 1)
    sink_VVV_at_sink = backend.asarray(get_mpi_data(sink_vvv_payload, mdtype="Bcast", root=t_sink_owner))

    # Fixed perambulator entries for this source/sink pair.
    src_src_payload = peram_all[t_src_abs, t_src_local, ..., :NEV_HADRON, :NEV_HADRON] if rank == t_src_owner else None
    sink_src_payload = peram_all[t_src_abs, t_sink_local, ..., :NEV_HADRON, :NEV_HADRON] if rank == t_sink_owner else None
    sink_sink_payload = peram_all[t_sink_abs, t_sink_local, ..., :NEV_HADRON, :NEV_HADRON] if rank == t_sink_owner else None
    peram_src_src = backend.asarray(get_mpi_data(src_src_payload, mdtype="Bcast", root=t_src_owner))
    peram_sink_src = backend.asarray(get_mpi_data(sink_src_payload, mdtype="Bcast", root=t_sink_owner))
    peram_sink_sink = backend.asarray(get_mpi_data(sink_sink_payload, mdtype="Bcast", root=t_sink_owner))

    PR.register("light", ("tsrc", "tsrc"), peram_src_src)
    PR.register("light", ("tsink", "tsrc"), peram_sink_src)
    PR.register("light", ("tsrc", "tsink"), seq_peram(peram_sink_src))
    PR.register("light", ("tsink", "tsink"), peram_sink_sink)
    VR.register("VDV_0", "tsrc", source_VDV)
    VR.register("VVV_0", "tsrc", source_VVV)

    requested_cur_abs = [(t_src_abs + dt) % NT for dt in INSERTION_OFFSETS]
    cur_abs_owned, cur_owner_labels, cur_local_indices = get_mpi_tlist(
        Nt=NT, t=requested_cur_abs, gtype="TScatter"
    )
    for t_cur_abs, t_cur_local in zip(cur_abs_owned, cur_local_indices):
        if t_cur_abs % size != rank:
            raise RuntimeError(f"invalid current owner for t={t_cur_abs}")
        peram_cur_src = backend.asarray(peram_all[t_src_abs, t_cur_local])
        peram_cur_sink = backend.asarray(peram_all[t_sink_abs, t_cur_local])
        PR.register("light", ("tcur0", "tsrc"), peram_cur_src)
        PR.register("light", ("tsrc", "tcur0"), seq_peram(peram_cur_src))
        PR.register("light", ("tcur0", "tsink"), peram_cur_sink)
        PR.register("light", ("tsink", "tcur0"), seq_peram(peram_cur_sink))
        for Mom_indx in range(0, Mom_len, MOMENTUM_CHUNK):
            Mom_slice = slice(Mom_indx, Mom_indx + MOMENTUM_CHUNK)
            VR.register("VDV_0", "tcur0", VdV_link[t_cur_local, Mom_slice])
            VR.register("VVV_0", "tsink", sink_VVV_at_sink[Mom_slice])
            for current_index, current_gamma in enumerate(CURRENT_GAMMA):
                GR.register("gamma_A", current_gamma)
                if dc is None:
                    dc = dynamic_contraction(
                        [[SINK_OPERATOR, SOURCE_OPERATOR, CURRENT_OPERATOR]],
                        peram_registry=PR,
                        v_registry=VR,
                        gamma_registry=GR,
                        Cpt=CPT,
                        Vindex=["N", "N", "M", "M"],
                        use_equivalence=USE_EQUIVALENCE,
                        ignore_dis=IGNORE_DISCONNECTED,
                        Projection=True,
                        optimize=["greedy", "dp", "auto"],
                        verbose=(rank == 0),
                    )
                    if len(dc.plan) != 12:
                        raise RuntimeError(f"expected 12 NJNp Wick entries, got {len(dc.plan)}")
                value = dc.calculate_all()
                expected_shape = (
                    Ns // 2, Ns // 2, MOMENTUM_CHUNK, Mom_len
                )
                if tuple(value.shape) != expected_shape:
                    raise RuntimeError(f"dynamic output {value.shape} != {expected_shape}")
                corr_NJN[0, t_src_abs, t_cur_local, current_index, ..., Mom_slice, :] = value

# ============================================================================
# MODULE 9: gather, source-time translation, and root-only save.
# ============================================================================
corr_NJN = get_mpi_data(corr_NJN, mdtype="TGather", root=0, axis=2)
if rank == 0:
    corr_NJN = loop_tsrc(
        corr_NJN,
        indx=[1, 2],
        Boundary_Conditions=BOUNDARY_CONDITION,
        Ctype=CPT,
        t_sep=TSEP,
        boundary_phase=FULL_CORRELATOR_WRAP_PHASE,
    )
    if tuple(corr_NJN.shape) != (1, 1, NT, 4, Ns // 2, Ns // 2, Mom_len, Mom_len):
        raise RuntimeError(f"unexpected translated output shape {corr_NJN.shape}")
    if not bool(backend.isfinite(corr_NJN).all()):
        raise FloatingPointError("non-finite correlator output")
    output_dir = OUTPUT_ROOT / ENSEMBLE / "Px0Py0Pz0" / f"ENV_{NEV_HADRON}" / f"conf{conf_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    backend.save(output_dir / f"corr_3pt_NJN_tsep{TSEP}.npy", corr_NJN)
    print(f"saved source-sum correlator: {output_dir}")
