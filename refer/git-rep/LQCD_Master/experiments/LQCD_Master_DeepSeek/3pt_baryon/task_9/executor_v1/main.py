# Run: mpirun -n 4 python3 main.py ~/.cache CFGNUM
#
# Lambda_b -> Lambda  three-point correlator via b->s vector current
#   Source (t=0):  Lambda_b = epsilon^{abc} (u^T C gamma_5 d) b   (point [0,0,0,0])
#   Sink   (t=8):  Lambda   = epsilon^{def} (u^T C gamma_5 d) s
#   Current:        J_x = s_bar gamma_x b  (flavour-changing b->s)
#   Projector:      T = (1 + gamma_4)/2 = P_plus
#   Sequential source via two-dagger convention: eta_seq = gamma_5 B^dag gamma_5
#   B-block has SINGLE topology (1u+1d in both Λ_b and Λ — no exchange term)
#
# Propagators (3 inversions per cfg):
#   prop_l   — light (u/d),  m=-0.277,     forward, point source, multigrid
#   prop_b   — bottom,       m=1.5,        forward, point source, BiCGStab
#   prop_seq — strange,      m=-0.2356,    sequential from Λ sink at t=8, multigrid
#
# Partially quenched: sea ms=-0.2400 (from cfg filename), valence ms=-0.2356
# Heavy quark: m_b=1.5 ~ 2.8 GeV (Fermilab-type; lighter than physical ~4.18 GeV)

import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ============================================================
# 1. Parameters (hard-coded — physics defines the computation)
# ============================================================

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

# Lattice
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

# Gauge configuration path template
cfg_path_template = (
    "/public/share/weiwang/clqcd/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Source — point source at origin, zero momentum
x_src = [0, 0, 0, 0]
t_seq = 8

# Clover action parameters (isotropic, C24P29 ensemble)
xi_0 = 1.0
csw = 1.160920226

# Quark masses
m_l = -0.277       # light (u/d) — also the sea light mass
m_s = -0.2356      # strange valence (tuned; sea ms = -0.2400, partially quenched)
m_b = 1.5          # bottom (Fermilab-type heavy quark, ~2.8 GeV)

# Solver parameters
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout link smearing (applied to gauge links BEFORE all inversions)
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# Output
out_dir = "."

# ============================================================
# 2. Initialize PyQUDA
# ============================================================
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ============================================================
# 3. Read gauge configuration
# ============================================================
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()

# Stout-smeared copy for Dirac inversions
gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ============================================================
# 4. Construct Dirac operators
# ============================================================
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)
# Bottom quark: heavy enough that BiCGStab converges quickly; no multigrid needed
dirac_b = core.getClover(latt_info, m_b, tol, maxiter, xi_0, csw, csw, None)

# ============================================================
# 5. Compute forward propagators
# ============================================================
pt_src = source.source12(latt_info, "point", x_src)

# Light propagator — reused for BOTH u and d spectator lines in B-block
# (isospin symmetry: S_u = S_d = S_l)
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# Bottom propagator — connects Lambda_b source b-quark to the current insertion
with dirac_b.useGauge(gauge_stout):
    prop_b = core.invertPropagator(dirac_b, pt_src)

# ============================================================
# 6. Sink block B(x) — SINGLE topology (no exchange term)
# ============================================================
# FROM generate_einsum (baryon_3pt)
# Lambda_b (udb) -> Lambda (uds) with current s_bar gamma_x b
# Gamma: g1  Forward: prop_b  Terms: 1
#
# B^{rf}_{sigma,rho}(0) = eps^{abc} eps^{def} (Cg5)_{ab} (Cg5)_{de}
#                          * T_{sigma,rho} * S_l^{ad} * S_l^{be}
# where r = s-quark colour (open), sigma = s-quark spin (open)

I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)  # gamma_x

# Levi-Civita epsilon_{abc} on GPU
eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

# Dirac matrices on GPU
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)  # C = gamma_2 * gamma_4
Cg5 = Cmat @ G5                                                           # C * gamma_5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)      # P_plus = (1+gamma_4)/2

# Single Wick topology: both u and d quarks go directly from source to sink
B = core.LatticePropagator(latt_info)
B.data = contract(
    "AB,abi,GH,fgj,JK,wtzyxHAga,wtzyxGBfb->wtzyxJKij",
    Cg5,          # (C gamma_5)_{alpha,beta}   for source diquark
    eps,          # epsilon^{abc}
    Cg5,          # (C gamma_5)_{alpha',beta'} for sink diquark
    eps,          # epsilon^{def}
    Tmat,         # P_plus projector: T_{sigma,rho}
    prop_l.data,  # S_l^{ad} — u-quark spectator
    prop_l.data,  # S_l^{be} — d-quark spectator (isospin: S_d = S_l)
)

# First dagger of the two-dagger convention:  B_tilde = gamma_5 * B^dag * gamma_5
B.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij",
    G5, B.data.conj(), G5,
)

# ============================================================
# 7. Sequential source and sequential solve (strange quark)
# ============================================================
# sequential12: builds the sequential source from B at t=t_seq, places at t=0
src_seq = source.sequential12(B, t_seq)

# Solve D_s * prop_seq = src_seq  (strange-quark Dirac operator)
with dirac_s.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_s, src_seq)

# Second dagger: prop_seq_dag = gamma_5 * prop_seq^dag * gamma_5
prop_seq_dag = core.LatticePropagator(latt_info)
prop_seq_dag.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij",
    G5, prop_seq.data.conj(), G5,
)

# ============================================================
# 8. Final contraction and MPI gather
# ============================================================
# C_3(tau) = sum_z Tr[ prop_seq_dag(z,tau) * gamma_x * prop_b(z,tau) ]
# No momentum phase (all momenta zero)

three_pt_local = contract(
    "wtzyxijba, jk, wtzyxkiab -> t",
    prop_seq_dag.data,   # sequential strange propagator (daggered)
    Gamma_cur,           # gamma_x — vector current
    prop_b.data,         # forward bottom propagator
)

# MPI gather: gather time dimension, reduce spatial
C3_t = core.gatherLattice(
    three_pt_local.get(), [0, -1, -1, -1],
)

# ============================================================
# 9. Save result — plain text, no header
# ============================================================
if core.getMPIRank() == 0:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"Lambda_b_to_Lambda_3pt_cfg{n_cfg:05d}_tseq{t_seq}.txt",
    )
    # C3_t has shape (Lt,) = (72,); save real and imaginary parts
    C3 = np.asarray(C3_t, dtype=np.complex128).reshape(-1)
    np.savetxt(
        out_path,
        np.column_stack((C3.real, C3.imag)),
        fmt=["%.16e", "%.16e"],
    )
