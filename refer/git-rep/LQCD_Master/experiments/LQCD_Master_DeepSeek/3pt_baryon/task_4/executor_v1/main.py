# Run: mpirun -n 4 python3 main.py ~/.cache 10000
#
# Lambda -> Lambda three-point function with axial current
# Current: J = bar{s} gamma_1 gamma_5 s  (CONNECTED DIAGRAM ONLY)
# Source: Lambda at [0,0,0,0], zero momentum
# Sink:   Lambda at t=8, projector P_plus = (1+gamma_4)/2

import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ═══════════════════════════════════════════════════════════
# 1. Parameter definitions (hard-coded)
# ═══════════════════════════════════════════════════════════

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

x_src = [0, 0, 0, 0]   # point source position
t_seq = 8               # sink time slice

# Clover action parameters
xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
m_s = -0.2356
tol = 1.0e-12
maxiter = 20000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout smearing parameters
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_file = "C3_axial_tau.txt"

# ═══════════════════════════════════════════════════════════
# 2. Read gauge configuration
# ═══════════════════════════════════════════════════════════

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Stout smear the gauge links (used for all Dirac inversions)
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════
# 3. Construct Dirac operators
# ═══════════════════════════════════════════════════════════

dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# ═══════════════════════════════════════════════════════════
# 4. Compute forward propagators
# ═══════════════════════════════════════════════════════════

pt_src = source.source12(latt_info, "point", x_src)

# Forward light propagator (u and d spectator quarks)
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# Forward strange propagator (s-quark from source to current insertion)
with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# ═══════════════════════════════════════════════════════════
# 5. Extract observable: Lambda 3pt with axial current
# ═══════════════════════════════════════════════════════════

# FROM generate_einsum (baryon_3pt)
# Lambda: gamma_1*gamma_5 current, P_plus projector, same-flavor (s->s)

# Gamma matrices and epsilon tensor (GPU)
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)  # gamma_5
# Axial current: gamma_1 * gamma_5  (gamma(1) = gamma_1, gamma(15) = gamma_5)
Gamma_cur = cp.asarray(gamma.gamma(1) @ gamma.gamma(15), dtype=cp.complex128)

# Levi-Civita epsilon_{abc}
eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = 1.0
eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1.0

# Dirac gamma matrices
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)  # C = gamma_2 * gamma_4
Cg5 = Cmat @ G5                                                        # C * gamma_5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)   # P_plus = (1+gamma_4)/2

# --- Sink block B(0) ---
# Lambda sink operator: epsilon^{abc} (u^T_a Cg5 d_b) s_c
# Both u and d are light quarks -> prop_l
# The strange-quark spin-color indices (JK, ij) are left open
B = core.LatticePropagator(latt_info)
B.data = contract(
    'AB,abi,GH,fgj,JK,wtzyxHAga,wtzyxGBfb->wtzyxJKij',
    Cg5, eps, Cg5, eps, Tmat, prop_l.data, prop_l.data,
)

# First G5-dagger: B_tilde = gamma_5 * B^dagger * gamma_5
# This shifts the sink block from the sink time slice to t=0
B.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, B.data.conj(), G5)

# --- Sequential source at t_sink = 8 ---
src_seq = source.sequential12(B, t_seq)

# --- Sequential solve (strange Dirac operator) ---
# The current is s->s, so the sequential propagator is strange
with dirac_s.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_s, src_seq)

# --- Second G5-dagger on sequential propagator ---
# G_s_seq_dag = gamma_5 * (prop_seq)^dagger * gamma_5
G_s_seq_dag = core.LatticePropagator(latt_info)
G_s_seq_dag.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_seq.data.conj(), G5)

# --- Final contraction ---
# C_3(tau) = sum_z Tr[ G_s_seq_dag(z,tau) * Gamma_cur * prop_s(z,tau) ]
three_pt_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    G_s_seq_dag.data, Gamma_cur, prop_s.data)

# ═══════════════════════════════════════════════════════════
# 6. Save the result
# ═══════════════════════════════════════════════════════════

C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"),
    [0, -1, -1, -1])

if core.getMPIRank() == 0:
    # Extract tau = 0..8 (inclusive).
    # NOTE: tau=0 and tau=8 contain unphysical contact terms.
    # Only tau in [1..7] should be used for physics analysis.
    C3_window = np.asarray(C3_t[:9], dtype=np.complex128)
    # Save real part only — one float per line, no header
    np.savetxt(out_file, C3_window.real, fmt="%.16e")
    print(f"Wrote {out_file} (tau = 0..8, real part only)")
