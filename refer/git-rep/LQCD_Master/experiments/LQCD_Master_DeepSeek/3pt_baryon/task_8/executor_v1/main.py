# Run: mpirun -np 4 python main.py <resource_path> <cfg_number>
#
# Lambda_b -> Lambda_c three-point correlator
#   Source: Lambda_b = eps_abc (u^a Cg5 d^b) b^c at t=0
#   Sink:   Lambda_c = eps_abc (u^a Cg5 d^b) c^c at t=8
#   Current: \bar{c} gamma_x  b  (vector, x-direction)
#   Projector: T = (I + gamma_4)/2  (P_plus)
#   Exactly ONE Wick topology (each quark flavor unique).

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
    "/public/share/weiwang/clqcd/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Source and sink
x_src = [0, 0, 0, 0]   # point source at t=0
t_seq = 8               # sink timeslice

# Clover parameters
xi_0 = 1.0
csw = 1.160920226

# Quark masses
m_l = -0.277      # light (u,d)
m_b = 1.5          # bottom
m_c = 0.4159       # charm

# Solver parameters
tol_l = 1.0e-12
tol_b = 1.0e-10
tol_c = 1.0e-10
maxiter_l = 10000
maxiter_b = 20000
maxiter_c = 15000
multigrid_l = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout link smearing
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ═══════════════════════════════════════════════════════════
# 2. Read gauge configuration
# ═══════════════════════════════════════════════════════════

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Build stout-smeared copy for Dirac inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════
# 3. Construct Dirac operators
# ═══════════════════════════════════════════════════════════

dirac_l = core.getClover(latt_info, m_l, tol_l, maxiter_l,
                         xi_0, csw, csw, multigrid_l)   # light, MG
dirac_b = core.getClover(latt_info, m_b, tol_b, maxiter_b,
                         xi_0, csw, csw, None)           # bottom, CG
dirac_c = core.getClover(latt_info, m_c, tol_c, maxiter_c,
                         xi_0, csw, csw, None)           # charm, CG

# ═══════════════════════════════════════════════════════════
# 4. Compute forward propagators
# ═══════════════════════════════════════════════════════════

pt_src = source.source12(latt_info, "point", x_src)

# Light propagator (u,d diquark lines)
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# Bottom propagator (source b-quark)
with dirac_b.useGauge(gauge_stout):
    prop_b = core.invertPropagator(dirac_b, pt_src)

# ═══════════════════════════════════════════════════════════
# 5. Extract observable — baryon 3pt sequential source
# ═══════════════════════════════════════════════════════════

# --- Gamma matrices and tensors on GPU ---
# FROM generate_einsum (baryon_3pt)
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)  # gamma_x

eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus

# --- Sink block B(x): single Wick topology ---
# Lambda_c sink: eps_abc (u^a Cg5 d^b) c^c, with projector Tmat
B = core.LatticePropagator(latt_info)
B.data = contract(
    'AB,abi,GH,fgj,JK,wtzyxHAga,wtzyxGBfb->wtzyxJKij',
    Cg5, eps, Cg5, eps, Tmat,
    prop_l.data,   # u-quark line
    prop_l.data,   # d-quark line
)

# --- First dagger: gamma5 @ B^dag @ gamma5 ---
B.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, B.data.conj(), G5)

# --- Sequential source at sink timeslice t=8 ---
src_seq = source.sequential12(B, t_seq)

# --- Sequential charm propagator ---
with dirac_c.useGauge(gauge_stout):
    prop_c_seq = core.invertPropagator(dirac_c, src_seq)

# --- Second dagger on sequential propagator ---
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_c_seq.data.conj(), G5)

# --- Final 3pt contraction: Tr[ tmp_prop · Gamma_cur · prop_b ] ---
three_pt_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    tmp_prop.data, Gamma_cur, prop_b.data)

C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"),
    [0, -1, -1, -1])

# ═══════════════════════════════════════════════════════════
# 6. Save the result
# ═══════════════════════════════════════════════════════════

if core.getMPIRank() == 0:
    # Extract insertion-time window tau = 0 .. t_seq
    C3_window = np.asarray(C3_t[0:t_seq + 1], dtype=np.complex128).reshape(-1)
    t_ins = np.arange(0, t_seq + 1, dtype=np.int32)
    tseq_col = np.full(t_ins.shape, t_seq, dtype=np.int32)
    out = np.column_stack((tseq_col, t_ins, C3_window.real, C3_window.imag))
    np.savetxt("Lambda_b_to_Lambda_c_3pt_corr.txt", out,
               fmt=["%d", "%d", "%.16e", "%.16e"])
