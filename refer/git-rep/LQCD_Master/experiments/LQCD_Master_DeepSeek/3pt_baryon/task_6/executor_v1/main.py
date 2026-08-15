# Run: mpirun -n 4 python3 main.py <resource_path> <cfg_number>
#
# Lambda_c+ -> Lambda  three-point function via (c->s) vector current (gamma_x).
# Source:  eps^{abc} (u^{Ta} C gamma_5 d^b) c^c     (Lambda_c+,  point,  t=0)
# Sink:    eps^{abc} (u^{Ta} C gamma_5 d^b) s^c     (Lambda,     wall,  t=8)
# Current: bar{s} gamma_x c                           (vector,    c -> s)
# Projector: P_+ = (1 + gamma_4) / 2
#
# Propagators:
#   prop_l      light  (m=-0.277)    point [0,0,0,0],  multigrid
#   prop_c      charm  (m= 0.4159)   point [0,0,0,0],  BiCGstab (default)
#   prop_s_seq  strange(m=-0.2356)   seq. wall t=8,    BiCGstab
#
# Sequential-source method (two-dagger convention):
#   1. B-block at t=8 encodes Lambda sink (ud diquark + projector)
#   2. eta_seq = gamma5 * B^dag * gamma5
#   3. D_s * G_seq = eta_seq   (BiCGstab)
#   4. G_seq_dag = gamma5 * G_seq^dag * gamma5
#   5. C3(tau) = sum_z Tr[ G_seq_dag(tau) * gamma_x * S_c(tau) ]

import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ============================================================
# 1. Parameter definitions (hard-coded)
# ============================================================
resource_path = sys.argv[1]
n_cfg         = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

x_src = [0, 0, 0, 0]
t_seq = 8

xi_0 = 1.0
csw  = 1.160920226
m_l  = -0.277
m_s  = -0.2356
m_c  = 0.4159
tol  = 1.0e-12
maxiter_l = 5000
maxiter_h = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho   = 0.125
stout_ndim  = 4

# ============================================================
# 2. Initialise PyQUDA and read gauge configuration
# ============================================================
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ============================================================
# 3. Construct Dirac operators
# ============================================================
# Light:   multigrid (critical slowing down)
# Charm:   no multigrid -> BiCGstab (well-conditioned positive mass)
# Strange: no multigrid -> BiCGstab (non-positive-definite at m=-0.2356)
dirac_l     = core.getDirac(latt_info, m_l, tol, maxiter_l, xi_0, csw, csw, multigrid)
dirac_c     = core.getDirac(latt_info, m_c, tol, maxiter_h, xi_0, csw, csw, None)
dirac_s_seq = core.getDirac(latt_info, m_s, tol, maxiter_h, xi_0, csw, csw, None)

# ============================================================
# 4. Compute forward propagators
# ============================================================
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# ============================================================
# 5. Three-point: sink block -> sequential source -> solve -> contract
# ============================================================

# --- Gamma matrices and epsilon on GPU ---
I4        = cp.eye(4, dtype=cp.complex128)
G5        = cp.asarray(gamma.gamma(15), dtype=cp.complex128)          # gamma5
Gamma_cur = cp.asarray(gamma.gamma(1),  dtype=cp.complex128)          # gamma_x

eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0,1,2] = eps[1,2,0] = eps[2,0,1] =  1.0
eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1.0

Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5  = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus

# --- Sink block B at t=8 (FROM generate_einsum: baryon_3pt) ---
# Encodes: epsilon^{abc}(u^T_a Cg5 d_b) at both source and sink,
# with Tmat bridging the s-quark (sink) and c-quark (source) spin indices.
# Single Wick topology: c->s current connects distinct flavours.
B = core.LatticePropagator(latt_info)
B.data = contract(
    "AB,abi,GH,fgj,JK,wtzyxHAga,wtzyxGBfb->wtzyxJKij",
    Cg5, eps, Cg5, eps, Tmat,
    prop_l.data, prop_l.data,
)

# --- First dagger: B_tilde = gamma5 * B^dag * gamma5 ---
B.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij",
    G5, B.data.conj(), G5,
)

# --- Sequential source (wall at t=8, sums over all spatial x) ---
src_seq = source.sequential12(B, t_seq)

# --- Sequential solve: D_s * G_seq = eta_seq (BiCGstab) ---
with dirac_s_seq.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_s_seq, src_seq)

# --- Second dagger: G_seq_dag = gamma5 * G_seq^dag * gamma5 ---
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij",
    G5, prop_seq.data.conj(), G5,
)

# --- Final contraction: C3(tau) = sum_z Tr[ G_seq_dag(tau) * gamma_x * S_c(tau) ] ---
three_pt_local = contract(
    "wtzyxijba, jk, wtzyxkiab -> t",
    tmp_prop.data, Gamma_cur, prop_c.data,
)

C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"),
    [0, -1, -1, -1],
)

# ============================================================
# 6. Save result (rank 0 only, tau = 0 .. t_seq)
# ============================================================
if core.getMPIRank() == 0:
    # Only current-insertion times up to the sink are physical
    C3_window = np.asarray(C3_t[0:t_seq+1], dtype=np.complex128).reshape(-1)
    out_path = "lambda_c_to_lambda_3pt.txt"
    with open(out_path, "w") as f:
        for val in C3_window:
            f.write(f"{val.real:.16e} {val.imag:.16e}\n")
