# Run: mpirun -np 4 python3 main.py ~/.cache 10000
# Lambda_b -> Lambda_c+ three-point function with axial current (b->c, gamma1*gamma5)
# Sequential source method; point source at [0,0,0,0]; zero momentum; t_seq=8

import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ==============================================================================
# 1. Parameter definitions (hard-coded)
# ==============================================================================

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Source position (point source at origin)
x_src = [0, 0, 0, 0]
t_seq = 8

# Clover action parameters
xi_0 = 1.0
csw = 1.160920226

# Quark masses (clover fermion mass parameter)
m_l = -0.277       # light (u/d) — multigrid
m_c = 0.4159       # charm — CG
m_b = 1.5          # bottom — CG

tol = 1.0e-12
maxiter_l = 10000
maxiter_c = 10000
maxiter_b = 20000

# Multigrid levels for light-quark solver
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout link smearing
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ==============================================================================
# 2. Read gauge configuration
# ==============================================================================

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()

# Stout smear the gauge links (used for all Dirac inversions)
gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ==============================================================================
# 3. Construct Dirac operators
# ==============================================================================

dirac_l = core.getClover(latt_info, m_l, tol, maxiter_l, xi_0, csw, csw, multigrid)
dirac_c = core.getClover(latt_info, m_c, tol, maxiter_c, xi_0, csw, csw, None)
dirac_b = core.getClover(latt_info, m_b, tol, maxiter_b, xi_0, csw, csw, None)

# ==============================================================================
# 4. Compute forward propagators
# ==============================================================================

# Point source at [0,0,0,0]
pt_src = source.source12(latt_info, "point", x_src)

# Forward light propagator (u,d spectator quarks) — multigrid
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# Forward bottom propagator — CG, appears in final current contraction
with dirac_b.useGauge(gauge_stout):
    prop_b = core.invertPropagator(dirac_b, pt_src)

# ==============================================================================
# 5. Extract observable / compute contraction
#    (sequential-source three-point for Lambda_b -> Lambda_c+)
# ==============================================================================

# --- Gamma matrices and tensors (GPU) ---
# FROM generate_einsum (baryon_3pt)

I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
# Axial-vector current: gamma1 * gamma5  (J_A^x = bar{c} gamma_1 gamma_5 b)
Gamma_cur = cp.asarray(gamma.gamma(1) @ gamma.gamma(15), dtype=cp.complex128)

# Levi-Civita epsilon_{abc}
eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = 1.0
eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1.0

# Charge conjugation and projectors
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus = (1+gamma4)/2

# --- Sink block B: Lambda_c+ at t=8 built from u,d diquark (prop_l only) ---
# The b->c transition goes through the current, so only spectator u,d appear here.
B = core.LatticePropagator(latt_info)
B.data = (
    + contract(
        'AB,abi,GH,fgj,JK,wtzyxHAga,wtzyxGBfb->wtzyxJKij',
        Cg5,
        eps,
        Cg5,
        eps,
        Tmat,
        prop_l.data,
        prop_l.data,
    ),
)

# --- First dagger: B_tilde = gamma5 * B^dag * gamma5 ---
B.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, B.data.conj(), G5)

# --- Sequential source at t_sink=8 (full spatial volume) ---
src_seq = source.sequential12(B, t_seq)

# --- Sequential solve: D_c * G_c_seq = eta_seq ---
with dirac_c.useGauge(gauge_stout):
    prop_c_seq = core.invertPropagator(dirac_c, src_seq)

# --- Second dagger on sequential propagator ---
prop_c_seq_dag = core.LatticePropagator(latt_info)
prop_c_seq_dag.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_c_seq.data.conj(), G5)

# --- Final contraction: C3(tau) = Tr[ G_c_seq_dag * Gamma_cur * S_b ] ---
three_pt_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    prop_c_seq_dag.data, Gamma_cur, prop_b.data)

C3_t = core.gatherLattice(
    three_pt_local.get(), [0, -1, -1, -1])

# ==============================================================================
# 6. Save the result
# ==============================================================================

if core.getMPIRank() == 0:
    # tau = 0 through t_seq = 8 inclusive (9 values)
    C3 = np.asarray(C3_t[0:9], dtype=np.complex128).reshape(-1)
    out_path = f"c3_Lb_Lc_axial_cfg{n_cfg:05d}_tseq{t_seq}.txt"
    out = np.column_stack((C3.real, C3.imag))
    np.savetxt(out_path, out, fmt=["%.16e", "%.16e"])
