import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract

from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# =============================================================================
# Parameters (hard-coded)
# =============================================================================

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Point source
x_src = [0, 0, 0, 0]

# Sequential source time
t_seq = 8

# Clover fermion parameters
xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
m_s = -0.2356
tol = 1.0e-12
maxiter = 20000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout smearing
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# Output
out_path = "lambda_to_proton_vector_3pt_result.txt"

# =============================================================================
# Initialize PyQUDA
# =============================================================================
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# =============================================================================
# Read gauge configuration
# =============================================================================
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Stout smear (copy before smearing)
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# =============================================================================
# Construct Dirac operators (Wilson-clover)
# =============================================================================
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# =============================================================================
# Compute forward propagators
# =============================================================================
pt_src = source.source12(latt_info, "point", x_src)

# Forward light propagator  S_l(x; 0)
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# Forward strange propagator  S_s(x; 0)
with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# =============================================================================
# Extract observable: Lambda -> p  three-point via sequential source
# =============================================================================

# ---- Gamma matrices and epsilon tensor (GPU) ----
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)

eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus

# ---- Sink block  B(0)  at t_f = 8  (from generate_einsum) ----
#   Two Wick topologies with a relative minus sign from fermion antisymmetry.
#   prop_l serves both u and d lines via flavour symmetry S_u = S_d = S_l.
B = core.LatticePropagator(latt_info)
B.data = (
    - contract(
        "AB,abi,KH,jge,JF,wtzyxFBeb,wtzyxHAga->wtzyxJKij",
        Cg5, eps, Cg5, eps, Tmat,
        prop_l.data, prop_l.data,
    )
    + contract(
        "AB,abi,GH,fgj,JK,wtzyxHAga,wtzyxGBfb->wtzyxJKij",
        Cg5, eps, Cg5, eps, Tmat,
        prop_l.data, prop_l.data,
    )
)

# ---- First dagger:  eta_seq = gamma5 * B^dag * gamma5 ----
B.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij",
    G5, B.data.conj(), G5,
)

# ---- Sequential source and solve ----
src_seq = source.sequential12(B, t_seq)

with dirac_l.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_l, src_seq)

# ---- Second dagger:  G_seq_dag = gamma5 * G_seq^dag * gamma5 ----
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij",
    G5, prop_seq.data.conj(), G5,
)

# ---- Final contraction:  C3(tau) = sum_z Tr[ G_seq_dag * gamma_x * S_s ] ----
#   wtzyx = (parity, t, z, y, x//2);  i,j = spin;  a,b = color
three_pt_local = contract(
    "wtzyxijba, jk, wtzyxkiab -> t",
    tmp_prop.data, Gamma_cur, prop_s.data,
)

# MPI gather over time dimension
C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"),
    [0, -1, -1, -1],
)

# =============================================================================
# Save result  (rank 0 only)
# =============================================================================
if core.getMPIRank() == 0:
    C3_full = np.asarray(C3_t, dtype=np.complex128)
    # tau = 0 .. t_seq  (contact terms at 0 and t_seq are included)
    tau = np.arange(t_seq + 1, dtype=np.int32)
    out = np.column_stack((tau, C3_full[: t_seq + 1].real, C3_full[: t_seq + 1].imag))
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
