# Lambda_c baryon two-point correlation function
# Run: mpirun -n 4 python3 main.py <resource_path> <cfg_number>

import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma


# ============================================================
# 1. Parameter definitions (hard-coded, physics-visible)
# ============================================================

# MPI / lattice partitioning
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

# Configuration file template (populated from Slurm args)
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])
cfg_path = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original"
    "/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
).format(n_cfg=n_cfg)

# Quark masses (Wilson-clover convention)
m_l  = -0.277      # light  (u, d)
m_c  =  0.4159     # charm

# Clover action parameters
xi_0 = 1.0
csw  = 1.160920226

# Solver parameters: CGNR on M^dag M (non-Hermitian Wilson-clover)
tol      = 1.0e-12
maxiter  = 20000
# Multigrid coarsening tuned for light-quark near-null space
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout link smearing (applied before inversion)
stout_nstep = 1
stout_rho   = 0.125
stout_ndim  = 4

# Point source at origin, zero momentum
x_src = [0, 0, 0, 0]


# ============================================================
# 2. Read gauge configuration
# ============================================================

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Apply stout smearing in-place (same smeared gauge for both inversions)
gauge.stoutSmear(stout_nstep, stout_rho, stout_ndim)


# ============================================================
# 3. Construct the Dirac operator (Wilson-clover)
# ============================================================

dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
# Charm: same multigrid coarsening; may be sub-optimal for heavy quark.
# If stalling is observed, fall back to plain CGNR (multigrid=None).
dirac_c = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, multigrid)


# ============================================================
# 4. Compute forward propagators
# ============================================================

with dirac_l.useGauge(gauge):
    prop_l = core.invert(dirac_l, "point", x_src)

with dirac_c.useGauge(gauge):
    prop_c = core.invert(dirac_c, "point", x_src)


# ============================================================
# 5. Contraction: Lambda_c (udc) baryon 2pt, single topology
# ============================================================

# Gamma matrices and epsilon on GPU
G5   = cp.asarray(gamma.gamma(15), dtype=cp.complex128)               # gamma_5
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)  # C = gamma_2 gamma_4
Cg5  = Cmat @ G5                                                       # C * gamma_5
Tmat = cp.asarray((cp.eye(4, dtype=cp.complex128) + gamma.gamma(8)) * 0.5,
                  dtype=cp.complex128)                                 # P^+ = (1+gamma_4)/2

# Levi-Civita epsilon_{abc}
epsilon = cp.zeros((3, 3, 3), dtype=cp.complex128)
epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1.0

# FROM generate_einsum (baryon_2pt)
#   Single Wick topology for udc (all three quark flavours distinct).
#   No exchange term exists because charm -> light contraction would be
#   flavour-violating.
#   sign = +1
C_t_local = contract(
    'AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFAfa, wtzyxEBeb -> t',
    Cg5, epsilon, Cg5, epsilon, Tmat,
    prop_c.data,   # charm propagator: spin (D,C), colour (d,c)
    prop_l.data,   # u-quark propagator: spin (F,A), colour (f,a)
    prop_l.data,   # d-quark propagator: spin (E,B), colour (e,b)
)

# MPI gather: collect time dimension, sum over spatial
C_t = core.gatherLattice(
    array.arrayAsNumpy(C_t_local, backend="cupy"),
    [0, -1, -1, -1],
)


# ============================================================
# 6. Save the result (rank 0 only)
# ============================================================

if core.getMPIRank() == 0:
    C_t_root = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    t_arr    = np.arange(C_t_root.shape[0], dtype=np.int32)
    out      = np.column_stack((t_arr, C_t_root.real, C_t_root.imag))
    out_path = f"lambda_c_2pt_cfg{n_cfg}.txt"
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
