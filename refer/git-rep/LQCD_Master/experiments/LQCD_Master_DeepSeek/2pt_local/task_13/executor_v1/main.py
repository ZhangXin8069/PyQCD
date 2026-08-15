# Run: mpirun -np 4 python main.py ~/.cache <cfg_num>

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
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Quark masses (kappa convention)
m_l = -0.277
m_s = -0.2356

# Solver parameters
tol = 1.0e-12
maxiter = 20000
xi_0 = 1.0
csw = 1.160920226

# Multigrid: 2-level
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout smearing on gauge links
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# Point source at origin
x_src = [0, 0, 0, 0]

# ============================================================
# 2. Read gauge configuration
# ============================================================

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Copy before smearing, then apply stout smearing for inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ============================================================
# 3. Construct the Dirac operators (clover fermions)
# ============================================================

dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# ============================================================
# 4. Compute forward propagators (point source at origin)
# ============================================================

pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# ============================================================
# 5. Sigma- (dds) baryon two-point contraction
#    Diquark: Cg5 = C * gamma_5 (required by Pauli principle —
#    antisymmetric in spin indices for the two identical d quarks)
#    Projector: P_plus = (1 + gamma_4)/2  (positive parity)
# ============================================================

# Gamma matrices and tensors on GPU
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)                      # gamma_5
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)   # C = gamma_2 gamma_4
Cg5 = Cmat @ G5                                                             # C gamma_5
I4 = cp.eye(4, dtype=cp.complex128)
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)        # P_plus = (I + gamma_4)/2

# Levi-Civita epsilon_{abc}
epsilon = cp.zeros((3, 3, 3), dtype=cp.complex128)
epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1.0

# FROM generate_einsum (baryon_2pt)
# Two Wick topologies for Sigma- (dds) with Cg5 diquark, P_plus projector
# prop_s is the strange spectator; prop_l provides both d quarks
C_t_local = (
    - contract(
        "AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFBfb, wtzyxEAea -> t",
        Cg5, epsilon, Cg5, epsilon, Tmat,
        prop_s.data, prop_l.data, prop_l.data,
    )
    + contract(
        "AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFAfa, wtzyxEBeb -> t",
        Cg5, epsilon, Cg5, epsilon, Tmat,
        prop_s.data, prop_l.data, prop_l.data,
    )
)

# MPI gather: sum over all ranks, keep time dimension
C_t = core.gatherLattice(C_t_local.get(), [0, -1, -1, -1])

# ============================================================
# 6. Save the result (rank 0 only)
# ============================================================

if core.getMPIRank() == 0:
    C_t_root = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    t_arr = np.arange(C_t_root.shape[0], dtype=np.int32)
    out = np.column_stack((t_arr, C_t_root.real, C_t_root.imag))
    np.savetxt(
        "sigma_minus_2pt_cfg{n_cfg}.txt".format(n_cfg=n_cfg),
        out,
        fmt=["%d", "%.16e", "%.16e"],
    )
