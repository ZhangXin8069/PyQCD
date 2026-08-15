# Run: mpirun -np 4 python3 main.py <resource_path> <n_cfg>

import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma

# ═══════════════════════════════════════════════════════════════════
# 1. Parameter definitions (hard-coded)
# ═══════════════════════════════════════════════════════════════════

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]            # 4 MPI ranks, partitioned in t-direction

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Point source at origin
x_src = [0, 0, 0, 0]

# Clover fermion parameters (C24P29 ensemble)
xi_0 = 1.0
csw  = 1.160920226
m_l  = -0.277      # light quark mass
m_s  = -0.2356     # strange quark mass
m_c  = 0.4159      # charm quark mass

tol     = 1.0e-12
maxiter = 20000

# Multigrid levels (validated for light/strange; primary for charm)
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout link smearing
stout_nstep = 1
stout_rho   = 0.125
stout_ndim  = 4

# ═══════════════════════════════════════════════════════════════════
# 2. Read gauge configuration
# ═══════════════════════════════════════════════════════════════════

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Copy before smearing: gauge_stout for Dirac inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════════════
# 3. Construct Dirac operators (Clover fermions)
# ═══════════════════════════════════════════════════════════════════

dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)
# Charm: use multigrid as primary (plan-specified); CG fallback would
# require exception handling which is avoided per coding constraints.
dirac_c = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, multigrid)

# ═══════════════════════════════════════════════════════════════════
# 4. Compute forward propagators (point source)
# ═══════════════════════════════════════════════════════════════════

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invert(dirac_l, "point", x_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invert(dirac_s, "point", x_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invert(dirac_c, "point", x_src)

# ═══════════════════════════════════════════════════════════════════
# 5. Baryon 2pt contraction — Xi_c+ (usc), Cγ5 diquark, P_+ projector
# ═══════════════════════════════════════════════════════════════════

# Gamma matrices and epsilon tensor (GPU-resident)
I4   = cp.eye(4, dtype=cp.complex128)
G5   = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5  = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus = (1+γ_t)/2

# Levi-Civita ε_{abc}  (normalised: ε_{123}=+1)
eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

# FROM generate_einsum (baryon_2pt) — single Wick topology (u,s,c all distinct)
#   O_snk = ε^{abc} (u_a Cγ5 s_b) c_c,   O_src^† → O_src with conjugate structure
#   Tmat = P_+ projects positive-parity 1/2^+ ground state
#   Args: Cg5(AB), eps(abc), Cg5(EF), eps(efd), Tmat(CD),
#         prop_c(wtzyxDCdc), prop_s(wtzyxFAfa), prop_l(wtzyxEBeb) -> t
C_t_local = contract(
    'AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFAfa, wtzyxEBeb -> t',
    Cg5, eps, Cg5, eps, Tmat,
    prop_c.data, prop_s.data, prop_l.data,
)

# MPI gather: collect time slices from all ranks
C_t = core.gatherLattice(
    array.arrayAsNumpy(C_t_local, backend="cupy"),
    [0, -1, -1, -1],
)

# ═══════════════════════════════════════════════════════════════════
# 6. Save result (rank 0 only)
# ═══════════════════════════════════════════════════════════════════

if core.getMPIRank() == 0:
    C_t_root = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    t_arr = np.arange(C_t_root.shape[0], dtype=np.int32)
    out = np.column_stack((t_arr, C_t_root.real, C_t_root.imag))
    out_path = f"xi_c_plus_2pt_cfg{n_cfg}.txt"
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
