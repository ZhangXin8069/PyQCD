# Run: mpirun -np 4 python main.py <resource_path> <cfg_number>
import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma

# ═══════════════════════════════════════════════════════════════
# 1. Parameter definitions
# ═══════════════════════════════════════════════════════════════

resource_path = os.path.expanduser(sys.argv[1])
n_cfg = int(sys.argv[2])

# Lattice geometry
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

# Configuration path
cfg_path = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/"
    "Configurations/Original/"
    f"beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Action parameters
xi_0 = 1.0
csw = 1.160920226

# Quark masses
m_l = -0.277       # light (Wilson-clover kappa convention)
m_c = 0.4159       # charm

# Light-quark solver (multigrid)
tol_l = 1.0e-8
maxiter_l = 20000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Charm-quark solver (no multigrid → BiCGStab)
tol_c = 1.0e-8
maxiter_c = 5000

# Stout link smearing parameters
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# Point source position
x_src = [0, 0, 0, 0]

# Output file name
out_path = f"D_plus_2pt_cfg{n_cfg}.txt"

# ═══════════════════════════════════════════════════════════════
# PyQUDA initialization
# ═══════════════════════════════════════════════════════════════
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ═══════════════════════════════════════════════════════════════
# 2. Read gauge configuration
# ═══════════════════════════════════════════════════════════════
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Apply stout smearing to a copy (keep raw gauge untouched)
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════════
# 3. Construct Dirac operators (Clover fermions)
# ═══════════════════════════════════════════════════════════════
dirac_l = core.getClover(latt_info, m_l, tol_l, maxiter_l, xi_0, csw, csw, multigrid)
dirac_c = core.getClover(latt_info, m_c, tol_c, maxiter_c, xi_0, csw, csw, None)

# ═══════════════════════════════════════════════════════════════
# 4. Compute forward propagators
# ═══════════════════════════════════════════════════════════════

# Light quark propagator (multigrid) on stout-smeared gauge
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invert(dirac_l, "point", x_src)

# Charm quark propagator (BiCGStab) on stout-smeared gauge
with dirac_c.useGauge(gauge_stout):
    prop_c = core.invert(dirac_c, "point", x_src)

# ═══════════════════════════════════════════════════════════════
# 5. Contraction: D+ meson two-point correlator
# ═══════════════════════════════════════════════════════════════

# FROM generate_einsum (meson_2pt)
# D+ = anti-d + c, gamma5 pseudoscalar interpolator at both source and sink.
# For gamma5, the Dirac adjoint at source and the gamma5-hermiticity
# factors cancel, leaving Tr[S_l^dag  S_c].
C_t_local = contract(
    "wtzyxCBba, wtzyxCBba -> t",
    prop_l.data.conj(),
    prop_c.data,
)

# MPI gather: reduce over spatial sites, gather time dimension
C_t = core.gatherLattice(
    array.arrayAsNumpy(C_t_local, backend="cupy"),
    [0, -1, -1, -1],
)

# ═══════════════════════════════════════════════════════════════
# 6. Save the result (plain txt, no header)
# ═══════════════════════════════════════════════════════════════
if core.getMPIRank() == 0:
    C_t_root = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    t = np.arange(C_t_root.shape[0], dtype=np.int32)
    out = np.column_stack((t, C_t_root.real, C_t_root.imag))
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
