# Run: mpirun -n 4 python main.py <resource_path> <n_cfg>
import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ============================================================
#  Parameters (hard-coded per physics specification)
# ============================================================
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]            # 4 MPI ranks in temporal direction

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Point source at the origin
x_src = [0, 0, 0, 0]

# Clover fermion parameters
xi_0 = 1.0
csw = 1.160920226
m_l = -0.277                       # light quark mass (kappa convention)
m_c = 0.4159                       # charm quark mass (kappa convention)
tol = 1.0e-12
maxiter = 20000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout link smearing parameters
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ============================================================
#  Initialization
# ============================================================
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ============================================================
#  Read gauge configuration
# ============================================================
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Apply 4D stout smearing (in-place, modifies gauge)
gauge.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ============================================================
#  Construct Dirac operators
# ============================================================
# Light quark: CG solver with two-level multigrid preconditioner
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)

# Charm quark: standard BiCGStab (multigrid=None), heavy quark does not
# benefit from multigrid acceleration
dirac_c = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, None)

# ============================================================
#  Compute forward propagators
# ============================================================
# Point source at [0,0,0,0], no momentum phase (rest frame)
pt_src = source.source12(latt_info, "point", x_src)

# Light propagator: reused for both u and d quark lines
with dirac_l.useGauge(gauge):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# Charm propagator
with dirac_c.useGauge(gauge):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# ============================================================
#  Contraction: Sigma_c+ (udc) baryon two-point function
# ============================================================
# Interpolating operator: epsilon^{abc} (u^{Ta} C*gamma_1 d^b) c^c
# Diquark structure Cg1 = C @ gamma_1  (vector diquark, I=1 ud pair)
# Parity projector: Tmat = (1 + gamma_t)/2 for positive parity
#
# Note: gamma_1 is antisymmetric under transpose in the DeGrand-Rossi
# basis (gamma_1^T = -gamma_1).  The generate_einsum tool accounts for
# the resulting sign in the contraction.  Only one Wick topology
# because u, d, c are all distinct flavors.

# Gamma matrices and tensor definitions (GPU arrays)
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg1 = Cmat @ cp.asarray(gamma.gamma(1), dtype=cp.complex128)  # C @ gamma_x
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus

# Levi-Civita epsilon_{abc} tensor (color anti-symmetrization)
epsilon = cp.zeros((3, 3, 3), dtype=cp.complex128)
epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1.0

# FROM generate_einsum (baryon_2pt)
# Single Wick topology: sign = +1, quark_lines = [prop_c, prop_l, prop_l]
C_t_local = contract(
    'AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFAfa, wtzyxEBeb -> t',
    Cg1, epsilon, Cg1, epsilon, Tmat, prop_c.data, prop_l.data, prop_l.data
)

# MPI gather along the time dimension; spatial dimensions already
# contracted by the einsum -> t
C_t = core.gatherLattice(
    array.arrayAsNumpy(C_t_local, backend="cupy"), [0, -1, -1, -1]
)

# ============================================================
#  Save result
# ============================================================
if core.getMPIRank() == 0:
    C_t_root = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    t_arr = np.arange(C_t_root.shape[0], dtype=np.int32)
    out = np.column_stack((t_arr, C_t_root.real, C_t_root.imag))
    out_path = f"sigma_c_plus_2pt_cfg{n_cfg}.txt"
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
