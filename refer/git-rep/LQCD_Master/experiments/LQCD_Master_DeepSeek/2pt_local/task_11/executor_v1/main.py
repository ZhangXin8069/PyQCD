# Run: mpirun -np 4 python3 main.py ~/.cache 10000
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma

# ── Parameters ──
resource_path = sys.argv[1]
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)
cfg_path = cfg_path.format(n_cfg=n_cfg)

# Source position (point source at origin, zero momentum)
x_src = [0, 0, 0, 0]

# Clover fermion parameters
xi_0 = 1.0
csw = 1.160920226

# Quark masses (valence = sea for strange, unitary calculation)
m_l = -0.277       # light (d quark)
m_s = -0.2400      # strange (s quark) — unitary with sea ms=-0.2400

# Solver parameters
tol = 1.0e-12
maxiter = 20000
# Safe MG blocking: single-level [[4,4,4,3]] avoids coarsest-level O(1) site/rank
multigrid = [[4, 4, 4, 3]]

# Stout link smearing
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ── Initialize PyQUDA ──
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── Read gauge configuration and apply stout smearing ──
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()
gauge.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── Construct Dirac operators (Clover fermions) ──
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# ── Compute forward propagators (point source, stout-smeared gauge) ──
with dirac_l.useGauge(gauge):
    prop_l = core.invert(dirac_l, "point", x_src)

with dirac_s.useGauge(gauge):
    prop_s = core.invert(dirac_s, "point", x_src)

# ── Gamma matrices and tensors (GPU) ──
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus = (1+gamma_4)/2

# Levi-Civita epsilon_{abc}
epsilon = cp.zeros((3, 3, 3), dtype=cp.complex128)
epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1.0

# ── Extract observable: Xi- (dss) baryon two-point contraction ──
# Xi- operator: epsilon^{abc} (s^{Ta} C gamma_5 d^b) s^c
# Quark-line mapping: a=s->prop_s, b=d->prop_l, c=s->prop_s
# Two Wick topologies (direct + exchange), Cg5 diquark, P_plus projector
# FROM generate_einsum (baryon_2pt)
C_t_local = (
    - contract(
        "AB, abc, EF, efd, CD, wtzyxDBdb, wtzyxFAfa, wtzyxECec -> t",
        Cg5, epsilon, Cg5, epsilon, Tmat, prop_s.data, prop_l.data, prop_s.data,
    )
    + contract(
        "AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFAfa, wtzyxEBeb -> t",
        Cg5, epsilon, Cg5, epsilon, Tmat, prop_s.data, prop_l.data, prop_s.data,
    )
)

# ── MPI gather: reduce over spatial dimensions, gather time ──
C_t = core.gatherLattice(
    array.arrayAsNumpy(C_t_local, backend="cupy"), [0, -1, -1, -1]
)

# ── Save result (rank 0 only) ──
if core.getMPIRank() == 0:
    t = np.arange(latt_size[3], dtype=np.int32)
    C_t_arr = np.asarray(C_t, dtype=np.complex128)
    out = np.column_stack((t, C_t_arr.real, C_t_arr.imag))
    np.savetxt(f"xi_minus_2pt_cfg{n_cfg}.txt", out, fmt=["%d", "%.16e", "%.16e"])
