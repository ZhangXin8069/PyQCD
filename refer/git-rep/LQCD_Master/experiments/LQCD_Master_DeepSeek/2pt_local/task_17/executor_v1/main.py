import sys
import numpy as np
import cupy as cp
from opt_einsum import contract

from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ── Parameters ────────────────────────────────────────────
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/"
    "Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

x_src = [0, 0, 0, 0]

xi_0 = 1.0
csw = 1.160920226
m_s = -0.2356
m_c = 0.4159

tol = 1.0e-12
maxiter_s = 10000
maxiter_c = 20000

multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# Per-config output file — avoids overwrite across SLURM array tasks
out_path = f"./omega_c0_2pt_result_{n_cfg}.txt"

# ── Initialize PyQUDA ────────────────────────────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── Load gauge configuration ─────────────────────────────
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Stout smear a copy for Dirac inversions (original gauge unchanged)
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── Dirac operators ──────────────────────────────────────
# Strange quark: multigrid solver for near-physical mass
dirac_s = core.getClover(latt_info, m_s, tol, maxiter_s, xi_0, csw, csw, multigrid)
# Charm quark: BiCGStab solver (multigrid=None).  Heavy mass → well-conditioned;
# the solver prints residual history to stdout — monitor for convergence failures.
dirac_c = core.getClover(latt_info, m_c, tol, maxiter_c, xi_0, csw, csw, None)

# ── Forward propagators from point source at origin ──────
pt_src = source.source12(latt_info, "point", x_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# ── Gamma matrices and tensors on GPU ────────────────────
# Cg5 = C @ gamma_5: antisymmetric under transpose,
# required for non-vanishing diquark with identical s-quarks (Pauli principle)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5

# Tmat = P_plus = (1 + gamma_4)/2: positive-parity projector
I4_gpu = cp.eye(4, dtype=cp.complex128)
Tmat = cp.asarray((I4_gpu + gamma.gamma(8)) * 0.5, dtype=cp.complex128)

# Levi-Civita epsilon_{abc} for color antisymmetrization
epsilon = cp.zeros((3, 3, 3), dtype=cp.complex128)
epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1.0

# ── Baryon 2pt contraction: Omega_c0 (ssc, Cg5 diquark, P_plus) ─
# FROM generate_einsum (baryon_2pt)
# Two Wick topologies from identical strange quarks:
#   topo0 (direct):  sign = -1
#   topo1 (exchange): sign = +1
C_t_local = (
    - contract(
        "AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFBfb, wtzyxEAea -> t",
        Cg5, epsilon, Cg5, epsilon, Tmat, prop_c.data, prop_s.data, prop_s.data,
    )
    + contract(
        "AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFAfa, wtzyxEBeb -> t",
        Cg5, epsilon, Cg5, epsilon, Tmat, prop_c.data, prop_s.data, prop_s.data,
    )
)

# ── MPI gather: sum spatial volume, concatenate time slices ─
C_t = core.gatherLattice(
    array.arrayAsNumpy(C_t_local, backend="cupy"), [0, -1, -1, -1]
)

# ── Save result (rank 0 only) ────────────────────────────
# Output: two columns (real, imag) per time slice, no header
if core.getMPIRank() == 0:
    np.savetxt(
        out_path,
        np.column_stack((C_t.real, C_t.imag)),
        fmt="%.16e",
    )
