# Run: mpirun -np 4 python main.py ~/.cache 10000
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

# Gauge configuration
cfg_path = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
).format(n_cfg=n_cfg)

# Quark and solver parameters
m_l = -0.277
csw = 1.160920226
xi_0 = 1.0
tol = 1.0e-12
maxiter = 5000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout smearing
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# Point source position
x_src = [0, 0, 0, 0]

# Output
out_path = "proton_2pt.txt"

# ── Initialize PyQUDA ─────────────────────────────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── Load gauge configuration ──────────────────────────────
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Create a smeared copy for Dirac inversion
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── Construct the Dirac operator ──────────────────────────
# Clover-Wilson fermion with multigrid solver
dirac_l = core.getDirac(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)

# ── Compute forward propagator (point source) ─────────────
# Single light-quark propagator reused for all three quark lines
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# ── Contraction: proton two-point correlator ──────────────
# C_p(p=0; t, 0) = sum over Wick topologies of epsilon * Cg5 * Tmat * S_l

# Gamma matrices and tensor definitions on GPU
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5                                     # C * gamma_5  (diquark structure)
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5,      # P_plus = (I + gamma_4)/2
                   dtype=cp.complex128)

# Levi-Civita epsilon_{abc} for color antisymmetrization
epsilon = cp.zeros((3, 3, 3), dtype=cp.complex128)
epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1.0

# FROM generate_einsum (baryon_2pt)
# Two Wick topologies for the proton 2pt correlator:
#   topo 0 (sign = -1): direct contraction
#   topo 1 (sign = +1): exchange contraction after color-index relabeling
C_t_local = (
    - contract(
        'AB, abc, EF, efd, CD, wtzyxDBdb, wtzyxFAfa, wtzyxECec -> t',
        Cg5, epsilon, Cg5, epsilon, Tmat,
        prop_l.data, prop_l.data, prop_l.data,
    )
    + contract(
        'AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFAfa, wtzyxEBeb -> t',
        Cg5, epsilon, Cg5, epsilon, Tmat,
        prop_l.data, prop_l.data, prop_l.data,
    )
)

# MPI gather: combine time dimension across ranks, sum over spatial sites
C_t = core.gatherLattice(C_t_local.get(), [0, -1, -1, -1])

# ── Save result ───────────────────────────────────────────
if core.getMPIRank() == 0:
    C_t_root = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    t_arr = np.arange(C_t_root.shape[0], dtype=np.int32)
    out = np.column_stack((t_arr, C_t_root.real, C_t_root.imag))
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
