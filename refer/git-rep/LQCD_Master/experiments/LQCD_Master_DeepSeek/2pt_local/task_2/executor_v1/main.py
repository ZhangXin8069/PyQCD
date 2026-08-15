# Kaon K+ two-point correlation function
# C(t) = sum_x Re Tr[ S_s^dag(x,t) S_l(x,t) ]
# Point source at origin [0,0,0,0], stout-smeared gauge, zero momentum.
#
# Run: mpirun -np 4 python main.py ~/.cache 10000

import sys
import numpy as np
from opt_einsum import contract

from pyquda_comm import array
from pyquda_utils import core, io, gamma, phase_v2

# ── 1. Parameter definitions ─────────────────────────────
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_"
    + str(n_cfg) + ".lime"
)

# Clover action parameters
xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
m_s = -0.2356
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout smearing parameters
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# Point source at origin, zero momentum
x_src = [0, 0, 0, 0]
mom = [0, 0, 0]

out_file = "corr_kaon_2pt.txt"

# ── Initialize PyQUDA ────────────────────────────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── 2. Read gauge configuration ─────────────────────────
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Apply stout smearing to a copy of the gauge field
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── 3. Construct Dirac operators ─────────────────────────
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# ── 4. Compute forward propagators ──────────────────────
# Zero-momentum phase (all ones)
phase = phase_v2.MomentumPhase(latt_info).getPhase(mom, x_src[:3])

# Light u-quark propagator
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invert(dirac_l, "point", x_src, phase.data)

# Strange s-quark propagator
with dirac_s.useGauge(gauge_stout):
    prop_s = core.invert(dirac_s, "point", x_src, phase.data)

# ── 5. Contraction: C(t) = sum_x Re Tr[ S_s^dag S_l ] ───
# FROM generate_einsum (meson_2pt)
# prop_s.data.conj()  ->  S_s^dag  (antiquark gets the dagger)
# prop_l.data          ->  S_l     (quark, no dagger)
C_t_local = contract(
    "wtzyxCBba, wtzyxCBba -> t",
    prop_s.data.conj(),
    prop_l.data,
)

# Gather across MPI ranks (grid is [1,1,1,4] so time is distributed)
C_t = core.gatherLattice(C_t_local.get(), [0, -1, -1, -1])

# ── 6. Save result ───────────────────────────────────────
if core.getMPIRank() == 0:
    # Extract real part; the imaginary part is machine-precision noise
    C_t_real = np.asarray(C_t.real, dtype=np.float64).reshape(-1)
    np.savetxt(out_file, C_t_real, fmt="%.16e")
