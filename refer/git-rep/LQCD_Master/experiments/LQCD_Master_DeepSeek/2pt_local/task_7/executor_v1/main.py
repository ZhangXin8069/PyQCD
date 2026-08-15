# D_s^+ meson two-point correlation function (c anti-s, pseudoscalar, gamma5)
# Run: mpirun -np 4 python main.py <resource_path> <n_cfg>

import sys
import numpy as np
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io

# ── Hard-coded physics parameters ─────────────────────────
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72/"
    "Configurations/Original/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

x_src = [0, 0, 0, 0]  # point source at origin

# Clover-improved Wilson fermion parameters
xi_0 = 1.0
csw = 1.160920226
m_s = -0.2356   # strange quark mass (antiquark in D_s^+)
m_c = 0.4159    # charm quark mass (quark in D_s^+)
tol = 1.0e-12
maxiter = 2000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout smearing parameters (applied to gauge links before inversion)
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ── Runtime arguments ─────────────────────────────────────
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

# ── Initialize PyQUDA ─────────────────────────────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── Read gauge configuration ──────────────────────────────
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Stout-smeared copy for Dirac inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── Construct Dirac operators ─────────────────────────────
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_c = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, multigrid)

# ── Forward propagators (point source at origin) ──────────
# Strange propagator — represents the anti-strange antiquark in D_s^+
with dirac_s.useGauge(gauge_stout):
    prop_s = core.invert(dirac_s, "point", x_src)

# Charm propagator — represents the charm quark in D_s^+
with dirac_c.useGauge(gauge_stout):
    prop_c = core.invert(dirac_c, "point", x_src)

# ── Contraction: Tr[prop_s^dag @ prop_c] → C(t) ──────────
# The Wick contraction for D_s^+ (c anti-s, gamma5) simplifies
# via gamma5-hermiticity and cyclicity of the trace:
#   C(t) = sum_x Tr[S_s^dag(x;0) S_c(x;0)]
# The dagger belongs on the antiquark (strange) propagator.
# FROM generate_einsum (meson_2pt)
C_t_local = contract(
    "wtzyxCBba, wtzyxCBba -> t",
    prop_s.data.conj(),   # S_s^dag — antiquark propagator
    prop_c.data,           # S_c   — quark propagator
)

# MPI gather: collect time-dimension data from all ranks
C_t = core.gatherLattice(
    array.arrayAsNumpy(C_t_local, backend="cupy"),
    [0, -1, -1, -1],
)

# ── Save result: 72 real values, one per time slice ──────
if core.getMPIRank() == 0:
    # Zero-momentum pseudoscalar correlator is real
    C_t_real = np.asarray(C_t, dtype=np.complex128).real.reshape(-1)
    out_path = f"ds_2pt_cfg{n_cfg}.txt"
    np.savetxt(out_path, C_t_real, fmt="%.16e")
