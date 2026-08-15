# Run: mpirun -n 4 python3 main.py ~/.cache 10000
import os
import sys
import numpy as np
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ── Parameters ────────────────────────────────────────────
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Point source at origin, zero momentum
x_src = [0, 0, 0, 0]

# Charm quark — Wilson-clover action
m_c = 0.4159
xi_0 = 1.0
csw = 1.160920226
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout link smearing before inversion
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ── Initialize PyQUDA ──────────────────────────────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── Read gauge configuration ───────────────────────────────
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Create stout-smeared copy for Dirac inversion
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── Construct Dirac operator (Wilson-clover, charm) ────────
dirac_c = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, multigrid)

# ── Compute forward charm propagator ───────────────────────
pt_src = source.source12(latt_info, "point", x_src)
with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# ── Extract observable: ηc two-point correlator ────────────
# Interpolating operator: O_ηc = c̄ γ5 c  (J^{PC}=0^{-+})
# After Wick contraction and γ5-hermiticity, the connected
# diagram reduces to Tr[S_c^† S_c].  Disconnected diagram
# is OZI-suppressed for charmonium and omitted.
# FROM generate_einsum (meson_2pt)
C_t_local = contract(
    "wtzyxCBba, wtzyxCBba -> t",
    prop_c.data.conj(),
    prop_c.data,
)

# MPI gather: sum over spatial sites, gather time dimension
C_t = core.gatherLattice(
    array.arrayAsNumpy(C_t_local, backend="cupy"), [0, -1, -1, -1]
)

# ── Save result ────────────────────────────────────────────
# Plain text: time slice, Re[C(t)], Im[C(t)] — no header
if core.getMPIRank() == 0:
    C_t_root = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    t = np.arange(C_t_root.shape[0], dtype=np.int32)
    out = np.column_stack((t, C_t_root.real, C_t_root.imag))
    out_path = f"etac_2pt_cfg{n_cfg:05d}.txt"
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
