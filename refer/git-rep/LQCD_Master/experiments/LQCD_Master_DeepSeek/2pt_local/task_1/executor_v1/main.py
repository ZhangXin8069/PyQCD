# Run: mpirun -n 4 python3 main.py ~/.cache 10000
import sys
import numpy as np
from opt_einsum import contract
from pyquda_utils import core, io, source
from pyquda_comm import array

# ── Parameters ────────────────────────────────────────────
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime".format(n_cfg=n_cfg)

# Dirac operator parameters (clover-improved Wilson, light quark)
mass = -0.277
tol = 1e-12
maxiter = 10000
xi_0 = 1.0
csw = 1.160920226
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout link smearing
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# Point source at spacetime origin — all 12 spin×color components (standard convention)
src_pos = [0, 0, 0, 0]

# Output: one floating-point number per time slice, no header
out_path = "pion_2pt_result.txt"

# ── Initialize PyQUDA ──────────────────────────────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── Read gauge configuration ───────────────────────────────
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# ── Stout-smear the gauge links ────────────────────────────
gauge.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── Construct the Dirac operator ───────────────────────────
dirac = core.getClover(latt_info, mass, tol, maxiter, xi_0, csw, csw, multigrid)

# ── Compute forward light-quark propagator ─────────────────
# Point source: unit delta at [0,0,0,0] on all spin×color components
pt_src = source.propagator(latt_info, "point", src_pos)
with dirac.useGauge(gauge):
    prop_l = core.invertPropagator(dirac, pt_src)

# ── Wick contraction: C(t) = Σ_x Tr[prop_l†(x,t) prop_l(x,t)] ──
# For the pion channel (γ₅ operator), isospin symmetry and γ₅-hermiticity
# make the γ₅ matrices cancel, leaving the simple trace squared.
# FROM generate_einsum (meson_2pt):
#   contract('wtzyxCBba, wtzyxCBba -> t', prop_l.data.conj(), prop_l.data)
C_t_local = contract(
    'wtzyxCBba, wtzyxCBba -> t',
    prop_l.data.conj(),
    prop_l.data,
)
C_t = core.gatherLattice(array.arrayAsNumpy(C_t_local, backend="cupy"), [0, -1, -1, -1])

# ── Save result (rank 0 only) ──────────────────────────────
if core.getMPIRank() == 0:
    # C(t) is real for the zero-momentum pion; output one value per time slice
    np.savetxt(out_path, C_t.real, fmt="%.16e")
