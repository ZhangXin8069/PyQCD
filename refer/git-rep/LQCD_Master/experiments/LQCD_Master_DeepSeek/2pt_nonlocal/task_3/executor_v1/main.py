# Run: mpirun -n 4 python3 main.py <resource_path> <n_cfg>
#
# Nonlocal two-point correlation function of eta_s (ssbar, J^PC = 0^-+)
# with s-quark field shifted by z in +z-direction via Wilson line.
# Connected diagram only; point source at [0,0,0,0].

import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source
from pyquda_utils.core import X, Y, Z, T

# ═══════════════════════════════════════════════════════════════════
# 1. Parameter definitions (hard-coded)
# ═══════════════════════════════════════════════════════════════════

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

# Lattice geometry
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]          # 4 MPI ranks in t-direction

# Gauge configuration path template
cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Point source position
x_src = [0, 0, 0, 0]

# Strange quark action parameters (Wilson-clover)
xi_0 = 1.0                       # gauge anisotropy
csw  = 1.160920226               # clover coefficient
m_s  = -0.2356                   # strange quark mass (kappa-critical)

# Solver parameters
tol       = 1e-12
maxiter   = 20000
multigrid = [[6, 6, 6, 3],       # level-1 block size
             [4, 4, 4, 6]]       # level-2 block size

# Stout smearing (for Dirac inversion only)
stout_nstep = 1
stout_rho   = 0.125
stout_ndim  = 4

# Nonlocal shift range
zmax = 10

# ═══════════════════════════════════════════════════════════════════
# 2. Read gauge configuration
# ═══════════════════════════════════════════════════════════════════

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()

# Keep a copy for stout smearing (used for Dirac inversion only)
gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════════════
# 3. Construct the Dirac operator
# ═══════════════════════════════════════════════════════════════════

dirac_s = core.getDirac(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# ═══════════════════════════════════════════════════════════════════
# 4. Compute forward strange-quark propagator (point source)
# ═══════════════════════════════════════════════════════════════════

pt_src = source.source12(latt_info, "point", x_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# ═══════════════════════════════════════════════════════════════════
# 5. Extract observable: nonlocal eta_s 2pt for z = 0 .. zmax
# ═══════════════════════════════════════════════════════════════════

# After gamma5-hermiticity, the connected correlator reduces to
#   C(z; t) = sum_x Tr[ S_s^dag(x,t) * W(x,x+z*e_z;t) * S_s(x+z*e_z,t) ]
# where W is the straight Wilson line (original unsmeared links).
# covDev on raw gauge shifts the propagator by one step and applies
# the gauge link, so z repeated covDev calls build W * S_s(x+z*e_z).

Lt_local = prop_s.data.shape[1]  # local time extent on this MPI rank
C_loc = cp.zeros((zmax + 1, Lt_local), dtype=cp.complex128)

# FROM generate_einsum (meson_2pt) — connected contribution for eta_s
# Einsum contracts parity(w), spatial(zyx), spin(CB), color(ba) → leaves t
einsum_str = "wtzyxCBba, wtzyxCBba -> t"

with gauge_raw.use() as dirac_shift:
    for zsep in range(zmax + 1):
        # Build shifted propagator: W(x, x+z*e_z) * prop_s(x+z*e_z)
        prop_shift = prop_s.copy()
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_shift.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, Z)
                    prop_shift.setFermion(tmp, spin, color)

        # Contract: C(z,t) = sum_{x,spin,color} Tr[ S^dag * S_shifted ]
        C_local = contract(einsum_str, prop_s.data.conj(), prop_shift.data)
        C_loc[zsep] = C_local

# ═══════════════════════════════════════════════════════════════════
# 6. MPI gather and save results
# ═══════════════════════════════════════════════════════════════════

for zsep in range(zmax + 1):
    # gatherLattice gathers dim 0 (t) across MPI ranks
    t_global = core.gatherLattice(C_loc[zsep].get(), [0, -1, -1, -1])

    if core.getMPIRank() == 0:
        Lt_global = len(t_global)
        out_path = f"etas_nonlocal_2pt_z{zsep:02d}.txt"
        out_data = np.column_stack((
            np.arange(Lt_global, dtype=np.int32),
            t_global.real,
            t_global.imag,
        ))
        np.savetxt(out_path, out_data, fmt=["%d", "%.16e", "%.16e"])
