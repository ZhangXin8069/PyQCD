# Run: mpirun -np 4 python3 main.py ~/.cache 10000
import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, source
from pyquda_utils.core import Z

# ═══════════════════════════════════════════════════════════════
# 1. Parameter definitions (hard-coded)
# ═══════════════════════════════════════════════════════════════
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

x_src = [0, 0, 0, 0]

# Clover ensemble parameters
xi_0 = 1.0
csw = 1.160920226
m_c = 0.4159
tol = 1e-12
maxiter = 20000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout smearing (for Dirac inverter only)
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# Nonlocal shift parameters
zmax = 10          # maximum z-separation (0..10 inclusive)
direction = Z      # +z direction

out_file = "etac_nonlocal_2pt_z0-10.txt"

# ═══════════════════════════════════════════════════════════════
# 2. Read gauge configuration
# ═══════════════════════════════════════════════════════════════
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Deep copy BEFORE any smearing — used exclusively for Wilson line construction
gauge_original = gauge.copy()
gauge_original.toDevice()

# Stout smear the active gauge field — used ONLY by the Dirac inverter
gauge.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════════
# 3. Construct the Dirac operator (clover, charm quark)
# ═══════════════════════════════════════════════════════════════
dirac_c = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, multigrid)

# ═══════════════════════════════════════════════════════════════
# 4. Compute forward propagator (point source, stout-smeared gauge)
# ═══════════════════════════════════════════════════════════════
pt_src = source.source12(latt_info, "point", x_src)

with dirac_c.useGauge(gauge):
    prop_c = core.invertPropagator(dirac_c, pt_src)
# Stout gauge context CLOSED — only gauge_original remains for covDev

# ═══════════════════════════════════════════════════════════════
# 5. Extract observable: nonlocal two-point function via covDev
# ═══════════════════════════════════════════════════════════════
# The Wilson line W(0,z;t) is baked into the propagator by applying
# covDev in +Z direction z times using the ORIGINAL unsmeared gauge links.
# After z covDev steps, the shifted propagator at position x contains
# the original propagator from position x−z·ê_z transported forward.
# Contracting S†(x,t) with the shifted S(x,t) over all spatial positions
# yields the nonlocal correlator Σ_x Tr[S†(x,t) · W·S(x+z,t)].
# For z=0 this reduces to the standard local η_c two-point function.
#
# Einsum string from generate_einsum(type="meson_2pt"):
# FROM generate_einsum (meson_2pt)
einsum_str = "wtzyxCBba, wtzyxCBba -> t"

# Store per-z results: shape (zmax+1, Lt_local)
C_loc = cp.zeros((zmax + 1, latt_info.Lt), dtype=cp.complex128)

with gauge_original.use() as dirac_shift:
    for zsep in range(zmax + 1):
        # Fresh copy of the original propagator for each z-separation
        prop_shift = prop_c.copy()

        # Apply covDev +Z zsep times — builds the straight Wilson line
        # from the original unsmeared gauge links
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_shift.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, direction)
                    prop_shift.setFermion(tmp, spin, color)

        # Contract S†(origin, t) with S_shifted(z, t):
        #   Tr_{spin,color}[S_c†(x,t) · W(x,x+z) · S_c(x+z,t)]
        # summed over all spatial positions x.
        # Result is 1D: shape (Lt_local,) — parity/spatial/spin/color
        # all contracted in the -> t einsum.
        C_loc[zsep] = contract(
            einsum_str,
            prop_c.data.conj(),   # S_c† at origin
            prop_shift.data,      # W·S_c shifted by zsep in +Z
        )
# Original gauge context CLOSED — safe for MPI communication now

# ═══════════════════════════════════════════════════════════════
# 6. Save the result (MPI gather to rank 0, then write txt)
# ═══════════════════════════════════════════════════════════════
C_full = np.zeros((zmax + 1, latt_size[3]), dtype=np.complex128)

for zsep in range(zmax + 1):
    t_field_global = core.gatherLattice(
        array.arrayAsNumpy(C_loc[zsep], backend="cupy"),
        [0, -1, -1, -1],
    )
    if core.getMPIRank() == 0:
        C_full[zsep, :] = t_field_global

if core.getMPIRank() == 0:
    with open(out_file, "w") as f:
        for zsep in range(zmax + 1):
            for t in range(latt_size[3]):
                val = C_full[zsep, t]
                f.write(f"{zsep} {t} {val.real:.16e} {val.imag:.16e}\n")
