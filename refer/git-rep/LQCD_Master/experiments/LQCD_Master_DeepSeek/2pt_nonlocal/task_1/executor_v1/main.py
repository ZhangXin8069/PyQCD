import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source
from pyquda_utils.core import X, Y, Z, T

# ═══════════════════════════════════════════════════════════════
# 1. Parameter definitions (hard-coded)
# ═══════════════════════════════════════════════════════════════

# Runtime args: resource_path and configuration number
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

# Lattice geometry
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]          # 4 MPI ranks, partitioned in t

# Configuration file path template
cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Source definition: point source at origin, zero momentum
x_src = [0, 0, 0, 0]

# Nonlocal shift range: z = 0 .. 10 (z=0 recovers local pion 2pt)
z_max = 10

# Quark action and solver parameters (Clover Wilson fermions, ensemble C24P29)
xi_0 = 1.0                       # gauge anisotropy
csw = 1.160920226                 # clover coefficient
m_l = -0.277                      # light quark mass (kappa-based)
tol = 1.0e-12                     # CG solver tolerance
maxiter = 10000                   # max CG iterations
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]  # multigrid levels

# Stout link smearing for Dirac operator inversion
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ═══════════════════════════════════════════════════════════════
# 2. Read gauge configuration
# ═══════════════════════════════════════════════════════════════

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()

# Preserve original (unsmeared) gauge links for Wilson line construction.
# Copy BEFORE smearing — stoutSmear modifies in place.
gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════════
# 3. Construct the Dirac operator (Clover with multigrid)
# ═══════════════════════════════════════════════════════════════

dirac_l = core.getDirac(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)

# ═══════════════════════════════════════════════════════════════
# 4. Compute forward propagator (point source, stout-smeared gauge)
# ═══════════════════════════════════════════════════════════════

pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# ═══════════════════════════════════════════════════════════════
# 5. Nonlocal shift and contraction (Wilson line on ORIGINAL gauge)
# ═══════════════════════════════════════════════════════════════

# The pion nonlocal operator O_π+(x;z) = d̄(x) γ₅ W(x,x+ẑ) u(x+ẑ).
# After Wick contraction with γ₅-hermiticity and u-d flavor symmetry,
# the gamma structure is fully absorbed, leaving Tr[S† W S].
#
# FROM generate_einsum (meson_2pt) — verified contraction:
#   contract('wtzyxCBba, wtzyxCBba -> t', prop_l.data.conj(), prop_l.data)
# This contracts parity(w), spatial(zyx), spin(CB), color(ba) → time-only.

einsum_str = "wtzyxCBba, wtzyxCBba -> t"

C_local_list = []  # one (Lt_local,) cupy array per z separation

# covDev on raw gauge builds the straight Wilson line W(x, x+ẑ).
# QUDA's covDev inherently handles periodic BC: the gauge link U_z at
# site 23 connects to site 0, so crossing the boundary is automatic.
with gauge_raw.use() as dirac_shift:
    for zsep in range(z_max + 1):
        # Fresh copy of the original propagator for each z separation
        prop_shift = prop_l.copy()

        # Apply zsep steps of covariant displacement in +z direction.
        # Each covDev call shifts by one lattice unit; the Wilson line
        # is built from the gauge links traversed along the way.
        for step in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_shift.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, Z)
                    prop_shift.setFermion(tmp, spin, color)

        # Contract: C(z;t) = Σ_x Tr[S†_l(x,t) · S_l,shifted(x,t)]
        # Result shape: (Lt_local,) — time dimension only
        result = contract(einsum_str, prop_l.data.conj(), prop_shift.data)
        C_local_list.append(result)

# ═══════════════════════════════════════════════════════════════
# MPI gather OUTSIDE the gauge_raw.use() context to avoid deadlock
# ═══════════════════════════════════════════════════════════════

C_full = np.zeros((z_max + 1, latt_size[3]), dtype=np.complex128)

for zsep in range(z_max + 1):
    # Transfer cupy array to numpy, then gather time slices across ranks.
    # gatherLattice with [0, -1, -1, -1]: gather dim 0 (time), sum spatial (no-op on 1D).
    # Grid [1,1,1,4] means each rank owns Lt_local=18 time slices.
    t_global = core.gatherLattice(C_local_list[zsep].get(), [0, -1, -1, -1])
    if core.getMPIRank() == 0:
        C_full[zsep, :] = t_global

# ═══════════════════════════════════════════════════════════════
# 6. Save the result (real part only, no header)
# ═══════════════════════════════════════════════════════════════

# Only the real part is physical; the imaginary part is zero up to
# roundoff after the full spatial sum (hermiticity of the pion 2pt).
# Output: plain text, 11 lines × 72 columns, no header or metadata.

if core.getMPIRank() == 0:
    out_path = f"pion_nonlocal_2pt_cfg{n_cfg}.txt"
    np.savetxt(out_path, C_full.real, fmt="%.16e")
