# Run: mpirun -n 4 python3 main.py ~/.cache 10000
#
# J/psi nonlocal two-point function — vector channel (γ_x, γ_y, γ_z averaged)
# Shifts charm-quark leg by z_len in +z (Wilson line from ORIGINAL unsmeared gauge).
# Contraction: Tr[S_c† (γ₅ γ_i) S_shifted (γ_i γ₅)], averaged over i=x,y,z.
# Output: nonlocal_2pt_jpsi.txt (z_len t Re[C] Im[C])

import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source
from pyquda_utils.core import X, Y, Z, T

# ============================================================
# Physical and ensemble parameters (hard-coded)
# ============================================================

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

x_src = [0, 0, 0, 0]
zmax = 10

# Clover charm quark
xi_0 = 1.0
csw = 1.160920226
m_c = 0.4159
tol = 1.0e-12
maxiter = 2000
multigrid = None  # CG for heavy charm

# Stout smearing for Dirac inversion
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_path = "nonlocal_2pt_jpsi.txt"

# ============================================================
# Initialize PyQUDA
# ============================================================
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ============================================================
# Load gauge configuration
# ============================================================
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# CRITICAL: copy BEFORE smearing — raw links for Wilson line
gauge_raw = gauge.copy()
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ============================================================
# Dirac operator (Clover, charm mass)
# ============================================================
dirac_c = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, multigrid)

# ============================================================
# Forward propagator — point source at [0,0,0,0]
# ============================================================
pt_src = source.source12(latt_info, "point", x_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# ============================================================
# Gamma matrices on GPU (DeGrand-Rossi basis)
# ============================================================
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)    # γ₅
G1 = cp.asarray(gamma.gamma(1), dtype=cp.complex128)     # γ₁ (γ_x)
G2 = cp.asarray(gamma.gamma(2), dtype=cp.complex128)     # γ₂ (γ_y)
G3 = cp.asarray(gamma.gamma(4), dtype=cp.complex128)     # γ₃ (γ_z) — note bit 4

# Precompute (γ₅ γ_i) and (γ_i γ₅) for each polarization
# These act on spin indices in the contraction Tr[S†(γ₅γ_i) S (γ_iγ₅)]
g5g1 = G5 @ G1
g1g5 = G1 @ G5
g5g2 = G5 @ G2
g2g5 = G2 @ G5
g5g3 = G5 @ G3
g3g5 = G3 @ G5

# ============================================================
# Contraction einsum
# ============================================================
# Derivation (see plan):
#   Operator: O_i = c̄(x) γ_i W(x,x+z) c(x+z)  (nonlocal, quark leg shifted)
#   Adjoint:  O_i† = -c̄(0) γ_i c(0)  (local)
#   Wick + γ₅-hermiticity → C_i = Tr[S† (γ₅γ_i) S (γ_iγ₅)]
#
# Index routing in PyQUDA convention:
#   prop_c.data shape: [w, t, z, y, x//2, spin_snk, spin_src, col_snk, col_src]
#   .conj() is element-wise conjugate (no transpose).
#   The physical Hermitian conjugate S† has swapped spin & color indices
#   relative to .conj(). The einsum accounts for this via index labeling:
#     first  arg (.conj):  C=spin_snk, B=spin_src, p=col_snk, q=col_src
#     second arg (data):   D=spin_snk, E=spin_src, q=col_snk, p=col_src (swapped color)
#   This is equivalent to the physical Tr[S† · A · S · B] with A=(γ₅γ_i), B=(γ_iγ₅).
#
# The generate_einsum tool provides the γ₅-hermiticity part (contracting through
# G5·S†·G5) but does not embed the operator-specific gamma matrices — those are
# applied here as explicit A and B operands.
einsum_str = "wtzyxCBpq, wtzyxDEqp, CD, EB -> t"

# ============================================================
# Nonlocal contraction: z_len = 0 .. zmax
# ============================================================
Lt_local = latt_info.Lt
C_loc = cp.zeros((zmax + 1, Lt_local), dtype=cp.complex128)

with gauge_raw.use() as dirac_shift:
    for zsep in range(zmax + 1):
        # Build shifted propagator: S_shift = W(x, x+z*ẑ) * S_c(x+z*ẑ)
        # One covDev call = one lattice step in +z. Periodic BC handled by QUDA.
        prop_shift = prop_c.copy()
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_shift.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, Z)
                    prop_shift.setFermion(tmp, spin, color)

        # Vector polarization average: (C_x + C_y + C_z) / 3
        C1 = contract(einsum_str, prop_c.data.conj(), prop_shift.data, g5g1, g1g5)
        C2 = contract(einsum_str, prop_c.data.conj(), prop_shift.data, g5g2, g2g5)
        C3 = contract(einsum_str, prop_c.data.conj(), prop_shift.data, g5g3, g3g5)
        C_loc[zsep] = (C1 + C2 + C3) / 3.0

# ============================================================
# MPI gather (OUTSIDE gauge_raw.use() context — REQUIRED)
# ============================================================
C_full = np.zeros((zmax + 1, latt_size[3]), dtype=np.complex128)
for zsep in range(zmax + 1):
    t_field_global = core.gatherLattice(C_loc[zsep].get(), [0, -1, -1, -1])
    if core.getMPIRank() == 0:
        C_full[zsep, :] = t_field_global

# ============================================================
# Save: z_len t Re[C] Im[C]  (no header, rank 0 only)
# ============================================================
if core.getMPIRank() == 0:
    with open(out_path, "w") as f:
        for zsep in range(zmax + 1):
            for t in range(latt_size[3]):
                val = C_full[zsep, t]
                f.write(f"{zsep} {t} {val.real:.16e} {val.imag:.16e}\n")
