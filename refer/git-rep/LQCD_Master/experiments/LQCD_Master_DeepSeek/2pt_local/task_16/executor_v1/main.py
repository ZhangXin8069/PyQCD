import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ═══════════════════════════════════════════════════════════════
# Step 1 — Parameter definitions (hard-coded physics parameters)
# ═══════════════════════════════════════════════════════════════

# Runtime parameters (passed from shell via mpirun)
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

# Lattice geometry
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]           # 4 MPI ranks partitioning the T-direction

# Gauge configuration path
cfg_path = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
).format(n_cfg=n_cfg)

# Source
x_src = [0, 0, 0, 0]               # point source at origin

# Clover action parameters
xi_0 = 1.0                          # gauge anisotropy (isotropic ensemble)
csw = 1.160920226                   # clover coefficient

# Quark masses (kappa-convention, negative for light/strange)
m_l = -0.277                        # light (d) quark
m_s = -0.2356                       # strange quark
m_c = 0.4159                        # charm quark (positive mass)

# Solver parameters
tol_l = 1.0e-12                     # light quark: multigrid, tight tolerance
tol_s = 1.0e-12                     # strange quark: multigrid, tight tolerance
tol_c = 1.0e-10                     # charm quark: CG/BiCGStab, moderate tolerance
maxiter = 2000

# Multigrid parameters (2-level, tuned for light quarks)
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout link smearing (applied before inversion)
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ═══════════════════════════════════════════════════════════════
# Step 2 — Initialize PyQUDA and load gauge configuration
# ═══════════════════════════════════════════════════════════════

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Keep raw gauge for possible future use; smeared copy for inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════════
# Step 3 — Construct Dirac operators
# ═══════════════════════════════════════════════════════════════

# Light and strange: multigrid solver (low-mode dominance makes MG effective)
dirac_l = core.getClover(latt_info, m_l, tol_l, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol_s, maxiter, xi_0, csw, csw, multigrid)

# Charm: standard BiCGStab (multigrid coarse-grid correction is ineffective
# for heavy quarks that lack low-mode dominance)
dirac_c = core.getClover(latt_info, m_c, tol_c, maxiter, xi_0, csw, csw, None)

# ═══════════════════════════════════════════════════════════════
# Step 4 — Compute forward propagators
# ═══════════════════════════════════════════════════════════════

# Point source at [0,0,0,0] for all three quark flavors
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# Propagator-to-quark-line mapping (for Xi_c0 operator):
#   prop_l -> d quark (epsilon index a)
#   prop_s -> s quark (epsilon index b)
#   prop_c -> c quark (epsilon index c)
# Operator: epsilon^{abc} (d^{Ta} Cγ₅ s^b) c^c

# ═══════════════════════════════════════════════════════════════
# Step 5 — Baryon two-point contraction (generate_einsum output)
# ═══════════════════════════════════════════════════════════════

# Gamma matrices and projectors on GPU
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)  # C = γ₂γ₄
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)                     # γ₅
Cg5 = Cmat @ G5                                                           # Cγ₅ diquark
I4 = cp.eye(4, dtype=cp.complex128)
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)       # P⁺=(1+γ₄)/2

# Levi-Civita epsilon_{abc} on GPU
epsilon = cp.zeros((3, 3, 3), dtype=cp.complex128)
epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1.0

# FROM generate_einsum (baryon_2pt)
# Only one Wick topology (sign=1): all three quark flavors (d,s,c) are distinct,
# so there is no exchange diagram.
C_t_local = contract(
    'AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFAfa, wtzyxEBeb -> t',
    Cg5, epsilon, Cg5, epsilon, Tmat,
    prop_c.data, prop_s.data, prop_l.data
)

# MPI gather: sum over spatial volume, collect full time extent to rank 0
C_t = core.gatherLattice(array.arrayAsNumpy(C_t_local, backend="cupy"),
                         [0, -1, -1, -1])

# ═══════════════════════════════════════════════════════════════
# Step 6 — Post-contraction sanity checks and save
# ═══════════════════════════════════════════════════════════════

if core.getMPIRank() == 0:
    C_t_arr = np.asarray(C_t, dtype=np.complex128)
    Lt = latt_size[3]
    half_T = Lt // 2

    # Sanity check 1: C(t=0) must be positive
    if C_t_arr[0].real <= 0.0:
        raise ValueError(
            f"Sanity check FAILED: C(t=0) = {C_t_arr[0].real:.6e} <= 0. "
            f"Possible sign error in contraction or failed inversion."
        )

    # Sanity check 2: |C(t)| should decrease monotonically for t < T/2
    # (allow ~5% tolerance for statistical fluctuations at late t)
    abs_C = np.abs(C_t_arr)
    for t in range(2, half_T):
        if abs_C[t] > abs_C[t - 1] * 1.05:
            raise ValueError(
                f"Sanity check FAILED: |C(t={t})| = {abs_C[t]:.6e} > "
                f"|C(t={t - 1})| = {abs_C[t - 1]:.6e}. "
                f"Non-monotonic correlator — possible propagator misassignment "
                f"or solver failure."
            )

    # Save: plain text, one real correlator value per time slice, no header
    out_path = f"xi_c0_2pt_cfg{n_cfg}.txt"
    np.savetxt(out_path, C_t_arr.real, fmt="%.16e")
    print(f"Saved Xi_c0 two-point correlator to {out_path}")
    print(f"  C(t=0) = {C_t_arr[0].real:.6e}")
    print(f"  C(t={half_T}) = {C_t_arr[half_T].real:.6e}")
