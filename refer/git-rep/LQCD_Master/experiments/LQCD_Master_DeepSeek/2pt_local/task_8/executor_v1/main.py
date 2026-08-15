# Run: mpirun -n 4 python3 main.py ~/.cache 10000
import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract

from pyquda_utils import core, io, gamma, source
from pyquda_comm import array

# ═══════════════════════════════════════════════════════════════
# 1. Parameter definitions (hard-coded)
# ═══════════════════════════════════════════════════════════════

# ── Runtime ──
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

# ── Lattice ──
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

# ── Gauge configuration ──
cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# ── Stout smearing ──
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ── Quark & action parameters ──
m_c = 0.4159                 # charm quark mass
xi_0 = 1.0                   # gauge anisotropy
csw = 1.160920226            # clover coefficient
tol = 1.0e-12                # solver tolerance
maxiter_mg = 20000           # max iterations (multigrid)
maxiter_cg = 40000           # max iterations (CG fallback)
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# ── Source ──
x_src = [0, 0, 0, 0]         # point source position

# ── Output ──
out_path = "./jpsi_2pt_result.txt"

# ═══════════════════════════════════════════════════════════════
# 2. Read gauge configuration
# ═══════════════════════════════════════════════════════════════

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Copy raw gauge BEFORE smearing, then stout-smear the working copy
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════════
# 3. Construct the Dirac operator
# ═══════════════════════════════════════════════════════════════

# Multigrid solver (primary)
dirac_c_mg = core.getClover(
    latt_info, m_c, tol, maxiter_mg, xi_0, csw, csw, multigrid
)
# CG/BiCGStab fallback (multigrid=None triggers BiCGStab in QUDA)
dirac_c_cg = core.getClover(
    latt_info, m_c, tol, maxiter_cg, xi_0, csw, csw, None
)

# ═══════════════════════════════════════════════════════════════
# 4. Compute forward propagator
# ═══════════════════════════════════════════════════════════════

pt_src = source.source12(latt_info, "point", x_src)

# Try multigrid first; fall back to CG/BiCGStab if it fails
try:
    with dirac_c_mg.useGauge(gauge_stout):
        prop_c = core.invertPropagator(dirac_c_mg, pt_src)
except Exception:
    if core.getMPIRank() == 0:
        print("[WARNING] Multigrid solver failed, falling back to CG/BiCGStab")
    with dirac_c_cg.useGauge(gauge_stout):
        prop_c = core.invertPropagator(dirac_c_cg, pt_src)

# ═══════════════════════════════════════════════════════════════
# 5. Extract observable / compute contraction
# ═══════════════════════════════════════════════════════════════

# Gamma matrices on GPU (DeGrand-Rossi basis)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)  # γ₅
g1 = cp.asarray(gamma.gamma(1),  dtype=cp.complex128)  # γ₁
g2 = cp.asarray(gamma.gamma(2),  dtype=cp.complex128)  # γ₂
g3 = cp.asarray(gamma.gamma(4),  dtype=cp.complex128)  # γ₃

# Contract for each polarization (connected diagram only;
# the disconnected diagram is OZI-suppressed for charmonium).
# Einsum strings from generate_einsum(type="meson_2pt", ...):
#   Tr[S† (γ₅γ_i) S (γ_i γ₅)]  →  summed to per-time-slice scalar.
# NOTE: the einsum "-> t" reduces parity(w), spatial(z,y,x), spin(C,D,A,B),
# and color(b,a), leaving only the LOCAL time dimension (Lt/4 = 18 per rank).
# Allocation must match this local extent, NOT the global latt_info.Lt.

C_t_local = cp.zeros(prop_c.data.shape[1], dtype=cp.complex128)

# Polarization γ₁  ── FROM generate_einsum (meson_2pt, gamma=gamma1)
C_t_local += contract(
    "wtzyxCBba, CD, wtzyxDAba, AB -> t",
    prop_c.data.conj(), G5 @ g1, prop_c.data, g1 @ G5,
)

# Polarization γ₂  ── FROM generate_einsum (meson_2pt, gamma=gamma2)
C_t_local += contract(
    "wtzyxCBba, CD, wtzyxDAba, AB -> t",
    prop_c.data.conj(), G5 @ g2, prop_c.data, g2 @ G5,
)

# Polarization γ₃  ── FROM generate_einsum (meson_2pt, gamma=gamma3)
C_t_local += contract(
    "wtzyxCBba, CD, wtzyxDAba, AB -> t",
    prop_c.data.conj(), G5 @ g3, prop_c.data, g3 @ G5,
)

# Average over three polarizations
C_t_local /= 3.0

# MPI gather: reduce spatial dims, gather time dim to rank 0
C_t = core.gatherLattice(
    array.arrayAsNumpy(C_t_local, backend="cupy"), [0, -1, -1, -1]
)

# ═══════════════════════════════════════════════════════════════
# 6. Save the result
# ═══════════════════════════════════════════════════════════════

if core.getMPIRank() == 0:
    C_t_root = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    t_arr = np.arange(C_t_root.shape[0], dtype=np.int32)
    out = np.column_stack((t_arr, C_t_root.real, C_t_root.imag))
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
    print(f"[DONE] Saved {C_t_root.shape[0]} time slices to {out_path}")
