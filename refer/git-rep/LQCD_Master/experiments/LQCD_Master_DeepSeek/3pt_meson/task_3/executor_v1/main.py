# ============================================================
# B- → D0 three-point function via b→c vector current (γ_x)
# Sequential source method — meson 3pt
#
# Source: B- meson (anti-u b), ūγ₅b interpolator at t=0
# Sink:   D0 meson (anti-u c), ūγ₅c interpolator at t=tseq=8
# Current: c̄ γ_x b inserted at τ = 1..7
#
# Physics note: γ_x at zero recoil (q=0) vanishes identically
# for a 0⁻→0⁻ pseudoscalar transition.  C3(τ) is expected to
# be consistent with zero modulo statistical noise.
#
# Run: mpirun -np 4 python3 main.py ~/.cache 10000
# ============================================================

import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ── Parameters (hard-coded, ensemble C24P29) ───────────────
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

# Lattice
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]          # 4 MPI ranks along t

# Gauge configuration
cfg_path_template = (
    "/public/share/weiwang/clqcd/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72/"
    "Configurations/Original/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Source
x_src = [0, 0, 0, 0]              # point source at origin
t_seq = 8                         # sink time slice

# Clover action
xi_0 = 1.0
csw  = 1.160920226

# Quark masses (lattice units)
m_l = -0.277                      # light (spectator u / d)
m_b =  1.5                        # bottom (forward, active)
m_c =  0.4159                     # charm  (sequential)

# Solver
tol     = 1.0e-12
maxiter = 20000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout link smearing
stout_nstep = 1
stout_rho   = 0.125
stout_ndim  = 4

# Output
out_path = f"B_to_D_3pt_gx_cfg{n_cfg:05d}.txt"

# ═══════════════════════════════════════════════════════════
#  0. Initialize PyQUDA
# ═══════════════════════════════════════════════════════════
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ═══════════════════════════════════════════════════════════
#  1. Read gauge configuration
# ═══════════════════════════════════════════════════════════
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Stout smear a copy for Dirac inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════
#  2. Construct Dirac operators
# ═══════════════════════════════════════════════════════════
# Light  — multigrid accelerated (spectator)
dirac_l = core.getDirac(latt_info, m_l, tol, maxiter,
                        xi_0, csw, csw, multigrid)

# Bottom — standard CG (forward, active quark)
dirac_b = core.getDirac(latt_info, m_b, tol, maxiter,
                        xi_0, csw, csw, None)

# Charm  — standard CG (sequential)
dirac_c = core.getDirac(latt_info, m_c, tol, maxiter,
                        xi_0, csw, csw, None)

# ═══════════════════════════════════════════════════════════
#  3. Compute forward propagators (point source at origin)
# ═══════════════════════════════════════════════════════════
pt_src = source.source12(latt_info, "point", x_src)

# Light spectator propagator
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# Bottom forward propagator (b-quark source → current insertion)
with dirac_b.useGauge(gauge_stout):
    prop_b = core.invertPropagator(dirac_b, pt_src)

# ═══════════════════════════════════════════════════════════
#  4. Meson 3pt sequential source
#     FROM generate_einsum (meson_3pt)
# ═══════════════════════════════════════════════════════════
# spectator  = u  (prop_l)
# forward    = b  (prop_b)
# sequential = c  (prop_seq)
# sink Gamma = G5
# src  Gamma = G5
# cur  Gamma = g1

# ── Gamma matrices (GPU, DeGrand-Rossi basis) ─────────────
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)       # γ₅
Gamma_snk = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_src = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1),  dtype=cp.complex128) # γ₁ = γ_x

# G5-conjugate gammas:  Γ̄ = γ₅ · Γ · γ₅  (appear in sink block)
Gamma_snk_bar = G5 @ Gamma_snk @ G5
Gamma_src_bar = G5 @ Gamma_src @ G5

# ── Step 1: Sink block  B(x) = Γ̄_snk · S_spectator · Γ̄_src ──
# No Wick summation needed: meson 3pt has a single topology.
B = core.LatticePropagator(latt_info)
B.data = contract(
    'AB, wtzyxBCab, CD -> wtzyxADab',
    Gamma_snk_bar, prop_l.data, Gamma_src_bar)

# ── Step 2: First dagger  η_seq = γ₅ · B† · γ₅ ────────────
# Required by the two-dagger sequential-source convention.
# Converts the sink-block spin structure into the correct
# orientation for the sequential Dirac solve.
B.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, B.data.conj(), G5)

# ── Step 3: Sequential source and solve ───────────────────
src_seq = source.sequential12(B, t_seq)

with dirac_c.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_c, src_seq)

# ── Step 4: Second dagger  G(0,x) = γ₅ · S_seq† · γ₅ ──────
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_seq.data.conj(), G5)

# ── Step 5: Final contraction  Tr[ G(0,x) · Γ_cur · S_fwd ] ──
three_pt_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    tmp_prop.data, Gamma_cur, prop_b.data)

# ── Step 6: MPI gather in time dimension ──────────────────
C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"),
    [0, -1, -1, -1])

# ═══════════════════════════════════════════════════════════
#  5. Save result (rank 0 only)
# ═══════════════════════════════════════════════════════════
if core.getMPIRank() == 0:
    C3_flat = np.asarray(C3_t, dtype=np.complex128).reshape(-1)
    C3_real = C3_flat.real
    vals = C3_real[1:8]   # τ = 1, 2, ..., 7

    with open(out_path, 'w') as f:
        f.write(' '.join([f"{x:.16e}" for x in vals]))

    print(f"Saved: {out_path}")
