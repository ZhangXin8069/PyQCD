# D⁰ → η_u three-point correlator via (c→u) vector current
# Sequential-source method, point source at [0,0,0,0], tseq=8
# Run: mpirun -np 4 python main.py <resource_path> <cfg_number>

import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ═══════════════════════════════════════════════════════════
# 1. Parameter definitions (hard-coded)
# ═══════════════════════════════════════════════════════════

# MPI & lattice
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

# Configuration path (format with n_cfg)
cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Source
x_src = [0, 0, 0, 0]       # point source position
tseq = 8                    # sink time slice

# Clover action
xi_0 = 1.0                  # gauge anisotropy
csw = 1.160920226           # clover coefficient (isotropic: csw_t = csw_r = csw)

# Quark masses (kappa convention)
m_l = -0.277                # light
m_c = 0.4159                # charm

# Solver parameters
tol  = 1.0e-12
maxiter = 2000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]
# NOTE: The same MG parameters are used for charm. If convergence fails
# for the heavy quark, relax tol to 1e-10 or switch to CG.

# Stout link smearing (applied to gauge before all inversions)
stout_nstep = 1
stout_rho   = 0.125
stout_ndim  = 4

# Output
out_dir = "."

# ── Runtime args ──
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

# ═══════════════════════════════════════════════════════════
# 2. Read gauge configuration
# ═══════════════════════════════════════════════════════════

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Copy BEFORE smearing — raw gauge kept unused but available
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════
# 3. Construct Dirac operators
# ═══════════════════════════════════════════════════════════

dirac_l = core.getDirac(latt_info, m_l, tol, maxiter,
                        xi_0, csw, csw, multigrid)
dirac_c = core.getDirac(latt_info, m_c, tol, maxiter,
                        xi_0, csw, csw, multigrid)

# ═══════════════════════════════════════════════════════════
# 4. Compute forward propagators
# ═══════════════════════════════════════════════════════════

pt_src = source.source12(latt_info, "point", x_src)

# Forward light propagator (spectator: ū → ū)
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# Forward charm propagator (c → current insertion point)
with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# ═══════════════════════════════════════════════════════════
# 5. Sequential source + 3pt contraction
#    FROM generate_einsum (meson_3pt)
#    spectator  = u  (prop_l)
#    forward    = c  (prop_c)
#    sequential = u  (prop_seq)
#    sink Gamma = G5,  src Gamma = G5,  cur Gamma = g1 (γ_x)
# ═══════════════════════════════════════════════════════════

# ── Gamma matrices on GPU ──
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)        # γ₅
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)  # γ₁ = γ_x

# G5-conjugate gammas for the sink block:  Γ̄ = γ₅ · Γ · γ₅
# For Γ=γ₅: Γ̄ = γ₅·γ₅·γ₅ = γ₅  (since γ₅² = I)
Gamma_snk_bar = G5 @ G5 @ G5   # = G5
Gamma_src_bar = G5 @ G5 @ G5   # = G5

# ── Step 5a: Sink block  B(x) = Γ̄_snk · S_spectator · Γ̄_src ──
B = core.LatticePropagator(latt_info)
B.data = contract(
    'AB, wtzyxBCab, CD -> wtzyxADab',
    Gamma_snk_bar, prop_l.data, Gamma_src_bar)

# ── Step 5b: Sequential source at t_sink = tseq ──
# sequential12 builds the full-volume source at t=tseq from B(x⃗,tseq)
src_seq = source.sequential12(B, tseq)

# Sequential inversion (light quark)
with dirac_l.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_l, src_seq)

# ── Step 5c: Second dagger — build G_seq_dag = γ₅ · G_seq† · γ₅ ──
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_seq.data.conj(), G5)

# ── Step 5d: Final trace  Tr[ G_seq_dag · Γ_cur · S_c ] ──
# Contracts all spin/color/space indices → local time array
three_pt_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    tmp_prop.data, Gamma_cur, prop_c.data)

# ── Step 5e: MPI gather (reduce over spatial ranks) ──
C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"),
    [0, -1, -1, -1])

# ═══════════════════════════════════════════════════════════
# 6. Save result
# ═══════════════════════════════════════════════════════════

if core.getMPIRank() == 0:
    # Save τ = 0 … tseq (inclusive)
    tau_range = list(range(tseq + 1))
    C3_window = np.asarray(C3_t[tau_range], dtype=np.complex128).reshape(-1)
    tau_col = np.asarray(tau_range, dtype=np.int32)
    out = np.column_stack((tau_col, C3_window.real, C3_window.imag))
    out_path = os.path.join(out_dir, f"d0_to_eta_u_3pt_cfg{n_cfg}.txt")
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
