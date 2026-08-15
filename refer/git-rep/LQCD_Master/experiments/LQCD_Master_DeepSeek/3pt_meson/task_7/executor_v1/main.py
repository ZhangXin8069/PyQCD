# Run: mpirun -n 4 python3 main.py <resource_path> <n_cfg>
import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ═══════════════════════════════════════════════════════════════
#  Parameters (hard-coded)
# ═══════════════════════════════════════════════════════════════
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72/"
    "Configurations/Original/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Source / sink geometry
t_src = 0
x_src = [0, 0, 0, 0]
t_sink = 8

# Wilson-clover parameters
xi_0 = 1.0
csw = 1.160920226

# Quark masses (kappa-convention negative for light/strange)
m_l = -0.277
m_s = -0.2356
m_b = 1.5

# Solver parameters
tol = 1.0e-12
maxiter_l = 10000
maxiter_s = 10000
maxiter_b = 50000   # raised for heavy bottom; monitor residual
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout link smearing
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ═══════════════════════════════════════════════════════════════
#  Initialise PyQUDA
# ═══════════════════════════════════════════════════════════════
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ═══════════════════════════════════════════════════════════════
#  Read gauge configuration
# ═══════════════════════════════════════════════════════════════
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Keep a raw copy before smearing (not needed here, but good practice)
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════════
#  Dirac operators
# ═══════════════════════════════════════════════════════════════
dirac_l = core.getDirac(latt_info, m_l, tol, maxiter_l, xi_0, csw, csw, multigrid)
dirac_s = core.getDirac(latt_info, m_s, tol, maxiter_s, xi_0, csw, csw, multigrid)
dirac_b = core.getDirac(latt_info, m_b, tol, maxiter_b, xi_0, csw, csw, multigrid)

# ═══════════════════════════════════════════════════════════════
#  Point source at t=0
# ═══════════════════════════════════════════════════════════════
pt_src = source.source12(latt_info, "point", x_src)

# ═══════════════════════════════════════════════════════════════
#  Forward propagators (on stout-smeared gauge)
# ═══════════════════════════════════════════════════════════════
# Light spectator (anti-u): source → all (x,t)
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# Bottom active quark (b): source → all (z,tau)
with dirac_b.useGauge(gauge_stout):
    prop_b = core.invertPropagator(dirac_b, pt_src)

# ═══════════════════════════════════════════════════════════════
#  Three-point correlator: B⁻ → K*⁻  (b → s vector current)
# ═══════════════════════════════════════════════════════════════
# FROM generate_einsum (meson_3pt)
#   spectator  = u  (prop_l)       — anti-u line, source→sink
#   forward    = b  (prop_b)       — bottom line, source→current
#   sequential = s  (prop_s_seq)   — strange line, sink→current
#   sink Gamma = g1 (gamma_x)      — K* vector interpolator
#   src  Gamma = G5 (gamma_5)      — B pseudoscalar interpolator
#   cur  Gamma = g1 (gamma_x)      — vector current  s̄ γ_x b

# --- Gamma matrices (GPU, DeGrand-Rossi basis) ---
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_snk = cp.asarray(gamma.gamma(1), dtype=cp.complex128)
Gamma_src = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)

# G5-conjugate gammas for the sink block:  Γ̄ = γ₅ · Γ · γ₅
Gamma_snk_bar = G5 @ Gamma_snk @ G5
Gamma_src_bar = G5 @ Gamma_src @ G5

# --- Step 1: Sink block B(x) = Γ̄_snk · S_l(x,0) · Γ̄_src ---
# This contracts the light spectator propagator with the K* sink
# operator at every spatial point x on time slice t_sink.
B = core.LatticePropagator(latt_info)
B.data = contract(
    "AB, wtzyxBCab, CD -> wtzyxADab",
    Gamma_snk_bar, prop_l.data, Gamma_src_bar,
)

# --- Step 2: Sequential source + solve ---
# sequential12 places the source at t_sink using the standard
# two-dagger convention internally.
src_seq = source.sequential12(B, t_sink)

# Sequential inversion: strange quark from effective sink → all (z,tau)
with dirac_s.useGauge(gauge_stout):
    prop_s_seq = core.invertPropagator(dirac_s, src_seq)

# --- Step 3: Final contraction ---
# C_3(τ) = Σ_z Tr[ γ₅ · S_seq†(z,τ) · γ₅ · Γ_cur · S_b(z,τ) ]

# 3a. Outer G5-dagger:  tmp = γ₅ · S_seq† · γ₅
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij",
    G5, prop_s_seq.data.conj(), G5,
)

# 3b. Trace over spin+colour, sum over spatial sites → (Lt_local,)
three_pt_local = contract(
    "wtzyxijba, jk, wtzyxkiab -> t",
    tmp_prop.data, Gamma_cur, prop_b.data,
)

# 3c. MPI gather the time dimension
C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"), [0, -1, -1, -1]
)

# ═══════════════════════════════════════════════════════════════
#  Save result
# ═══════════════════════════════════════════════════════════════
if core.getMPIRank() == 0:
    # Exclude contact terms: current insertion times tau = 1 … t_sink-1
    C3_window = np.asarray(C3_t[1:t_sink], dtype=np.complex128)
    out_path = f"C3_B_to_Kstar_cfg{n_cfg:05d}.txt"
    out = np.column_stack((C3_window.real, C3_window.imag))
    np.savetxt(out_path, out, fmt=["%.16e", "%.16e"])
