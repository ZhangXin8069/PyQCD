# K- -> pi- 3pt correlator via (s->d) vector current
# Run: mpirun -np 4 python main.py ~/.cache 10000

import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ============================================================================
# Parameter definitions
# ============================================================================

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/"
    "Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

x_src = [0, 0, 0, 0]
t_seq = 8

xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
m_s = -0.2356
tol_l = 1.0e-12
tol_s = 1.0e-10
maxiter = 20000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_filename = "K_to_pi_3pt_result.txt"

# ============================================================================
# Initialize PyQUDA
# ============================================================================

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ============================================================================
# Read gauge configuration
# ============================================================================

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ============================================================================
# Construct Dirac operators
# ============================================================================

dirac_l = core.getClover(latt_info, m_l, tol_l, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol_s, maxiter, xi_0, csw, csw, multigrid)

# ============================================================================
# Compute forward propagators
# ============================================================================

pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# ============================================================================
# Meson 3pt contraction — FROM generate_einsum (meson_3pt)
# ============================================================================
#   spectator  = u  (prop_l)
#   forward    = s  (prop_s)
#   sequential = d  (prop_seq)
#   sink Gamma = G5
#   src Gamma  = G5
#   cur Gamma  = g1 (gamma_x)
# ============================================================================

# Gamma matrices
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_snk = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_src = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)

# G5-conjugate gammas:  Gamma_bar = gamma_5 . Gamma . gamma_5
Gamma_snk_bar = G5 @ Gamma_snk @ G5
Gamma_src_bar = G5 @ Gamma_src @ G5

# Step 1 — Sink block
#   B(x) = Gamma_snk_bar . S_spectator(x;0) . Gamma_src_bar
#   spectator = u (prop_l)
B = core.LatticePropagator(latt_info)
B.data = contract(
    'AB, wtzyxBCab, CD -> wtzyxADab',
    Gamma_snk_bar, prop_l.data, Gamma_src_bar)

# Step 2 — Two-dagger convention: first dagger
#   B <- gamma_5 . B^dag . gamma_5
B.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, B.data.conj(), G5)

# Step 3 — Sequential source + solve
#   Sequential source at t_sink (second dagger handled by sequential12)
src_seq = source.sequential12(B, t_seq)

# Sequential inversion: light-quark Dirac operator (same mass as prop_l)
with dirac_l.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_l, src_seq)

# Step 4 — Final contraction
#   C_3pt(tau) = sum_z Tr[ gamma_5 . S_seq^dag . gamma_5  . Gamma_cur . S_s ]

# 4a. Outer G5-dagger: tmp = gamma_5 . S_seq^dag . gamma_5
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_seq.data.conj(), G5)

# 4b. Per-site trace + sum over spatial sites -> time only
C3_t_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    tmp_prop.data, Gamma_cur, prop_s.data)

# 4c. MPI gather across time-sliced ranks
C3_t = core.gatherLattice(
    array.arrayAsNumpy(C3_t_local, backend="cupy"), [0, -1, -1, -1])

# ============================================================================
# Save the result
# ============================================================================

if core.getMPIRank() == 0:
    # tau = 1..7 (skip contact terms at tau=0 and tau=t_seq=8)
    tau_list = list(range(1, t_seq))
    C3_window = np.asarray(C3_t[tau_list], dtype=np.complex128).reshape(-1)
    np.savetxt(out_filename, C3_window.real, fmt="%.16e")
