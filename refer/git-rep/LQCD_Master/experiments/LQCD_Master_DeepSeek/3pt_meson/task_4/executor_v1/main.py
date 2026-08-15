# B⁻ → K⁻ three-point function via vector current s̄γₓb
# Run: mpirun -np 4 python3 main.py ~/.cache 10000

import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ═══════════════════════════════════════════════════════════
# 1. Parameters
# ═══════════════════════════════════════════════════════════
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/"
    "Configurations/Original/"
    f"beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Source — point at origin, zero momentum
x_src = [0, 0, 0, 0]
t_seq = 8

# Gauge / clover
xi_0 = 1.0
csw = 1.160920226

# Quark masses
m_l = -0.277      # light (u spectator)
m_s = -0.2356     # strange (sequential)
m_b = 1.5         # bottom (heavy, forward)

# Solver — light spectator (multigrid CG)
tol_l = 1.0e-12
maxiter_l = 20000
mg_levels = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Solver — strange sequential (CG, no multigrid)
tol_s = 1.0e-10
maxiter_s = 10000

# Solver — bottom heavy (CG, well-conditioned)
tol_b = 1.0e-10
maxiter_b = 5000

# Stout smearing
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ═══════════════════════════════════════════════════════════
# 2. Initialize PyQUDA
# ═══════════════════════════════════════════════════════════
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ═══════════════════════════════════════════════════════════
# 3. Read gauge configuration
# ═══════════════════════════════════════════════════════════
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Copy before smearing: gauge_stout for Dirac inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════
# 4. Dirac operators (Clover fermions)
# ═══════════════════════════════════════════════════════════
dirac_l = core.getClover(latt_info, m_l, tol_l, maxiter_l,
                         xi_0, csw, csw, mg_levels)

dirac_s = core.getClover(latt_info, m_s, tol_s, maxiter_s,
                         xi_0, csw, csw, None)

dirac_b = core.getClover(latt_info, m_b, tol_b, maxiter_b,
                         xi_0, csw, csw, None)

# ═══════════════════════════════════════════════════════════
# 5. Forward propagators
# ═══════════════════════════════════════════════════════════
pt_src = source.source12(latt_info, "point", x_src)

# Light spectator u quark — multigrid CG
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# Heavy bottom quark — standard CG
with dirac_b.useGauge(gauge_stout):
    prop_b = core.invertPropagator(dirac_b, pt_src)

# ═══════════════════════════════════════════════════════════
# 6. Gamma matrices (DeGrand-Rossi basis, GPU tensors)
# ═══════════════════════════════════════════════════════════
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_snk = cp.asarray(gamma.gamma(15), dtype=cp.complex128)    # γ₅
Gamma_src = cp.asarray(gamma.gamma(15), dtype=cp.complex128)    # γ₅
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)     # γ₁ = γₓ

# G5-conjugate gammas:  Γ̄ = γ₅ · Γ · γ₅
Gamma_snk_bar = G5 @ Gamma_snk @ G5
Gamma_src_bar = G5 @ Gamma_src @ G5

# ═══════════════════════════════════════════════════════════
# 7. Sink block B(x) = Γ̄_snk · S_spectator · Γ̄_src
#    Single topology (connected diagram only).
#    spectator = u (prop_l)
# ═══════════════════════════════════════════════════════════
B = core.LatticePropagator(latt_info)
B.data = contract(
    'AB, wtzyxBCab, CD -> wtzyxADab',
    Gamma_snk_bar, prop_l.data, Gamma_src_bar)

# ═══════════════════════════════════════════════════════════
# 8. First dagger:  η_seq = γ₅ · B† · γ₅
#    Two-dagger convention — REQUIRED before sequential12.
# ═══════════════════════════════════════════════════════════
B.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, B.data.conj(), G5)

# ═══════════════════════════════════════════════════════════
# 9. Sequential source + sequential inversion
#    Sequential source on time slice t_seq = 8
#    Sequential flavour = s (strange)
# ═══════════════════════════════════════════════════════════
src_seq = source.sequential12(B, t_seq)

with dirac_s.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_s, src_seq)

# ═══════════════════════════════════════════════════════════
# 10. Second dagger:  G_seq_dag = γ₅ · S_seq† · γ₅
#     Converts backward propagator to forward convention.
# ═══════════════════════════════════════════════════════════
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_seq.data.conj(), G5)

# ═══════════════════════════════════════════════════════════
# 11. Final contraction:  C₃(τ) = Σ_z Tr[ G_seq_dag · γₓ · S_b ]
#     forward = b (prop_b)
# ═══════════════════════════════════════════════════════════
three_pt_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    tmp_prop.data, Gamma_cur, prop_b.data)

# ═══════════════════════════════════════════════════════════
# 12. MPI gather and save result
# ═══════════════════════════════════════════════════════════
C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"),
    [0, -1, -1, -1])

if core.getMPIRank() == 0:
    # τ = 1,…,7  (drop τ=0 contact term, drop τ=8 sink location)
    C3_window = np.asarray(C3_t[1:8]).real
    out_path = f"BtoK_3pt_cfg{n_cfg}.txt"
    np.savetxt(out_path, C3_window, fmt="%.16e")
    print(f"Saved {out_path}")
