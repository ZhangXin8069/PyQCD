# Run: mpirun -np 4 python3 main.py ~/.cache 10000
import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ============================================================
# 1. Parameter definitions (hard-coded)
# ============================================================

resource_path = os.path.expanduser(sys.argv[1])
n_cfg = int(sys.argv[2])

# Lattice
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

# Gauge configuration path template
cfg_path_template = (
    "/public/share/weiwang/clqcd/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72/"
    "Configurations/Original/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Source: point at origin, t=0
x_src = [0, 0, 0, 0]
t_seq = 8

# Clover action parameters
xi_0 = 1.0
csw = 1.160920226

# Quark masses
m_l = -0.277       # light (u/d)
m_b = 1.5          # bottom

# Light-quark solver: multigrid
tol_l = 1.0e-12
maxiter_l = 5000
multigrid_l = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Bottom-quark solver: CG (no multigrid), tight tolerance
tol_b = 1.0e-10
maxiter_b = 20000

# Stout link smearing (applied to a copy, original gauge untouched)
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# Output directory
out_dir = "."

# ============================================================
# 2. Read gauge configuration
# ============================================================

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Stout-smear a copy for Dirac inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ============================================================
# 3. Construct Dirac operators
# ============================================================

dirac_l = core.getClover(
    latt_info, m_l, tol_l, maxiter_l, xi_0, csw, csw, multigrid_l
)
dirac_b = core.getClover(
    latt_info, m_b, tol_b, maxiter_b, xi_0, csw, csw, None
)

# ============================================================
# 4. Compute forward propagators
# ============================================================

pt_src = source.source12(latt_info, "point", x_src)

# Light quark forward propagator (multigrid)
with dirac_l.useGauge(gauge_stout):
    prop_l_fwd = core.invertPropagator(dirac_l, pt_src)

# Bottom quark forward propagator (CG, convergence-checked)
# PyQUDA raises an exception on solver failure, which naturally
# aborts and flags the configuration as required by the plan.
with dirac_b.useGauge(gauge_stout):
    prop_b_fwd = core.invertPropagator(dirac_b, pt_src)

# ============================================================
# 5. Three-point correlator
#    B- -> pi- via (b->d) vector current J_x = bar{d} gamma_x b
# ============================================================

# FROM generate_einsum (meson_3pt)
# ----------------------------------------------------------
#   spectator  = u  (prop_l_fwd)   -- anti-u in pi- sink
#   forward    = b  (prop_b_fwd)   -- b quark from B- source
#   sequential = d  (prop_l_seq)   -- d quark from current
#   sink Gamma = gamma_5
#   src  Gamma = gamma_5
#   cur  Gamma = gamma_x  (gamma(1) in DeGrand-Rossi basis)
# ----------------------------------------------------------

# Gamma matrices on GPU
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_snk = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_src = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)

# G5-conjugate gammas:  Gamma_bar = gamma_5 . Gamma . gamma_5
# These appear in the sink block construction.
Gamma_snk_bar = G5 @ Gamma_snk @ G5
Gamma_src_bar = G5 @ Gamma_src @ G5

# Step 1 -- Sink block B(x) at t = t_seq
#   B(x) = Gamma_bar_snk . S_spectator(x, t_f; 0, 0) . Gamma_bar_src
#   spectator = u  (prop_l_fwd)
B = core.LatticePropagator(latt_info)
B.data = contract(
    'AB, wtzyxBCab, CD -> wtzyxADab',
    Gamma_snk_bar, prop_l_fwd.data, Gamma_src_bar,
)

# Step 2 -- Sequential source at t_sink and sequential solve
#   Sequential source spans ALL spatial points at t=8
#   (zero-momentum projection of the pi- sink interpolator).
src_seq = source.sequential12(B, t_seq)

# Sequential inversion: light-quark Dirac operator (d quark = light)
with dirac_l.useGauge(gauge_stout):
    prop_l_seq = core.invertPropagator(dirac_l, src_seq)

# Step 3 -- Final contraction
#   C_3(tau) = Tr[ gamma_5 . S_seq^dag . gamma_5  .  Gamma_cur  .  S_fwd ]
#   = - sum_z Tr[ G_l_seq(z, tau) gamma_x S_b(z, tau) ]
#   (overall minus sign from single closed fermion loop,
#    handled by the generate_einsum code generator)

# 3a. Outer G5-dagger:  tmp = gamma_5 . S_seq^dag . gamma_5
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_l_seq.data.conj(), G5,
)

# 3b. Trace contraction over spin, color, spatial, and parity
#     -> yields C3 as a function of insertion time only
three_pt_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    tmp_prop.data, Gamma_cur, prop_b_fwd.data,
)

# 3c. MPI gather: combine time slices across ranks
C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"),
    [0, -1, -1, -1],
)

# ============================================================
# 6. Save the result
# ============================================================

if core.getMPIRank() == 0:
    # Raw correlator: one row per insertion time tau
    # Columns: tau, Re[C3(tau)], Im[C3(tau)]
    C3 = np.asarray(C3_t, dtype=np.complex128).reshape(-1)
    tau = np.arange(C3.shape[0], dtype=np.int32)
    out = np.column_stack((tau, C3.real, C3.imag))
    out_path = os.path.join(
        out_dir,
        f"B_to_pi_3pt_cfg{n_cfg:05d}.txt",
    )
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
