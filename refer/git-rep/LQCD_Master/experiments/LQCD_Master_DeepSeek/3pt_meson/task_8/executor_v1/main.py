# Run: mpirun -np 4 python main.py <resource_path> <cfg_number>
import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract

from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ==============================================================================
#  Parameters (hard-coded, physics-visible)
# ==============================================================================
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/"
    "Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
).format(n_cfg=n_cfg)

# Source: point source at origin, zero momentum
x_src = [0, 0, 0, 0]
t_sep = 8

# Clover parameters
xi_0 = 1.0
csw = 1.160920226

# Quark masses (kappa convention)
m_s = -0.2356   # strange
m_c = 0.4159    # charm

# Solver parameters
tol = 1.0e-10
maxiter = 20000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout smearing for gauge links
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ==============================================================================
#  1. Initialize PyQUDA
# ==============================================================================
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ==============================================================================
#  2. Load gauge configuration and apply stout smearing
# ==============================================================================
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ==============================================================================
#  3. Dirac operators
#     Strange: multigrid-preconditioned CG (both forward and sequential)
#     Charm:   standard CG, no multigrid (heavy quark)
# ==============================================================================
dirac_s = core.getDirac(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_c = core.getDirac(latt_info, m_c, tol, maxiter, xi_0, csw, csw, None)

# ==============================================================================
#  4. Forward propagators from point source
#     prop_s: spectator anti-s quark line (source -> sink)
#     prop_c: active charm quark line (source -> current insertion)
# ==============================================================================
pt_src = source.source12(latt_info, "point", x_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# ==============================================================================
#  5. Three-point correlator: Ds+ -> phi via c->s vector current
#     Sequential source method.  Sink block already embeds G5-conjugated
#     gammas (from gamma5-hermiticity of source/sink operators), so no
#     extra G5-dagger on B is needed before constructing the sequential
#     source.
#
#     FROM generate_einsum (meson_3pt):
#       spectator  = s  (prop_s / strange forward)
#       forward    = c  (prop_c / charm forward)
#       sequential = s  (prop_seq / strange sequential)
#       sink Gamma = gamma_x (phi interpolator)
#       src Gamma  = gamma5  (Ds+ interpolator, daggered)
#       cur Gamma  = gamma_x (vector current)
# ==============================================================================

# -- Gamma matrices (GPU tensors, DeGrand-Rossi basis) --
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)          # gamma5
Gamma_snk = cp.asarray(gamma.gamma(1), dtype=cp.complex128)    # gamma_x
Gamma_src = cp.asarray(gamma.gamma(15), dtype=cp.complex128)   # gamma5
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)    # gamma_x

# G5-conjugate:  Gamma_bar = gamma5 @ Gamma @ gamma5
#   gamma_x:  gamma5 @ gamma_x @ gamma5 = -gamma_x  (anticommutation)
#   gamma5:   gamma5 @ gamma5 @ gamma5 = gamma5     (self)
# These appear in the sink block and encode the gamma5-hermiticity
# of the source and sink operators.
Gamma_snk_bar = G5 @ Gamma_snk @ G5
Gamma_src_bar = G5 @ Gamma_src @ G5

# -- Step 5a: Sink block B(x) = Gamma_snk_bar · S_s · Gamma_src_bar --
# B encodes the phi sink operator (gamma_x, G5-conjugated) and the
# daggered Ds source operator (gamma5, G5-conjugated), sandwiching
# the spectator anti-s quark propagator.
B = core.LatticePropagator(latt_info)
B.data = contract(
    'AB, wtzyxBCab, CD -> wtzyxADab',
    Gamma_snk_bar, prop_s.data, Gamma_src_bar)

# -- Step 5b: Sequential source at t_sink = 8 --
# The G5-dagger is already embedded in Gamma_snk_bar / Gamma_src_bar,
# so sequential12 is called directly on B.
src_seq = source.sequential12(B, t_sep)

# -- Step 5c: Sequential inversion (strange, multigrid) --
with dirac_s.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_s, src_seq)

# -- Step 5d: Outer G5-dagger on sequential propagator --
# G(0,x) = gamma5 · S_seq_dagger · gamma5
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_seq.data.conj(), G5)

# -- Step 5e: Final contraction C3(tau) = Tr[ G(0,x) · Gamma_cur · S_c ] --
# Contract spin/color/spatial -> only time dimension remains
three_pt_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    tmp_prop.data, Gamma_cur, prop_c.data)

# -- Step 5f: MPI gather (spatial sum, full temporal extent) --
C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"), [0, -1, -1, -1])

# ==============================================================================
#  6. Save result (rank 0 only) — three columns: tau, Re(C3), Im(C3)
# ==============================================================================
if core.getMPIRank() == 0:
    # Window: tau = 0 .. t_sep (inclusive)
    C3_window = np.asarray(C3_t[:t_sep + 1], dtype=np.complex128).reshape(-1)
    tau = np.arange(t_sep + 1, dtype=np.int32)
    out = np.column_stack((tau, C3_window.real, C3_window.imag))
    out_path = "ds_phi_3pt_cfg{n_cfg}.txt".format(n_cfg=n_cfg)
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
