import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# Run with MPI, e.g.:
# mpirun -n 4 python3 main.py ~/.cache 10000

# ------------------------------------------------------------------
# 1. Parameter definitions (hard-coded)
# ------------------------------------------------------------------
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
cfg_path = cfg_path_template.format(n_cfg=n_cfg)

x_src = [0, 0, 0, 0]
tseq = 8

anisotropy = 1.0
xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
m_s = -0.2356
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_path = f"./c3_kminus_to_piminus_vx_cfg{n_cfg}_tseq8.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

# ------------------------------------------------------------------
# 2. Read gauge configuration
# ------------------------------------------------------------------
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ------------------------------------------------------------------
# 3. Construct the Dirac operators
# ------------------------------------------------------------------
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_d = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)

# ------------------------------------------------------------------
# 4. Compute forward propagators
# ------------------------------------------------------------------
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# ------------------------------------------------------------------
# 5. Extract observable / compute contraction
# ------------------------------------------------------------------
# FROM generate_einsum (meson_3pt)
# ═══════════════════════════════════════════════════════
# Meson 3pt:
#   spectator  = u  (prop_l)
#   forward    = s  (prop_s)
#   sequential = d (prop_seq)
#   sink Gamma = G5
#   src Gamma  = G5
#   cur Gamma  = g1
# ═══════════════════════════════════════════════════════

# ------------------------------------------------------------------
#  Gamma matrices
# ------------------------------------------------------------------
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_snk = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_src = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)

# ------------------------------------------------------------------
#  G5-conjugate gammas: Γ̄ = γ5 . Γ . γ5
# ------------------------------------------------------------------
Gamma_snk_bar = G5 @ Gamma_snk @ G5
Gamma_src_bar = G5 @ Gamma_src @ G5

# ------------------------------------------------------------------
#  Sink block for pi- at zero momentum and fixed tseq
# ------------------------------------------------------------------
B = core.LatticePropagator(latt_info)
B.data = contract(
    "AB, wtzyxBCab, CD -> wtzyxADab",
    Gamma_snk_bar, prop_l.data, Gamma_src_bar,
)

# ------------------------------------------------------------------
#  Sequential source from the sink block at t = tseq
# ------------------------------------------------------------------
src_seq = source.sequential12(B, tseq)

with dirac_d.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_d, src_seq)

# ------------------------------------------------------------------
#  Final K- -> pi- three-point contraction with J_x = dbar gamma_x s
# ------------------------------------------------------------------
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij",
    G5, prop_seq.data.conj(), G5,
)

three_pt_local = contract(
    "wtzyxijba, jk, wtzyxkiab -> t",
    tmp_prop.data, Gamma_cur, prop_s.data,
)

C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"), [0, -1, -1, -1]
)

# ------------------------------------------------------------------
# 6. Save the result
# ------------------------------------------------------------------
t_ins = np.arange(1, tseq, dtype=np.int32)

if core.getMPIRank() == 0:
    c3_window = np.asarray(C3_t[t_ins], dtype=np.complex128).reshape(-1)
    tseq_col = np.full(t_ins.shape, tseq, dtype=np.int32)
    out = np.column_stack((tseq_col, t_ins, c3_window.real, c3_window.imag))
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
