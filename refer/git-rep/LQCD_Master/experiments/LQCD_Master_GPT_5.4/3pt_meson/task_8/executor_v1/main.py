import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# Run with MPI, for example:
# mpirun -n 4 python3 main.py ~/.cache 10000

# ------------------------------------------------------------------
# 1. Parameter definitions
# ------------------------------------------------------------------
resource_path = sys.argv[1]
cfg_label = sys.argv[2]
n_cfg = int(cfg_label)

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
cfg_path = cfg_path_template.format(n_cfg=cfg_label)

x_src = [0, 0, 0, 0]
tseq = 8

anisotropy = 1.0
xi_0 = 1.0
csw = 1.160920226
m_s = -0.2356
m_c = 0.4159
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_path = os.path.join(os.getcwd(), f"ds_to_phi_3pt_cfg{cfg_label}_tseq8.txt")

# ------------------------------------------------------------------
# 2. Read gauge configuration
# ------------------------------------------------------------------
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ------------------------------------------------------------------
# 3. Construct the Dirac operators
# ------------------------------------------------------------------
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_c = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, multigrid)

# ------------------------------------------------------------------
# 4. Compute forward propagators
# ------------------------------------------------------------------
pt_src = source.source12(latt_info, "point", x_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# ------------------------------------------------------------------
# 5. Compute the connected Ds -> phi meson 3pt correlator
#    Source:  sbar * gamma5 * c
#    Sink:    sbar * gamma_x * s
#    Current: sbar * gamma_x * c
#    Spectator line: strange
# ------------------------------------------------------------------
# FROM generate_einsum (meson_3pt)
# ═══════════════════════════════════════════════════════
# Meson 3pt:
#   spectator  = s  (prop_s)
#   forward    = c  (prop_c)
#   sequential = s (prop_seq)
#   sink Gamma = g1
#   src Gamma  = G5
#   cur Gamma  = g1
# ═══════════════════════════════════════════════════════

# ------------------------------------------------------------------
#  Gamma matrices
# ------------------------------------------------------------------
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_snk = cp.asarray(gamma.gamma(1), dtype=cp.complex128)
Gamma_src = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)

# ------------------------------------------------------------------
#  G5-conjugate gammas:  Γ̄ = γ5 . Γ . γ5
#  These appear in the sink block.
# ------------------------------------------------------------------
Gamma_snk_bar = G5 @ Gamma_snk @ G5
Gamma_src_bar = G5 @ Gamma_src @ G5

# ------------------------------------------------------------------
#  Step 1 -- Sink block
#  B(x) = Γ̄_snk . S_spectator . Γ̄_src
# ------------------------------------------------------------------
B = core.LatticePropagator(latt_info)
B.data = contract(
    "AB, wtzyxBCab, CD -> wtzyxADab",
    Gamma_snk_bar, prop_s.data, Gamma_src_bar
)

# ------------------------------------------------------------------
#  Step 2 -- Sequential source + solve
# ------------------------------------------------------------------
src_seq = source.sequential12(B, tseq)

with dirac_s.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_s, src_seq)

# ------------------------------------------------------------------
#  Step 3 -- Final contraction
#  C3(t) = Tr[ gamma5 * G_seq^dag * gamma5 * Gamma_cur * S_c ]
# ------------------------------------------------------------------
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij",
    G5, prop_seq.data.conj(), G5
)

three_pt_local = contract(
    "wtzyxijba, jk, wtzyxkiab -> t",
    tmp_prop.data, Gamma_cur, prop_c.data
)

C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"),
    [0, -1, -1, -1]
)

# ------------------------------------------------------------------
# 6. Save the result
#    Save only tau = 1, ..., tseq-1 with columns:
#    tseq, t_insert, Re, Im
# ------------------------------------------------------------------
if core.getMPIRank() == 0:
    t_insert = np.arange(1, tseq, dtype=np.int32)
    c3_window = np.asarray(C3_t[1:tseq], dtype=np.complex128).reshape(-1)
    tseq_col = np.full(t_insert.shape, tseq, dtype=np.int32)
    out = np.column_stack((tseq_col, t_insert, c3_window.real, c3_window.imag))
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
