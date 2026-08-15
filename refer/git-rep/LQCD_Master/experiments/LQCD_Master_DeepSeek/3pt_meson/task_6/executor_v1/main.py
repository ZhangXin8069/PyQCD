import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# =============================================================================
#  Parameters — D0 → K*-  three-point function  (c → s  vector current)
# =============================================================================
resource_path = os.path.expanduser(sys.argv[1])
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)
x_src = [0, 0, 0, 0]
t_seq = 8

xi_0 = 1.0
csw  = 1.160920226
m_l  = -0.277
m_s  = -0.2356
m_c  = 0.4159
tol      = 1.0e-12
maxiter  = 20000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho   = 0.125
stout_ndim  = 4

out_path = f"D0_to_Kstar_3pt_result_cfg{n_cfg:05d}.txt"

# =============================================================================
#  1.  Initialise PyQUDA
# =============================================================================
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# =============================================================================
#  2.  Read gauge configuration
# =============================================================================
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Stout-smeared copy for Dirac inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# =============================================================================
#  3.  Dirac operators  (Clover, isotropic)
# =============================================================================
dirac_l = core.getDirac(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_c = core.getDirac(latt_info, m_c, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getDirac(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# =============================================================================
#  4.  Forward propagators  (point source at origin)
# =============================================================================
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# =============================================================================
#  5.  Three-point correlator  (sequential-source method)
# =============================================================================
# FROM generate_einsum (meson_3pt)
#   spectator  = u  (prop_l)
#   forward    = c  (prop_c)
#   sequential = s  (prop_seq)
#   sink Gamma = g1  (γ_x)
#   src  Gamma = G5  (γ₅)
#   cur  Gamma = g1  (γ_x)

# ------------------------------------------------------------------
#  Gamma matrices  (DeGrand-Rossi basis, on GPU)
# ------------------------------------------------------------------
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)

Gamma_snk = cp.asarray(gamma.gamma(1), dtype=cp.complex128)   # γ_x  at sink
Gamma_src = cp.asarray(gamma.gamma(15), dtype=cp.complex128)  # γ₅   at source
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)   # γ_x  current

# G5-conjugate gammas:  Γ̄ = γ₅ · Γ · γ₅  (appear in the sink block)
Gamma_snk_bar = G5 @ Gamma_snk @ G5
Gamma_src_bar = G5 @ Gamma_src @ G5

# ------------------------------------------------------------------
#  Step 1 — Sink block  B(x) = Γ̄_snk · S_l · Γ̄_src
# ------------------------------------------------------------------
B = core.LatticePropagator(latt_info)
B.data = contract(
    'AB, wtzyxBCab, CD -> wtzyxADab',
    Gamma_snk_bar, prop_l.data, Gamma_src_bar)

# ------------------------------------------------------------------
#  Step 2 — Sequential source and solve
#           sequential12 applies γ₅·B†·γ₅ internally and
#           restricts to time slice t_seq.
# ------------------------------------------------------------------
src_seq = source.sequential12(B, t_seq)

with dirac_s.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_s, src_seq)

# ------------------------------------------------------------------
#  Step 3 — Second dagger:  G_seq_dag = γ₅ · G_seq† · γ₅
# ------------------------------------------------------------------
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_seq.data.conj(), G5)

# ------------------------------------------------------------------
#  Step 4 — Final contraction:  C₃(τ) = Σ_x Tr[ G_seq_dag · Γ_cur · S_c ]
# ------------------------------------------------------------------
C3_t_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    tmp_prop.data, Gamma_cur, prop_c.data)

# MPI gather across time-partitioned ranks
C3_t = core.gatherLattice(
    array.arrayAsNumpy(C3_t_local, backend="cupy"), [0, -1, -1, -1])

# =============================================================================
#  6.  Save result  (rank 0 only)
# =============================================================================
if core.getMPIRank() == 0:
    t_list = np.arange(t_seq + 1, dtype=np.int32)
    tseq_col = np.full(t_list.shape, t_seq, dtype=np.int32)
    C3_window = np.asarray(C3_t, dtype=np.complex128).reshape(-1)
    out = np.column_stack((tseq_col, t_list, C3_window.real, C3_window.imag))
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
