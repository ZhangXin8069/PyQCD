# Ds+ EM 3pt — Connected contribution, c→c vector current (gamma_x)
# Launch: mpirun -n 4 python3 main.py <resource_path> <n_cfg>
import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ═══════════════════════════════════════════════════════════════
#  Parameters
# ═══════════════════════════════════════════════════════════════
resource_path = sys.argv[1]
n_cfg         = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [ 1,  1,  1,  4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Source position
x_src = [0, 0, 0, 0]
t_seq = 8

# Clover fermion parameters
xi_0   = 1.0
csw    = 1.160920226
m_s    = -0.2356
m_c    =  0.4159
tol    = 1.0e-10
maxiter = 3000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout smearing
stout_nstep = 1
stout_rho   = 0.125
stout_ndim  = 4

# ═══════════════════════════════════════════════════════════════
#  Initialise PyQUDA
# ═══════════════════════════════════════════════════════════════
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ═══════════════════════════════════════════════════════════════
#  Load gauge configuration
# ═══════════════════════════════════════════════════════════════
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Keep a smeared copy for inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════════
#  Dirac operators (clover)
# ═══════════════════════════════════════════════════════════════
dirac_s = core.getClover(latt_info, m_s, tol, maxiter,
                          xi_0, csw, csw, multigrid)
dirac_c = core.getClover(latt_info, m_c, tol, maxiter,
                          xi_0, csw, csw, multigrid)

# ═══════════════════════════════════════════════════════════════
#  Forward propagators (point source at origin, zero momentum)
#  Spin-colour Kronecker delta — no gamma insertion here.
#  All gamma structures go into the sink block and final
#  contraction via generate_einsum.
# ═══════════════════════════════════════════════════════════════
pt_src = source.source12(latt_info, "point", x_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# ═══════════════════════════════════════════════════════════════
#  FROM generate_einsum (meson_3pt)
#  Meson 3pt:
#    spectator  = s  (prop_s)
#    forward    = c  (prop_c)
#    sequential = c  (prop_seq)
#    sink Gamma = G5
#    src Gamma  = G5
#    cur Gamma  = g1 (gamma_x)
# ═══════════════════════════════════════════════════════════════

# ------------------------------------------------------------------
#  Gamma matrices (DeGrand-Rossi basis)
# ------------------------------------------------------------------
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)           # gamma_5
Gamma_snk = cp.asarray(gamma.gamma(15), dtype=cp.complex128)    # sink  gamma_5
Gamma_src = cp.asarray(gamma.gamma(15), dtype=cp.complex128)    # source gamma_5
Gamma_cur = cp.asarray(gamma.gamma(1),  dtype=cp.complex128)    # current gamma_1 = gamma_x

# ------------------------------------------------------------------
#  G5-conjugate gammas:  Γ̄ = γ₅ · Γ · γ₅
#  These appear in the sink block.
# ------------------------------------------------------------------
Gamma_snk_bar = G5 @ Gamma_snk @ G5
Gamma_src_bar = G5 @ Gamma_src @ G5

# ------------------------------------------------------------------
#  Step 1 — Sink block
#  B(x) = Γ̄_snk · S_spectator · Γ̄_src
#  The strange quark is the spectator, carrying BOTH source and
#  sink gamma_5 insertions.
# ------------------------------------------------------------------
B = core.LatticePropagator(latt_info)
B.data = contract(
    'AB, wtzyxBCab, CD -> wtzyxADab',
    Gamma_snk_bar, prop_s.data, Gamma_src_bar)

# ------------------------------------------------------------------
#  Step 2 — Sequential source + solve
#  Two-dagger convention:
#    η_seq = γ₅ · B† · γ₅   (sequential source at t_seq=8)
#    D_c · G_seq = η_seq    (sequential propagator)
#  Wall source spanning all spatial points at t=8 for correct
#  zero-momentum sink projection.
# ------------------------------------------------------------------
src_seq = source.sequential12(B, t_seq)

with dirac_c.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_c, src_seq)

# ------------------------------------------------------------------
#  Step 3 — Final contraction
#  C_3pt(τ) = Σ_z Tr[ γ₅·G†_seq·γ₅ · Γ_cur · S_fwd ]
#  Zero sink momentum → no Fourier phase factor.
# ------------------------------------------------------------------
# 3a. Outer G5-dagger: tmp = γ₅ · S_seq† · γ₅ = G(0,x)
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_seq.data.conj(), G5)

# 3b. Trace contraction: Tr[ G(0,x) · Γ_cur · S_fwd ]
three_pt_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    tmp_prop.data, Gamma_cur, prop_c.data)

# ------------------------------------------------------------------
#  MPI gather — reduce spatial dimensions, keep time
# ------------------------------------------------------------------
C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"),
    [0, -1, -1, -1])

# ═══════════════════════════════════════════════════════════════
#  Save — rank 0 only
#  Format: tau  Re[C3]  Im[C3]   (τ = 0 … t_seq)
#  No header, no metadata.
# ═══════════════════════════════════════════════════════════════
if core.getMPIRank() == 0:
    C3_window = np.asarray(C3_t[:t_seq + 1], dtype=np.complex128)
    tau = np.arange(t_seq + 1, dtype=np.int32)
    out = np.column_stack((tau, C3_window.real, C3_window.imag))
    out_path = f"C3_Ds_EM_gammax_cfg{n_cfg:05d}.txt"
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
