import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_utils import core, io, gamma, source

# Run: mpirun -n 4 python3 main.py ~/.cache 10000

# 1. parameter definitions (hard-coded)
resource_path = sys.argv[1]
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

x_src = [0, 0, 0, 0]
tseq = 8

anisotropy = 1.0
xi_0 = 1.0
csw = 1.160920226

mass_u = -0.277
mass_s = -0.2356
mass_c = 0.4159

tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_path = f"d0_to_kst_minus_vec_x_3pt_cfg{int(n_cfg):05d}_tseq{tseq}.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

# 2. read gauge configuration
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# 3. construct the Dirac operator
dirac_u = core.getClover(latt_info, mass_u, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, mass_s, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_c = core.getClover(latt_info, mass_c, tol, maxiter, xi_0, csw, csw, multigrid)

# 4. compute forward propagators
pt_src = source.source12(latt_info, "point", x_src)

with dirac_u.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_u, pt_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# 5. extract observable / compute contraction
# FROM generate_einsum (meson_3pt)
# ═══════════════════════════════════════════════════════
# Meson 3pt:
#   spectator  = u  (prop_l)
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
#  G5-conjugate gammas:  Γ̄ = γ₅ . Γ . γ₅
#  These appear in the sink block.
# ------------------------------------------------------------------
Gamma_snk_bar = G5 @ Gamma_snk @ G5
Gamma_src_bar = G5 @ Gamma_src @ G5

# ------------------------------------------------------------------
#  Step 1 -- Sink block
#  B(x) = Γ̄_snk . S_spectator . Γ̄_src
#  No Wick summation needed: meson 3pt has a single topology.
# ------------------------------------------------------------------
# spectator = u  (prop_l)
B = core.LatticePropagator(latt_info)
B.data = contract(
    "AB, wtzyxBCab, CD -> wtzyxADab",
    Gamma_snk_bar, prop_l.data, Gamma_src_bar,
)

# ------------------------------------------------------------------
#  Step 2 -- Sequential source + solve
# ------------------------------------------------------------------
src_seq = source.sequential12(B, tseq)

with dirac_s.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_s, src_seq)

# ------------------------------------------------------------------
#  Step 3 -- Final contraction
#  C_3pt(t) = Tr[ γ₅·G†_seq·γ₅ · Γ_cur · S_fwd ]
# ------------------------------------------------------------------
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij",
    G5, prop_seq.data.conj(), G5,
)

three_pt_local = contract(
    "wtzyxijba, jk, wtzyxkiab -> t",
    tmp_prop.data, Gamma_cur, prop_c.data,
)

C3_t = core.gatherLattice(cp.asnumpy(three_pt_local), [0, -1, -1, -1])

# 6. save the result
if core.getMPIRank() == 0:
    t_list = np.arange(1, tseq, dtype=np.int32)
    c3_window = np.asarray(C3_t[1:tseq], dtype=np.complex128).reshape(-1)
    out = np.column_stack((
        np.full(t_list.shape, tseq, dtype=np.int32),
        t_list,
        c3_window.real,
        c3_window.imag,
    ))
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
