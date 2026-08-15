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

mass_light = -0.277
mass_charm = 0.4159
mass_bottom = 1.5

tol_light = 1.0e-10
tol_charm = 1.0e-10
tol_bottom = 1.0e-10

maxiter_light = 10000
maxiter_charm = 12000
maxiter_bottom = 20000

multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_dir = "."
out_path = os.path.join(out_dir, f"B_to_D_vector_3pt_cfg{int(n_cfg):05d}_tseq{tseq}.txt")

# 2. read gauge configuration
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# 3. construct the Dirac operator
try:
    dirac_light = core.getClover(latt_info, mass_light, tol_light, maxiter_light, xi_0, csw, csw, multigrid)
    dirac_bottom = core.getClover(latt_info, mass_bottom, tol_bottom, maxiter_bottom, xi_0, csw, csw, multigrid)
    dirac_charm = core.getClover(latt_info, mass_charm, tol_charm, maxiter_charm, xi_0, csw, csw, multigrid)
except AttributeError:
    dirac_light = core.getDirac(latt_info, mass_light, tol_light, maxiter_light, xi_0, csw, csw, multigrid)
    dirac_bottom = core.getDirac(latt_info, mass_bottom, tol_bottom, maxiter_bottom, xi_0, csw, csw, multigrid)
    dirac_charm = core.getDirac(latt_info, mass_charm, tol_charm, maxiter_charm, xi_0, csw, csw, multigrid)

# 4. compute forward propagators
pt_src = source.source12(latt_info, "point", x_src)

with dirac_light.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_light, pt_src)

with dirac_bottom.useGauge(gauge_stout):
    prop_b = core.invertPropagator(dirac_bottom, pt_src)

# 5. extract observable / compute contraction
# FROM generate_einsum (meson_3pt)
# ═══════════════════════════════════════════════════════
# Meson 3pt:
#   spectator  = u  (prop_l)
#   forward    = b  (prop_b)
#   sequential = c (prop_seq)
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

with dirac_charm.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_charm, src_seq)

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
    tmp_prop.data, Gamma_cur, prop_b.data,
)

C3_t = core.gatherLattice(cp.asnumpy(three_pt_local), [0, -1, -1, -1])

# 6. save the result
if core.getMPIRank() == 0:
    t_list = np.arange(1, tseq, dtype=np.int32)
    C3_window = np.asarray(C3_t[t_list], dtype=np.complex128).reshape(-1)
    out = np.column_stack((t_list, C3_window.real, C3_window.imag))
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
