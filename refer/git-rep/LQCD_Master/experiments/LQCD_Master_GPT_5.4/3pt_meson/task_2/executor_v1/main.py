import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

resource_path = sys.argv[1]
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

x_src = [0, 0, 0, 0]
tseq = 8

t_boundary = -1
anisotropy = 1.0
xi_0 = 1.0
csw = 1.160920226
m_light = -0.277
m_charm = 0.4159
tol = 1.0e-12
maxiter = 10000
multigrid_light = [[6, 6, 6, 3], [4, 4, 4, 6]]
multigrid_charm = None

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_path = "./d0_to_piminus_vx_3pt_tseq8_cfg10000.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=t_boundary, anisotropy=anisotropy)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

dirac_l = core.getClover(latt_info, m_light, tol, maxiter, xi_0, csw, csw, multigrid_light)
dirac_c = core.getClover(latt_info, m_charm, tol, maxiter, xi_0, csw, csw, multigrid_charm)
dirac_d = core.getClover(latt_info, m_light, tol, maxiter, xi_0, csw, csw, multigrid_light)

pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# FROM generate_einsum (meson_3pt)
# ═══════════════════════════════════════════════════════
# Meson 3pt:
#   spectator  = u  (prop_l)
#   forward    = c  (prop_c)
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
#  G5-conjugate gammas for the sink block
# ------------------------------------------------------------------
Gamma_snk_bar = G5 @ Gamma_snk @ G5
Gamma_src_bar = G5 @ Gamma_src @ G5

# ------------------------------------------------------------------
#  Sink block for pi-(ubar d) at t_sink = tseq
# ------------------------------------------------------------------
B = core.LatticePropagator(latt_info)
B.data = contract(
    "AB, wtzyxBCab, CD -> wtzyxADab",
    Gamma_snk_bar, prop_l.data, Gamma_src_bar,
)

# ------------------------------------------------------------------
#  Sequential source and light sequential inversion through sink d leg
# ------------------------------------------------------------------
src_seq = source.sequential12(B, tseq)

with dirac_d.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_d, src_seq)

# ------------------------------------------------------------------
#  Final 3pt contraction with J_x = bar(d) gamma_x c
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

C3_t = core.gatherLattice(array.arrayAsNumpy(three_pt_local, backend="cupy"), [0, -1, -1, -1])

if core.getMPIRank() == 0:
    tau = np.arange(1, tseq, dtype=np.int32)
    data = np.asarray(C3_t[tau], dtype=np.complex128).reshape(-1)
    out = np.column_stack((tau, data.real, data.imag))
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
