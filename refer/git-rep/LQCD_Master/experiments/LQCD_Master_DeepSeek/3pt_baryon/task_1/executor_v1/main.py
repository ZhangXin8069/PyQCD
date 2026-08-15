import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ── Parameters ──
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
m_l = -0.277
csw = 1.160920226
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4
x_src = [0, 0, 0, 0]
t_seq = 8
xi_0 = 1.0
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

resource_path = sys.argv[1]
n_cfg = sys.argv[2]

# ── Init ──
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── Read gauge configuration ──
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Keep a raw copy before smearing (not needed here since no covDev, but kept for clarity)
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── Dirac operator (Clover fermion) ──
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)

# ── Forward light propagator from point source at [0,0,0,0] ──
pt_src = source.source12(latt_info, "point", x_src)
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# ── Gamma matrices and epsilon tensor (GPU) ──
# FROM generate_einsum (baryon_3pt): proton -> proton, gamma1, P_plus
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)             # gamma_5
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)       # gamma_x

# Levi-Civita epsilon_{abc}
eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = 1.0
eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1.0

Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)  # C = gamma_2 gamma_4
Cg5 = Cmat @ G5                                                        # C gamma_5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)    # P_plus = (1+gamma_4)/2

# ── Sink block B(x) at t_sink = 8 ──
# Sum over 4 Wick topologies (direct + exchange, antisymmetrised over both u-quark lines)
B = core.LatticePropagator(latt_info)
B.data = (
    + contract(
        "AJ,aic,KH,jge,CF,wtzyxFCec,wtzyxHAga->wtzyxJKij",
        Cg5, eps, Cg5, eps, Tmat,
        prop_l.data, prop_l.data,
    )
    - contract(
        "AJ,aic,GH,fgj,CK,wtzyxHAga,wtzyxGCfc->wtzyxJKij",
        Cg5, eps, Cg5, eps, Tmat,
        prop_l.data, prop_l.data,
    )
    - contract(
        "AB,abi,KH,jge,JF,wtzyxFBeb,wtzyxHAga->wtzyxJKij",
        Cg5, eps, Cg5, eps, Tmat,
        prop_l.data, prop_l.data,
    )
    + contract(
        "AB,abi,GH,fgj,JK,wtzyxHAga,wtzyxGBfb->wtzyxJKij",
        Cg5, eps, Cg5, eps, Tmat,
        prop_l.data, prop_l.data,
    )
)

# ── First dagger: eta_seq = gamma_5 B^dag gamma_5 ──
B.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij",
    G5, B.data.conj(), G5)

# ── Sequential source at t_sink = 8 ──
src_seq = source.sequential12(B, t_seq)

# ── Sequential propagator solve ──
with dirac_l.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_l, src_seq)

# ── Second dagger on sequential propagator: G_l_seq_dag ──
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij",
    G5, prop_seq.data.conj(), G5)

# ── Final 3pt contraction: C3(tau) = sum_z Tr[ G_l_seq_dag  gamma_x  prop_l ] ──
three_pt_local = contract(
    "wtzyxijba, jk, wtzyxkiab -> t",
    tmp_prop.data, Gamma_cur, prop_l.data)

C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"),
    [0, -1, -1, -1])

# ── Save result (rank 0 only) ──
if core.getMPIRank() == 0 and C3_t is not None:
    C3_window = np.asarray(C3_t[1:t_seq], dtype=np.complex128)
    t_ins = np.arange(1, t_seq, dtype=np.int32)
    tseq_col = np.full(t_ins.shape, t_seq, dtype=np.int32)
    out = np.column_stack((tseq_col, t_ins, C3_window.real, C3_window.imag))
    np.savetxt("proton_3pt_vector_gammax_result.txt", out,
               fmt=["%d", "%d", "%.16e", "%.16e"])