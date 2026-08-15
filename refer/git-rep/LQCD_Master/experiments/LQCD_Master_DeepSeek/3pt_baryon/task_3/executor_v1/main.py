import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ── Parameters ──
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

m_l = -0.277
m_s = -0.2356
csw = 1.160920226
xi_0 = 1.0
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

x_src = [0, 0, 0, 0]
t_seq = 8

out_path = f"Lambda_3pt_vector_corr_cfg{n_cfg}.txt"

# ── Init ──
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── Load gauge configuration ──
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# ── Stout smear ──
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── Dirac operators ──
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# ── Forward propagators ──
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# ── 3pt: Sequential source method ──
# Gamma matrices on GPU (DeGrand-Rossi basis)
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)  # gamma_x

# Epsilon tensor (Levi-Civita in color space)
eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = 1.0
eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1.0

# Dirac matrices
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)  # C = gamma_2 gamma_4
Cg5 = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus = (I + gamma_4)/2

# ── Sink block B at t=t_seq ──
# FROM generate_einsum (baryon_3pt):
# Lambda (uds) has only ONE Wick topology — no exchange term unlike proton (uud).
# The sink block contracts two light propagators with Cg5 and epsilon,
# leaving the strange-quark spin-color indices (JKij) open.
B = core.LatticePropagator(latt_info)
B.data = (
    + contract(
        'AB,abi,GH,fgj,JK,wtzyxHAga,wtzyxGBfb->wtzyxJKij',
        Cg5, eps, Cg5, eps, Tmat,
        prop_l.data, prop_l.data,
    )
)

# First dagger: B_tilde = gamma5 @ B^dag @ gamma5  (two-dagger convention)
B.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, B.data.conj(), G5)

# Sequential source at sink time
src_seq = source.sequential12(B, t_seq)

# Sequential solve on strange quark
# The s-quark line goes through the current insertion; sequential source
# is built from the Lambda sink and solved on the strange Dirac operator.
with dirac_s.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_s, src_seq)

# Second dagger on sequential propagator
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_seq.data.conj(), G5)

# ── Final 3pt contraction ──
# C_3(tau) = sum_z Tr[ G_s_seq(z,tau) · gamma_x · S_s(z,tau) ]
# where G_s_seq is the (daggered) sequential strange propagator
# and S_s is the forward strange propagator.
three_pt_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    tmp_prop.data, Gamma_cur, prop_s.data)

C3_t = core.gatherLattice(
    three_pt_local.get(), [0, -1, -1, -1])

# ── Save result ──
# Output columns: t_seq, insertion_time(tau), Re[C3], Im[C3]
# Contact terms at tau=0 and tau=t_seq are included.
if core.getMPIRank() == 0:
    t_list = np.arange(t_seq + 1, dtype=np.int32)
    C3_window = np.asarray(C3_t[:t_seq + 1], dtype=np.complex128).reshape(-1)
    tseq_col = np.full(t_list.shape, t_seq, dtype=np.int32)
    out = np.column_stack((tseq_col, t_list, C3_window.real, C3_window.imag))
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
