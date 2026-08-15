import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
out_name_template = "xi_cc_to_xi_c_vec_x_tseq8_cfg{n_cfg}.txt"

x_src = [0, 0, 0, 0]
tseq = 8

t_boundary = -1
anisotropy = 1.0
xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
m_s = -0.2356
m_c = 0.4159
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=t_boundary, anisotropy=anisotropy)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_c = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, multigrid)

pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# FROM generate_einsum (baryon_3pt)
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)

eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)

ones_phase = cp.ones(prop_l.data.shape[:5], dtype=cp.complex128)

B = core.LatticePropagator(latt_info)
B.data = (
    contract(
        "wtzyx,JB,ibc,GK,fje,CF,wtzyxFCec,wtzyxGBfb->wtzyxJKij",
        ones_phase,
        Cg5,
        eps,
        Cg5,
        eps,
        Tmat,
        prop_c.data,
        prop_l.data,
    )
    - contract(
        "wtzyx,AB,abi,GK,fje,JF,wtzyxFAea,wtzyxGBfb->wtzyxJKij",
        ones_phase,
        Cg5,
        eps,
        Cg5,
        eps,
        Tmat,
        prop_c.data,
        prop_l.data,
    )
)

B.data = contract("AB,wtzyxCBji,CD->wtzyxADij", G5, B.data.conj(), G5)
src_seq = source.sequential12(B, tseq)

with dirac_s.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_s, src_seq)

tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract("AB,wtzyxCBji,CD->wtzyxADij", G5, prop_seq.data.conj(), G5)

three_pt_local = contract("wtzyxijba,jk,wtzyxkiab->t", tmp_prop.data, Gamma_cur, prop_c.data)
C3_t = core.gatherLattice(array.arrayAsNumpy(three_pt_local, backend="cupy"), [0, -1, -1, -1])

out_path = out_name_template.format(n_cfg=n_cfg)
t_list = np.arange(1, tseq, dtype=np.int32)

if core.getMPIRank() == 0:
    C3_window = np.asarray(C3_t[t_list], dtype=np.complex128).reshape(-1)
    tseq_col = np.full(t_list.shape, tseq, dtype=np.int32)
    out = np.column_stack((tseq_col, t_list, C3_window.real, C3_window.imag))
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
