# Run: mpirun -n 4 python3 main.py ~/.cache 10000
import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_utils import core, io, gamma, source

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

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

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
out_path = f"c3_lambda_to_p_axial_x_cfg{n_cfg}_tseq{tseq}.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# FROM generate_einsum (baryon_3pt), adapted to the requested axial current gamma_x @ gamma_5.
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_x = cp.asarray(gamma.gamma(1), dtype=cp.complex128)
Gamma_cur = Gamma_x @ G5

eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)

# Proton sink block from the two light spectator lines.
B = core.LatticePropagator(latt_info)
B.data = (
    - contract(
        "AB,abi,KH,jge,JF,wtzyxFBeb,wtzyxHAga->wtzyxJKij",
        Cg5,
        eps,
        Cg5,
        eps,
        Tmat,
        prop_l.data,
        prop_l.data,
    )
    + contract(
        "AB,abi,GH,fgj,JK,wtzyxHAga,wtzyxGBfb->wtzyxJKij",
        Cg5,
        eps,
        Cg5,
        eps,
        Tmat,
        prop_l.data,
        prop_l.data,
    )
)

# First dagger to form the sequential source.
B.data = contract("AB,wtzyxCBji,CD->wtzyxADij", G5, B.data.conj(), G5)
src_seq = source.sequential12(B, tseq)

with dirac_l.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_l, src_seq)

# Second dagger before the final current contraction.
G_l_seq_dag = core.LatticePropagator(latt_info)
G_l_seq_dag.data = contract("AB,wtzyxCBji,CD->wtzyxADij", G5, prop_seq.data.conj(), G5)

three_pt_local = contract(
    "wtzyxijba,jk,wtzyxkiab->t",
    G_l_seq_dag.data,
    Gamma_cur,
    prop_s.data,
)

C3_t = core.gatherLattice(three_pt_local.get(), [0, -1, -1, -1])
t_list = np.arange(1, tseq, dtype=np.int32)

if core.getMPIRank() == 0:
    c3_window = np.asarray(C3_t[t_list], dtype=np.complex128).reshape(-1)
    tseq_col = np.full(t_list.shape, tseq, dtype=np.int32)
    out = np.column_stack((tseq_col, t_list, c3_window.real, c3_window.imag))
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
