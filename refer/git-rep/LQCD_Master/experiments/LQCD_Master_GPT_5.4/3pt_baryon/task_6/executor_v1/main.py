import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_utils import core, io, gamma, source

# Run: mpirun -n 4 python3 main.py ~/.cache 10000

# 1. parameter definitions
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
m_l = -0.277
m_s = -0.2356
m_c = 0.4159
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_path = f"lambda_c_to_lambda_c3_vx_cfg{n_cfg}_tseq{tseq}.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=t_boundary, anisotropy=anisotropy)

# 2. read gauge configuration
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# 3. construct the Dirac operator
try:
    dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
    dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)
    dirac_c = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, multigrid)
except AttributeError:
    dirac_l = core.getDirac(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
    dirac_s = core.getDirac(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)
    dirac_c = core.getDirac(latt_info, m_c, tol, maxiter, xi_0, csw, csw, multigrid)

# 4. compute forward propagators
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)
    with dirac_c.useGauge(gauge_stout):
        prop_c = core.invertPropagator(dirac_c, pt_src)

# 5. extract observable / compute contraction
# FROM generate_einsum (baryon_3pt)
# Lambda_c+ -> Lambda, J = sbar gamma_x c, zero spatial momentum, tseq = 8
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)

# Epsilon tensor and spin projectors for the Lambda sink block
eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)

# Build the Lambda sink block from the two light spectator lines
B = core.LatticePropagator(latt_info)
B.data = contract(
    "AB,abi,GH,fgj,JK,wtzyxHAga,wtzyxGBfb->wtzyxJKij",
    Cg5,
    eps,
    Cg5,
    eps,
    Tmat,
    prop_l.data,
    prop_l.data,
)

# First dagger for the sequential source construction
B.data = contract("AB,wtzyxCBji,CD->wtzyxADij", G5, B.data.conj(), G5)
src_seq = source.sequential12(B, tseq)

# Strange sequential solve on the fixed Lambda sink block
with dirac_s.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_s, src_seq)

# Second dagger and final current contraction with the charm forward line
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract("AB,wtzyxCBji,CD->wtzyxADij", G5, prop_seq.data.conj(), G5)
three_pt_local = contract("wtzyxijba,jk,wtzyxkiab->t", tmp_prop.data, Gamma_cur, prop_c.data)
C3_t = core.gatherLattice(three_pt_local.get(), [0, -1, -1, -1])

# 6. save the result
if core.getMPIRank() == 0:
    t_ins = np.arange(1, tseq, dtype=np.int32)
    c3_window = np.asarray(C3_t[1:tseq], dtype=np.complex128).reshape(-1)
    tseq_col = np.full(t_ins.shape, tseq, dtype=np.int32)
    out = np.column_stack((tseq_col, t_ins, c3_window.real, c3_window.imag))
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
