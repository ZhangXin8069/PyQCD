# Lambda -> proton axial-vector 3pt correlation function
# Run: mpirun -n 4 python main.py <resource_path> <cfg_number>
import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ═══ Parameters ════════════════════════════════════════════════════════
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Source and sink
x_src = [0, 0, 0, 0]   # point source at origin
t_seq = 8               # sink time slice

# Gauge / fermion parameters
xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
m_s = -0.2356
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout smearing
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ═══ Initialize PyQUDA ════════════════════════════════════════════════
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ═══ Read gauge configuration ═════════════════════════════════════════
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══ Construct Dirac operators ════════════════════════════════════════
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# ═══ Forward propagators from point source ════════════════════════════
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# ═══ Gamma matrices and epsilon tensor (GPU) ═════════════════════════
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
# Axial-vector current: bar{u} gamma_x gamma_5 s
Gamma_cur = cp.asarray(gamma.gamma(1) @ gamma.gamma(15), dtype=cp.complex128)

# Levi-Civita epsilon_{abc}
eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = 1.0
eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1.0

# Dirac matrices for baryon operators
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus

# ═══ Sink block B(x) at t=t_seq ══════════════════════════════════════
# FROM generate_einsum (baryon_3pt): Lambda(uds) -> proton(uud)
# Two Wick topologies: spectator u,d in both Lambda and proton are light
B = core.LatticePropagator(latt_info)
B.data = (
    - contract(
        'AB,abi,KH,jge,JF,wtzyxFBeb,wtzyxHAga->wtzyxJKij',
        Cg5, eps, Cg5, eps, Tmat, prop_l.data, prop_l.data)
    + contract(
        'AB,abi,GH,fgj,JK,wtzyxHAga,wtzyxGBfb->wtzyxJKij',
        Cg5, eps, Cg5, eps, Tmat, prop_l.data, prop_l.data)
)

# ═══ First dagger: B_tilde = gamma_5 · B^dagger · gamma_5 ════════════
B.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, B.data.conj(), G5)

# ═══ Sequential source and solve ═════════════════════════════════════
src_seq = source.sequential12(B, t_seq)

with dirac_l.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_l, src_seq)

# ═══ Second dagger on sequential propagator ══════════════════════════
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_seq.data.conj(), G5)

# ═══ Final contraction: Tr[G_seq_dag · Gamma_cur · prop_s] ═══════════
# Gamma_cur = gamma_x @ gamma_5 (axial-vector current, x-component)
# Sum over spatial sites (zero momentum transfer)
three_pt_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    tmp_prop.data, Gamma_cur, prop_s.data)

# ═══ MPI gather ══════════════════════════════════════════════════════
C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"),
    [0, -1, -1, -1])

# ═══ Save result (rank 0 only, no header) ════════════════════════════
if core.getMPIRank() == 0:
    # Drop contact terms at tau=0 and tau=t_seq
    C3_window = np.asarray(C3_t[1:t_seq], dtype=np.complex128)
    tau = np.arange(1, t_seq, dtype=np.int32)
    out_data = np.column_stack((tau, C3_window.real, C3_window.imag))
    np.savetxt(
        f"c3_lambda_p_axial_cfg{n_cfg:05d}.txt",
        out_data,
        fmt=["%d", "%.16e", "%.16e"])
