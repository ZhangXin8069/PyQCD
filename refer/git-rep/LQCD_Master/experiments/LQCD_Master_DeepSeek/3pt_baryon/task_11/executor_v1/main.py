# Run: mpirun -n 4 python3 main.py ~/.cache 10000
import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ── Parameters ─────────────────────────────────────────────
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

# Source position (point source at origin)
x_src = [0, 0, 0, 0]

# Quark masses (valence, sea-matched)
m_l = -0.277
m_s = -0.2400

# Clover parameters
xi_0 = 1.0
csw = 1.160920226

# Solver
tol = 1.0e-12
maxiter = 5000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout smearing
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# Sequential source time
t_seq = 8

# ── Initialize PyQUDA ──────────────────────────────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── Load gauge configuration ───────────────────────────────
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Stout-smeared gauge for Dirac inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── Dirac operators ────────────────────────────────────────
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# ── Forward propagators (point source at origin) ───────────
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# ── Baryon 3pt: Xi- -> Lambda, vector current gamma_x ─────
# FROM generate_einsum (baryon_3pt)
#   Gamma: g1  (gamma_x / gamma_1)
#   Forward propagator entering current: prop_s

I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)           # gamma_5
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)     # gamma_x

# Epsilon tensor (GPU)
eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = 1.0
eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1.0

# Dirac gamma matrices (GPU)
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus = (1+gamma_4)/2

# ── Sink block: B = B_1 + B_2 (sum over 2 Wick topologies) ─
# Both topologies route the two s-quarks through the sink/current
B = core.LatticePropagator(latt_info)
B.data = (
    - contract(
        'JB,ibc,KH,jge,CF,wtzyxFCec,wtzyxHBgb->wtzyxJKij',
        Cg5, eps, Cg5, eps, Tmat, prop_s.data, prop_l.data)
    + contract(
        'AB,abi,KH,jge,JF,wtzyxFAea,wtzyxHBgb->wtzyxJKij',
        Cg5, eps, Cg5, eps, Tmat, prop_s.data, prop_l.data)
)

# ── First dagger: B_tilde = gamma_5 · B^dag · gamma_5 ──────
B.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, B.data.conj(), G5)

# ── Sequential wall source at t_seq = 8 ────────────────────
src_seq = source.sequential12(B, t_seq)

# ── Sequential solve (light quark, u line) ─────────────────
with dirac_l.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_l, src_seq)

# ── Second dagger on sequential propagator ─────────────────
prop_seq_dag = core.LatticePropagator(latt_info)
prop_seq_dag.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_seq.data.conj(), G5)

# ── Final contraction: Tr[ G_l_seq_dag · gamma_x · S_s ] ────
three_pt_site = contract(
    'wtzyxijba, jk, wtzyxkiab -> wtzyx',
    prop_seq_dag.data, Gamma_cur, prop_s.data)

# Sum over parity and spatial dimensions (zero-momentum)
C3_t_local = contract('wtzyx -> t', three_pt_site)

# MPI gather over time dimension
C3_t = core.gatherLattice(
    array.arrayAsNumpy(C3_t_local, backend="cupy"),
    [0, -1, -1, -1])

# ── Save result (root rank only) ───────────────────────────
if core.getMPIRank() == 0:
    # C3(tau) for tau = 0, 1, ..., t_seq
    C3_window = np.asarray(C3_t[:t_seq+1], dtype=np.complex128).reshape(-1)
    out_path = f"c3_xi2lam_v_gx_cfg{n_cfg}.txt"
    np.savetxt(out_path, np.column_stack((C3_window.real, C3_window.imag)),
               fmt=["%.16e", "%.16e"])
