import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ═══════════════════════════════════════════════════════════════
# Parameters — Xi_cc++(ucc) → Xi_c+(usc) via (c→s) vector current
# ═══════════════════════════════════════════════════════════════
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

x_src = [0, 0, 0, 0]       # point source at origin
t_seq = 8                    # sink time slice for sequential source

xi_0 = 1.0
csw = 1.160920226
m_l = -0.277                 # light quark mass
m_s = -0.2356                # strange quark mass
m_c = 0.4159                 # charm quark mass
tol = 1.0e-10
maxiter_l = 10000
maxiter_s = 10000
maxiter_c = 5000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ═══════════════════════════════════════════════════════════════
# Initialize PyQUDA and read gauge configuration
# ═══════════════════════════════════════════════════════════════
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Stout-smeared copy for Dirac inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════════
# Dirac operators — clover fermions
# ═══════════════════════════════════════════════════════════════
dirac_l = core.getClover(latt_info, m_l, tol, maxiter_l, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter_s, xi_0, csw, csw, multigrid)
dirac_c = core.getClover(latt_info, m_c, tol, maxiter_c, xi_0, csw, csw, None)

# ═══════════════════════════════════════════════════════════════
# Forward propagators — point source at origin
# ═══════════════════════════════════════════════════════════════
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# ═══════════════════════════════════════════════════════════════
# Gamma matrices and epsilon tensor on GPU
# ═══════════════════════════════════════════════════════════════
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)   # gamma_x

eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus

# ═══════════════════════════════════════════════════════════════
# Sink block B(x) — two Wick topologies
# FROM generate_einsum (baryon_3pt): Xi_cc++(ucc) → Xi_c+(usc)
#   Gamma: g1  Forward: prop_c  Projector: P_plus on spectator charm
# ═══════════════════════════════════════════════════════════════
B = core.LatticePropagator(latt_info)
B.data = (
    + contract(
        'JB,ibc,GK,fje,CF,wtzyxFCec,wtzyxGBfb->wtzyxJKij',
        Cg5, eps, Cg5, eps, Tmat, prop_c.data, prop_l.data,
    )
    - contract(
        'AB,abi,GK,fje,JF,wtzyxFAea,wtzyxGBfb->wtzyxJKij',
        Cg5, eps, Cg5, eps, Tmat, prop_c.data, prop_l.data,
    )
)

# First dagger:  B̃ = γ₅ · B† · γ₅
B.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, B.data.conj(), G5)

# ═══════════════════════════════════════════════════════════════
# Sequential source (volume at t=8) and strange-quark solve
# ═══════════════════════════════════════════════════════════════
src_seq = source.sequential12(B, t_seq)

with dirac_s.useGauge(gauge_stout):
    prop_s_seq = core.invertPropagator(dirac_s, src_seq)

# Second dagger on sequential propagator
prop_s_seq_dag = core.LatticePropagator(latt_info)
prop_s_seq_dag.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_s_seq.data.conj(), G5)

# ═══════════════════════════════════════════════════════════════
# Final contraction: C₃(τ) = Σ_z Tr[ G_s_seq_dag(z,τ) · γ_x · S_c(z,τ) ]
# ═══════════════════════════════════════════════════════════════
three_pt_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    prop_s_seq_dag.data, Gamma_cur, prop_c.data)

C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"), [0, -1, -1, -1])

# ═══════════════════════════════════════════════════════════════
# Save result — tau = 1..7, no header
# ═══════════════════════════════════════════════════════════════
if core.getMPIRank() == 0:
    C3_window = np.asarray(C3_t[1:8], dtype=np.complex128)
    out_path = f"c3_xi_cc_to_xi_c_cfg{n_cfg:05d}.txt"
    out_data = np.column_stack((C3_window.real, C3_window.imag))
    np.savetxt(out_path, out_data, fmt=["%.16e", "%.16e"])
