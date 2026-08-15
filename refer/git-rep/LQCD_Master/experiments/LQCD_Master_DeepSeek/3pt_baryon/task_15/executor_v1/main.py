# Run: mpirun -n 4 python3 main.py ~/.cache 10000
import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ═══════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

x_src = [0, 0, 0, 0]   # point source at origin
t_seq = 8               # sink time slice

# Clover fermion parameters
xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
m_s = -0.2356
tol = 1.0e-12
maxiter = 20000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout smearing
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ═══════════════════════════════════════════════════════════
# Initialize PyQUDA
# ═══════════════════════════════════════════════════════════
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ═══════════════════════════════════════════════════════════
# Read gauge configuration
# ═══════════════════════════════════════════════════════════
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Stout-smeared copy for inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════
# Dirac operators (Clover)
# ═══════════════════════════════════════════════════════════
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# ═══════════════════════════════════════════════════════════
# Forward propagators (point source at origin)
# ═══════════════════════════════════════════════════════════
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# ═══════════════════════════════════════════════════════════
# Gamma matrices and tensors (GPU)
# ═══════════════════════════════════════════════════════════
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)          # gamma_5
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)  # C = gamma_2 * gamma_4
Cg5 = Cmat @ G5                                                # C * gamma_5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus = (1+gamma_4)/2

# Axial-vector current: gamma_1 * gamma_5  (DeGrand-Rossi: gamma_x = gamma_1)
Gamma_cur = cp.asarray(gamma.gamma(1) @ gamma.gamma(15), dtype=cp.complex128)

# Levi-Civita epsilon_{abc}
eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = 1.0
eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1.0

# ═══════════════════════════════════════════════════════════
# Sink block B  (from generate_einsum, baryon_3pt)
# Xi-(dss) -> Lambda(uds), current s->u, g1@G5
# 2 Wick topologies from source-side antisymmetrization
# ═══════════════════════════════════════════════════════════
B = core.LatticePropagator(latt_info)
B.data = (
    - contract(
        'JB,ibc,KH,jge,CF,wtzyxFCec,wtzyxHBgb->wtzyxJKij',
        Cg5,
        eps,
        Cg5,
        eps,
        Tmat,
        prop_s.data,
        prop_l.data,
    )  # topo 0: sink s -> source s_c, sink d -> source d_b
    + contract(
        'AB,abi,KH,jge,JF,wtzyxFAea,wtzyxHBgb->wtzyxJKij',
        Cg5,
        eps,
        Cg5,
        eps,
        Tmat,
        prop_s.data,
        prop_l.data,
    )  # topo 1: antisymmetrized partner
)

# ═══════════════════════════════════════════════════════════
# First dagger:  B_tilde = gamma_5 * B^dag * gamma_5
# ═══════════════════════════════════════════════════════════
B.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, B.data.conj(), G5)

# ═══════════════════════════════════════════════════════════
# Sequential source at t_seq=8
# ═══════════════════════════════════════════════════════════
src_seq = source.sequential12(B, t_seq)

# Diagnostic: L2 norm of sequential source (local portion)
eta_norm = float(cp.sqrt(cp.sum(src_seq.data.conj() * src_seq.data)).real)
if core.getMPIRank() == 0:
    print(f"||eta_seq||_2 (local) = {eta_norm:.6e}")

# ═══════════════════════════════════════════════════════════
# Sequential solve: D_l * G_seq = eta_seq
# ═══════════════════════════════════════════════════════════
with dirac_l.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_l, src_seq)

# ═══════════════════════════════════════════════════════════
# Second dagger: G_seq_dag = gamma_5 * G_seq^dag * gamma_5
# ═══════════════════════════════════════════════════════════
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_seq.data.conj(), G5)

# ═══════════════════════════════════════════════════════════
# Final contraction: Tr[ G_seq_dag * Gamma_cur * S_s ]
# ═══════════════════════════════════════════════════════════
three_pt_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    tmp_prop.data, Gamma_cur, prop_s.data)

# Diagnostic: sample contraction at tau=4
C3_local = three_pt_local.get()
if core.getMPIRank() == 0:
    print(f"Sample C3(tau=4) local = {C3_local[4]:.6e}")

# ═══════════════════════════════════════════════════════════
# MPI gather
# ═══════════════════════════════════════════════════════════
C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"), [0, -1, -1, -1])

# ═══════════════════════════════════════════════════════════
# Save result: C3(tau) for tau=1..7, one line per tau
# ═══════════════════════════════════════════════════════════
if core.getMPIRank() == 0:
    C3_window = np.asarray(C3_t[1:t_seq], dtype=np.complex128)
    out_path = f"C3_Xi_to_Lambda_axial_cfg{n_cfg:05d}.txt"
    out_data = np.column_stack((C3_window.real, C3_window.imag))
    np.savetxt(out_path, out_data, fmt=["%.16e", "%.16e"])
    print(f"Saved C3 to {out_path}")
