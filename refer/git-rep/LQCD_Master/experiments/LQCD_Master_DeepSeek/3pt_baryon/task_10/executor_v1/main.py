# Run: mpirun -np 4 python main.py /path/to/tunecache 10000
import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_utils import core, io, gamma, source

# ── runtime inputs ──
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

# ═══════════════════  Hard-coded parameters  ═══════════════════
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

x_src = [0, 0, 0, 0]        # point source position
t_seq = 8                   # sink timeslice for sequential source

xi_0 = 1.0
csw = 1.160920226
m_l = -0.277                # light quark mass
m_b = 1.5                   # bottom quark mass (heavy, lighter than physical)
tol = 1.0e-12
maxiter = 20000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]   # multigrid for light quark

stout_nstep = 1
stout_rho   = 0.125
stout_ndim  = 4

# ═══════════════════  Initialize  ═══════════════════
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ═══════════════════  Read gauge configuration  ═══════════════════
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Keep raw gauge for possible future use; smeared gauge for Dirac operators
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════  Construct Dirac operators  ═══════════════════
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_b = core.getClover(latt_info, m_b, tol, maxiter, xi_0, csw, csw, None)  # plain CG

# ═══════════════════  Compute forward propagators  ═══════════════════
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)   # light forward

with dirac_b.useGauge(gauge_stout):
    prop_b = core.invertPropagator(dirac_b, pt_src)   # bottom forward

# ═══════════════════  Gamma matrices & epsilon (GPU)  ═══════════════════
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)            # γ₅
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)     # γ₁ (vector current)

eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = 1.0
eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1.0

Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5  = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus = (1+γ₄)/2

# ═══════════════════  Sink block B at t_f = 8  ═══════════════════
# FROM generate_einsum (baryon_3pt): Λ_b → p, current ū γ₁ b, projector P_plus
B = core.LatticePropagator(latt_info)
B.data = (
    - contract("AB,abi,KH,jge,JF,wtzyxFBeb,wtzyxHAga->wtzyxJKij",
               Cg5, eps, Cg5, eps, Tmat, prop_l.data, prop_l.data)
    + contract("AB,abi,GH,fgj,JK,wtzyxHAga,wtzyxGBfb->wtzyxJKij",
               Cg5, eps, Cg5, eps, Tmat, prop_l.data, prop_l.data)
)

# First dagger:  B̃ = γ₅ · B† · γ₅
B.data = contract("AB, wtzyxCBji, CD -> wtzyxADij",
                  G5, B.data.conj(), G5)

# Sequential source: volume source placed on time slice t_seq = 8
src_seq = source.sequential12(B, t_seq)

# ═══════════════════  Sequential solve  ═══════════════════
with dirac_l.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_l, src_seq)

# Second dagger:  G_seq_dag = γ₅ · G_seq† · γ₅
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract("AB, wtzyxCBji, CD -> wtzyxADij",
                          G5, prop_seq.data.conj(), G5)

# ═══════════════════  Final contraction  ═══════════════════
# C₃(τ) = Σ_z  Tr[ G_seq_dag(z,τ) · γ₁ · S_b(z,τ; 0) ]
# Contract spin, color, parity, spatial → 1D time series
three_pt_local = contract("wtzyxijba, jk, wtzyxkiab -> t",
                          tmp_prop.data, Gamma_cur, prop_b.data)

# MPI-gather time dimension
C3_t = core.gatherLattice(three_pt_local.get(), [0, -1, -1, -1])

# ═══════════════════  Save the result  ═══════════════════
if core.getMPIRank() == 0:
    t_list = list(range(1, t_seq))           # τ = 1 … 7
    C3_window = np.asarray(C3_t[t_list], dtype=np.complex128).reshape(-1)
    t_ins = np.asarray(t_list, dtype=np.int32)
    out = np.column_stack((t_ins, C3_window.real, C3_window.imag))
    out_path = f"Lambda_b_to_p_3pt_Vx_cfg{n_cfg:05d}.txt"
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
