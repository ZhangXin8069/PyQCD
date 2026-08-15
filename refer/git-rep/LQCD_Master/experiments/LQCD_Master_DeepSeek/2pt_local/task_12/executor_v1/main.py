# Run: mpirun -np 4 python3 main.py ~/.cache 10000

import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ── Parameters ────────────────────────────────────────────
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/"
    "Configurations/Original/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

x_src = [0, 0, 0, 0]

xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
m_s = -0.2356
tol = 1.0e-12
maxiter = 20000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ── Initialize PyQUDA ─────────────────────────────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── Load gauge configuration ──────────────────────────────
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Copy before smearing; stoutSmear modifies in place
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── Dirac operators (clover-improved Wilson, CGNR with multigrid) ──
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# ── Forward propagators (point source at origin, stout-smeared gauge) ──
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# ── Gamma matrices and epsilon tensor (GPU) ───────────────
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)   # C = γ₂ γ₄
Cg1 = Cmat @ cp.asarray(gamma.gamma(1), dtype=cp.complex128)              # Cγ₁  diquark
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)      # P⁺ = (1+γ₄)/2

eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

# ── Contraction: Σ⁺ two-point (uus, Cγ₁ diquark, P⁺ projector) ──
# Two Wick topologies: direct + exchange.  Both from generate_einsum.
# FROM generate_einsum (baryon_2pt)
C_t_local = (
    -contract(
        'AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFBfb, wtzyxEAea -> t',
        Cg1, eps, Cg1, eps, Tmat, prop_s.data, prop_l.data, prop_l.data
    )
    + contract(
        'AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFAfa, wtzyxEBeb -> t',
        Cg1, eps, Cg1, eps, Tmat, prop_s.data, prop_l.data, prop_l.data
    )
)

# MPI gather: reduce spatial dimensions, keep time
C_t = core.gatherLattice(
    array.arrayAsNumpy(C_t_local, backend="cupy"), [0, -1, -1, -1]
)

# ── Save result (two columns: t, Re[C(t)]) ────────────────
if core.getMPIRank() == 0:
    C_t_flat = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    t_arr = np.arange(C_t_flat.shape[0], dtype=np.int32)
    out = np.column_stack((t_arr, C_t_flat.real))
    out_path = "sigma_plus_2pt_cfg{n_cfg}.txt".format(n_cfg=n_cfg)
    np.savetxt(out_path, out, fmt=["%d", "%.16e"])
