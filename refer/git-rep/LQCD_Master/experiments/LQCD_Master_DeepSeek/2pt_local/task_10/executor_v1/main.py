import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ═══════════════════════════════════════════════════════════════
# 1. Parameter definitions
# ═══════════════════════════════════════════════════════════════

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Point source at origin
x_src = [0, 0, 0, 0]

# Quark masses (Wilson-clover)
m_l = -0.277       # light (u/d)
m_s = -0.2356      # strange

# Clover coefficient and anisotropy
csw = 1.160920226
xi_0 = 1.0

# Solver parameters
tol = 1.0e-12
maxiter = 2000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout link smearing parameters (applied to gauge before Dirac op)
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# Output directory
out_dir = "."

# ═══════════════════════════════════════════════════════════════
# 2. Read gauge configuration
# ═══════════════════════════════════════════════════════════════

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Stout-smear the gauge links BEFORE building the Dirac operator
gauge.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════════
# 3. Construct the Dirac operator (Wilson-clover)
# ═══════════════════════════════════════════════════════════════

dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# ═══════════════════════════════════════════════════════════════
# 4. Compute forward propagators
# ═══════════════════════════════════════════════════════════════

pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_s.useGauge(gauge):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# ═══════════════════════════════════════════════════════════════
# 5. Extract observable — Lambda baryon 2pt contraction
# ═══════════════════════════════════════════════════════════════
# The Lambda operator: O_Lambda = eps_abc (u^T_a Cg5 d_b) s_c
# With distinct u, d, s flavours there is exactly ONE Wick topology.
# The projector T = (I + gamma_4)/2 selects the positive-parity channel.

# Gamma matrix and tensor definitions (GPU tensors, DeGrand-Rossi basis)
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Cg5 = Cmat @ G5
# Tmat = (I + gamma_4)/2 = P_plus projector
Tmat = cp.asarray(
    (cp.eye(4, dtype=cp.complex128) + gamma.gamma(8)) * 0.5,
    dtype=cp.complex128,
)

# Levi-Civita epsilon_{abc} for color contraction
eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

# FROM generate_einsum (baryon_2pt)
# Single Wick topology for Lambda (ud s distinct → no exchange):
#   eps_abc (Cg5)_AB · eps_efd (Cg5)_EF · Tmat_CD ·
#     S_s(wtzyx, D_snk=d, C_src=c) · S_l(wtzyx, F_snk=f, A_src=a) ·
#     S_l(wtzyx, E_snk=e, B_src=b)  →  t
C_t_local = contract(
    'AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFAfa, wtzyxEBeb -> t',
    Cg5, eps, Cg5, eps, Tmat,
    prop_s.data, prop_l.data, prop_l.data,
)

# MPI gather: each rank has Lt_local time slices; gather along dim 0
C_t_cpu = array.arrayAsNumpy(C_t_local, backend="cupy")
C_t = core.gatherLattice(C_t_cpu, [0, -1, -1, -1])

# ═══════════════════════════════════════════════════════════════
# 6. Save the result
# ═══════════════════════════════════════════════════════════════

if core.getMPIRank() == 0:
    C_t_root = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    t_arr = np.arange(C_t_root.shape[0], dtype=np.int32)
    out = np.column_stack((t_arr, C_t_root.real, C_t_root.imag))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"lambda_2pt_cfg{n_cfg}.txt")
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
