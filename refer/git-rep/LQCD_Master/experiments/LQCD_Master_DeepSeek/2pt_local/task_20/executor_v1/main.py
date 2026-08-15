# Run: mpirun -n 4 python main.py <resource_path> <cfg_number>
import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ═══════════════════════════════════════════════════════════════
# 1. Parameter definitions (hard-coded)
# ═══════════════════════════════════════════════════════════════

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

# Lattice geometry
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]          # 4 MPI ranks, partitioned in time

# Configuration file path
cfg_path_template = (
    "/public/share/weiwang/clqcd/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Point source at origin
x_src = [0, 0, 0, 0]

# Clover fermion parameters
xi_0 = 1.0
csw = 1.160920226

# Strange quark: mass = -0.2356 (from ensemble registry)
m_s = -0.2356
tol_s = 1.0e-12
maxiter_s = 2000

# Charm quark: mass = 0.4159, relaxed tolerance for heavy quark
m_c = 0.4159
tol_c = 1.0e-10
maxiter_c = 4000

# Multigrid parameters (two-level)
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout link smearing: 1 step, rho=0.125, 4-dim
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ═══════════════════════════════════════════════════════════════
# 2. Read gauge configuration
# ═══════════════════════════════════════════════════════════════

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Copy before smearing; stoutSmear modifies in place
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════════
# 3. Construct the Dirac operator (Clover-improved Wilson)
# ═══════════════════════════════════════════════════════════════

dirac_s = core.getClover(latt_info, m_s, tol_s, maxiter_s, xi_0, csw, csw, multigrid)
dirac_c = core.getClover(latt_info, m_c, tol_c, maxiter_c, xi_0, csw, csw, multigrid)

# ═══════════════════════════════════════════════════════════════
# 4. Compute forward propagators
# ═══════════════════════════════════════════════════════════════

# Point source at [0, 0, 0, 0]
pt_src = source.source12(latt_info, "point", x_src)

# Strange propagator prop_s — quark line 'a' in the sc diquark
with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# Charm propagator prop_c — quark lines 'b' (diquark) and 'c' (spectator)
with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# ═══════════════════════════════════════════════════════════════
# 5. Extract observable: Omega_cc baryon 2pt contraction
# ═══════════════════════════════════════════════════════════════
# Operator: epsilon^{abc} (s^{Ta} C gamma_5 c^b) c^c
# Diquark: Cg5 = C @ gamma_5, projector: Tmat = (1+gamma_4)/2 = P_plus

# Gamma matrices and color tensors on GPU
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)            # gamma_5
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)  # C = gamma_2 gamma_4
Cg5 = Cmat @ G5                                                   # C gamma_5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus = (1+gamma_4)/2

# Levi-Civita antisymmetric tensor epsilon_{abc}
eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

# FROM generate_einsum (baryon_2pt)
# Two Wick topologies (direct + exchange) sum to restore full
# antisymmetrization of the two identical charm quarks.
#
# Quark-line mapping:
#   prop_c (1st arg) — charm in diquark  (line 'b')
#   prop_c (2nd arg) — spectator charm   (line 'c')
#   prop_s (3rd arg) — strange in diquark (line 'a')

C_t_local = cp.zeros(latt_info.Lt, dtype=cp.complex128)

# Topology 0: direct attachment, sign = -1
C_t_local += -contract(
    'AB, abc, EF, efd, CD, wtzyxDAda, wtzyxFCfc, wtzyxEBeb -> t',
    Cg5, eps, Cg5, eps, Tmat,
    prop_c.data, prop_c.data, prop_s.data,
)

# Topology 1: exchange attachment, sign = +1
C_t_local += contract(
    'AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFAfa, wtzyxEBeb -> t',
    Cg5, eps, Cg5, eps, Tmat,
    prop_c.data, prop_c.data, prop_s.data,
)

# MPI gather: sum over spatial volume, gather time slices to rank 0
C_t = core.gatherLattice(
    array.arrayAsNumpy(C_t_local, backend="cupy"), [0, -1, -1, -1]
)

# ═══════════════════════════════════════════════════════════════
# 6. Save the result (rank 0 only)
# ═══════════════════════════════════════════════════════════════

if core.getMPIRank() == 0:
    t = np.arange(latt_size[3], dtype=np.int32)
    C_t_root = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    out = np.column_stack((t, C_t_root.real, C_t_root.imag))
    out_path = f"omega_cc_2pt_cfg{n_cfg}.txt"
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
