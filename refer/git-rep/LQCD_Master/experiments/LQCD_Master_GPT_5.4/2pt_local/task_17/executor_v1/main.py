import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# Run: mpirun -n 4 python3 main.py ~/.cache 10000

# 1. parameter definitions (hard-coded)
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

x_src = [0, 0, 0, 0]

anisotropy = 1.0
xi_0 = 1.0
csw = 1.160920226
m_s = -0.2356
m_c = 0.4159
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_dir = "."
out_path = os.path.join(out_dir, f"omega_c0_2pt_cfg{n_cfg}.txt")

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

# 2. read gauge configuration
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# 3. construct the Dirac operator
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_c = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, None)

# 4. compute forward propagators
pt_src = source.source12(latt_info, "point", x_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# 5. extract observable / compute contraction
# User-fixed Omega_c0 operator: epsilon^{abc} (s^{Ta} C gamma_x s^b) c^c
# Positive-parity projection with Tmat = (I + gamma_t) / 2.
# This keeps one vector-diquark polarization component and does not apply a separate spin-3/2 projection.
I4 = cp.eye(4, dtype=cp.complex128)
epsilon = cp.zeros((3, 3, 3), dtype=cp.complex128)
epsilon[0, 1, 2] = 1.0
epsilon[1, 2, 0] = 1.0
epsilon[2, 0, 1] = 1.0
epsilon[0, 2, 1] = -1.0
epsilon[2, 1, 0] = -1.0
epsilon[1, 0, 2] = -1.0
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Gamma_x = cp.asarray(gamma.gamma(1), dtype=cp.complex128)
Gamma_t = cp.asarray(gamma.gamma(8), dtype=cp.complex128)
Cg1 = Cmat @ Gamma_x
Tmat = 0.5 * (I4 + Gamma_t)

# FROM generate_einsum (baryon_2pt)
C_t_local = (
    -contract('AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFBfb, wtzyxEAea -> t', Cg1, epsilon, Cg1, epsilon, Tmat, prop_c.data, prop_s.data, prop_s.data)
    +contract('AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFAfa, wtzyxEBeb -> t', Cg1, epsilon, Cg1, epsilon, Tmat, prop_c.data, prop_s.data, prop_s.data)
)

C_t = core.gatherLattice(array.arrayAsNumpy(C_t_local, backend="cupy"), [0, -1, -1, -1])

# 6. save the result
if core.getMPIRank() == 0:
    C_t_root = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    t = np.arange(C_t_root.shape[0], dtype=np.int32)
    out = np.column_stack((t, C_t_root.real, C_t_root.imag))
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
