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
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

x_src = [0, 0, 0, 0]

anisotropy = 1.0
xi_0 = 1.0
csw = 1.160920226

mass_light = -0.277
mass_charm = 0.4159

tol_light = 1.0e-10
tol_charm = 1.0e-12
maxiter = 10000
multigrid_light = [[6, 6, 6, 3], [4, 4, 4, 6]]
multigrid_charm = None

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_path = "./xi_cc_dcc_2pt.txt"

# 2. read gauge configuration
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# 3. construct the Dirac operator
dirac_light = core.getClover(latt_info, mass_light, tol_light, maxiter, xi_0, csw, csw, multigrid_light)
dirac_charm = core.getClover(latt_info, mass_charm, tol_charm, maxiter, xi_0, csw, csw, multigrid_charm)

# 4. compute forward propagators
pt_src = source.source12(latt_info, "point", x_src)

with dirac_light.useGauge(gauge_stout):
    prop_light = core.invertPropagator(dirac_light, pt_src)

with dirac_charm.useGauge(gauge_stout):
    prop_charm = core.invertPropagator(dirac_charm, pt_src)

# 5. extract observable / compute contraction
I4 = cp.eye(4, dtype=cp.complex128)
Gt = cp.asarray(gamma.gamma(8), dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = 0.5 * (I4 + Gt)

epsilon = cp.zeros((3, 3, 3), dtype=cp.complex128)
epsilon[0, 1, 2] = 1.0
epsilon[1, 2, 0] = 1.0
epsilon[2, 0, 1] = 1.0
epsilon[0, 2, 1] = -1.0
epsilon[2, 1, 0] = -1.0
epsilon[1, 0, 2] = -1.0

prop_light_data = prop_light.data
prop_charm_data = prop_charm.data

# FROM generate_einsum (baryon_2pt)
C_t_local = (
    -contract('AB, abc, EF, efd, CD, wtzyxDAda, wtzyxFCfc, wtzyxEBeb -> t', Cg5, epsilon, Cg5, epsilon, Tmat, prop_charm_data, prop_charm_data, prop_light_data)
    +contract('AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFAfa, wtzyxEBeb -> t', Cg5, epsilon, Cg5, epsilon, Tmat, prop_charm_data, prop_charm_data, prop_light_data)
)

C_t = core.gatherLattice(array.arrayAsNumpy(C_t_local, backend="cupy"), [0, -1, -1, -1])

# 6. save the result
if core.getMPIRank() == 0:
    C_t_root = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    t = np.arange(C_t_root.shape[0], dtype=np.int32)
    out = np.column_stack((t, C_t_root.real, C_t_root.imag))
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
