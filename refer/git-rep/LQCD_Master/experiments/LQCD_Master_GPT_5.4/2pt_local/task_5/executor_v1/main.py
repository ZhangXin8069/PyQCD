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
mass_l = -0.277
csw = 1.160920226
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_dir = "."
out_path = os.path.join(out_dir, f"rho_2pt_cfg{n_cfg}.txt")

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

# 2. read gauge configuration
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# 3. construct the Dirac operator
dirac_l = core.getClover(latt_info, mass_l, tol, maxiter, xi_0, csw, csw, multigrid)

# 4. compute forward propagators
pt_src = source.source12(latt_info, "point", x_src)
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# 5. extract observable / compute contraction
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
g1 = cp.asarray(gamma.gamma(1), dtype=cp.complex128)
g2 = cp.asarray(gamma.gamma(2), dtype=cp.complex128)
g3 = cp.asarray(gamma.gamma(4), dtype=cp.complex128)

# FROM generate_einsum (meson_2pt)
C_x_local = contract('wtzyxCBba, CD, wtzyxDAba, AB -> t', prop_l.data.conj(), G5 @ g1, prop_l.data, g1 @ G5)
# FROM generate_einsum (meson_2pt)
C_y_local = contract('wtzyxCBba, CD, wtzyxDAba, AB -> t', prop_l.data.conj(), G5 @ g2, prop_l.data, g2 @ G5)
# FROM generate_einsum (meson_2pt)
C_z_local = contract('wtzyxCBba, CD, wtzyxDAba, AB -> t', prop_l.data.conj(), G5 @ g3, prop_l.data, g3 @ G5)

C_rho_local = (C_x_local + C_y_local + C_z_local) / 3.0
C_rho = core.gatherLattice(array.arrayAsNumpy(C_rho_local, backend="cupy"), [0, -1, -1, -1])

# 6. save the result
if core.getMPIRank() == 0:
    C_rho = np.asarray(C_rho, dtype=np.complex128).reshape(-1)
    t = np.arange(C_rho.shape[0], dtype=np.int32)
    out = np.column_stack((t, C_rho.real, C_rho.imag))
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
