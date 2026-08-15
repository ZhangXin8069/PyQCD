import os
import sys
import numpy as np
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, source

# Run: mpirun -n 4 python3 main.py ~/.cache 10000

# 1. parameter definitions (hard-coded)
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
source_position = [0, 0, 0, 0]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
out_path = f"./ds_2pt_cfg{n_cfg}.txt"

anisotropy = 1.0
xi_0 = 1.0
csw = 1.160920226
mass_s = -0.2356
mass_c = 0.4159
tol = 1.0e-10
maxiter = 10000
multigrid_s = [[6, 6, 6, 3], [4, 4, 4, 6]]
multigrid_c = None

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# 2. read gauge configuration
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# 3. construct the Dirac operator
dirac_s = core.getClover(latt_info, mass_s, tol, maxiter, xi_0, csw, csw, multigrid_s)
dirac_c = core.getClover(latt_info, mass_c, tol, maxiter, xi_0, csw, csw, multigrid_c)

# 4. compute forward propagators
pt_src = source.source12(latt_info, "point", source_position)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# 5. extract observable / compute contraction
# Local pseudoscalar D_s two-point function: O_Ds = sbar gamma5 c
# FROM generate_einsum (meson_2pt)
C_t_local = contract('wtzyxCBba, wtzyxCBba -> t', prop_s.data.conj(), prop_c.data)
C_t = core.gatherLattice(array.arrayAsNumpy(C_t_local, backend="cupy"), [0, -1, -1, -1])

# 6. save the result
if core.getMPIRank() == 0:
    C_t_root = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    t = np.arange(latt_size[3], dtype=np.int32)
    out = np.column_stack((t, C_t_root.real, C_t_root.imag))
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
