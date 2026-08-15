import os
import sys
import numpy as np
from pyquda_utils import core, io, source
from pyquda_comm import array
from opt_einsum import contract

# Run: mpirun -n 4 python3 main.py <resource_path> <cfg_number>

# 1. parameter definitions (hard-coded)
resource_path = sys.argv[1]
n_cfg = sys.argv[2]

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

out_name = f"pion_2pt_ubar_g5_d_cfg{n_cfg}.txt"
out_path = os.path.join(os.getcwd(), out_name)

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
# Exact channel: O(x) = \bar{u}(x) gamma_5 d(x), with degenerate light propagator reuse.
# FROM generate_einsum (meson_2pt)
C2_t_local = contract('wtzyxCBba, wtzyxCBba -> t', prop_l.data.conj(), prop_l.data)
C2_t = core.gatherLattice(array.arrayAsNumpy(C2_t_local, backend="cupy"), [0, -1, -1, -1])

# 6. save the result
if core.getMPIRank() == 0:
    C2_t = np.asarray(C2_t, dtype=np.complex128).reshape(-1)
    t = np.arange(C2_t.shape[0], dtype=np.int32)
    out = np.column_stack((t, C2_t.real, C2_t.imag))
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
