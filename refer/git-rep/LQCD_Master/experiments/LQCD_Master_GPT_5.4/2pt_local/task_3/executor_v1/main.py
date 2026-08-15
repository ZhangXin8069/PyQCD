import os
import sys
import numpy as np
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, source

# Run: mpirun -n 4 python3 main.py <resource_path> <cfg>

# 1. parameter definitions (hard-coded)
resource_path = sys.argv[1]
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
source_position = [0, 0, 0, 0]

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
out_file = f"eta_s_connected_2pt_cfg_{n_cfg}.txt"

anisotropy = 1.0
xi_0 = 1.0
mass_s = -0.2356
csw = 1.160920226
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

backend = "cupy"

core.init(grid_size, latt_size, backend=backend, resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

# 2. read gauge configuration
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# 3. construct the Dirac operator
dirac_s = core.getClover(latt_info, mass_s, tol, maxiter, xi_0, csw, csw, multigrid)

# 4. compute forward propagators
pt_src = source.source12(latt_info, "point", source_position)
with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# 5. extract observable / compute contraction
# connected eta_s proxy correlator: sum_x Tr[S_s^dagger(x,t;0) S_s(x,t;0)]
# FROM generate_einsum (meson_2pt)
C_t_local = contract('wtzyxCBba, wtzyxCBba -> t', prop_s.data.conj(), prop_s.data)
C_t = core.gatherLattice(array.arrayAsNumpy(C_t_local, backend=backend), [0, -1, -1, -1])

# 6. save the result
if core.getMPIRank() == 0:
    np.savetxt(out_file, np.asarray(C_t.real, dtype=np.float64).reshape(latt_size[3]), fmt='%.16e')
