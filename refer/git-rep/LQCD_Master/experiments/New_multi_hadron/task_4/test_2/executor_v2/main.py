import os
import sys
import numpy as np
from pyquda_utils import core, io, source

# Run with MPI, e.g.:
# mpirun -n 8 python3 main.py ~/.cache 10000

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 2, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
source_position = [0, 0, 0, 0]

anisotropy = 1.0
xi_0 = 1.0
csw = 1.160920226
mass_l = -0.277
mass_s = -0.2356
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_n_steps = 1
stout_rho = 0.125
stout_ndim = 4

out_path = "./three_baryon_local_9q_corr.txt"
cfg_path = cfg_path_template.format(n_cfg=n_cfg)

core.init(grid_size=grid_size, latt_size=latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_n_steps, stout_rho, stout_ndim)

dirac_l = core.getClover(latt_info, mass_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, mass_s, tol, maxiter, xi_0, csw, csw, multigrid)

point_src = source.source12(latt_info, "point", source_position)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, point_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, point_src)

sink_file = "sink_three_baryon_local_9q_corr.py"
exec(open(sink_file).read())

if core.getMPIRank() == 0:
    corr = np.asarray(two_pt_result, dtype=np.complex128).reshape(-1)
    t = np.arange(corr.shape[0], dtype=np.int32)
    out = np.column_stack((t, corr.real, corr.imag))
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
