import os
import sys
import numpy as np
import cupy as cp
from pyquda_utils import core, io, source

# Run with: mpirun -n 8 python3 main.py ~/.cache 10000

resource_path = sys.argv[1]
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 2, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

x_src = [0, 0, 0, 0]

anisotropy = 1.0
xi_0 = 1.0
clover_coeff = 1.160920226
mass_l = -0.277
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

sink_file = "sink_uud_uds_local_9q_2pt.py"
out_path = f"multi_hadron_2pt_uud_uuds_cfg{int(n_cfg):05d}.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)

gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

dirac_l = core.getClover(latt_info, mass_l, tol, maxiter, xi_0, clover_coeff, clover_coeff, multigrid)

pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

exec(open(sink_file).read())

if core.getMPIRank() == 0:
    two_pt_array = np.asarray(two_pt_result, dtype=np.complex128).reshape(-1)
    t = np.arange(two_pt_array.shape[0], dtype=np.int32)
    out = np.column_stack((t, two_pt_array.real, two_pt_array.imag))
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
