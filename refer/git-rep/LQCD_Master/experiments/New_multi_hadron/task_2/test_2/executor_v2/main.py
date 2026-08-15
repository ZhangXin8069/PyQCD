import os
import sys
import numpy as np
from pyquda_utils import core, io, source

# Run: mpirun -n 8 python3 main.py ~/.cache 10000

resource_path = sys.argv[1]
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 2, 2, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

mass_l = -0.277
xi_0 = 1.0
csw = 1.160920226
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

x_src = [0, 0, 0, 0]
obs_name = "local_9q_npn_two_point_function"
out_dir = "."
sink_file = "sink_local_9q_npn_two_point_function.py"

two_pt_result = None
C_t = None
corr = None

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

dirac_l = core.getClover(latt_info, mass_l, tol, maxiter, xi_0, csw, csw, multigrid)

pt_src = source.source12(latt_info, "point", x_src)
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

exec(open(sink_file).read())

if two_pt_result is not None:
    two_pt_root = two_pt_result
elif C_t is not None:
    two_pt_root = C_t
elif corr is not None:
    two_pt_root = corr
else:
    two_pt_root = None

if core.getMPIRank() == 0:
    two_pt_root = np.asarray(two_pt_root, dtype=np.complex128).reshape(-1)
    t = np.arange(two_pt_root.shape[0], dtype=np.int32)
    out = np.column_stack((t, two_pt_root.real, two_pt_root.imag))
    out_path = os.path.join(out_dir, f"{obs_name}_cfg{int(n_cfg):05d}.txt")
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
