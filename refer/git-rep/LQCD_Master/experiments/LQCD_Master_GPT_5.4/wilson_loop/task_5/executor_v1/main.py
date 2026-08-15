import sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# Run: mpirun -n 4 python3 main.py ~/.cache 10000

resource_path = sys.argv[1]
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
Nc = 3
R = 1
T_len = 3

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
out_path = f"wilson_loop_R1_T3_cfg{n_cfg}.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Rectangular Wilson loop W(R=1,T=3), averaged over XT, YT, ZT planes
path_XT = [X, T, T, T, -X, -T, -T, -T]
path_YT = [Y, T, T, T, -Y, -T, -T, -T]
path_ZT = [Z, T, T, T, -Z, -T, -T, -T]

groups = [[path_XT], [path_YT], [path_ZT], [path_XT]]
weights = [1, 1, 1, 0]

loop_fields = gauge.loop(groups, weights)

plane_sum_local = None
for i in range(3):
    U_host = loop_fields[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(U_host, axis1=-2, axis2=-1).real
    if plane_sum_local is None:
        plane_sum_local = re_tr
    else:
        plane_sum_local += re_tr

plane_avg_local = plane_sum_local / 3.0

grid = core.getGridSize()
local_shape = (
    2,
    latt_size[3] // grid[3],
    latt_size[2] // grid[2],
    latt_size[1] // grid[1],
    (latt_size[0] // grid[0]) // 2,
)
plane_avg_field = plane_avg_local.reshape(local_shape)
plane_avg_global = core.gatherLattice(plane_avg_field, [-1, -1, -1, -1])

if core.getMPIRank() == 0:
    total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]
    wilson_loop_value = float(plane_avg_global.sum()) / (total_sites * Nc)
    np.savetxt(out_path, np.asarray([wilson_loop_value]), fmt="%.16e")
