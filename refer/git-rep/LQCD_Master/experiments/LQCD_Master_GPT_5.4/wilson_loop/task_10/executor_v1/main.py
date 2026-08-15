import numpy as np
import sys
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# Run: mpirun -n 4 python3 main.py ~/.cache 10000

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
Nc = 3
R = 1
Tlen = 4

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
out_name_template = "wilson_loop_R1_T4_avg_cfg{n_cfg}.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

path_XT = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
path_YT = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
path_ZT = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

res = gauge.loop([[path_XT], [path_YT], [path_ZT], [path_XT]], [1, 1, 1, 0])

grid = core.getGridSize()
local_shape = (
    2,
    latt_size[3] // grid[3],
    latt_size[2] // grid[2],
    latt_size[1] // grid[1],
    (latt_size[0] // grid[0]) // 2,
)
total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]

plane_values = []
for i in range(3):
    U = res[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(U, axis1=-2, axis2=-1).real
    re_tr_field = re_tr.reshape(local_shape)
    re_tr_global = core.gatherLattice(re_tr_field, [-1, -1, -1, -1])
    if core.getMPIRank() == 0:
        plane_values.append(float(re_tr_global.sum()) / (total_sites * Nc))

if core.getMPIRank() == 0:
    w_avg = (plane_values[0] + plane_values[1] + plane_values[2]) / 3.0
    out_path = out_name_template.format(n_cfg=n_cfg)
    np.savetxt(out_path, np.array([w_avg], dtype=np.float64), fmt="%.16e")
