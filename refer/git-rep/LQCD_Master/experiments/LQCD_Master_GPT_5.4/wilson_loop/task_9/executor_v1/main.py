import sys
import math
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# Run: mpirun -n 4 python3 main.py ~/.cache 10000

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
Nc = 3
R = 3
Tlen = 3
local_shape = (2, 18, 24, 24, 12)
volume = 24 * 24 * 24 * 72

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
out_path_template = "wilson_loop_R3_T3_avg_cfg{n_cfg}.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
out_path = out_path_template.format(n_cfg=n_cfg)

gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

path_XT = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
path_YT = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
path_ZT = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

res = gauge.loop([[path_XT], [path_YT], [path_ZT], [path_XT]], [1, 1, 1, 0])

plane_values = []
for i in [0, 1, 2]:
    U = res[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(U, axis1=-2, axis2=-1).real
    field = re_tr.reshape(local_shape)
    global_field = core.gatherLattice(field, [-1, -1, -1, -1])
    if core.getMPIRank() == 0:
        plane_values.append(float(global_field.sum()) / float(volume * Nc))

if core.getMPIRank() == 0:
    if len(plane_values) == 3:
        finite_ok = True
        for value in plane_values:
            if not math.isfinite(value):
                finite_ok = False
        if finite_ok:
            w_avg = (plane_values[0] + plane_values[1] + plane_values[2]) / 3.0
            with open(out_path, "w") as f:
                f.write(f"{w_avg:.16e}\n")
