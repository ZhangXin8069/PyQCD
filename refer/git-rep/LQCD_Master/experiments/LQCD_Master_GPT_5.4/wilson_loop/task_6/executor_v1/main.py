import os
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
mpi_num = 4
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
out_path = os.path.join(os.getcwd(), "wilson_loop_R3_T1.txt")

R = 3
Tlen = 1

cfg_path = cfg_path_template.format(n_cfg=n_cfg)

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

if core.getMPISize() != mpi_num or core.getMPISize() != math.prod(grid_size):
    raise SystemExit

gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

path_XT = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
path_YT = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
path_ZT = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

res = gauge.loop([[path_XT], [path_YT], [path_ZT], [path_XT]], [1, 1, 1, 0])

Nc = gauge.latt_info.Nc

re_tr_sum = None
for i in range(3):
    U = res[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(U, axis1=-2, axis2=-1).real
    if re_tr_sum is None:
        re_tr_sum = re_tr
    else:
        re_tr_sum += re_tr

grid = core.getGridSize()
local_shape = (
    2,
    latt_size[3] // grid[3],
    latt_size[2] // grid[2],
    latt_size[1] // grid[1],
    (latt_size[0] // grid[0]) // 2,
)
field_local = re_tr_sum.reshape(local_shape)
field_global = core.gatherLattice(field_local, [-1, -1, -1, -1])

if core.getMPIRank() == 0:
    volume = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]
    wilson_loop_avg = float(field_global.sum()) / (3.0 * volume * Nc)
    with open(out_path, "w") as f:
        f.write(f"{wilson_loop_avg:.16e}\n")
