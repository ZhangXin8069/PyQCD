import os
import sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# Run: mpirun -n 4 python3 main.py <resource_path> <n_cfg>

# 1. parameter definitions (hard-coded)
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
Nc = 3
R = 2
Tlen = 3
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
out_path = "wilson_loop_R2_T3_avg.txt"

resource_path = sys.argv[1]
n_cfg = sys.argv[2]

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

# 2. read gauge configuration
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# 3. construct the Dirac operator (skip if gauge-only)

# 4. compute forward propagators (skip if gauge-only)

# 5. extract observable / compute contraction
path_xt = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
path_yt = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
path_zt = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

# gauge.loop requires exactly four outer groups in PyQUDA
loop_result = gauge.loop([[path_xt], [path_yt], [path_zt], [path_xt]], [1, 1, 1, 0])

re_tr_sum = None
for i in range(3):
    link_host = loop_result[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(link_host, axis1=-2, axis2=-1).real
    if re_tr_sum is None:
        re_tr_sum = re_tr
    else:
        re_tr_sum = re_tr_sum + re_tr

re_tr_avg = re_tr_sum / 3.0

grid = core.getGridSize()
local_shape = (
    2,
    latt_size[3] // grid[3],
    latt_size[2] // grid[2],
    latt_size[1] // grid[1],
    (latt_size[0] // grid[0]) // 2,
)
re_tr_field = re_tr_avg.reshape(local_shape)
re_tr_global = core.gatherLattice(re_tr_field, [-1, -1, -1, -1])

# 6. save the result
if core.getMPIRank() == 0:
    volume = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]
    wilson_loop_avg = float(re_tr_global.sum()) / (volume * Nc)
    with open(out_path, "w") as f:
        f.write(f"{wilson_loop_avg:.16e}\n")
