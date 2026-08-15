import os
import sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# Run: mpirun -n 4 python3 main.py ~/.cache 10000

# 1. parameter definitions (hard-coded)
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

R = 1
Tlen = 2
out_filename = f"wilson_loop_R1_T2_avg_xt_yt_zt_cfg{n_cfg}.txt"

# 2. read gauge configuration
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# 3. construct the Dirac operator (skip if gauge-only)

# 4. compute forward propagators (skip if gauge-only)

# 5. extract observable / compute contraction
path_xt = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
path_yt = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
path_zt = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

groups = [[path_xt], [path_yt], [path_zt], [path_xt]]
weights = [1, 1, 1, 0]
res = gauge.loop(groups, weights)

Nc = getattr(gauge.latt_info, "Nc", 3)
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
    loop_host = res[i].getHost()
    loop_flat = loop_host.reshape(-1, Nc, Nc)
    re_tr = np.trace(loop_flat, axis1=-2, axis2=-1).real
    local_field = re_tr.reshape(local_shape)
    global_field = core.gatherLattice(local_field, [-1, -1, -1, -1])
    if core.getMPIRank() == 0:
        plane_values.append(global_field.sum() / (total_sites * Nc))

# 6. save the result
if core.getMPIRank() == 0:
    wilson_loop_value = (plane_values[0] + plane_values[1] + plane_values[2]) / 3.0
    with open(out_filename, "w") as f:
        f.write(f"{wilson_loop_value:.16e}\n")
