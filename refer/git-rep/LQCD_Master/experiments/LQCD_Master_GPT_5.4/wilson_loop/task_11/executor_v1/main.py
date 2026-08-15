import os
import sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# Run: mpirun -n 4 python3 main.py ~/.cache 10000

# 1. parameter definitions (hard-coded)
resource_path = sys.argv[1]
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

R = 4
Tlen = 1
out_path = f"./wilson_loop_R4_T1_cfg{n_cfg}.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)

# 2. read gauge configuration
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()
Nc = gauge.latt_info.Nc

# 3. construct the Dirac operator (skip if gauge-only)

# 4. compute forward propagators (skip if gauge-only)

# 5. extract observable / compute contraction (task-specific)
path_XT = [X, X, X, X, T, -X, -X, -X, -X, -T]
path_YT = [Y, Y, Y, Y, T, -Y, -Y, -Y, -Y, -T]
path_ZT = [Z, Z, Z, Z, T, -Z, -Z, -Z, -Z, -T]

groups = [[path_XT], [path_YT], [path_ZT], [path_XT]]
weights = [1, 1, 1, 0]

loop_fields = gauge.loop(groups, weights)

total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]
grid = core.getGridSize()
local_shape = (
    2,
    latt_size[3] // grid[3],
    latt_size[2] // grid[2],
    latt_size[1] // grid[1],
    (latt_size[0] // grid[0]) // 2,
)

W_planes = []
for i in [0, 1, 2]:
    link_host = loop_fields[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(link_host, axis1=-2, axis2=-1).real
    re_tr_field = re_tr.reshape(local_shape)
    re_tr_global = core.gatherLattice(re_tr_field, [-1, -1, -1, -1])
    if core.getMPIRank() == 0:
        S_global = re_tr_global.sum()
        W_planes.append(S_global / (Nc * total_sites))

# 6. save the result
if core.getMPIRank() == 0:
    W_avg = (W_planes[0] + W_planes[1] + W_planes[2]) / 3.0
    with open(out_path, "w") as f:
        f.write(f"{W_avg:.16e}\n")
