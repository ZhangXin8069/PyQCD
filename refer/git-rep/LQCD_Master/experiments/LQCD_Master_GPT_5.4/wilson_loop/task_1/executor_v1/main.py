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
Nc = 3
R = 1
Tlen = 1
anisotropy = 1.0
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
out_path = f"wilson_loop_W_R1_T1_cfg{n_cfg}.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

# 2. read gauge configuration
gauge = io.readChromaQIOGauge(cfg_path_template.format(n_cfg=n_cfg))
gauge.toDevice()

# 3. construct the Dirac operator (skip for gauge-only task)

# 4. compute forward propagators (skip for gauge-only task)

# 5. extract observable / compute contraction
# W(R=1,T=1) averaged equally over XT, YT, and ZT planes.
path_XT = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
path_YT = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
path_ZT = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

# gauge.loop() requires exactly 4 outer groups; the last one is API padding only.
res = gauge.loop([[path_XT], [path_YT], [path_ZT], [path_XT]], [1, 1, 1, 0])

re_tr_sum = None
for i in range(3):
    U = res[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(U, axis1=-2, axis2=-1).real
    re_tr_sum = re_tr if re_tr_sum is None else (re_tr_sum + re_tr)

# Equal physics average over the three temporal planes.
re_tr_avg = re_tr_sum / 3.0

grid = core.getGridSize()
local_shape = (
    2,
    latt_size[3] // grid[3],
    latt_size[2] // grid[2],
    latt_size[1] // grid[1],
    (latt_size[0] // grid[0]) // 2,
)
re_tr_avg_field = re_tr_avg.reshape(local_shape)
global_sum = core.gatherLattice(re_tr_avg_field, [-1, -1, -1, -1])

# 6. save the result
if core.getMPIRank() == 0:
    total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]
    W_avg = float(global_sum.sum()) / (total_sites * Nc)
    with open(out_path, "w") as f:
        f.write(f"{W_avg:.16e}\n")
