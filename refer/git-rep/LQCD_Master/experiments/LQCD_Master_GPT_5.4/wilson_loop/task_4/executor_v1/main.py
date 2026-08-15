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
Nc = 3

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

R = 2
Tlen = 2
out_dir = "."

# Pure-gauge Wilson loop measurement uses only gauge links.
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
out_path = os.path.join(out_dir, f"wilson_loop_R{R}_T{Tlen}_cfg_{n_cfg}.txt")

# 2. read gauge configuration
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# 3. construct the Dirac operator (skip for gauge-only task)

# 4. compute forward propagators (skip for gauge-only task)

# 5. extract observable / compute contraction
# Rectangular Wilson loop W(R=2,T=2), averaged over XT, YT, ZT planes.
path_XT = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
path_YT = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
path_ZT = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

# gauge.loop() expects four outer groups in PyQUDA; the fourth entry is padded with zero weight.
res = gauge.loop([[path_XT], [path_YT], [path_ZT], [path_XT]], [1, 1, 1, 0])

re_tr_sum = None
for i in range(3):
    U = res[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(U, axis1=-2, axis2=-1).real
    if re_tr_sum is None:
        re_tr_sum = re_tr
    else:
        re_tr_sum = re_tr_sum + re_tr

re_tr_avg = re_tr_sum / 3.0

grid = core.getGridSize()
local_shape = (
    2,
    latt_size[3] // grid[3],
    latt_size[2],
    latt_size[1],
    (latt_size[0] // grid[0]) // 2,
)
re_tr_field = re_tr_avg.reshape(local_shape)
global_sum = core.gatherLattice(re_tr_field, [-1, -1, -1, -1])

total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]

# 6. save the result
if core.getMPIRank() == 0:
    wilson_loop_value = float(global_sum.sum()) / (total_sites * Nc)
    np.savetxt(out_path, np.asarray([wilson_loop_value], dtype=np.float64), fmt="%.16e")
