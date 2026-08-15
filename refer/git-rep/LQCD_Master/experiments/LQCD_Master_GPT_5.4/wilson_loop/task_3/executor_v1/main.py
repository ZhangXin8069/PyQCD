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
R = 2
Tlen = 1

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
out_path = os.path.join(os.getcwd(), "wilson_loop_R2_T1_avg.txt")

total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

# 2. read gauge configuration
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# 3. construct the Dirac operator (skip for gauge-only task)

# 4. compute forward propagators (skip for gauge-only task)

# 5. extract observable / compute contraction
# Rectangular Wilson loops W(R=2,T=1) in XT, YT, ZT planes.
path_xt = [X, X, T, -X, -X, -T]
path_yt = [Y, Y, T, -Y, -Y, -T]
path_zt = [Z, Z, T, -Z, -Z, -T]

# PyQUDA gauge.loop() requires exactly 4 outer groups.
loop_fields = gauge.loop([[path_xt], [path_yt], [path_zt], [path_xt]], [1, 1, 1, 0])

grid = core.getGridSize()
local_shape = (
    2,
    latt_size[3] // grid[3],
    latt_size[2],
    latt_size[1],
    (latt_size[0] // grid[0]) // 2,
)

plane_values = []
for i_plane in [0, 1, 2]:
    U_host = loop_fields[i_plane].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(U_host, axis1=-2, axis2=-1).real
    re_tr_field = re_tr.reshape(local_shape)
    re_tr_global = core.gatherLattice(re_tr_field, [-1, -1, -1, -1])
    if core.getMPIRank() == 0:
        plane_values.append(float(re_tr_global.sum()) / (total_sites * Nc))

# 6. save the result
if core.getMPIRank() == 0:
    w_avg = (plane_values[0] + plane_values[1] + plane_values[2]) / 3.0
    np.savetxt(out_path, np.asarray([w_avg], dtype=np.float64), fmt="%.16e")
