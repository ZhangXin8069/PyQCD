# Wilson loop W(R=2,T=2) averaged over XT, YT, ZT planes
# Pure-gauge measurement, no link smearing
# Run: mpirun -np 4 python3 main.py <resource_path> <cfg_num>

import sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# ── Parameters ────────────────────────────────────────────
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
Nc = 3
R = 2
Tlen = 2

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# ── Read configuration from command line ──────────────────
resource_path = sys.argv[1]
n_cfg = sys.argv[2]

# ── Initialize PyQUDA ─────────────────────────────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

# ── Load gauge configuration (no smearing) ────────────────
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# ── Construct Wilson loop paths for R=2, T=2 ──────────────
# XT plane: forward X^2 → forward T^2 → backward X^2 → backward T^2
path_XT = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
# YT plane
path_YT = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
# ZT plane
path_ZT = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

# gauge.loop() requires exactly 4 outer groups; 4th is dummy with weight 0
res = gauge.loop(
    [[path_XT], [path_YT], [path_ZT], [path_XT]],
    [1, 1, 1, 0],
)

# ── Extract per-site ReTr for each of the 3 active planes ─
re_tr_sum = None
for i in range(3):
    U = res[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(U, axis1=-2, axis2=-1).real
    re_tr_sum = re_tr if re_tr_sum is None else re_tr_sum + re_tr

# Average over XT, YT, ZT planes with equal weight 1/3
re_tr_avg = re_tr_sum / 3.0

# ── MPI gather: sum over all sublattices to rank 0 ────────
grid = core.getGridSize()
# Local field shape in even-odd layout: (2, Lt_local, Lz, Ly, Lx_local//2)
local_shape = (
    2,
    latt_size[3] // grid[3],
    latt_size[2],
    latt_size[1],
    (latt_size[0] // grid[0]) // 2,
)
field = re_tr_avg.reshape(local_shape)
global_sum = core.gatherLattice(field, [-1, -1, -1, -1])

total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]

# ── Save result: single float, no header ──────────────────
if core.getMPIRank() == 0:
    W_val = float(global_sum.sum()) / (total_sites * Nc)
    out_path = f"wl_R2_T2_cfg{n_cfg}.txt"
    with open(out_path, "w") as f:
        f.write(str(W_val))
