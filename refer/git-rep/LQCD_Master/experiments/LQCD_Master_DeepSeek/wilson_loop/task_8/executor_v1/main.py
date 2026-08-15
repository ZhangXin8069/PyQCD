# Run: mpirun -np 4 python3 main.py ~/.cache 10000
import sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# ── Parameters ────────────────────────────────────────────
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
Nc = 3
R = 3
Tlen = 2

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

resource_path = sys.argv[1]
n_cfg = sys.argv[2]

# ── Initialize PyQUDA ─────────────────────────────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

# ── Load gauge configuration (unsmeared raw links) ────────
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# ── Build Wilson loop paths for XT, YT, ZT planes ────────
#  R=3 spatial steps forward, T=2 temporal steps forward,
#  R=3 spatial steps backward, T=2 temporal steps backward.
path_XT = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
path_YT = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
path_ZT = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

# gauge.loop() requires exactly 4 outer groups.
# Three active planes (XT, YT, ZT) with weight 1; 4th dummy with weight 0.
res = gauge.loop(
    [[path_XT], [path_YT], [path_ZT], [path_XT]],
    [1, 1, 1, 0],
)

# ── Extract per-site ReTr and average over 3 planes ───────
re_tr_sum = None
for i in range(3):
    # Transfer LatticeLink to CPU, reshape to per-site Nc×Nc matrices
    U = res[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(U, axis1=-2, axis2=-1).real
    re_tr_sum = re_tr if re_tr_sum is None else re_tr_sum + re_tr

re_tr_avg = re_tr_sum / 3.0

# ── MPI gather: sum across all ranks to rank 0 ────────────
grid = core.getGridSize()
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

# ── Compute final Wilson loop value and save ──────────────
if core.getMPIRank() == 0:
    W_val = float(global_sum.sum()) / (total_sites * Nc)
    with open("wl_R3_T2.txt", "w") as f:
        f.write(str(W_val) + "\n")
