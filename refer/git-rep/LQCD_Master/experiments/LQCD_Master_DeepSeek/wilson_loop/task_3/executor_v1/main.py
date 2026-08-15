# Run: mpirun -np 4 python3 main.py ~/.cache 10000
import os
import sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# ── Parameters ────────────────────────────────────────────
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
Nc = 3
R = 2
Tlen = 1

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

# ── Read command-line args ─────────────────────────────────
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

# ── Initialize PyQUDA ──────────────────────────────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

# ── Load gauge configuration (unsmeared) ───────────────────
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# No link smearing applied — use original gauge links directly.

# ── Build Wilson loop paths: R=2, T=1 ──────────────────────
# XT plane: forward x (2) → forward t (1) → backward x (2) → backward t (1)
path_XT = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
path_YT = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
path_ZT = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

# gauge.loop() requires exactly 4 outer groups; 4th is dummy with weight 0
res = gauge.loop([[path_XT], [path_YT], [path_ZT], [path_XT]], [1, 1, 1, 0])

# ── Extract per-site ReTr and average over 3 planes ────────
re_tr_sum = None
for i in range(3):
    U = res[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(U, axis1=-2, axis2=-1).real
    re_tr_sum = re_tr if re_tr_sum is None else re_tr_sum + re_tr

re_tr_avg = re_tr_sum / 3.0

# ── MPI gather to rank 0 ───────────────────────────────────
# Grid [1,1,1,4] splits t-direction over 4 ranks.
# Each rank: local_shape = (2, Lt/4, Lz, Ly, Lx//2) = (2, 18, 24, 24, 12)
grid = core.getGridSize()
local_shape = (2, latt_size[3] // grid[3], latt_size[2],
               latt_size[1], (latt_size[0] // grid[0]) // 2)
field = re_tr_avg.reshape(local_shape)
global_sum = core.gatherLattice(field, [-1, -1, -1, -1])

# ── Normalize and save result ──────────────────────────────
total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]

if core.getMPIRank() == 0:
    W_val = float(global_sum.sum()) / (total_sites * Nc)
    out_path = f"wl_R{R}_T{Tlen}_cfg{n_cfg}.txt"
    with open(out_path, "w") as f:
        f.write(str(W_val) + "\n")
