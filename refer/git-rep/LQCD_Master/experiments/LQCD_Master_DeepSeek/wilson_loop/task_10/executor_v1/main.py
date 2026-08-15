# Run: mpirun -np 4 python3 main.py ~/.cache 10000
import sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# ── 1. Parameter definitions ──────────────────────────────────
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
Nc = 3
R = 1
Tlen = 4

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)
out_path = "./W_R1_T4.txt"

# ── Read command-line arguments ───────────────────────────────
resource_path = sys.argv[1]
n_cfg = sys.argv[2]

# ── Initialize PyQUDA (MPI + GPU) ─────────────────────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

# ── 2. Read gauge configuration (unsmeared) ───────────────────
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# No smearing applied — use original gauge links directly

# ── 5. Extract observable: Wilson loop W(R=1, T=4) ────────────
# Build rectangular paths for XT, YT, ZT planes
# Each: forward spatial ×1 → forward temporal ×4 → backward spatial ×1 → backward temporal ×4
path_XT = [X] + [T] * Tlen + [-X] + [-T] * Tlen
path_YT = [Y] + [T] * Tlen + [-Y] + [-T] * Tlen
path_ZT = [Z] + [T] * Tlen + [-Z] + [-T] * Tlen

# gauge.loop() requires exactly 4 outer groups; 4th group is dummy with weight 0
res = gauge.loop([[path_XT], [path_YT], [path_ZT], [path_XT]], [1, 1, 1, 0])

# Extract per-site ReTr and average over the 3 active planes
re_tr_sum = None
for i in range(3):
    U = res[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(U, axis1=-2, axis2=-1).real
    re_tr_sum = re_tr if re_tr_sum is None else re_tr_sum + re_tr

re_tr_avg = re_tr_sum / 3.0

# MPI gather: sum over all lattice sites across all ranks
local_shape = (
    2,
    latt_size[3] // grid_size[3],
    latt_size[2],
    latt_size[1],
    (latt_size[0] // grid_size[0]) // 2,
)
field = re_tr_avg.reshape(local_shape)
global_sum = core.gatherLattice(field, [-1, -1, -1, -1])

total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]

# ── 6. Save the result ───────────────────────────────────────
if core.getMPIRank() == 0:
    W_val = float(global_sum.sum()) / (total_sites * Nc)
    with open(out_path, "w") as f:
        f.write(f"{W_val:.16e}\n")
