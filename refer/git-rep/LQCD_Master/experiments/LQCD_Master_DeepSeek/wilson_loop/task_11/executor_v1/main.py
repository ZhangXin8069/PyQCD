# Compute rectangular Wilson loop W(R=4, T=1) averaged over XT, YT, ZT planes.
# Pure-gauge measurement on unsmeared gauge links.
# Run: mpirun -n 4 python main.py <resource_path> <cfg_number>
import sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# ── 1. Hard-coded parameters ──────────────────────────────────
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
Nc = 3
R = 4
Tlen = 1

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

out_filename = "wl_R4_T1.txt"

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

# ── 2. Initialise PyQUDA and load gauge configuration ─────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# No link smearing — use original unsmeared gauge links directly.

# ── 3. Build Wilson loop paths ────────────────────────────────

# Build rectangular paths for three spatial-temporal planes.
# Each path: R steps forward in mu, Tlen steps forward in nu,
#            R steps backward in mu, Tlen steps backward in nu.
path_XT = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
path_YT = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
path_ZT = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

# gauge.loop() requires exactly 4 outer groups.
# Group 3 is a dummy (XT path again) with weight 0.
res = gauge.loop(
    [[path_XT], [path_YT], [path_ZT], [path_XT]],
    [1, 1, 1, 0],
)

# ── 4. Extract per-site ReTr from each active plane ───────────
re_tr_sum = None
for i in range(3):
    U = res[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(U, axis1=-2, axis2=-1).real
    re_tr_sum = re_tr if re_tr_sum is None else re_tr_sum + re_tr

# Average over the three planes element-wise.
re_tr_avg = re_tr_sum / 3.0

# Reshape to field layout for MPI gather.
# Local shape: [2, Lt_local, Lz, Ly, Lx//2]
grid = core.getGridSize()
local_shape = (
    2,
    latt_size[3] // grid[3],
    latt_size[2],
    latt_size[1],
    (latt_size[0] // grid[0]) // 2,
)
re_tr_field = re_tr_avg.reshape(local_shape)

# MPI gather: sum contributions over all 4 ranks onto rank 0.
global_sum = core.gatherLattice(re_tr_field, [-1, -1, -1, -1])

total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]

# ── 5. Save result ────────────────────────────────────────────
if core.getMPIRank() == 0:
    W_val = float(global_sum.sum()) / (total_sites * Nc)
    with open(out_filename, "w") as f:
        f.write(str(W_val))
