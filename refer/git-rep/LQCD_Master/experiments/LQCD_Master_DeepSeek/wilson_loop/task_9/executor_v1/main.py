# Compute rectangular Wilson loop W(R=3,T=3) averaged over XT, YT, ZT planes
# Pure-gauge measurement on unsmeared gauge links — no quark propagators
# Run: mpirun -np 4 python main.py ~/.cache 10000

import os, sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# ═══════════════════════════════════════════════════════════════
# 1. Parameter definitions (hard-coded)
# ═══════════════════════════════════════════════════════════════
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
Nc = 3
R = 3
Tlen = 3

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# ── Runtime arguments: resource_path and configuration number ──
resource_path = sys.argv[1]
n_cfg = sys.argv[2]

out_filename = f"wl_R3_T3_cfg{n_cfg}.txt"

# ═══════════════════════════════════════════════════════════════
# 2. Read gauge configuration
# ═══════════════════════════════════════════════════════════════
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# ═══════════════════════════════════════════════════════════════
# 3-4. Skipped — pure-gauge measurement, no Dirac operator or
#       quark propagators
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# 5. Extract observable — Wilson loop W(R=3, T=3) on XT, YT, ZT
# ═══════════════════════════════════════════════════════════════

# Build closed rectangular paths:
#   R steps along μ → T steps along ν → R steps along -μ → T steps along -ν
path_XT  = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
path_YT  = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
path_ZT  = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

# PyQUDA gauge.loop() requires exactly 4 outer groups.
# 4th group repeats path_XT as a dummy with weight 0.
groups  = [[path_XT], [path_YT], [path_ZT], [path_XT]]
weights = [1, 1, 1, 0]

res = gauge.loop(groups, weights)

# Extract per-site ReTr from each active plane (i = 0, 1, 2)
re_tr_planes = []
for i in range(3):
    U = res[i].getHost()                          # GPU → CPU
    U_flat = U.reshape(-1, Nc, Nc)                # flatten to (N_sites_local, 3, 3)
    re_tr = np.trace(U_flat, axis1=-2, axis2=-1).real
    re_tr_planes.append(re_tr)

# Plane average: element-wise mean over XT, YT, ZT
re_tr_avg = (re_tr_planes[0] + re_tr_planes[1] + re_tr_planes[2]) / 3.0

# ═══════════════════════════════════════════════════════════════
# 6. MPI gather and save result
# ═══════════════════════════════════════════════════════════════

# Reshape to PyQUDA even-odd local-lattice layout for gatherLattice
# local_shape = (parity, Lt_local, Lz, Ly, Lx_local//2)
grid = core.getGridSize()  # [1, 1, 1, 4]
local_shape = (
    2,                                        # parity (even/odd)
    latt_size[3] // grid[3],                  # Lt_local = 72 // 4 = 18
    latt_size[2],                             # Lz = 24
    latt_size[1],                             # Ly = 24
    (latt_size[0] // grid[0]) // 2,           # Lx_local//2 = 12
)

field = re_tr_avg.reshape(local_shape)
global_sum = core.gatherLattice(field, [-1, -1, -1, -1])

if core.getMPIRank() == 0:
    total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]
    W_val = float(global_sum.sum()) / (total_sites * Nc)
    with open(out_filename, "w") as f:
        f.write(str(W_val))
    print(f"W(R={R}, T={Tlen}) = {W_val:.12e}")
    print(f"Saved to {out_filename}")
