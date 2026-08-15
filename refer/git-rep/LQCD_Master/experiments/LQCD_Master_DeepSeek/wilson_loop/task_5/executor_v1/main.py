# Run: mpirun -np 4 python3 main.py <resource_path> <cfg_number>
#
# Computes rectangular Wilson loop W(R=1, T=3) averaged over XT, YT, ZT planes.
# Pure-gauge measurement — no quark propagators, Dirac solvers, or link smearing.

import sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# =============================================================================
# 1. Parameter definitions (hard-coded)
# =============================================================================
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]          # 4 MPI ranks split along t-direction
Nc = 3                             # SU(3) gauge group
R = 1                              # spatial extent of Wilson loop
Tlen = 3                           # temporal extent of Wilson loop

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Runtime arguments: resource_path and configuration number
resource_path = sys.argv[1]
n_cfg = sys.argv[2]

# =============================================================================
# 2. Read gauge configuration
# =============================================================================
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# =============================================================================
# 3. Compute Wilson loop W(R=1, T=3)
# =============================================================================
# No smearing applied — use raw unsmeared gauge links directly.

# Build rectangular Wilson loop paths for the three spatial-temporal planes.
# Convention: R steps forward along spatial direction, T steps forward along
# temporal direction, then R steps backward, T steps backward to close the loop.
path_XT = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
path_YT = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
path_ZT = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

# PyQUDA API requires exactly 4 groups in gauge.loop(); pack 3 active planes
# plus one dummy (weight 0) to satisfy the 4-group constraint.
res = gauge.loop(
    [[path_XT], [path_YT], [path_ZT], [path_XT]],
    [1, 1, 1, 0]
)

# Extract per-site Re Tr for each active plane (i = 0, 1, 2) and accumulate.
re_tr_sum = None
for i in range(3):
    # Transfer SU(3) matrix per site from GPU to CPU, flatten spatial dimensions
    U = res[i].getHost().reshape(-1, Nc, Nc)
    # Real part of the color trace at each lattice site
    re_tr = np.trace(U, axis1=-2, axis2=-1).real
    if re_tr_sum is None:
        re_tr_sum = re_tr
    else:
        re_tr_sum = re_tr_sum + re_tr

# Average over the three spatial-temporal planes (XT, YT, ZT)
re_tr_avg = re_tr_sum / 3.0

# =============================================================================
# 4. MPI gather and normalization
# =============================================================================
# Reshape to local lattice field layout: (parity, Lt_local, Lz, Ly, Lx//2)
# With grid [1,1,1,4]: Lt_local = 72/4 = 18, Lx//2 = 24/2 = 12
grid = core.getGridSize()
local_shape = (
    2,
    latt_size[3] // grid[3],
    latt_size[2],
    latt_size[1],
    (latt_size[0] // grid[0]) // 2,
)
field = re_tr_avg.reshape(local_shape)

# MPI gather: sum all spatial and temporal contributions to rank 0
global_sum = core.gatherLattice(field, [-1, -1, -1, -1])

total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]

if core.getMPIRank() == 0:
    # Normalize: W = (1/V) * (1/Nc) * sum_x Re Tr[U_loop(x)]
    W_val = float(global_sum.sum()) / (total_sites * Nc)
    # Per-config output — one file per configuration to avoid overwrites
    out_name = f"wilson_loop_R1_T3_cfg{n_cfg}.txt"
    with open(out_name, "w") as f:
        f.write(f"{W_val:.12f}\n")
