import os, sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# ── Physics parameters ──
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
Nc = 3

# Wilson loop sizes: R ∈ {1,2,3,4}, T ∈ {1,2,3,4}
R_values = [1, 2, 3, 4]
T_values = [1, 2, 3, 4]

# Gauge configuration path template
cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# ── Runtime arguments (passed by Slurm submission script) ──
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

# ── Step 1: Initialize PyQUDA ──
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

# ── Step 2: Load gauge configuration ──
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()  # transfer to GPU memory — required before gauge.loop()

# No link smearing is applied; use original unsmeared gauge links directly.

# ── Step 5: Compute Wilson loops (pure-gauge observable) ──
total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]

# Local lattice shape for gatherLattice after trace reduction:
#   (parity, Lt_local, Lz, Ly, Lx_local//2)
#   = (2, 72//4, 24, 24, (24//1)//2) = (2, 18, 24, 24, 12)
local_shape = (
    2,
    latt_size[3] // grid_size[3],
    latt_size[2],
    latt_size[1],
    (latt_size[0] // grid_size[0]) // 2,
)

results = []

for R in R_values:
    for Tlen in T_values:
        # Build Wilson loop paths for the three spatial-temporal planes
        # Path: R steps in μ, T steps in ν, R steps in -μ, T steps in -ν
        path_XT = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
        path_YT = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
        path_ZT = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

        # gauge.loop() requires exactly 4 outer groups; pad unused group with weight 0
        groups = [[path_XT], [path_YT], [path_ZT], [path_XT]]
        weights = [1, 1, 1, 0]

        res = gauge.loop(groups, weights)

        # Plane-average over XT, YT, ZT: sum ReTr from the first 3 result slots
        re_tr_sum = None
        for i in range(3):
            U = res[i].getHost().reshape(-1, Nc, Nc)  # GPU → CPU, flatten sites
            re_tr = np.trace(U, axis1=-2, axis2=-1).real
            if re_tr_sum is None:
                re_tr_sum = re_tr
            else:
                re_tr_sum = re_tr_sum + re_tr

        re_tr_avg = re_tr_sum / 3.0

        # Reshape to local lattice for MPI gather
        field = re_tr_avg.reshape(local_shape)
        global_sum = core.gatherLattice(field, [-1, -1, -1, -1])

        if core.getMPIRank() == 0:
            W_val = float(global_sum.sum()) / (total_sites * Nc)
            results.append((R, Tlen, W_val))

# ── Step 6: Save results as plain three-column text ──
if core.getMPIRank() == 0:
    out_path = os.path.join(os.getcwd(), f"wilson_loops_cfg{n_cfg}.txt")
    with open(out_path, "w") as f:
        for R, Tlen, W_val in results:
            f.write(f"{R} {Tlen} {W_val:.16e}\n")
    print(f"Saved {len(results)} Wilson loops to {out_path}")
