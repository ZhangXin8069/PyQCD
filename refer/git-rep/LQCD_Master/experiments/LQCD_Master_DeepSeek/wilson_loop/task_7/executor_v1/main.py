# Run: mpirun -np 4 python3 main.py ~/.cache 10000
import sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# ── Parameters ────────────────────────────────────────────
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
Nc = 3
R = 2
Tlen = 3
cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# ── Runtime arguments ─────────────────────────────────────
resource_path = sys.argv[1]
n_cfg = sys.argv[2]

# ── Initialize PyQUDA (MPI + GPU) ─────────────────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

# ── Load unsmeared gauge configuration ────────────────────
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# ── Build Wilson loop paths: R=2 forward, T=3 forward ─────
#     then R=2 backward, T=3 backward to close the rectangle.
# XT plane: forward in X then T, backward in X then T
path_XT = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
# YT plane
path_YT = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
# ZT plane
path_ZT = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

# gauge.loop() demands exactly 4 outer groups; pad 4th as dummy weight 0
res = gauge.loop(
    [[path_XT], [path_YT], [path_ZT], [path_XT]],
    [1, 1, 1, 0],
)

# ── Per-site extraction: ReTr for each of 3 active planes ─
re_tr_sum = None
for i in range(3):
    # res[i] is a LatticeLink: [2, Lt, Lz, Ly, Lx//2, Nc, Nc]
    U = res[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(U, axis1=-2, axis2=-1).real
    if re_tr_sum is None:
        re_tr_sum = re_tr
    else:
        re_tr_sum = re_tr_sum + re_tr

# Average over the three spatial-temporal planes
re_tr_avg = re_tr_sum / 3.0

# ── MPI gather: sum per-site contributions to rank 0 ──────
# Reshape into local lattice field: [parity, t, z, y, x//2]
local_shape = (
    2,
    latt_size[3] // grid_size[3],
    latt_size[2],
    latt_size[1],
    (latt_size[0] // grid_size[0]) // 2,
)
field = re_tr_avg.reshape(local_shape)
# [-1, -1, -1, -1] reduces all four space-time dimensions
global_sum = core.gatherLattice(field, [-1, -1, -1, -1])

total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]

# ── Save result as one-line plain text ────────────────────
if core.getMPIRank() == 0:
    W_val = float(global_sum.sum()) / (total_sites * Nc)
    out_path = f"wilson_loop_R{R}_T{Tlen}_cfg{n_cfg}.txt"
    with open(out_path, "w") as f:
        f.write(f"{W_val:.16e}\n")
    print(f"W(R={R}, T={Tlen}) = {W_val:.16e}  ->  {out_path}")
