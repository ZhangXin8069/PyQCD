# Run: mpirun -np 4 python3 main.py ~/.cache 10000
import os, sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# ── Parameters ──────────────────────────────────────────────────────
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]           # 4 MPI ranks in t-direction
Nc = 3
R = 1                              # spatial extent of Wilson loop
Tlen = 1                           # temporal extent of Wilson loop

cfg_path_template = (
    "/public/share/weiwang/clqcd/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# ── Runtime arguments ────────────────────────────────────────────────
resource_path = sys.argv[1]        # QUDA tunecache directory
n_cfg = int(sys.argv[2])           # configuration number

# ── Initialize PyQUDA ────────────────────────────────────────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── Load gauge configuration (NO smearing) ───────────────────────────
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()
# Explicitly no stoutSmear / APE / HYP — use raw unsmeared links

# ── Build Wilson loop paths: R=1, T=1 for three spatial-temporal planes ─
#   XT plane: forward x → forward t → backward x → backward t
path_XT = [X, T, -X, -T]
#   YT plane
path_YT = [Y, T, -Y, -T]
#   ZT plane
path_ZT = [Z, T, -Z, -T]

# gauge.loop() requires exactly 4 outer groups; pad with dummy (weight=0)
res = gauge.loop(
    [[path_XT], [path_YT], [path_ZT], [path_XT]],
    [1.0, 1.0, 1.0, 0.0]
)

# ── Extract per-site Re Tr for each active plane (i=0,1,2) ───────────
re_tr_sum = None
for i in range(3):
    U = res[i].getHost()                     # GPU → CPU
    U_flat = U.reshape(-1, Nc, Nc)           # flatten all sites
    re_tr = np.trace(U_flat, axis1=-2, axis2=-1).real
    if re_tr_sum is None:
        re_tr_sum = re_tr
    else:
        re_tr_sum = re_tr_sum + re_tr

re_tr_avg = re_tr_sum / 3.0                  # plane-averaged per-site Re Tr

# ── MPI global reduction ──────────────────────────────────────────────
grid = core.getGridSize()
local_shape = (
    2,                                    # parity (even/odd)
    latt_size[3] // grid[3],              # Lt_local = 72 / 4 = 18
    latt_size[2],                         # Lz = 24
    latt_size[1],                         # Ly = 24
    (latt_size[0] // grid[0]) // 2,       # Lx_local/2 = 24/1/2 = 12
)
field = re_tr_avg.reshape(local_shape)
global_sum = core.gatherLattice(field, [-1, -1, -1, -1])  # reduce all dims

total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]  # 995328

# ── Save result: single float, no header ─────────────────────────────
if core.getMPIRank() == 0:
    W_val = float(global_sum.sum()) / (total_sites * Nc)
    out_path = f"wilson_loop_R1_T1_cfg{n_cfg}.txt"
    with open(out_path, "w") as f:
        f.write(f"{W_val:.16e}")
    print(f"W(R=1,T=1) = {W_val:.16e}  →  {out_path}")
