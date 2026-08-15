# Run: mpirun -np 4 python main.py ~/.cache 10000
import sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# ── Physical and lattice parameters ────────────────────────
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]          # MPI partitioning: 4 ranks in t-direction
Nc = 3
R = 3                             # spatial extent of Wilson loop
Tlen = 1                          # temporal extent of Wilson loop

# ── I/O paths ──────────────────────────────────────────────
cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/"
    "Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)
out_file = "wilson_loop_R3_T1.txt"

# ── Read command-line arguments ────────────────────────────
resource_path = sys.argv[1]
n_cfg = sys.argv[2]

# ── Initialize PyQUDA with autotuning cache ───────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

# ── Load gauge configuration (Chroma QIO LIME format) ─────
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# ── No link smearing — use original unsmeared gauge links ──

# ── Construct Wilson loop paths (R=3, T=1) ─────────────────
# Each path: 3 forward spatial → 1 forward time →
#            3 backward spatial → 1 backward time  (8 hops total)
path_XT = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
path_YT = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
path_ZT = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

# ── Evaluate with gauge.loop() (4-group packing convention) ─
# Only first 3 groups are active (XT, YT, ZT); 4th is a dummy with weight 0
res = gauge.loop(
    [[path_XT], [path_YT], [path_ZT], [path_XT]],
    [1, 1, 1, 0]
)

# ── Extract per-site ReTr and average over the 3 planes ───
re_tr_sum = None
for i in range(3):
    # Transfer LatticeLink GPU→CPU, flatten sites, trace over color
    U = res[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(U, axis1=-2, axis2=-1).real
    re_tr_sum = re_tr if re_tr_sum is None else re_tr_sum + re_tr
re_tr_avg = re_tr_sum / 3.0

# ── MPI gather: reduce per-site ReTr across all ranks ──────
grid = core.getGridSize()
local_shape = (
    2,                              # parity (even/odd)
    latt_size[3] // grid[3],       # Lt local
    latt_size[2],                   # Lz
    latt_size[1],                   # Ly
    (latt_size[0] // grid[0]) // 2  # Lx//2 (even-odd preconditioned)
)
field = re_tr_avg.reshape(local_shape)
global_sum = core.gatherLattice(field, [-1, -1, -1, -1])

# ── Normalize to obtain W(R=3, T=1) on rank 0 ─────────────
total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]

if core.getMPIRank() == 0:
    W_avg = float(global_sum.sum()) / (total_sites * Nc)
    # Write bare scalar with no header or metadata
    with open(out_file, "w") as f:
        f.write(str(W_avg))
