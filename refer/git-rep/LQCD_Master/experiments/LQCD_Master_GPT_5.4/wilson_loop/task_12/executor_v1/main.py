import sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# Run: mpirun -n 4 python3 main.py ~/.cache 10000

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
Nc = 3
R_values = [1, 2, 3, 4]
T_values = [1, 2, 3, 4]

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
out_path = f"wilson_loops_rt_cfg{n_cfg}.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

results = []
grid = core.getGridSize()
local_shape = (
    2,
    latt_size[3] // grid[3],
    latt_size[2] // grid[2],
    latt_size[1] // grid[1],
    (latt_size[0] // grid[0]) // 2,
)
total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]
local_sites = np.prod(local_shape)

for R in R_values:
    for Tlen in T_values:
        path_xt = [X] * R + [T] * Tlen + [-X] * R + [-T] * Tlen
        path_yt = [Y] * R + [T] * Tlen + [-Y] * R + [-T] * Tlen
        path_zt = [Z] * R + [T] * Tlen + [-Z] * R + [-T] * Tlen

        loops = gauge.loop([[path_xt], [path_yt], [path_zt], [path_xt]], [1, 1, 1, 0])

        plane_sum = None
        for i in range(3):
            host = loops[i].getHost()
            if host.shape[-2:] != (Nc, Nc):
                raise ValueError(f"Unexpected color shape for gauge.loop output: {host.shape}")
            if host.size != local_sites * Nc * Nc:
                raise ValueError(f"Unexpected local volume for gauge.loop output: {host.shape}")
            re_tr = np.trace(host.reshape(-1, Nc, Nc), axis1=-2, axis2=-1).real
            field = re_tr.reshape(local_shape)
            plane_sum = field if plane_sum is None else plane_sum + field

        plane_avg = plane_sum / (3.0 * Nc)
        gathered = core.gatherLattice(plane_avg, [-1, -1, -1, -1])

        if core.getMPIRank() == 0:
            value = float(gathered.sum()) / total_sites
            results.append((R, Tlen, value))

if core.getMPIRank() == 0:
    if len(results) != 16:
        raise ValueError(f"Expected 16 Wilson loop values, got {len(results)}")
    out = np.asarray(results, dtype=np.float64)
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e"])
    with open(out_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    if line_count != 16:
        raise ValueError(f"Expected 16 output lines, got {line_count}")
