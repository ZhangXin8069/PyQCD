import os
import sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

# Run: mpirun -n 4 python3 main.py ~/.cache 10000

# 1. parameter definitions (hard-coded)
resource_path = sys.argv[1]
n_cfg = str(sys.argv[2])

latt_size = [24, 24, 24, 72]
preferred_grid_size = [1, 1, 1, 4]
single_rank_grid_size = [1, 1, 1, 1]
backend = "cupy"

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

R = 3
Tlen = 2
Nc = 3
out_filename = f"wilson_loop_R3_T2_avg_cfg{n_cfg}.txt"
out_path = os.path.join(os.getcwd(), out_filename)

total_sites = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3]

mpi_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1")))
if mpi_size == 4:
    grid_size = preferred_grid_size
else:
    grid_size = single_rank_grid_size

core.init(grid_size, latt_size, backend=backend, resource_path=resource_path)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)

# 2. read gauge configuration
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# 3. construct the Dirac operator (skip if gauge-only)

# 4. compute forward propagators (skip if gauge-only)

# 5. extract observable / compute contraction
path_XT = [X, X, X, T, T, -X, -X, -X, -T, -T]
path_YT = [Y, Y, Y, T, T, -Y, -Y, -Y, -T, -T]
path_ZT = [Z, Z, Z, T, T, -Z, -Z, -Z, -T, -T]

res = gauge.loop([[path_XT], [path_YT], [path_ZT], [path_XT]], [1, 1, 1, 0])

grid = core.getGridSize()
local_shape = (
    2,
    latt_size[3] // grid[3],
    latt_size[2] // grid[2],
    latt_size[1] // grid[1],
    (latt_size[0] // grid[0]) // 2,
)

plane_values = []
for i in range(3):
    U = res[i].getHost().reshape(-1, Nc, Nc)
    re_tr = np.trace(U, axis1=-2, axis2=-1).real
    re_tr_field = re_tr.reshape(local_shape)

    if np.prod(grid) == 1:
        plane_value = float(re_tr_field.sum()) / float(total_sites * Nc)
    else:
        global_field = core.gatherLattice(re_tr_field, [-1, -1, -1, -1])
        if core.getMPIRank() == 0:
            plane_value = float(global_field.sum()) / float(total_sites * Nc)
        else:
            plane_value = None

    plane_values.append(plane_value)

W_XT = plane_values[0]
W_YT = plane_values[1]
W_ZT = plane_values[2]

if core.getMPIRank() == 0:
    plane_array = np.asarray([W_XT, W_YT, W_ZT], dtype=np.float64)
    np.isfinite(plane_array).all()
    W_avg = float((W_XT + W_YT + W_ZT) / 3.0)
    np.isfinite(W_avg)
    (-1.0 <= W_avg <= 1.0)

    # 6. save the result
    with open(out_path, "w") as f:
        f.write(f"{W_avg:.16e}\n")
