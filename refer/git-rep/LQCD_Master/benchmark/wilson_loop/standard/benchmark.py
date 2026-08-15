import os
import sys
import numpy as np

from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T as tdir

resource_path = sys.argv[1]
cfg = sys.argv[2]

ensemble = "C24P29"

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
backend = "cupy"

nc = 3

lx, ly, lz, lt = latt_size
volume = lx * ly * lz * lt

cfg_template = (
    "/public/share/weiwang/clqcd/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72/"
    "Configurations/Original/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{cfg}.lime"
)

out_dir = "./output_wilson_C24P29"
out_name = "wl_C24P29_cfg{cfg}.txt"

r_min, r_max = 1, 20
t_min, t_max = 1, 20



core.init(
    grid_size,
    latt_size,
    backend=backend,
    resource_path=resource_path
)

rank = core.getMPIRank()
mpi_size = core.getMPISize()
grid = core.getGridSize()

cfg_path = cfg_template.format(cfg=cfg)

gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()


def make_path(mu, nu, r_len, t_len):
    return [mu] * r_len + [nu] * t_len + [-mu] * r_len + [-nu] * t_len


lt_local = lt // grid[3]
lx_local_half = (lx // grid[0]) // 2
field_shape = (2, lt_local, lz, ly, lx_local_half)


def get_loop_value(loop_obj):
    mat = loop_obj.getHost().reshape(-1, nc, nc)
    tr = np.trace(mat, axis1=-2, axis2=-1)

    re_field = tr.real.reshape(field_shape)
    gathered = core.gatherLattice(re_field, [-1, -1, -1, -1])

    if rank == 0:
        return float(np.sum(gathered)) / float(volume * nc)

    return None


def calc_one_rt(r_len, t_len):
    path_xt = make_path(X, tdir, r_len, t_len)
    path_yt = make_path(Y, tdir, r_len, t_len)
    path_zt = make_path(Z, tdir, r_len, t_len)

    res = gauge.loop(
        [[path_xt], [path_yt], [path_zt], [path_xt]],
        [1.0, 1.0, 1.0, 0.0]
    )

    return (
        get_loop_value(res[0]),
        get_loop_value(res[1]),
        get_loop_value(res[2])
    )


if rank == 0:
    shape = (r_max - r_min + 1, t_max - t_min + 1)

    w_xt_all = np.zeros(shape, dtype=np.float64)
    w_yt_all = np.zeros(shape, dtype=np.float64)
    w_zt_all = np.zeros(shape, dtype=np.float64)
    w_avg_all = np.zeros(shape, dtype=np.float64)

    print("# R T w_xt w_yt w_zt w_avg")
else:
    w_xt_all = w_yt_all = w_zt_all = w_avg_all = None


for r_len in range(r_min, r_max + 1):
    for t_len in range(t_min, t_max + 1):

        w_xt, w_yt, w_zt = calc_one_rt(r_len, t_len)

        if rank == 0:
            ir = r_len - r_min
            it = t_len - t_min

            w_xt_all[ir, it] = w_xt
            w_yt_all[ir, it] = w_yt
            w_zt_all[ir, it] = w_zt
            w_avg_all[ir, it] = (w_xt + w_yt + w_zt) / 3.0

            print(
                f"{r_len:2d} {t_len:2d} "
                f"{w_xt:.16e} "
                f"{w_yt:.16e} "
                f"{w_zt:.16e} "
                f"{w_avg_all[ir, it]:.16e}"
            )


if rank == 0:
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, out_name.format(cfg=cfg))

    with open(out_path, "w") as f:
        f.write("# R T w_xt w_yt w_zt w_avg\n")

        for r_len in range(r_min, r_max + 1):
            for t_len in range(t_min, t_max + 1):
                ir = r_len - r_min
                it = t_len - t_min

                f.write(
                    f"{r_len:2d} {t_len:2d} "
                    f"{w_xt_all[ir, it]:.16e} "
                    f"{w_yt_all[ir, it]:.16e} "
                    f"{w_zt_all[ir, it]:.16e} "
                    f"{w_avg_all[ir, it]:.16e}\n"
                )

    print(f"# saved to {out_path}")
