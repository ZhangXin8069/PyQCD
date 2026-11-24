#!/public/home/huangcl/.venv3_10/bin/python
"""读取 c32p29, P=4 的 3ptdisc 数据并画图."""

import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve()
project_dir = here.parents[1]
sys.path.insert(0, str(project_dir / "98_tools"))
from analysis_tools import calc_momentum_GeV  # noqa: E402  # type: ignore
from ratio_tools import plot_ratio_per_z  # noqa: E402  # type: ignore


CONF_SHORT = "c32p29"
MOM = 4
Z_LIST = [0, 1, 5, 10, 15]
DT_LIST = list(range(6, 12))
XLIM = [-7, 7]
YLIM = [-0.001, 0.005]
DATA_PATH = Path(
    "/public/home/huangcl/07_pion_gluon_pdf/test01_check_selfratio/01_data/c32p29/P4/3ptdisc.npy")
FIG_DIR = here.parent / "02_fig" / CONF_SHORT / f"P{MOM}"

p_gev = calc_momentum_GeV(MOM, CONF_SHORT)
print(f"loading 3ptdisc from: {DATA_PATH}")
# para: sample, dt, dtau, z
corr3_disc = np.load(DATA_PATH, mmap_mode="r")
print(f"loaded 3ptdisc shape: {corr3_disc.shape}")

for _z in Z_LIST:
    # para: sample, dt, dtau
    corr3_disc_z = corr3_disc[:, :, :, _z].real
    title = f"3ptdisc, z={_z}, |P|={p_gev:.3g}GeV, {CONF_SHORT}"
    fig_path = FIG_DIR / f"3ptdisc_{CONF_SHORT}_P{MOM}_z{_z}.png"
    plot_ratio_per_z(
        corr3_disc_z, fig_path, title, XLIM, YLIM, DT_LIST,
        ylabel=r"$\mathrm{Re}\,C_{3,\mathrm{disc}}$", name="3ptdisc",
    )
