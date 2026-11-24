#!/public/home/huangcl/.venv3_10/bin/python
"""绘制 z=0, tsep=10 时不同 tau 的逐样本 ratio."""

from pathlib import Path
import sys

import numpy as np

here = Path(__file__).resolve()
sys.path.insert(0, str(here.parents[1] / "98_tools"))
from figure_tools import plot_scatter  # noqa: E402  # type: ignore

CONF_SHORT = "c32p29"
MOM = 4
TSEP = 10
Z = 1
TAU_LIST = [0, 1, 2, 8, 9, 10]
DATA_PATH = Path(
    "/public/home/huangcl/07_pion_gluon_pdf/test01_check_selfratio/"
    "01_data/c32p29/P4/ratio.npy")
FIG_PATH = Path(
    "/public/home/huangcl/07_pion_gluon_pdf/test01_check_selfratio/"
    f"02_fig/c32p29/P4/ratio_scat_t{TSEP}_z{Z}.png")

# para: sample, dt, dtau, z
ratio = np.load(DATA_PATH)
print(f"loaded ratio shape: {ratio.shape}")
sample_id = np.arange(ratio.shape[0])
y_data = {
    f"tau={tau}": ratio[:, TSEP, tau, Z]
    for tau in TAU_LIST
}
plot_scatter(
    sample_id, y_data, FIG_PATH,
    xlabel="sample_id",
    ylabel="ratio",
    title=f"ratio, tsep={TSEP}, z={Z}, {CONF_SHORT}, P={MOM}",
    x_offset=0,
)
