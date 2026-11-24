#!/public/home/huangcl/.venv3_10/bin/python
"""剔除每个 ratio 误差棒中离中位数最远的样本后画图."""

import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve()
project_dir = here.parents[1]
sys.path.insert(0, str(project_dir / "98_tools"))
from output_tools import save_npy  # noqa: E402  # type: ignore

sys.path.insert(0, str(here.parent))
from ratio_tools import plot_ratio_per_z  # noqa: E402  # type: ignore

CONF_SHORT = "c32p29"
MOM = 4
Z_LIST = [0, 1, 2, 5, 10, 15]
DT_LIST = list(range(6, 12))
XLIM = [-7, 7]
YLIM = [-10, 10]
DATA_DIR = here.parent / "01_data" / CONF_SHORT / f"P{MOM}"
FIG_DIR = here.parent / "ex"


def exclude_farthest(ratio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐个 (dt, tau, z) 剔除离中位数最远的 sample."""
    filtered = ratio.copy()
    # para: dt, tau, z; -1 表示该位置不参与计算.
    excluded_ids = np.full(ratio.shape[1:], -1, dtype=int)
    for dt in range(ratio.shape[1]):
        for tau in range(dt + 1):
            for z in range(ratio.shape[3]):
                samples = ratio[:, dt, tau, z]
                sample_id = np.argmax(np.abs(samples - np.median(samples)))
                filtered[sample_id, dt, tau, z] = np.nan
                excluded_ids[dt, tau, z] = sample_id
    return filtered, excluded_ids


# para: sample, dt, dtau, z
ratio = np.load(DATA_DIR / "ratio.npy", mmap_mode="r")
print(f"loaded ratio shape: {ratio.shape}")
ratio_filtered, excluded_ids = exclude_farthest(ratio)
save_npy(DATA_DIR / "excluded_sample_ids.npy", excluded_ids,
         name="excluded sample ids")

for z in Z_LIST:
    # para: sample, dt, dtau
    plot_ratio_per_z(
        ratio_filtered[:, :, :, z],
        FIG_DIR / f"ratio_{CONF_SHORT}_P{MOM}_z{z}.png",
        f"ratio without farthest sample, z={z}, {CONF_SHORT}",
        XLIM, YLIM, DT_LIST,
    )
