#!/public/home/huangcl/.venv/bin/python
"""恢复 Feynman-Hellmann 裸矩阵元的绝对归一化.

输入:
    - 04_feyn_hell/01_data/<conf_short>/P<P>/fit_cand.npz
    - 06_feyn_hell/01_data/<conf_short>/P<P>/fit_cand.npz
输出:
    - 06_feyn_hell/01_data/<conf_short>/P<P>/hB.npy (para: sample, z)
"""

import importlib
import time
from pathlib import Path

import numpy as np


here = Path(__file__).resolve()


def init() -> tuple[Path, Path, Path]:
    """初始化输入和输出路径.

    输出:
    -----
    reference_path: 04_feyn_hell 的拟合结果路径.
    normalized_path: 06_feyn_hell 的归一化拟合结果路径.
    output_path: 恢复绝对归一化后的结果路径.
    """

    CONF_SHORT = "c32p29"
    MOM = 4

    project_dir = here.parent.parent
    ref_path = (
        project_dir / "04_feyn_hell" / "01_data" / CONF_SHORT
        / f"P{MOM}" / "fit_cand.npz")
    data_dir = here.parent / "01_data" / CONF_SHORT / f"P{MOM}"
    normalized_path = data_dir / "fit_cand.npz"
    output_path = data_dir / "hB.npy"
    return ref_path, normalized_path, output_path


def get_baremat(
        reference_path: Path, normalized_path: Path, output_path: Path) -> None:
    """逐样本恢复 h_B(z) 并保存.

    参数:
    -----
    reference_path: 04_feyn_hell 的 cand 拟合结果路径.
    normalized_path: 06_feyn_hell 的 cand 拟合结果路径.
    output_path: h_B(z) 数组保存路径.
    """

    ref_c0 = np.load(reference_path)["c0"]
    normalized_c0 = np.load(normalized_path)["c0"]
    # para: sample
    hB_zero = ref_c0[:, 0]
    # para: sample, z
    hB = normalized_c0 * hB_zero[:, np.newaxis]
    np.save(output_path, hB)
    print(f"saved hB: {output_path}, shape={hB.shape}")


if __name__ == "__main__":
    t_total_begin = time.perf_counter()

    reference_path, normalized_path, output_path = init()
    get_baremat(reference_path, normalized_path, output_path)

    pf = importlib.import_module("5_plot_fh")
    pf.plot_c0_z(pf.init(), norm=False)

    print(f"job done, time={time.perf_counter() - t_total_begin:.0f}s")
