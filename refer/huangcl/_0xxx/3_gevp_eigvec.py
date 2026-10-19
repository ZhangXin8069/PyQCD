#!/public/home/huangcl/.venv/bin/python
"""读取 C32P29 已保存的基态 GEVP 本征矢, 不做相位对齐, 直接画各分量实部与虚部以及模长.

输入: <项目目录>/01_gevp/01_result/<conf_short>/eigvec/state0_tref<tref>.npy,
      数据的 para: sample, t, k; tref 为可修改的输入参数.
输出: 01_result/fig 下 k=0,1 的 Re/Im errorbar 图, 以及本征矢模长 |v| 的 errorbar 图.
"""

from pathlib import Path
import sys

import numpy as np

# ---- 从项目目录下的 98_tools 导入通用函数 ----
# sys.path.insert 必须在 import 之前, 否则运行时 ImportError
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "98_tools")
)

# noqa: E402 抑制 autopep8 的 import 顺序检查, 防止 import 被上移到 sys.path.insert 之前
# type: ignore 抑制 Pylance 无法解析导入的报错
from analysis_tools import format_mean_err, plot_errbar, sem  # noqa: E402  # type: ignore


def init():
    """根据 conf_short 和 tref 初始化输入输出路径, 拆散返回.

    CONF_SHORT 和 TREF 为可给用户修改的配置参数.
    返回: tref, eigvec_dir, output_dir, CONF_SHORT.
    """

    # 用户可修改的配置参数
    CONF_SHORT = "c32p29"

    if CONF_SHORT == "c32p29":
        TREF = 5
    else:
        raise ValueError(f"unsupported conf_short: {CONF_SHORT}")

    project_dir = Path(__file__).resolve().parent.parent
    eigvec_dir = project_dir / "01_gevp" / "01_result" / CONF_SHORT / "eigvec"
    output_dir = Path(__file__).resolve().parent / \
        "01_result" / "fig" / CONF_SHORT

    return TREF, eigvec_dir, output_dir, CONF_SHORT


def load_eigvec(eigvec_dir: Path, tref: int) -> np.ndarray:
    """读取 state 0 的本征矢样本, 返回 sample,t,k."""

    eigvec_path = eigvec_dir / f"state0_tref{tref}.npy"
    # para: sample, t, k
    eigvec = np.load(eigvec_path)
    # tref 时间点本征矢不定, 设为 nan, 后续处理保持不变
    eigvec[:, tref] = np.nan
    print(f"loaded eigvec: {eigvec_path} {eigvec.shape}")
    return eigvec


def plot_ground_state_eigvec(
    eigvec: np.ndarray,
    tref: int,
    output_dir: Path,
    conf_short: str,
) -> None:
    """
    画基态本征矢各分量实部和虚部随 t 的 errorbar 图, 以及模长 |v| 随 t 的 errorbar 图.

    参数:
    -----
    eigvec: para: sample, t, k.
    tref: 参考时间点, 用于文件名和标题.
    output_dir: 图片输出目录, 父目录由 save_fig 自动创建.
    conf_short: 组态短名, 拼接到图片文件名末尾.

    输出:
    -----
    图片保存到 output_dir.
    """

    # para: t, k
    mean_eigvec = eigvec.mean(axis=0)
    error_real = sem(eigvec.real)
    error_imag = sem(eigvec.imag)
    t = np.arange(mean_eigvec.shape[0])

    for k in range(mean_eigvec.shape[1]):
        for part, part_label, mean_part, error_part in (
            ("re", "Re", mean_eigvec[:, k].real, error_real[:, k]),
            ("im", "Im", mean_eigvec[:, k].imag, error_imag[:, k]),
        ):
            figure_path = (
                output_dir
                / f"state0_tref{tref}_eigvec_k{k}_{part}_{conf_short}.png"
            )
            plot_errbar(
                t,
                {f"k={k}": (mean_part, error_part)},
                figure_path,
                xlabel="t",
                ylabel=f"{part_label}(v_{k})",
                title=rf"$\mathrm{{{part_label}}}\ v_{{0{k}}}$, tref={tref}",
                grid=True,
            )

    # 计算本征矢的模长 |v| = sqrt(v^dag v)
    # para: sample, t
    norm = np.sqrt(np.sum(np.conj(eigvec) * eigvec, axis=2).real)

    # para: t
    mean_norm = norm.mean(axis=0)
    error_norm = sem(norm)

    # 打印各时间点 |v| 的均值和误差
    for _t in t:
        print(
            f"t={_t:2d}  |v|={format_mean_err(mean_norm[_t], error_norm[_t], 7)}"
        )

    figure_path = output_dir / f"norm_state0_tref{tref}_{conf_short}.png"
    plot_errbar(
        t,
        {"|v|": (mean_norm, error_norm)},
        figure_path,
        xlabel="t",
        ylabel="|v|",
        title=rf"$\sqrt{{v^\dag_0 v_0}}$, tref={tref}",
        grid=True,
    )


if __name__ == "__main__":
    # 拆散接收 init() 返回的参数和路径
    tref, eigvec_dir, output_dir, conf_short = init()

    eigvec = load_eigvec(eigvec_dir, tref)
    plot_ground_state_eigvec(eigvec, tref, output_dir, conf_short)
