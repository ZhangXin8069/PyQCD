#!/public/home/huangcl/.venv/bin/python
"""读取 C32P29 氘核关联函数, 投影, 求 GEVP 有效质量并画图.

不使用 dataclass 配置, init_config() 直接把参数和路径拆散返回.
CONF_SHORT 和 T0 为可给用户修改的配置参数, 使用全大写.
计算函数只接收传入的小写参数, 不使用全局配置.
"""

from pathlib import Path
import sys

import numpy as np
import scipy.linalg as spla

# ---- 从上级目录 98_tools 导入通用函数 ----
# sys.path.insert 必须在 import 之前, 否则运行时 ImportError
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "98_tools"))

# noqa: E402 抑制 autopep8 的 import 顺序检查, 防止 import 被上移到 sys.path.insert 之前
# type: ignore 抑制 Pylance 无法解析导入的报错
from analysis_tools import plot_errbar, resample, sem  # noqa: E402  # type: ignore
from gamma_matrix_DP import gamma  # noqa: E402  # type: ignore
from output_tools import save_npy  # noqa: E402  # type: ignore
from snsc_params import get_conf  # noqa: E402  # type: ignore


def init():
    """根据 conf_short 初始化 C32P29 的分析参数和结果路径, 拆散返回.

    CONF_SHORT 和 T0 为可给用户修改的配置参数.
    返回: conf_ids, T0, energy_unit, nmom, sink_mom, source_mom,
          data_dir, eigval_dir, eigvec_dir, figure_dir, CONF_SHORT.
    """

    # 用户可修改的配置参数
    CONF_SHORT = "c32p29"

    if CONF_SHORT == "c32p29":
        conf_ids = np.array(
            [c for c in range(11000, 11000 + 50 * 297, 50) if c not in (18350, 23050)])

        T0 = 5

    else:
        raise ValueError(f"unsupported conf_short: {CONF_SHORT}")

    energy_unit = 0.1973 / get_conf(CONF_SHORT)["lattice_spacing"]

    nmom = 2
    sink_mom = (0, 1, 1, 1)
    source_mom = (0, 1, 1, 1, -1, 1, 1, 1)

    data_dir = Path(
        f"/public/home/huangcl/00_data/00_NP_2pt/mom0/{CONF_SHORT}")
    result_dir = Path(__file__).resolve().parent / "01_result" / CONF_SHORT
    eigval_dir = result_dir / "eigval"
    eigvec_dir = result_dir / "eigvec"
    figure_dir = Path(__file__).resolve().parent / \
        "01_result" / "fig" / CONF_SHORT

    return (
        CONF_SHORT,
        conf_ids,
        T0,
        energy_unit,
        nmom,
        sink_mom,
        source_mom,
        data_dir,
        eigval_dir,
        eigvec_dir,
        figure_dir,
    )


def load_correlators(conf_ids: np.ndarray, data_dir: Path) -> np.ndarray:
    """纯读取本地 NP 2pt 关联函数, 不改变轴序.

    返回 para: conf, p_snk, p_src, dt, t_src, r, q, z, y.
    """

    correlator_list = []
    for conf_id in conf_ids:
        file_path = data_dir / str(conf_id) / f"NPcorr_conf{conf_id}_P0.npy"
        if not file_path.is_file():
            raise FileNotFoundError(f"not found: {file_path}")

        # para: p_snk, p_src, dt, t_src, r, q, z, y
        correlator_list.append(np.load(file_path))

    correlators = np.asarray(correlator_list, dtype=np.complex128)
    print(f"file_path: {file_path}, {correlators.shape}")
    return correlators


def project_deuteron_spin_channels(
    correlators: np.ndarray,
) -> np.ndarray:
    """构造 Cgamma 自旋投影, 并逐通道投影 s=-1,0,+1 三条氘核自旋通道.

    输入 correlators 的末尾 4 轴为 r,q,z,y 旋量指标, 之前的轴原样保留.
    返回 para: <others>, sz; sz 末轴依次为 -1, 0, 1.
    """

    charge_conjugation = -np.asarray(gamma(10))
    c_gamma_1 = np.matmul(charge_conjugation, gamma(1))[:2, :2]
    c_gamma_2 = np.matmul(charge_conjugation, gamma(2))[:2, :2]
    c_gamma_3 = np.matmul(charge_conjugation, gamma(3))[:2, :2]
    # 按从小到大 -1,0,+1 排列三条自旋通道
    projectors = {
        "s_minus1": (c_gamma_1 + 1j * c_gamma_2) / np.sqrt(2.0),
        "s0": c_gamma_3,
        "s_plus1": (c_gamma_1 - 1j * c_gamma_2) / np.sqrt(2.0),
    }

    # para: conf, p, k, <others>
    channel_list = [
        np.einsum(
            "...rqzy,rz,yq->...",
            correlators,
            projector,
            np.conj(projector).T,
            optimize=True,
        )
        for projector in projectors.values()
    ]
    # 在末尾追加 sz 轴, para: <others>, sz
    return np.stack(channel_list, axis=-1)


def build_corr_matrix(
    projected_correlator: np.ndarray,
    nmom: int,
    sink_mom: tuple[int, ...],
    source_mom: tuple[int, ...],
) -> np.ndarray:
    """汇总等价动量方向并补齐 Hermitian 矩阵.

    要求输入 projected_correlator 的前 3 轴为 conf,p,k, 之后所有轴原样保留.
    返回 para: conf, |p|, |k|, <others>.
    """

    # 算符 O0: 两核子相对动量为 0
    # 算符 O1: 两核子相对动量为 2, 对核子 1 的动量方向 +x, +y, +z, -x, -y, -z 求和
    corr_matrix = np.zeros(
        (projected_correlator.shape[0], nmom, nmom)
        + projected_correlator.shape[3:],
        dtype=np.complex128,
    )
    for sink_id, p in enumerate(sink_mom):
        for source_id, k in enumerate(source_mom):
            if k < 0:
                continue
            corr_matrix[:, p, k] += projected_correlator[:, sink_id, source_id]

    # (1,1) 完整应为 6x6=36 个方向对的关联函数, 数据只算了 sink=+xyz 的 3x6=18 个,
    # 缺的 18 个 (sink=-xyz) 是已有 18 个的空间反射, 利用对称性乘 2 补成 36
    corr_matrix[:, 1, 1] *= 2

    # 补齐二点函数的 Hermitian 矩阵, para: conf, |p|, |k|, <others>
    diagonal = np.arange(nmom)
    corr_matrix[:, diagonal, diagonal] = corr_matrix[:,
                                                     diagonal, diagonal].real
    upper = np.triu_indices(nmom, k=1)
    corr_matrix[:, upper[1], upper[0]] = np.conj(
        corr_matrix[:, upper[0], upper[1]])
    return corr_matrix


def solve_gevp(
    corr_matrix: np.ndarray,
    nmom: int,
    t0: int,
) -> tuple[np.ndarray, np.ndarray]:
    """重采样并求 GEVP, 返回本征值与本征矢.

    返回值:
        本征值, para: sample, t, state
        本征矢, para: sample, t, k, state
    """

    # resample 默认 3000 个 bootstrap 样本
    corr_samples = resample(corr_matrix)
    n_sample = corr_samples.shape[0]

    # dt 轴长度由数据决定, 兼容任意 dt 长度
    n_dt = corr_matrix.shape[1]
    eigenvalues_raw = np.empty(
        (n_sample, n_dt, nmom),
        dtype=float,
    )
    eigenvectors_raw = np.empty(
        (n_sample, n_dt, nmom, nmom),
        dtype=np.complex128,
    )
    for sample_id in range(n_sample):
        reference_matrix = corr_samples[sample_id, t0]
        for time_id in range(n_dt):
            (
                eigenvalues_raw[sample_id, time_id],
                eigenvectors_raw[sample_id, time_id],
            ) = spla.eigh(
                corr_samples[sample_id, time_id],
                b=reference_matrix,
            )

    # 与原 gevp.py 一致: t>t0 时将本征值与本征矢按从大到小排列
    eigenvalues = np.stack(
        [
            eigenvalues_raw[:, time_id, ::-1]
            if time_id > t0
            else eigenvalues_raw[:, time_id, :]
            for time_id in range(n_dt)
        ],
        axis=1,
    )
    # para: sample, t, k, state
    eigenvectors = np.stack(
        [
            eigenvectors_raw[:, time_id, :, ::-1]
            if time_id > t0
            else eigenvectors_raw[:, time_id, :, :]
            for time_id in range(n_dt)
        ],
        axis=1,
    )
    return eigenvalues, eigenvectors


def align_eigvec_phase(eigenvectors: np.ndarray) -> np.ndarray:
    """调整每个本征矢的复角, 使 k=0 分量变为正实数.

    输入/输出的 para 均为 sample, t, k, state.
    """

    # para: sample, t, state
    v0 = eigenvectors[:, :, 0, :]
    # 相位因子 conj(v0)/|v0|, 使 k=0 分量旋转为正实数
    phase = np.conj(v0) / np.abs(v0)
    # 广播到所有分量: para: sample, t, k, state
    eigenvectors_aligned = eigenvectors * phase[:, :, None, :]
    return eigenvectors_aligned


def calc_effective_energy(
    data_samples: np.ndarray,
    energy_unit: float,
) -> np.ndarray:
    """由重采样样本求有效能量, 时间轴为 axis=1.

    输入 data_samples 的 para: sample, t, state, 也兼容 sample, t.
    """

    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            np.log(np.roll(data_samples, shift=1, axis=1) / data_samples)
            * energy_unit
        )


if __name__ == "__main__":
    # 拆散接收 init() 返回的参数和路径
    (
        conf_short,
        conf_ids,
        t0,
        energy_unit,
        nmom,
        sink_mom,
        source_mom,
        data_dir,
        eigval_dir,
        eigvec_dir,
        figure_dir,
    ) = init()
    figure_name = f"cmp_gevp_tref{t0}_{conf_short}.png"

    # para: conf, p, k, dt, t_src, r, q, z, y
    correlators = load_correlators(conf_ids, data_dir)

    # 对 t_src 求平均, para: conf, p, k, dt, r, q, z, y
    correlators = correlators.mean(axis=4)

    # para: conf, p, k, dt, sz; sz[:] = -1, 0, 1
    spin_channels = project_deuteron_spin_channels(correlators)

    # para: conf, p, k, dt
    average_correlator = spin_channels.mean(axis=-1)

    # para: conf, |p|, |k|, dt
    corr_matrix = build_corr_matrix(
        average_correlator, nmom, sink_mom, source_mom)

    # para: conf, dt, |p|, |k| -> conf, t, |p|, |k|
    corr_matrix = corr_matrix.transpose(0, 3, 1, 2)

    # 求 GEVP, 返回本征值 sample,t,state 与本征矢 sample,t,k,state
    eigenvalues, eigenvectors = solve_gevp(corr_matrix, nmom, t0)

    # 对齐本征矢相位, 使 k=0 分量为正实数
    eigenvectors = align_eigvec_phase(eigenvectors)

    # 按能级保存本征值与本征矢, 每个 state 两个文件
    for state_id in range(nmom):
        eigval_path = eigval_dir / f"state{state_id}_tref{t0}.npy"
        save_npy(eigval_path, eigenvalues[:, :, state_id], name="eigval")

        # para: sample, t, k
        eigvec_path = eigvec_dir / f"state{state_id}_tref{t0}.npy"
        save_npy(eigvec_path, eigenvectors[:, :, :, state_id], name="eigvec")

    effective_energy = calc_effective_energy(eigenvalues, energy_unit)
    mean_energy = effective_energy.mean(axis=0)
    error_energy = sem(effective_energy)

    # 不做 GEVP 的对照: p=k=0 与 p=k=1 的对角两点函数分别重采样, para: sample,t
    correlator_p0 = resample(corr_matrix[:, :, 0, 0])
    correlator_p1 = resample(corr_matrix[:, :, 1, 1])
    effmass_p0 = calc_effective_energy(correlator_p0, energy_unit)
    effmass_p1 = calc_effective_energy(correlator_p1, energy_unit)

    # 非 GEVP 的 p=k=0,p=k=1 与 GEVP 的 state 0,state 1 画在同一张图, t0 记录在标题
    effmass_data = {
        "p=k=0": (effmass_p0.mean(axis=0), sem(effmass_p0)),
        "p=k=1": (effmass_p1.mean(axis=0), sem(effmass_p1)),
    }
    for state_id in range(nmom):
        effmass_data[f"state {state_id}"] = (
            mean_energy[:, state_id],
            error_energy[:, state_id],
        )
    plot_errbar(
        np.arange(mean_energy.shape[0]),
        effmass_data,
        figure_dir / figure_name,
        xlabel="t",
        ylabel="effective mass (GeV)",
        xlim=[2.5, 20.5],
        ylim=[1.5, 2.5],
        x_offset=0.15,
        title=f"effective mass, tref={t0}",
    )
