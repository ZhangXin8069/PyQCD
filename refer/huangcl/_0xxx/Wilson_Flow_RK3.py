"""Wilson flow RK3, clover Gmunu 和单配置 E 验证.

本程序从 SU(3) gauge links U 出发, 使用 Luescher 三阶段 Lie-group RK3
实现 Wilson flow. 每个完整 flow time 都测量四叶 clover 场强得到的体积平均
能量 E, 并展示 reduced Wilson action, SU(3) 残差和 action 增量.

输入是与现有 Calc_ope_unpol.py 相同风格的参数文件. 参数文件至少包含
Nt, Nx 和 conf_file. 原始二进制 gauge file 默认采用大端 float64, 实部和虚部
按参考程序的欧氏时间, z, y, x, link direction, color row, color column 顺序
排列. 空间三个方向统一使用 Nx.

结果只在屏幕上展示, 不保存局域 E, Gmunu, flowed-link history 或其他结果文件.
可选的 RK3 收敛检查在相同终点比较 epsilon, epsilon / 2 和 epsilon / 4.
"""

from __future__ import annotations

import fileinput
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from opt_einsum import contract


DEBUG = False


INDEPENDENT_MU_NU = tuple(
    (mu, nu) for mu in range(4) for nu in range(mu + 1, 4)
)


# ==================== 配置数据结构 ====================

@dataclass(frozen=True)
class GeometryConfig:
    """保存格点几何和边界条件."""

    Nt: int
    Nx: int


@dataclass(frozen=True)
class FlowConfig:
    """保存 Wilson flow 和独立收敛检查的数值参数."""

    epsilon: float
    n_steps: int
    include_t0: bool
    convergence_t_star: Optional[float]


@dataclass(frozen=True)
class FieldConventionConfig:
    """保存 clover 场强的归一化和 SU(3) 投影约定."""

    traceless_clover: bool
    link_convention: str


@dataclass(frozen=True)
class ValidationConfig:
    """保存每个完整 flow state 的数值验证参数."""

    unitarity_tolerance: float
    determinant_tolerance: float
    action_tolerance: float
    failure_policy: str


@dataclass(frozen=True)
class PathConfig:
    """保存输入参数, gauge links 和结果文件路径."""

    conf_file: str
    conf_id: str
    output_dir: str
    output_format: str


@dataclass(frozen=True)
class FlowStateChecks:
    """记录一个完整 flow state 的标量诊断."""

    reduced_action: float
    unitarity_residual: float
    determinant_residual: float
    action_delta: float
    unitarity_passed: bool
    determinant_passed: bool
    action_monotonic: bool
    checks_passed: bool


@dataclass(frozen=True)
class RK3ConvergenceResult:
    """记录固定终点的三组步长收敛比较."""

    step_size_epsilon: float
    step_size_epsilon_half: float
    step_size_epsilon_quarter: float
    energy_epsilon: float
    energy_epsilon_half: float
    energy_epsilon_quarter: float
    delta_1: float
    delta_2: float
    delta_ratio: float


# ==================== 输入参数和 gauge 读取 ====================

def init_config() -> Tuple[
    GeometryConfig,
    FlowConfig,
    FieldConventionConfig,
    ValidationConfig,
    PathConfig,
]:
    """读取输入参数并集中设置 Wilson flow 参数."""

    epsilon = 0.01
    n_steps = 10
    include_t0 = True
    convergence_t_star = None
    traceless_clover = True
    link_convention = "coupling_absorbed"
    unitarity_tolerance = 1e-10
    determinant_tolerance = 1e-10
    action_tolerance = 1e-10
    failure_policy = "record"
    output_format = "npz"

    infile = fileinput.input()
    for line in infile:
        tmp = line.split()
        if not tmp:
            continue
        if tmp[0] == "Nt":
            Nt = int(tmp[1])
        if tmp[0] == "Nx":
            Nx = int(tmp[1])
        if tmp[0] == "conf_id":
            conf_id = tmp[1]
        if tmp[0] == "conf_file":
            conf_file = tmp[1]
        if tmp[0] == "output_dir":
            output_dir = tmp[1]
        if tmp[0] == "epsilon":
            epsilon = float(tmp[1])
        if tmp[0] == "n_steps":
            n_steps = int(tmp[1])
        if tmp[0] == "include_t0":
            include_t0 = tmp[1] == "True"
        if tmp[0] == "convergence_t_star":
            convergence_t_star = float(tmp[1])
        if tmp[0] == "unitarity_tolerance":
            unitarity_tolerance = float(tmp[1])
        if tmp[0] == "determinant_tolerance":
            determinant_tolerance = float(tmp[1])
        if tmp[0] == "action_tolerance":
            action_tolerance = float(tmp[1])
        if tmp[0] == "failure_policy":
            failure_policy = tmp[1]
    infile.close()

    os.makedirs(output_dir, exist_ok=True)

    geometry = GeometryConfig(
        Nt=Nt,
        Nx=Nx,
    )
    flow = FlowConfig(
        epsilon=epsilon,
        n_steps=n_steps,
        include_t0=include_t0,
        convergence_t_star=convergence_t_star,
    )
    field_convention = FieldConventionConfig(
        traceless_clover=traceless_clover,
        link_convention=link_convention,
    )
    validation = ValidationConfig(
        unitarity_tolerance=unitarity_tolerance,
        determinant_tolerance=determinant_tolerance,
        action_tolerance=action_tolerance,
        failure_policy=failure_policy,
    )
    paths = PathConfig(
        conf_file=conf_file,
        conf_id=conf_id,
        output_dir=output_dir,
        output_format=output_format,
    )
    return geometry, flow, field_convention, validation, paths


def load_gauge_links(
    paths: PathConfig,
    geometry: GeometryConfig,
) -> np.ndarray:
    """按参考代码的大端 float64 格式读取 SU(3) gauge links."""

    file_path = paths.conf_file
    gauge = np.fromfile(file_path, dtype=">f8")
    gauge = gauge.reshape(
        geometry.Nt, geometry.Nx, geometry.Nx, geometry.Nx, 4, 3, 3, 2
    )
    gauge = gauge[..., 0] + gauge[..., 1] * 1j  # para: t, z, y, x, mu, color, color
    print(f"loaded gauge links: {file_path}")
    # para: t, z, y, x, mu, color, color
    return gauge


# ==================== 方向和坐标轴约定 ====================

# mu=0,1,2,3 分别表示 x,y,z,t, gauge 的坐标轴顺序为 t,z,y,x
# 因此 link 方向到 gauge 坐标轴的映射为 mu -> (3,2,1,0)[mu]
COORDINATE_AXIS = (3, 2, 1, 0)

# ==================== Wilson flow RK3 ====================

def wilson_staples(gauge: np.ndarray, mu: int) -> np.ndarray:
    """直接计算固定 mu 下所有 nu != mu 的六条 Wilson staples.

    输入:
        gauge: 当前完整的四维 gauge links U.
        mu: 当前 link U_mu(x) 的方向.
    输出:
        sigma_mu: 三个 nu 方向的 C_plus+C_minus 之和.
    """

    coor_mu = COORDINATE_AXIS[mu]
    sigma_mu = np.zeros_like(gauge[..., mu, :, :])
    for nu in range(4):
        if nu == mu:
            continue

        coor_nu = COORDINATE_AXIS[nu]
        gauge_rightup = gauge
        gauge_rightdown = np.roll(gauge, 1, axis=coor_nu)

        # C_plus = U_nu(x+mu) U_mu^dagger(x+nu) U_nu^dagger(x)
        c_plus = contract(
            "tzyxab,tzyxcb->tzyxac",
            np.roll(gauge_rightup, -1, axis=coor_mu)[..., nu, :, :],
            np.roll(gauge_rightup, -1, axis=coor_nu)[..., mu, :, :].conj(),
        )
        c_plus = contract(
            "tzyxab,tzyxcb->tzyxac",
            c_plus,
            gauge_rightup[..., nu, :, :].conj(),
        )

        # C_minus = U_nu^dagger(x+mu-nu) U_mu^dagger(x-nu) U_nu(x-nu)
        c_minus = contract(
            "tzyxba,tzyxcb->tzyxac",
            np.roll(gauge_rightdown, -1, axis=coor_mu)[..., nu, :, :].conj(),
            gauge_rightdown[..., mu, :, :].conj(),
        )
        c_minus = contract(
            "tzyxab,tzyxbc->tzyxac",
            c_minus,
            gauge_rightdown[..., nu, :, :],
        )
        sigma_mu += c_plus + c_minus
    # para: t, z, y, x, color, color
    return sigma_mu


def _traceless_antihermitian(matrix_field: np.ndarray) -> np.ndarray:
    """计算式(4)的 traceless anti-Hermitian projection."""

    # 用颜色矩阵的 dagger 构造 anti-Hermitian 部分.
    antihermitian = 0.5 * (
        matrix_field - np.swapaxes(matrix_field.conj(), -1, -2)
    )
    trace = contract(
        "tzyxcc->tzyx",
        antihermitian,
    )
    # para: t, z, y, x, color, color
    return antihermitian - (trace / 3)[..., None, None] * np.eye(
        3, dtype=antihermitian.dtype
    )


def wilson_flow_vector_field(gauge: np.ndarray) -> np.ndarray:
    """按六个 staple 构造不含显式 g0 的 Wilson-flow vector field.

    输入:
        gauge: 当前完整 flowed configuration V.
    输出:
        vector_field: 每条 link 上的 traceless anti-Hermitian Z(V).

    C_plus 和 C_minus 分别对应笔记式(3)的两个 staple.
    """

    vector_field = np.empty_like(gauge)
    for mu in range(4):
        sigma_mu = wilson_staples(gauge, mu)
        # 将当前 link 与 staple 相乘并收缩中间颜色指标.
        force_matrix = contract(
            "tzyxab,tzyxbc->tzyxac",
            gauge[..., mu, :, :],
            sigma_mu,
        )
        vector_field[..., mu, :, :] = -1.0 * _traceless_antihermitian(force_matrix)

    # para: t, z, y, x, mu, color, color
    return vector_field


def _exp_antihermitian(algebra_field: np.ndarray) -> np.ndarray:
    """用批量 Hermitian eigendecomposition 计算 SU(3) 群指数."""

    hermitian_field = -1j * algebra_field
    # 用颜色矩阵的 Hermitian 部分抑制数值误差.
    hermitian_field = 0.5 * (
        hermitian_field + np.swapaxes(hermitian_field.conj(), -1, -2)
    )

    # 当前用 np.linalg.eigh 对每个局部 3 x 3 Hermitian 颜色矩阵批量对角化.
    # 该函数调用底层 LAPACK, 不是 Python 层逐个格点循环.
    # 可选的矩阵指数实现:
    # 1. scipy.linalg.expm: 通用矩阵指数, 内部使用 scaling-and-squaring 和 Pade
    #    逼近. 适合单个矩阵或结果交叉验证, 但对大量 3 x 3 矩阵不保证更快.
    # 2. cupy.linalg.eigh: GPU 上的批量 Hermitian 对角化. 需要 gauge 和中间量
    #    全程保存在 GPU, 否则 CPU/GPU 数据拷贝可能抵消加速.
    # 3. SU(3) 专用的 Cayley-Hamilton 指数: 利用 traceless 3 x 3 矩阵满足的
    #    特征多项式, 将 \exp(A) 写成 c0 * I + c1 * A + c2 * A^2
    #    这种实现可能更快, 但要专门处理本征值接近简并时的系数稳定性.
    # 当前先保留 eigh, 便于优先验证 RK3 演化和 SU(3) 残差; 后续可用基准测试
    # 再决定是否切换到 GPU 或 SU(3) 专用实现.
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian_field)
    phases = np.exp(1j * eigenvalues)
    # para: t, z, y, x, mu, color, color
    return contract(
        "...ab,...b,...cb->...ac",
        eigenvectors,
        phases,
        eigenvectors.conj(),
    )


def _left_group_update(
    gauge: np.ndarray,
    algebra_increment: np.ndarray,
) -> np.ndarray:
    """按 exp(algebra_increment) 从左侧更新全部 links."""

    group_increment = _exp_antihermitian(algebra_increment)
    # 将群增量从左侧乘到每条 link 上, 收缩中间颜色指标.
    # para: t, z, y, x, mu, color, color
    return contract(
        "tzyxmab,tzyxmbc->tzyxmac",
        group_increment,
        gauge,
    )


def rk3_flow_step(
    gauge: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """执行一个完整的 Luescher Lie-group RK3 flow step.

    输入:
        gauge: 当前 flow state V_t.
        epsilon: 格点单位下的 flow-time 步长.
    输出:
        next_gauge: 完整推进后的 flow state V_{t+epsilon}.

    每个 stage 都从完整的 stage configuration 重新计算 vector field. Z0,
    Z1 和 Z2 已经包含一次 epsilon, 因而指数内不再重复乘 epsilon.
    """

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    z0 = epsilon * wilson_flow_vector_field(gauge)
    w1 = _left_group_update(gauge, 0.25 * z0)

    z1 = epsilon * wilson_flow_vector_field(w1)
    w2_increment = (8.0 / 9.0) * z1 - (17.0 / 36.0) * z0
    w2 = _left_group_update(w1, w2_increment)

    z2 = epsilon * wilson_flow_vector_field(w2)
    final_increment = (
        (3.0 / 4.0) * z2
        - (8.0 / 9.0) * z1
        + (17.0 / 36.0) * z0
    )
    next_gauge = _left_group_update(w2, final_increment)
    # para: t, z, y, x, mu, color, color
    return next_gauge


# ==================== Clover Gmunu 和 E ====================

def clover_field_strength_munu(
    gauge: np.ndarray,
    mu: int,
    nu: int,
    traceless: bool = True,
) -> np.ndarray:
    """构造一个指定 mu, nu 平面的四叶 clover Gmunu.

    输入:
        gauge: 同一个完整 flow time 的四维 gauge links U.
        mu, nu: 独立方向对, 要求 mu != nu.
        traceless: 是否去掉颜色 trace 部分.
    输出:
        field_strength: 当前平面的全时空 clover 场强矩阵.

    四个 plaquette 直接用正向 link U 及其 dagger 构造为
    P_mu,nu, P_nu,-mu, P_-mu,-nu 和 P_-nu,mu. 四项相加得到 Q_mu,nu,
    格点单位制下再由 Q_mu,nu-Q_mu,nu^dagger 实现
    -i(Q_mu,nu-Q_mu,nu^dagger)/8.0.
    """

    if mu == nu:
        raise ValueError("clover field strength requires mu != nu")

    coor_mu = COORDINATE_AXIS[mu]
    coor_nu = COORDINATE_AXIS[nu]

    gauge_rightup = gauge # U(x)
    gauge_leftup = np.roll(gauge, 1, axis=coor_mu) # U(x-mu)
    gauge_rightdown = np.roll(gauge, 1, axis=coor_nu) # U(x-nu)
    gauge_leftdown = np.roll(gauge_leftup, 1, axis=coor_nu) # U(x-mu-nu)

    # P_mu,nu(x) = U_mu(x) U_nu(x+mu) U_mu^dagger(x+nu) U_nu^dagger(x)
    p_mu_nu = contract(
        "tzyxab,tzyxbc->tzyxac",
        gauge_rightup[..., mu, :, :],
        np.roll(gauge_rightup, -1, axis=coor_mu)[..., nu, :, :],
    )
    p_mu_nu = contract(
        "tzyxab,tzyxcb->tzyxac",
        p_mu_nu,
        np.roll(gauge_rightup, -1, axis=coor_nu)[..., mu, :, :].conj(),
    )
    p_mu_nu = contract(
        "tzyxab,tzyxcb->tzyxac",
        p_mu_nu,
        gauge_rightup[..., nu, :, :].conj(),
    )

    # P_nu,-mu(x) = U_nu(x) U_mu^dagger(x+nu-mu)
    #                U_nu^dagger(x-mu) U_mu(x-mu)
    p_nu_minus_mu = contract(
        "tzyxab,tzyxcb->tzyxac",
        np.roll(gauge_leftup, -1, axis=coor_mu)[..., nu, :, :],
        np.roll(gauge_leftup, -1, axis=coor_nu)[..., mu, :, :].conj(),
    )
    p_nu_minus_mu = contract(
        "tzyxab,tzyxcb->tzyxac",
        p_nu_minus_mu,
        gauge_leftup[..., nu, :, :].conj(),
    )
    p_nu_minus_mu = contract(
        "tzyxab,tzyxbc->tzyxac",
        p_nu_minus_mu,
        gauge_leftup[..., mu, :, :],
    )

    # P_-mu,-nu(x) = U_mu^dagger(x-mu) U_nu^dagger(x-mu-nu)
    #                  U_mu(x-mu-nu) U_nu(x-nu)
    p_minus_mu_minus_nu = contract(
        "tzyxba,tzyxcb->tzyxac",
        np.roll(gauge_leftdown, -1, axis=coor_nu)[..., mu, :, :].conj(),
        gauge_leftdown[..., nu, :, :].conj(),
    )
    p_minus_mu_minus_nu = contract(
        "tzyxab,tzyxbc->tzyxac",
        p_minus_mu_minus_nu,
        gauge_leftdown[..., mu, :, :],
    )
    p_minus_mu_minus_nu = contract(
        "tzyxab,tzyxbc->tzyxac",
        p_minus_mu_minus_nu,
        np.roll(gauge_leftdown, -1, axis=coor_mu)[..., nu, :, :],
    )

    # P_-nu,mu(x) = U_nu^dagger(x-nu) U_mu(x-nu)
    #                U_nu(x+mu-nu) U_mu^dagger(x)
    p_minus_nu_mu = contract(
        "tzyxba,tzyxbc->tzyxac",
        gauge_rightdown[..., nu, :, :].conj(),
        gauge_rightdown[..., mu, :, :],
    )
    p_minus_nu_mu = contract(
        "tzyxab,tzyxbc->tzyxac",
        p_minus_nu_mu,
        np.roll(gauge_rightdown, -1, axis=coor_mu)[..., nu, :, :],
    )
    p_minus_nu_mu = contract(
        "tzyxab,tzyxcb->tzyxac",
        p_minus_nu_mu,
        np.roll(gauge_rightdown, -1, axis=coor_nu)[..., mu, :, :].conj(),
    )

    q_mu_nu = (
        p_mu_nu
        + p_nu_minus_mu
        + p_minus_mu_minus_nu
        + p_minus_nu_mu
    )

    # 用 clover sum 的 dagger 构造反对称部分.
    field_strength = (
        -1j
        * (q_mu_nu - np.swapaxes(q_mu_nu.conj(), -1, -2))
        / 8.0
    )
    if traceless:
        trace = contract("tzyxcc->tzyx", field_strength)
        field_strength -= (trace / 3)[..., None, None] * np.eye(
            3, dtype=field_strength.dtype
        )
    # para: t, z, y, x, color, color
    return field_strength


def clover_field_strength_all(
    gauge: np.ndarray,
    traceless: bool = True,
) -> np.ndarray:
    """反复调用单个平面函数, 构造完整的 clover Fmunu.

    mu, nu = 0, 1, 2, 3 分别表示 x, y, z, t 方向.

    输入:
        gauge: 同一个完整 flow time 的四维 gauge links U.
        traceless: 是否去掉颜色 trace 部分.
    输出:
        field_strength_all: 按 mu, nu, t, z, y, x, color, color 排列的完整场强.
    """

    field_strength_all = np.zeros(
        (4, 4) + gauge.shape[:4] + (3, 3),
        dtype=gauge.dtype,
    )  # para: mu, nu, t, z, y, x, color, color
    for mu, nu in INDEPENDENT_MU_NU:
        field_strength_munu = clover_field_strength_munu(
            gauge,
            mu,
            nu,
            traceless=traceless,
        )
        field_strength_all[mu, nu] = field_strength_munu
        field_strength_all[nu, mu] = -field_strength_munu
    # para: mu, nu, t, z, y, x, color, color
    return field_strength_all


def clover_energy_mean(
    field_strengths: Sequence[np.ndarray],
) -> float:
    """只由六个独立 clover Gmunu 计算体积平均 E=GG/4.

    输入:
        field_strengths: 六个 mu < nu 平面的 clover 场强.
    输出:
        energy_mean: 一个配置在当前 flow time 的体积平均能量密度.

    当前只对已经去掉颜色 trace 的 Gmunu 做颜色收缩, 对六个独立平面求
    ReTr(Gmunu Gmunu).
    每个平面的局域收缩只在本函数内短暂存在, 不进入长期结果.
    """

    energy_sum = 0.0
    n_site: Optional[int] = None
    for field_strength in field_strengths:
        trace_g_squared = contract(
            "tzyxcd,tzyxdc->tzyx",
            field_strength,
            field_strength,
        )
        energy_sum += float(np.sum(trace_g_squared.real, dtype=np.float64))
        if n_site is None:
            n_site = trace_g_squared.size

    if n_site is None:
        raise ValueError("field_strengths must contain the six independent planes")
    return float(energy_sum / n_site)


def measure_clover_energy(
    gauge: np.ndarray,
    field_convention: FieldConventionConfig,
) -> float:
    """构造完整的 clover Fmunu, 再返回当前配置的体积平均 E."""

    field_strength_all = clover_field_strength_all(
        gauge,
        traceless=field_convention.traceless_clover,
    )
    field_strengths = [
        field_strength_all[mu, nu] for mu, nu in INDEPENDENT_MU_NU
    ]
    energy_mean = clover_energy_mean(field_strengths)
    del field_strength_all
    del field_strengths
    return energy_mean


# ==================== Wilson action 和 SU(3) 验证 ====================

def reduced_wilson_action(gauge: np.ndarray) -> float:
    """计算不含显式 g0 的 reduced Wilson action Sred=6 Ap."""

    action_sum = 0.0
    for mu, nu in INDEPENDENT_MU_NU:
        coor_mu = COORDINATE_AXIS[mu]
        coor_nu = COORDINATE_AXIS[nu]
        link_mu = gauge[..., mu, :, :]
        link_nu = gauge[..., nu, :, :]

        # P_mu,nu = V_mu(x) V_nu(x+mu) V_mu^dagger(x+nu) V_nu^dagger(x)
        plaquette = contract(
            "tzyxab,tzyxbc->tzyxac",
            link_mu,
            np.roll(link_nu, -1, axis=coor_mu),
        )
        plaquette = contract(
            "tzyxab,tzyxcb->tzyxac",
            plaquette,
            np.roll(link_mu, -1, axis=coor_nu).conj(),
        )
        plaquette = contract(
            "tzyxab,tzyxcb->tzyxac",
            plaquette,
            link_nu.conj(),
        )
        plaquette_trace = contract(
            "tzyxcc->tzyx",
            plaquette,
        )
        action_sum += float(
            6.0
            * np.sum(
                1.0 - plaquette_trace.real / 3,
                dtype=np.float64,
            )
        )
    return float(action_sum)


def su3_unitarity_residual(gauge: np.ndarray) -> float:
    """计算所有 flowed links 的最大 unitarity Frobenius 残差."""

    unitary_product = contract(
        "tzyxmab,tzyxmbc->tzyxmac",
        gauge,
        gauge.conj().swapaxes(-1, -2),
    )
    residual = np.sqrt(
        np.sum(
            np.abs(unitary_product - np.eye(3, dtype=gauge.dtype)) ** 2,
            axis=(-2, -1),
        )
    )
    return float(np.max(residual))


def flow_state_checks(
    gauge: np.ndarray,
    previous_action: Optional[float],
    validation: ValidationConfig,
) -> FlowStateChecks:
    """计算当前完整 flow state 的 action 和 SU(3) 验证量.

    输入:
        gauge: 当前完整 flowed configuration.
        previous_action: 前一个已保存 flow time 的 reduced action, 初始时为 None.
        validation: SU(3) 和 action 单调性容差.
    输出:
        checks: 当前 state 的标量诊断和通过状态.
    """

    reduced_action = reduced_wilson_action(gauge)
    unitarity_residual = su3_unitarity_residual(gauge)
    # 直接计算所有 link 的 determinant 偏离 1 的最大值.
    determinant_residual = float(np.max(np.abs(np.linalg.det(gauge) - 1.0)))

    if previous_action is None:
        action_delta = float("nan")
        action_monotonic = True
    else:
        action_delta = reduced_action - previous_action
        action_monotonic = action_delta <= validation.action_tolerance

    unitarity_passed = unitarity_residual <= validation.unitarity_tolerance
    determinant_passed = determinant_residual <= validation.determinant_tolerance
    checks_passed = (
        unitarity_passed and determinant_passed and action_monotonic
    )
    return FlowStateChecks(
        reduced_action=reduced_action,
        unitarity_residual=unitarity_residual,
        determinant_residual=determinant_residual,
        action_delta=action_delta,
        unitarity_passed=unitarity_passed,
        determinant_passed=determinant_passed,
        action_monotonic=action_monotonic,
        checks_passed=checks_passed,
    )


# ==================== 单配置 flow 和展示 ====================

def _measure_flow_state(
    gauge: np.ndarray,
    flow_time: float,
    previous_action: Optional[float],
    field_convention: FieldConventionConfig,
    validation: ValidationConfig,
) -> FlowStateChecks:
    """测量并展示一个完整 flow state 的 E 和 SU(3) 诊断, 不保存结果."""

    energy = measure_clover_energy(gauge, field_convention)
    checks = flow_state_checks(gauge, previous_action, validation)
    print(
        "flow state: "
        f"t={flow_time:.16g}, "
        f"E={energy:.12e}, "
        f"action={checks.reduced_action:.12e}, "
        f"action_delta={checks.action_delta:.6e}, "
        f"unitarity={checks.unitarity_residual:.6e}, "
        f"determinant={checks.determinant_residual:.6e}, "
        f"passed={checks.checks_passed}"
    )
    return checks


def run_configuration_flow(
    gauge: np.ndarray,
    flow: FlowConfig,
    field_convention: FieldConventionConfig,
    validation: ValidationConfig,
) -> None:
    """组织单个配置的 flow-time 演化并逐时刻展示标量诊断, 不保存结果.

    输入:
        gauge: 未 flow 的 SU(3) links U.
        flow: RK3 步长, 步数和 t=0 测量开关.
        field_convention: clover 场强约定.
        validation: 每个完整 state 的检查约定.
    """

    current_gauge = np.array(gauge, dtype=np.complex128, copy=True)
    previous_action: Optional[float] = None

    if flow.include_t0:
        checks = _measure_flow_state(
            current_gauge,
            0.0,
            previous_action,
            field_convention,
            validation,
        )
        previous_action = checks.reduced_action
        if not checks.checks_passed and validation.failure_policy == "stop":
            del current_gauge
            return

    for step_index in range(flow.n_steps):
        next_gauge = rk3_flow_step(current_gauge, flow.epsilon)
        del current_gauge
        current_gauge = next_gauge
        flow_time = (step_index + 1) * flow.epsilon

        checks = _measure_flow_state(
            current_gauge,
            flow_time,
            previous_action,
            field_convention,
            validation,
        )
        previous_action = checks.reduced_action

        if not checks.checks_passed and validation.failure_policy == "stop":
            break

    del current_gauge


# ==================== RK3 收敛检查 ====================

def _step_count_for_endpoint(t_star: float, step_size: float) -> int:
    """取得恰好到达固定终点所需的整数步数."""

    step_count = int(round(t_star / step_size))
    if not np.isclose(
        step_count * step_size,
        t_star,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError("t_star must be an integer multiple of every step size")
    return step_count


def rk3_convergence_check(
    gauge: np.ndarray,
    epsilon: float,
    t_star: float,
    field_convention: FieldConventionConfig,
) -> RK3ConvergenceResult:
    """在固定 flow time 比较 E 的三阶全局误差收敛.

    输入:
        gauge: 同一个初始配置 U.
        epsilon: 粗步长.
        t_star: 三组演化共同到达的终点.
        field_convention: E 的 clover 场强约定.
    输出:
        convergence: delta_1, delta_2 和 delta_1 / delta_2.

    三次演化顺序独立执行, 不长期保存任何一组 flowed configuration.
    """

    if epsilon <= 0.0 or t_star <= 0.0:
        raise ValueError("epsilon and t_star must be positive")

    step_sizes = (epsilon, epsilon / 2.0, epsilon / 4.0)
    energies: List[float] = []
    for step_size in step_sizes:
        step_count = _step_count_for_endpoint(t_star, step_size)
        flowed_gauge = np.array(gauge, dtype=np.complex128, copy=True)
        for _ in range(step_count):
            next_gauge = rk3_flow_step(flowed_gauge, step_size)
            del flowed_gauge
            flowed_gauge = next_gauge
        energies.append(measure_clover_energy(flowed_gauge, field_convention))
        del flowed_gauge

    delta_1 = abs(energies[0] - energies[1])
    delta_2 = abs(energies[1] - energies[2])
    delta_ratio = float("nan") if delta_2 == 0.0 else delta_1 / delta_2
    return RK3ConvergenceResult(
        step_size_epsilon=step_sizes[0],
        step_size_epsilon_half=step_sizes[1],
        step_size_epsilon_quarter=step_sizes[2],
        energy_epsilon=energies[0],
        energy_epsilon_half=energies[1],
        energy_epsilon_quarter=energies[2],
        delta_1=delta_1,
        delta_2=delta_2,
        delta_ratio=delta_ratio,
    )


# ==================== 主程序 ====================

if __name__ == "__main__":
    # fileinput 直接读取命令行中给出的 input 文件, 与参考代码一致.
    geometry, flow, field_convention, validation, paths = init_config()
    gauge = load_gauge_links(paths, geometry)
    run_configuration_flow(
        gauge,
        flow,
        field_convention,
        validation,
    )

    if flow.convergence_t_star is not None:
        convergence = rk3_convergence_check(
            gauge,
            flow.epsilon,
            flow.convergence_t_star,
            field_convention,
        )
        print(
            "RK3 convergence: "
            f"t_star={flow.convergence_t_star:.16g}, "
            f"E(epsilon)={convergence.energy_epsilon:.12e}, "
            f"E(epsilon/2)={convergence.energy_epsilon_half:.12e}, "
            f"E(epsilon/4)={convergence.energy_epsilon_quarter:.12e}, "
            f"delta_1={convergence.delta_1:.6e}, "
            f"delta_2={convergence.delta_2:.6e}, "
            f"ratio={convergence.delta_ratio:.6e}"
        )
