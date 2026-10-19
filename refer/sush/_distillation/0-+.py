#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J^{PC} = 0^{-+} (格点 A_1^{-+}) 胶球两点关联函数
E32P29 组态: beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64 (32^3 x 64, a=0.0897 fm)

构造流程 (对照 三篇胶球格点文章分析.md §2.3 / §2.4 / §3.2):
  1) 组态读取: lqcddb_update 的 read_gauge/readin_gauge (ILDG/LIME -> (4,Nt,Nz,Ny,Nx,3,3), 方向 X,Y,Z,T)
  2) 局域场强: 中心对称改进 clover (Type-II, 2006 Chen)
        C^{(a,b)}_{μν} = (1/8)Σ_四象限 (P^{a×b} + P^{b×a})
        F̂_{μν} = (1/3)(5·ImC^{(1,1)} - ImC^{(1,2)}),  ImC = (C-C†)/2i,  无迹 Hermitian
  3) 0^{-+} 算符: 对原型 Tr[F̂_{T,X}F̂_{Y,Z}] 施加立方群 O 的全部 24 个转动并做 A_1 投影:
        O(x,t) = (1/24) Σ_{g∈O} s_g Tr[F̂_{T,ê_{p_g(0)}} F̂_{ê_{p_g(1)},ê_{p_g(2)}}]
      (转动 g 由有号置换矩阵 R 描述, p_g(i) = R 第 i 行的非零列号, s_g = Π_i R_{i,p_g(i)};
       该投影恒等于 (1/6)ε_{ijk}Tr[F̂_{T,i}F̂_{j,k}] ∝ Tr(E·B), 宇称奇、C=+)
  4) 零动量 + 两点函数:  O(t)=Σ_x O(x,t),  C(τ)=(1/Nt)Σ_t O(t+τ)O(t)
     (0^{-+} 宇称奇, 系综真空期望严格为零 ⟨O⟩≡0, 不做真空减除;
      分析文档 §2.5 也只要求 A_1^{++} 通道减除非零真空期望值)
  5) 有效质量 (log / cosh) 并输出 ASCII (L. Liu 格式) 与图

注: 场强型 (E·B 拓扑密度型) 算符在格点上满足 A_1 代数投影, 但严格转动协变性
    只到 O(a) 级 (闭合环起始点循环置换不保持 Im 部分, 连续极限消失), 这是该类
    算符的标准性质 (见 2006 Chen 的 O^{(PS)}); 如需格点精确 A_1^{-+}, 应改用
    空间 Wilson 环的直接 24 转动几何投影 (2018 Sun 的 BDB 型算符), 作为后续扩展。
"""

# ---- 标准库导入 ----
import sys                       # sys.argv: 从命令行读取组态编号 (默认 cfg_1000)
import itertools                 # permutations/product: 枚举立方群 24 个转动所需的有号置换
import numpy as np               # 所有数组运算 (张量收缩、roll 周期平移、einsum 迹收缩)

# ---- 绘图库: 使用无显示后端 (服务器环境无 X 窗口) ----
import matplotlib                # matplotlib 顶层模块
matplotlib.use("Agg")            # 选择 Agg 后端: 只输出 PNG 文件, 不弹窗口
import matplotlib.pyplot as plt  # pyplot: 绘制关联函数与有效质量图

# ---- lqcddb_update (distillation 包) 导入 ----
from lqcddb import set_backend   # 全局数值后端切换函数 (numpy / cupy)
set_backend("numpy")             # 本脚本使用 CPU (numpy) 后端

# readin_gauge: 读取 ILDG/LIME 规范组态 -> (4,Nt,Nz,Ny,Nx,3,3) 复数数组
# write_data_ascii: 把关联函数按 L. Liu 的 ASCII 格式写出 (首行 "nsamples T complex L 1")
from lqcddb.io import readin_gauge, write_data_ascii

# ---------------------------------------------------------------- 常数
# 组态目录: E32P29 的所有组态文件 (.lime)
CONF_DIR = "/public/group/lqcd/configurations/CLOVER/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64"
# 组态文件名前缀: {ENS}_cfg_{编号}.lime
ENS = "beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64"
A_FM = 0.0897      # E32P29 各向同性格距 a (fm), 见 lqcddb physics.md
FM2GEV = 0.197     # 转换常数: 1 fm^{-1} = 0.197 GeV, 用于 a^{-1}[GeV] = fm2GeV/a_fm

# 方向 -> 数组轴 映射: U[mu] 子数组的形状是 (Nt,Nz,Ny,Nx,3,3),
#   方向 X 沿最后空间轴 (轴 3), Y 沿轴 2, Z 沿轴 1, T 沿轴 0
AXIS_OF_DIR = {0: 3, 1: 2, 2: 1, 3: 0}
# 方向名称 (仅用于打印)
DIRS = ("X", "Y", "Z", "T")


# ---------------------------------------------------------------- 立方群
def cubic_group_A1_terms():
    """立方群 O 的全部 24 个转动 (det=+1 的有号置换矩阵)。

    每个转动 R 返回 (p0, p1, p2, s):
      p_i = 第 i 行的非零列号 (空间方向 X,Y,Z 的像, 0 基),
      s   = Π_i R_{i,p_i} (行列式符号; 作用在原型 Tr[F̂_{T,X}F̂_{Y,Z}] 上产生
            s·Tr[F̂_{T,ê_{p0}}F̂_{ê_{p1},ê_{p2}}]).
    """
    terms = []                                        # 收集 24 个 (p0,p1,p2,s) 元组
    # 对 3! = 6 个行排列 perm 遍历 (R 的每一行恰有一个非零元, 列位置由 perm 给出)
    for perm in itertools.permutations(range(3)):
        # 对 2^3 = 8 组符号遍历 (每行非零元的取值 ±1)
        for signs in itertools.product((-1, 1), repeat=3):
            R = np.zeros((3, 3), dtype=int)           # 初始化 3x3 整数矩阵
            for i, j in enumerate(perm):              # i = 行号, j = 该行非零元的列号
                R[i, j] = signs[i]                    # 填非零元: R_{i,perm(i)} = ±1 (有号置换矩阵)
            # 立方群元素要求 det(R) = +1 (纯转动, 不含反射);
            # det = 符号乘积 × 置换奇偶, 只有一半 (24 个) 满足
            if int(round(np.linalg.det(R))) != 1:
                continue                               # 跳过反射 (det = -1)
            # p_i = 第 i 行的非零列号: 即空间基矢 ê_{i+1} 在转动 g 下的像 (行->列映射)
            # 该映射决定原型算符 (T,X) x (Y,Z) 旋转后变成哪个平面组合
            p = tuple(int(np.argwhere(R[i, :] != 0)[0, 0]) for i in range(3))
            # s = Π_i R_{i,p_i}: 三个非零元的乘积 = 转动带来的符号因子 s_g
            # (旋转原型 Tr[F̂_TX F̂_YZ] 得到 s_g·Tr[F̂_{T,ê_{p0}} F̂_{ê_{p1},ê_{p2}}])
            s = int(R[0, p[0]] * R[1, p[1]] * R[2, p[2]])
            terms.append((p[0], p[1], p[2], s))       # 记录该转动对应的 (平面像, 符号)
    # 立方群 O 恰有 24 个元素, 断言防止构造错误
    assert len(terms) == 24, "立方群转动数应为 24"
    return terms                                       # 返回 24 个转动的系数表


# ---------------------------------------------------------------- 场强
# def clover_field_strength(U, mu, nu, improved=True):
#     """Type-II 改进局域场强 F̂_{μν} = (1/3)(5·ImC^{(1,1)} - ImC^{(1,2)}).

#     U    : 规范场 (4, Nt, Nz, Ny, Nx, 3, 3), 方向 (X,Y,Z,T)
#     mu,nu: 平面指标 (0..3)
#     返回 : 无迹 Hermitian 3x3 矩阵场, 形状 (Nt, Nz, Ny, Nx, 3, 3)
#     """
#     # 平面 (μ,ν) 两方向对应的数组轴: 用于 np.roll 沿该轴平移链接
#     a_mu, a_nu = AXIS_OF_DIR[mu], AXIS_OF_DIR[nu]
#     Um, Un = U[mu], U[nu]                              # 链接 U_μ(x), U_ν(x) (全体积数组)
#     Umd = Um.conj().swapaxes(-1, -2)   # U_μ†: 颜色指标转置+复共轭 (U_μ(x) 的逆链接)
#     Und = Un.conj().swapaxes(-1, -2)   # U_ν†

#     # ---- 1x1 plaquette (角 x) ----
#     # 基本 plaquette: P_{μν}(x) = U_μ(x) U_ν(x+μ̂) U_μ†(x+ν̂) U_ν†(x)
#     # np.roll(A, -1, 轴)[x] = A[x+1]: "沿 +μ̂ 平移一个格距" 的实现
#     P = Um @ np.roll(Un, -1, a_mu) @ np.roll(Umd, -1, a_nu) @ Und
#     # 中心对称 clover C^{(1,1)}(x) = (1/4) Σ_四象限 P(角):
#     #   四个角 = x, x-μ̂, x-ν̂, x-μ̂-ν̂ (np.roll(·, +1, 轴) = 沿 -μ̂ 平移)
#     # 物理意义: 对角点求平均, 抵消环的奇次导数项, 使场强更局域
#     C11 = (P + np.roll(P, +1, a_mu) + np.roll(P, +1, a_nu)
#            + np.roll(np.roll(P, +1, a_mu), +1, a_nu)) / 4.0

#     if not improved:
#         # 未改进版: 场强直接取 1x1 clover 的反厄米部分
#         # Im C = (C - C†)/2i : 取 plaquette 的虚部, 对应 F_{μν} (线性项), 去掉单位矩阵部分
#         F = (C11 - C11.conj().swapaxes(-1, -2)) / (2j)
#     else:
#         # ---- 1x2 矩形 (沿 ν 方向长 2, 沿 μ 方向长 1) ----
#         # 路径: x -> x+μ̂ -> x+μ̂+ν̂ -> x+μ̂+2ν̂ -> x+2ν̂ -> x+ν̂ -> x
#         #   = U_μ(x) U_ν(x+μ̂) U_ν(x+μ̂+ν̂) U_μ†(x+2ν̂) U_ν†(x+ν̂) U_ν†(x)
#         R12 = (Um @ np.roll(Un, -1, a_mu)              # U_ν(x+μ̂)
#                @ np.roll(np.roll(Un, -1, a_mu), -1, a_nu)   # U_ν(x+μ̂+ν̂)
#                @ np.roll(Umd, -2, a_nu)                # U_μ†(x+2ν̂)
#                @ np.roll(Und, -1, a_nu) @ Und)         # U_ν†(x+ν̂), U_ν†(x)
#         # ---- 2x1 矩形 (沿 μ 方向长 2, 沿 ν 方向长 1) ----
#         # 路径: x -> x+μ̂ -> x+2μ̂ -> x+2μ̂+ν̂ -> x+μ̂+ν̂ -> x+ν̂ -> x
#         #   = U_μ(x) U_μ(x+μ̂) U_ν(x+2μ̂) U_μ†(x+μ̂+ν̂) U_μ†(x+ν̂) U_ν†(x)
#         R21 = (Um @ np.roll(Um, -1, a_mu)              # U_μ(x+μ̂)
#                @ np.roll(Un, -2, a_mu)                 # U_ν(x+2μ̂)
#                @ np.roll(np.roll(Umd, -1, a_mu), -1, a_nu)   # U_μ†(x+μ̂+ν̂)
#                @ np.roll(Umd, -1, a_nu) @ Und)         # U_μ†(x+ν̂), U_ν†(x)
#         Rsum = R12 + R21                               # C^{(1,2)} 的分子: P^{1×2} + P^{2×1}
#         # 对称化 clover: C^{(1,2)}(x) = (1/8) Σ_四象限 (P^{1×2} + P^{2×1})
#         # (1×2 与 2×1 两个取向 + 4 个角点 -> 除以 8)
#         C12 = (Rsum + np.roll(Rsum, +1, a_mu) + np.roll(Rsum, +1, a_nu)
#                + np.roll(np.roll(Rsum, +1, a_mu), +1, a_nu)) / 8.0
#         # 取反厄米 (虚) 部分: ImC^{(1,1)}, ImC^{(1,2)}
#         # 1x1 环展开: ImC^{(1,1)} = ga²F + O(a⁴D²F); 1x2 环: ImC^{(1,2)} = 2ga²F + (5/6)ga⁴D²F
#         ImC11 = (C11 - C11.conj().swapaxes(-1, -2)) / (2j)
#         ImC12 = (C12 - C12.conj().swapaxes(-1, -2)) / (2j)
#         # Type-II 改进 (2006 Chen): F̂ = (1/3)(5·ImC^{(1,1)} - ImC^{(1,2)})
#         # 系数选择使 O(a²) 导数项 5×(1/6) - (5/6) = 0 精确抵消, 得 F̂ = ga²F + O(a⁶)
#         F = (5 * ImC11 - ImC12) / 3.0

#     # 无迹化: F̂ -= (Tr F̂ / 3)·I, 使场强矩阵无迹 (规范场 F_{μν} 无迹的基本性质)
#     F -= (np.trace(F, axis1=-2, axis2=-1) / 3.0)[..., None, None] * np.eye(3)
#     return F                                           # 返回无迹 Hermitian 场强场

def clover_field_strength(U, mu, nu, improved=True):
    """共同基点的 clover 场强；假定周期边界。

    U: (4, Nt, Nz, Ny, Nx, 3, 3)，方向顺序为 X,Y,Z,T。
    mu, nu: 0..3，且 mu != nu。
    返回无迹 Hermitian 矩阵场，形状 (Nt, Nz, Ny, Nx, 3, 3)。
    improved=True 时采用 (5 ImC11 - ImC12)/3。
    """
    if U.ndim != 7 or U.shape[0] != 4 or U.shape[-2:] != (3, 3):
        raise ValueError("U 必须具有形状 (4, Nt, Nz, Ny, Nx, 3, 3)")
    if mu not in range(4) or nu not in range(4) or mu == nu:
        raise ValueError("mu、nu 必须是不同的 0..3 方向")

    def dagger(A):
        return A.conj().swapaxes(-1, -2)

    def loop(path):
        # ±1, ±2, ±3, ±4 分别表示 ±X, ±Y, ±Z, ±T。
        # offset 按数组坐标 (t,z,y,x) 记录相对基点的位置。
        offset = [0, 0, 0, 0]
        W = None

        for step in path:
            d = abs(step) - 1
            axis = 3 - d

            # 负方向：U_d^dagger(y-d)，先退一步。
            if step < 0:
                offset[axis] -= 1

            link = U[d]
            if any(offset):
                # roll(A, -offset)[x] = A[x + offset]
                link = np.roll(
                    link,
                    tuple(-n for n in offset),
                    axis=(0, 1, 2, 3),
                )

            if step < 0:
                link = dagger(link)

            W = link if W is None else W @ link

            # 正方向：乘 U_d(y) 后再前进一步。
            if step > 0:
                offset[axis] += 1

        if any(offset):
            raise ValueError("Wilson 环路径未闭合")
        return W

    def clover(m, n):
        # 沿 mu 方向长度 m，沿 nu 方向长度 n。
        p = (mu + 1,) * m
        q = (nu + 1,) * n
        pm = (-(mu + 1),) * m
        qm = (-(nu + 1),) * n

        # 每条路径均从 x 出发、回到 x，且环绕方向相同。
        paths = (
            p + q + pm + qm,
            q + pm + qm + p,
            pm + qm + p + q,
            qm + p + q + pm,
        )

        C = loop(paths[0])
        for path in paths[1:]:
            C += loop(path)
        return C / 4.0

    # 四个 1×1 环。
    C11 = clover(1, 1)
    F = (C11 - dagger(C11)) / (2j)
    del C11

    if improved:
        # 四个 1×2 环 + 四个 2×1 环，共八个环。
        C12 = clover(1, 2)
        C12 += clover(2, 1)
        C12 *= 0.5

        ImC12 = (C12 - dagger(C12)) / (2j)
        F = (5.0 * F - ImC12) / 3.0

    # 无迹化。
    F -= (
        np.trace(F, axis1=-2, axis2=-1) / 3.0
    )[..., None, None] * np.eye(3)

    return F

# ---------------------------------------------------------------- 0^{-+} 算符
def getF(F, mu, nu):
    """F[mu,nu] 的规范读取 (F 只存 mu<nu 的规范平面, 反称性 F[nu,mu] = -F[mu,nu])."""
    if mu < nu:                                        # 规范顺序 (mu<nu) 直接读取
        return F[(mu, nu)]
    return -F[(nu, mu)]                                # 反序平面: 利用 F_{νμ} = -F_{μν} 反号


def glueball_operator(F, terms):
    """O(t) = (1/24) Σ_g s_g Σ_x Tr[F̂_{T,a} F̂_{b,c}],  (a,b,c) = (p0,p1,p2) 为空间方向 0..2,
    时间方向取数组方向 3 (T)."""
    # 取任意一个场强的 Nt (所有平面时间长度相同)
    Nt = next(iter(F.values())).shape[0]
    O = np.zeros(Nt, dtype=float)                      # 每时间片一个算符值 (实数)
    # 遍历 24 个转动: 每个转动贡献 s_g·Σ_x Tr[F̂_{T,ê_{p0}} F̂_{ê_{p1},ê_{p2}}]
    for p0, p1, p2, s in terms:
        Fa = getF(F, 3, p0)                            # 电场部分: F̂_{T,ê_{p0}} (时间-空间平面)
        Fbc = getF(F, p1, p2)                          # 磁场部分: F̂_{ê_{p1},ê_{p2}} (纯空间平面)
        # einsum "tzyxab,tzyxba->t": 对格点坐标 (z,y,x) 与颜色指标 (a,b) 全部求和
        #   = Σ_x Tr_c[F̂_{T,ê_{p0}}(x) F̂_{ê_{p1},ê_{p2}}(x)], 结果只按时间片 t 索引
        # .real: Tr 对两个 Hermitian 矩阵是实数 (算符 C=+)
        O += s * np.einsum("tzyxab,tzyxba->t", Fa, Fbc).real
    return O / 24.0                                    # A_1 投影: 除以群阶 24, 得到 (1/6)ε_ijk 组合


def epsilon_combination(F):
    """直接计算 (1/6) ε_ijk Σ_x Tr[F̂_{T,i} F̂_{j,k}], 用于验证 24 转动投影等价性."""
    # 独立计算同一算符的 ε 收缩形式, 用于数值验证 24 转动投影的正确性
    Nt = next(iter(F.values())).shape[0]
    O = np.zeros(Nt, dtype=float)
    # ε_ijk 的 6 个非零项 = (i,j,k) 的全部排列, 符号 = 排列奇偶
    for perm in itertools.permutations(range(3)):
        inv = 0                                        # 逆序数计数 (冒泡式两两比较)
        for a in range(3):
            for b in range(a + 1, 3):
                if perm[a] > perm[b]:                  # 出现逆序对 -> 逆序数 +1
                    inv += 1
        sign = 1 if inv % 2 == 0 else -1               # 奇排列取 -1, 偶排列取 +1 (即 ε 符号)
        i, j, k = perm                                 # (i,j,k) 一个排列
        # Σ_x Tr[F̂_{T,i}F̂_{j,k}]: 同上 einsum 的迹-求和, 乘 ε 符号后累加
        O += sign * np.einsum("tzyxab,tzyxba->t",
                              getF(F, 3, i), getF(F, j, k)).real
    return O / 6.0                                     # 除以 6 (6 项 ε 求和) -> (1/6)ε_ijk 组合


# ---------------------------------------------------------------- 两点函数
def two_point(O):
    """C(τ) = (1/Nt) Σ_t O(t+τ) O(t).

    0^{-+} 算符宇称奇, 系综真空期望值严格为零 (⟨O⟩ ≡ 0, 宇称对称性),
    因此不做真空减除 (分析文档 §2.5: 只有 A_1^{++} 通道必须减去非零真空期望值).
    单组态上 O(t) 的时间平均可能非零 (纯统计涨落), 它不进入关联函数的定义.
    """
    Nt = O.size                                        # 时间方向格点数 (64)
    C = np.empty(Nt)                                   # 关联函数数组 C(τ), τ = 0..Nt-1
    for tau in range(Nt):
        # np.roll(O, -tau)[t] = O(t+τ) (mod Nt, 周期边界):
        # C(τ) = (1/Nt) Σ_t O(t+τ) O(t)  = mean( roll(O,-τ) * O )
        # 零动量投影已由 O(t)=Σ_x O(x,t) 完成, 此式给出胶球谱分解
        #   C(τ) = Σ_n |Z_n|²/(2E_n V) · [e^{-E_n τ} + e^{-E_n (Nt-τ)}]
        C[tau] = np.mean(np.roll(O, -tau) * O)
    return C


def meff_log(C):
    # log 型有效质量: m_eff(τ) = ln[C(τ)/C(τ+1)]  (格点单位, 对应单指数衰减 e^{-mτ})
    # 仅适用于 τ 远小于 Nt/2 (反向传播项可忽略的区域)
    with np.errstate(invalid="ignore", divide="ignore"):   # 负值/零值产生 NaN 时不报警告
        return np.log(C[:-1] / C[1:])                      # C(τ)/C(τ+1) 逐点求对数


def meff_cosh(C):
    # cosh 型有效质量 (周期边界下对单态双项形式 C(t)=A[e^{-mt}+e^{-m(Nt-t)}]/(1+e^{-mNt})):
    #   由恒等式 C(t-1) + C(t+1) = 2 cosh(m) C(t)  推出
    #   m_eff(t) = arccosh{[C(t-1) + C(t+1)] / [2 C(t)]}
    # 注意: 仅对内部点 1 <= t <= Nt-2 有效 (边界点 C(-1) 或 C(Nt) 的周期
    # 回绕不满足双项谱分解形式), 与 lqcddb meff 的 cosh 有效区间一致
    Nt = C.size
    m = np.full(Nt, np.nan)                            # 全部初始化为 NaN (无效点)
    with np.errstate(invalid="ignore", divide="ignore"):
        for t in range(1, Nt - 1):                     # 内部点 1..Nt-2
            num = C[t - 1] + C[t + 1]                  # 分子: C(t-1) + C(t+1)
            den = 2 * C[t]                             # 分母: 2C(t)
            r = num / den                              # 比值 (物理上应 > 1)
            m[t] = np.arccosh(r) if r > 1 else np.nan  # arccosh 定义域 r ≥ 1, 否则置 NaN
    return m


# ---------------------------------------------------------------- 主流程
def main():
    # 命令行参数: python 0-+.py <组态编号>, 缺省用 cfg_1000
    conf_id = sys.argv[1] if len(sys.argv) > 1 else "1000"
    # 拼出组态文件完整路径: {目录}/{前缀}_cfg_{编号}.lime
    path = f"{CONF_DIR}/{ENS}_cfg_{conf_id}.lime"
    print(f"[1] 读取组态: {path}")
    # 用 lqcddb_update 的 readin_gauge 读取 ILDG/LIME 组态:
    # 返回 (4, Nt, Nz, Ny, Nx, 3, 3) complex 数组 (方向 X,Y,Z,T)
    U = readin_gauge(path)
    print(f"    形状 {U.shape}, dtype {U.dtype}")

    improved = True                                    # 使用 Type-II 改进场强
    print(f"[2] 计算改进 clover 场强 F̂_μν (Type-II: 5/3·ImC11 - 1/3·ImC12, improved={improved})")
    # 需要的 6 个平面: 3 个时间-空间 (E 部分) + 3 个纯空间 (B 部分)
    planes = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    F = {}                                             # 场强字典: 键 (mu,nu) (mu<nu 规范顺序)
    for mu, nu in planes:
        # 对每个平面调用改进 clover 公式计算 F̂_{μν}(x)
        F[(mu, nu)] = clover_field_strength(U, mu, nu, improved=improved)
        f = F[(mu, nu)]
        # 数值自检: 打印 Hermitian 性残差 |F - F†| 与无迹性残差 |Tr F|
        # (F̂ 应为无迹 Hermitian 矩阵, 残差 ~1e-16 表示构造正确)
        print(f"    F̂_{DIRS[mu]}{DIRS[nu]}: max|Hermit残差|="
              f"{np.abs(f - f.conj().swapaxes(-1, -2)).max():.2e}, "
              f"max|Tr|={np.abs(np.trace(f, axis1=-2, axis2=-1)).max():.2e}")

    print("[3] 立方群 O 的 24 个转动 (行→列像 (p0,p1,p2), 符号 s):")
    # 枚举立方群全部 24 个转动, 得到 A_1 投影系数 (p0,p1,p2,s)
    terms = cubic_group_A1_terms()
    # 打印每个转动对应的平面组合与符号, 便于人工核对
    for p0, p1, p2, s in terms:
        print(f"    g: (p0,p1,p2) = ({p0},{p1},{p2})  平面 (T,{DIRS[p0]}) x ({DIRS[p1]},{DIRS[p2]}),  s = {s}")

    print("[4] 构造 0^{-+} 算符 O(t) (A_1 投影) 并验证与 (1/6)ε_ijk Tr(F̂_Ti F̂_jk) 等价")
    # O(t) = (1/24) Σ_g s_g Σ_x Tr[F̂_{T,ê_{p0}}F̂_{ê_{p1},ê_{p2}}]: 24 转动 A_1 投影算符
    # (原型 Tr[F̂_TX F̂_YZ] 已宇称奇, 故 A_1^{-+} 投影中的宇称部分自动满足:
    #  (1/48)Σ_{g∈O_h}(1-P̂)O = (1/48)Σ_r r·2O = (1/24)Σ_r r·O, 即 24 转动平均)
    O = glueball_operator(F, terms)
    # 独立用 ε_ijk 收缩计算同一算符, 两者应逐时间片相等 (数值验证投影系数正确)
    O_eps = epsilon_combination(F)
    print(f"    max|O_24rot - O_epsilon| = {np.abs(O - O_eps).max():.3e}")
    # 诊断打印: 宇称奇算符的系综真空期望严格为 0; 单组态时间平均仅统计涨落, 不作减除
    print(f"    时间平均 <O>_t = {O.mean():+.6e}  (宇称奇算符, 系综 <O>=0, 不参与关联函数定义)")

    print("[5] 两点函数 C(τ) 与有效质量")
    # 两点函数: C(τ) = (1/Nt) Σ_t O(t+τ) O(t)  (不做真空减除: 0^{-+} 系综 <O>≡0)
    C = two_point(O)
    Nt = C.size                                        # Nt = 64
    # log 型有效质量: m(τ) = ln[C(τ)/C(τ+1)]  (格点单位)
    m_log = meff_log(C)
    # cosh 型有效质量: m(t) = arccosh[(C(t-1)+C(t+1))/(2C(t))]  (周期边界精确)
    m_cosh = meff_cosh(C)
    # 格点单位 -> 物理单位: m[GeV] = m_lat × (fm2GeV / a_fm), 即乘 a^{-1}[GeV]
    GeV = FM2GEV / A_FM
    print(f"    a = {A_FM} fm,  a^{-1} = {GeV:.3f} GeV  (E32P29, 各向同性)")
    print(f"    {'tau':>4} {'C(tau)':>14} {'m_eff(log) [GeV]':>18} {'m_eff(cosh) [GeV]':>18}")
    # 只打印 τ < Nt/2 (前半段, 反向传播污染较小)
    for t in range(Nt // 2):
        # 无效值 (NaN, 来自负的 C 比值) 以 nan 显示
        ml = m_log[t] * GeV if np.isfinite(m_log[t]) else float("nan")
        mc = m_cosh[t] * GeV if np.isfinite(m_cosh[t]) else float("nan")
        print(f"    {t:4d} {C[t]:14.6e} {ml:18.4f} {mc:18.4f}")

    # 保存 ASCII (L. Liu 格式): 首行 "nsamples T complex L 1"
    # data 形状 (1, Nt): 1 个样本, 每个样本 Nt=64 个时间点
    out_name = f"glueball_0mplus_conf{conf_id}.dat"
    write_data_ascii(C.reshape(1, -1), T=Nt, L=32, filename=out_name, complex=False)
    print(f"    关联函数已写入 {out_name}")

    # 图: 左图为关联函数 (对数坐标), 右图为有效质量
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    t = np.arange(Nt)                                  # 时间轴 τ = 0..Nt-1
    # 左图: |C(τ)| 半对数图 (胶球信号应呈指数衰减 -> 直线; 噪声主导时无规律)
    axes[0].semilogy(t, np.abs(C), "o-")
    axes[0].set_xlabel(r"$\tau/a_t$")                  # 横轴: 格点时间
    axes[0].set_ylabel(r"$|C(\tau)|$")                 # 纵轴: 关联函数绝对值
    axes[0].set_title(r"$0^{-+}$ glueball correlator, conf %s" % conf_id)
    # 右图: log 型与 cosh 型有效质量 (物理单位 GeV), 平台值即胶球质量
    axes[1].plot(t[:Nt // 2], m_log[:Nt // 2] * GeV, "s-", label="log")
    axes[1].plot(t[:Nt // 2], m_cosh[:Nt // 2] * GeV, "o-", label="cosh")
    axes[1].set_xlabel(r"$\tau/a_t$")                  # 横轴: 格点时间
    axes[1].set_ylabel(r"$m_{\rm eff}$ [GeV]")         # 纵轴: 有效质量
    axes[1].legend()                                   # 图例
    axes[1].set_title(r"$0^{-+}$ glueball effective mass, conf %s" % conf_id)
    fig.tight_layout()                                 # 自动调整子图间距
    fig_name = f"glueball_0mplus_conf{conf_id}.png"
    fig.savefig(fig_name, dpi=150)                     # 保存 PNG (150 dpi)
    print(f"    图已保存 {fig_name}")


if __name__ == "__main__":
    main()                                             # 直接运行时执行主流程