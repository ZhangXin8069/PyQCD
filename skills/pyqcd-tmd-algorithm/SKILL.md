---
name: pyqcd-tmd-algorithm
description: |
  Use when implementing, porting, debugging, or validating PyQCD's gradient-flow
  renormalized nucleon gluon TMD-PDF algorithm, especially flowed Clover fields,
  finite staple geometry, disconnected matrix elements, soft/rapidity layers,
  Collins–Soper evolution, matching, or continuum-limit tests; keep quasi-PDF
  prototypes distinct from a complete TMD result.
metadata:
  openclaw:
    emoji: 🧬
---

# pyqcd-tmd-algorithm — 物理到实现的算法契约

## 目的与边界

本技能把

\[
f_{g/N}^{[\Gamma]}(x,\boldsymbol b_\perp;\mu,\zeta)
\longleftarrow h_g^{\mathrm{flow}}(z,\boldsymbol b_\perp;P_z,\tau,\ell)
\]

映射为数组、接口、统计量和验证门，要求“方程—离散几何—代码—证据”闭环。它补充
`pyqcd-tmd-chain` 的六步总览；纯算符/关联函数推导转 `pyqcd-physics-correlator`，
纯谱式转 `pyqcd-physics-spectrum`，统计执行转 `pyqcd-statistics`。

参考代码只用于理解；实现必须留在 PyQCD 内，不得 import `refer/` 或 `examples/`。

## 按需参考

| 当前问题 | 必读 reference | 产出 |
|---|---|---|
| `tau/z/b`、Clover、staple、表示、路径、正负方向 | [`references/geometry.md`](references/geometry.md) | 几何与算符契约 |
| `C2*Lg`、soft/Z_R、CS、匹配、外推 | [`references/renormalization.md`](references/renormalization.md) | 重整化与提取方案 |
| 状态升级、门控顺序、失败边界 | [`references/validation.md`](references/validation.md) | 可审计的验收矩阵 |

不涉及对应模式时只读相关 reference；跨层任务依次读取 geometry → renormalization →
validation。

## 四种状态

代码、日志和报告只能使用与证据相符的最高状态：

| 状态 | 最低证据 | 允许的结论 |
|---|---|---|
| 实现存在 | 函数或文件可调用 | 接口存在 |
| 测试通过 | 受控输入满足形状/数值断言 | 该断言通过 |
| 方案闭合 | soft、rapidity、流方案、匹配和尺度明确 | 可讨论方案内物理量 |
| 真实数据验证 | 真实逐组态三点、系统扫描和误差账本完成 | 可报告物理结果 |

当前已有 Wilson flow、Clover/对偶场强、直线 OPE、部分 staple/quasi-TMD、质子 2pt
和 Fourier/matching 原型；接口存在不等于完整核子胶子 TMD-PDF 完成。

## Step 0：先写算法契约

配置或元数据必须先固定：通道与 Lorentz 投影、纵向/横向轴、`z`、`b_perp`、独立
`staple_length`、`tau`、`Pz`、`mu`、`zeta`、表示、匹配阶数和 Fourier 约定。当前
`b_perp/b_dir` 只覆盖一个横向轴；完整二维 TMD 需显式支持
`b_perp=(b_x,b_y)` 与旋转对称性对照。中心几何与锚定几何不能混用，复数不能在证明
对称性前取实部或强行偶化。

每个产物记录 `a_fm`、格点体积、`tau`、流步长、`Pz_gev`、`z`、`b_perp`、`staple_length`、
方向、表示、Lorentz 通道、`mu`、`zeta`、阶数、seed、重采样、cov/SVD、命令行和代码版本。

未指定扫描时，用 `b_perp=0`/非零横向向量、两个 `Pz`、两个有限 `ell`、两个邻近 `tau`
组成 smoke 矩阵；它只暴露几何/统计问题，不代表物理结果，且所有格点共享组态索引。

## 推荐数据流

每一步只消费上一步已通过的对象：

1. 输入：验证规范场形状、边界、SU(3) 近似幺正性和元数据。
2. 流化：从同一 `U` 计算并缓存各 `V_tau`，记录步长收敛和流能量判据。
3. 算符：在同一 `V_tau` 上构造 `F/F_tilde`、有限 staple 和 Lorentz 分量，先保留复数。
4. 软因子：独立计算同表示、同几何、同流时间的真空对象。
5. 外态：保存逐组态 `C2`、`Lg`、`C2*Lg`，做真空扣除和多个 `t_sep` 平台。
6. 统计：对 `z/b/ell/tau/±z/Pz` 使用共享索引的 block-jackknife/bootstrap。
7. 重整化：按明确方案应用 soft、短距比值、Z_R/混合和流到 MS 的系数。
8. 提取：按已证明的偶奇性质做 cos/sin 或复 Fourier，再用两动量提取 CS/匹配。
9. 外推：检查 `ell`、`tau→0`、`Pz→∞`、`a→0`、体积和拟合窗，分离统计/系统误差。
10. 归档：HDF5/JSON 保存数组、维度、单位、参数、seed、命令、版本和每个门的结果。

## 完整链的停止条件

只有“非零 `b_perp` + 有限 `ell` + 真实核子胶子三点 + 同几何 soft/rapidity 处理 +
胶子匹配 + 多尺度误差验证”全部有证据时，才称第一版完整链。否则使用“梯度流胶子
quasi-PDF 原型”“带 staple 的裸准关联函数”或“接口/测试骨架”等准确名称。逐门证据、
典型错误与当前 API 边界见上述 references。

## 路由与交接

- 统一轴序、单位、γ 和边界 → `pyqcd-conventions`；核子 2pt/3pt/Wick →
  `pyqcd-physics-correlator`；传播子与顺序源 → `pyqcd-propagator`。
- 重采样、协方差和窗口 → `pyqcd-statistics`；批量运行/IO/后端 → `pyqcd-pipeline` /
  `pyqcd-infra`；结果成文 → `pyqcd-docs`。
- 六步物理链和现有入口地图 → `pyqcd-tmd-chain`；本技能只在实现契约、几何边界或
  物理验收需要时被调用。
