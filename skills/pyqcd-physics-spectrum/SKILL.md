---
name: pyqcd-physics-spectrum
description: |
  Use when a lattice-QCD task needs a spectral decomposition, dispersion likelihood,
  finite-time backward states, excited-state model, two- or three-point fit template,
  or a decision about identifiable energy or overlap parameters; use
  pyqcd-physics-correlator for operator/Wick definitions and pyqcd-statistics for fits.
metadata:
  openclaw:
    emoji: 🎼
---

# pyqcd-physics-spectrum — 谱分解与拟合模板

## 目的与边界

本技能把已经确定的关联函数变成“态、能量、重叠因子和拟合模型”。它不重新定义
内插算符、Wick 缩并、传播子求解或重采样算法：前者交给
`pyqcd-physics-correlator`，后者分别交给 `pyqcd-propagator`、`pyqcd-statistics`。
时间轴、γ 矩阵、边界和符号以 `pyqcd-conventions` 为准。

## 谱分解的最小推导

采用有限体积相对论归一化

\[
\langle n,\vec p|m,\vec q\rangle=2E_nV\,\delta_{nm}\delta_{\vec p\vec q},
\qquad
\langle0|\mathcal O|n,\vec p\rangle=Z_n(\vec p).
\]

插入完备关系后，长时间两点函数为

\[
C_2(\vec p,t)=\sum_n A_n(\vec p)e^{-E_n(\vec p)t},
\qquad A_n=\frac{|Z_n|^2}{2E_n},
\]

其中归一化因子可以吸收进 $Z_n$，但必须在交接表说明。有限时间方向 $T$ 时，不能
把 backward 项默认删除：

| 通道 | 最小模型 | 物理解释 |
|---|---|---|
| 介子 | $\sum_n A_n[e^{-E_nt}+e^{-E_n(T-t)}]$ | 偶数夸克的有效周期边界 |
| 重子 $P^+$ | $\sum_n A_n^+e^{-E_n^+t}-\sum_n A_n^-e^{-E_n^-(T-t)}$ | 奇数夸克的反周期边界；backward 为反宇称伙伴 |

重叠因子不是装饰：smear 源的目标是增大 $|Z_0|/|Z_1|$，而首个激发态污染量级为
$(A_1/A_0)e^{-\Delta E t}$。

## 三点函数模型

对 $0<\tau<t_{\rm sep}$ 两侧插入完备关系：

\[
C_3(\tau,t_{\rm sep})=
\sum_{n,m}B_{nm}e^{-E_n^f(t_{\rm sep}-\tau)}e^{-E_m^i\tau},
\]

\[
B_{nm}=\frac{Z_n^f\,\langle n|J|m\rangle\,(Z_m^i)^*}{4E_n^fE_m^i}.
\]

因此 C₂+C₃ 联合拟合应共享 $E_n$ 和 $Z_n$，把基态矩阵元留作独立参数；不得把
三点振幅直接当作矩阵元。若 $t_{\rm sep}\ll T$，热 backward 项可以作为已声明的
近似；接近 $T$ 时必须扩展模型。

多态模型的 `data_identifiability`、`fit_quality_status`、`physical_result_status`、
条件数和逐窗口/重采样统计元数据由 `pyqcd-statistics` 定义；本技能只规定谱式、态的
物理含义、边界项和参数化。满秩或优化器收敛不自动等于物理结果可靠，具体状态契约见
[`pyqcd-statistics`](../pyqcd-statistics/SKILL.md) 与
[`identifiability.md`](../pyqcd-statistics/references/identifiability.md)。

当前质子能谱入口用 AICc 比较一态与两态时，一态胜出表示该窗口没有支持显式激发态项：
保留一态的 `c0/E0`，固定输出 schema 中的 `c1/dE` 写 NaN，并把激发态状态记为
`practically_unidentifiable`；不得用两态优化器给出的有限数补回它们。这个选择只说明
给定候选与窗口中的相对偏好，不验证被模型省略的有限时 backward 反宇称项，也不自动
把 `E0` 升级为已验证物理基态能量。

## 给分析层的交接表

| 输入 | 拟合函数 | 至少记录 |
|---|---|---|
| 介子 C₂ | cosh 型 | $T$、$\{E_n,A_n\}$、状态数 |
| 重子投影 C₂ | forward − backward 反宇称 | $T$、投影、$\{E_n^\pm,A_n^\pm\}$ |
| C₃ | 双指数或联合多态模型 | $t_{\rm sep}$、$\tau$ 范围、共享的 $E,Z$、$B_{nm}$ |
| 有效质量/能量 | 由上面模型导出的诊断量 | 定义、时间窗、单位和边界处理 |

## 色散拟合契约

当前三参数模型为

\[
E(P)=\sqrt{m^2+k_2P^2+k_3a^2P^4}.
\]

`fit_dispersion` 在测得的能量 $E$ 空间构造加权残差，不对 $E^2$ 使用一阶误差传播。
用于可辨识诊断的 $(m^2,k_2,k_3)$ 设计先按 $\sigma_E$ 加权，再做列缩放后判断秩与
条件数，避免不同量纲造成伪病态。

三个独立动量点若设计满秩，可唯一估计三参数，但 `dof=0`，因此
`goodness_of_fit_available=False` 且 reduced $\chi^2$ 为 `NaN`；这与“不可辨识”不同。
若质量平方或任一预测能量平方的正性约束活跃，普通无约束曲率 covariance 无效，必须
返回全 `NaN`，并同时报告 `constraint_active=True`、`covariance_valid=False`。只有内点
最优解才可解释该 covariance。

优先用能量差参数化
$E_n=E_0+\sum_{k=1}^n\Delta E_k$，并以正参数（如 log 参数化）保持能级顺序。
拟合窗口、协方差和 SVD 由 `pyqcd-statistics` 记录，具体入口由
`pyqcd-analysis` 实现。

## 工作流程

1. 检查上游是否已给出算符、源汇顺序、投影、$T$ 和 Fourier 约定；缺项先回到
   `pyqcd-physics-correlator`。
2. 按复合态的夸克数和投影选择周期/反周期模板，明确是否保留 backward 态。
3. 插入完备关系，列出能量、重叠和矩阵元的依赖关系；做量纲和 $t\to0/T/2$ 极限检查。
4. 输出拟合函数、自由参数、先验/约束、拟合窗口和未覆盖热项，交给
   `pyqcd-analysis` 与 `pyqcd-statistics`。

## 常见陷阱

| 现象 | 处理 |
|---|---|
| 把重子 backward 当普通激发态 | 保留反宇称项，核对 `parity_and_boundary` 与投影 |
| 拟合振幅符号跳变 | 检查 Fourier/算符整体相位；不要凭“应为正”强改数据 |
| $C<0$ 导致拟合盆地跳变 | 记录全局相位并对所有相关数据一致处理，之后复核物理符号 |
| 三点振幅被直接解释为矩阵元 | 除去两侧 $Z$ 和归一化，或做 C₂+C₃ 联合拟合 |
| 窗口能拟合但参数由先验主导 | 依 `pyqcd-statistics` 的状态契约记录 prior 主导、窗口和条件数；不要把它写成数据或物理结果通过 |

## 交接

交付“谱式、状态/边界解释、参数化、可拟合轴和未建模项”。拟合与画图用
`pyqcd-analysis`，重采样和相关拟合用 `pyqcd-statistics`；若算符或缩并尚未固定，
停止在 `pyqcd-physics-correlator`，不要用谱式补猜输入。
