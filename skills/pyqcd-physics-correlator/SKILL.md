---
name: pyqcd-physics-correlator
description: |
  Use when a lattice-QCD task must derive an observable into interpolating operators,
  two- or three-point correlators, Wick contractions, a propagator inventory, or
  an einsum contraction; use pyqcd-physics-spectrum for spectral forms and
  pyqcd-propagator for PyQUDA execution.
metadata:
  openclaw:
    emoji: 🧮
---

# pyqcd-physics-correlator — 观测量到关联函数

## 目的与边界

本技能输出可交给求解器和分析层的推理契约：

\[
\text{观测量}\;\to\;\text{算符}\;\to\;\text{关联函数}\;\to\;
\text{Wick 缩并}\;\to\;\text{传播子清单}\;\to\;\text{einsum 结构}.
\]

它不决定时间谱分解、拟合模板或 PyQUDA API。γ 矩阵、轴序、单位、符号和边界条件
以 `pyqcd-conventions` 为唯一共享来源；若任务只处理规范链接，转 `pyqcd-gauge`。

## 推导顺序

### 1. 固定量子数与内插算符

先写目标的 $J^{PC}$、味内容、动量、源/汇时间和归一化。例：

\[
\mathcal O_{\pi^+}=\bar d\gamma_5u,\qquad
\mathcal O_p=\epsilon^{abc}(u^{Ta}C\gamma_5d^b)u^c.
\]

非局部算符必须显式写 Wilson 线，例如
`bar(d)(0) gamma5 W(0,z) u(z)`；三点函数还要固定流插入、动量转移和端点。

### 2. 定义关联函数

\[
C_2(\vec p;t_f,t_i)=\langle O(\vec p,t_f)O^\dagger(\vec p,t_i)\rangle,
\]

\[
C_3(\vec q;t_f,t_i,\tau)=
\langle O_{snk}(\vec p_f,t_f)J(\vec q,\tau)O^\dagger_{src}(\vec p_i,t_i)\rangle,
\quad\vec q=\vec p_i-\vec p_f.
\]

写清 Fourier 符号、投影、归一化、时间边界和每一轴的意义；能量依赖交给
`pyqcd-physics-spectrum`。

### 3. Wick 缩并与拓扑

对每个味分别配对夸克/反夸克，记录费米子置换号、颜色/自旋自由指标、连接分量、
是否 disconnected 以及共轭方向。常用恒等式为

\[
S_f(x,y)=\gamma_5 S_f^\dagger(y,x)\gamma_5,
\qquad S_u=S_d=S_l\quad(\text{简并轻味}).
\]

不得因数组数值相等就抹掉味标签；味单态或胶子 loop 的 disconnected 必须单列，不能
用 charged pion 的 connected 结构代替。若依赖 $gamma_5$ 厄米性省求逆，必须把
转置、共轭和源汇方向写入交接表。

### 4. 输出传播子需求表

| 字段 | 必须给出 |
|---|---|
| 味与质量 | `l/s` 或实际 flavor、质量/kappa、是否简并 |
| 源 | 点/壁/体积/随机、位置、源数、smear 参数 |
| 几何 | 动量相位、Wilson 线、位移方向和长度 |
| 三点 | 流、`p_i/p_f/q`、`t_sep`、顺序源种类与次数 |
| 缩并 | 每个传播子源汇、自由指标、共轭/转置、输出轴序 |

点源适合快速检查；重子谱通常用 smear 源；非连通使用 Z2/Z4 随机源；每个固定汇
动量、sink smearing 和投影的顺序源都要计入求逆清单。此处只决定“需要什么”，不
擅自选择求解器后端或 MPI 布局。

## 三点顺序源交接

对于 $Λ\to p$ 一类过程，交接顺序是：两条 light spectator 线构造 sink block
`B` → 第一次 `gamma5 B† gamma5` → 轻味顺序源求逆 → 第二次 dagger → 与 current-side
propagator 和 $Γ$ 收缩。PyQUDA 上下文、轴序和伪代码见
[`pyqcd-propagator/references/sequential-and-covdev.md`](../pyqcd-propagator/references/sequential-and-covdev.md)。

## 源策略

| 场景 | 默认建议 | 原因 |
|---|---|---|
| 调试/快速首看 | 点源，单源时 | 成本低、便于核对符号 |
| 介子量产 | smear 源，多源时刻 | 增强基态重叠与统计 |
| 重子谱 | smear 或 smeared-point | 压低激发态污染 |
| 非连通 | 随机体积源 | 估计全空间 loop |
| 形状因子/3pt | 顺序源或随机估计 | 固定 sink 后控制求逆成本 |

## 常见物理边界与自检

- 介子和重子的时间边界不同；重子 backward 是反宇称伙伴，具体模板转
  `pyqcd-physics-spectrum`，不要在本层折叠掉。
- P2 的相位负号或算符整体符号必须保留到分析层；先做独立极限核对，不能凭正值修正。
- einsum 中每个重复指标应恰好收缩两次，输出自由指标必须与关联函数轴序一致；需要
  自动 Wick 时仍要人工核对 flavor、符号、gamma 转置和 connected topology。
- 在进入求解器前用极端情形检查：局部算符的零位移极限、简并味替换、$gamma_5$
  厄米性和已知零动量通道；任何未决符号都在交接表标为假设。

## 交接

完成后只交付“算符定义、关联函数公式、Wick 项、传播子表、缩并字符串及未决假设”，
并附轴序、单位、边界和验证证据。传播子落地用 `pyqcd-propagator`，谱模板用
`pyqcd-physics-spectrum`，重采样和拟合用 `pyqcd-statistics`，数据产品用
`pyqcd-analysis`，纯规范对象用 `pyqcd-gauge`。
