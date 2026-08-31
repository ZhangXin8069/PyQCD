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
用 charged pion 的 connected 结构代替。若依赖 $\gamma_5$ 厄米性省求逆，必须把
转置、共轭和源汇方向写入交接表。

### 重子动态缩并实现契约

`GammaRegistry` 的 `Projector` 是源/汇成对的 $(P_+,P_+)$，不是只在一端施加的
单个投影。单动量重子 2pt 的输出只保留动量标签，必须使用 `Oindex='M'`；含
vector current 的 3pt/4pt 同时保留流与动量标签，必须使用 `Oindex='GM'`。

缩并结束时，每个自旋指标必须由显式 projector 或 trace 闭合。若输出仍有自由自旋轴，
应修正 `einsum` 标签或补上有物理定义的投影/迹；不得用 `ravel()[:4]`，也不得用未声明
指标语义的无标签 `sum` 掩盖残留轴。

验证证据必须按拓扑分别陈述：

- `PJN` 3pt 已有 `Nev=2` 的四张 Wick 独立显式 oracle；它不调用动态 contraction
  plan，并能捕获四项符号翻转与旧 `ravel()[:4]` 路径。
- `PJNNJNP` 4pt 目前只有动态 plan 参考。已有参考 current 是
  `gamma5 gamma_mu`（$\gamma_5\gamma_\mu$），生产实现是 `vector gamma_mu`
  （$\gamma_\mu$）；因此尚无独立 12 图 oracle，不得写成已独立验证。

这些结论限定于当前实现的索引闭合与测试独立性；它们不替代算符定义、流选择、动量约定
或真实组态上的物理验证，也不能把 3pt 的 oracle 证据外推到 4pt。

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

## 蒸馏 basis、vertex 与 perambulator 交接

蒸馏本征矢按 `E (Nev,Lz,Ly,Lx,Nc)` 或其 C-order 展平形式存储。局部 basis 相位

\[
E_p(z,y,x)=\exp\!\left[-i2\pi\left(
\frac{p_z z}{L_z}+\frac{p_y y}{L_y}+\frac{p_x x}{L_x}\right)\right]E(z,y,x)
\]

只是在已定义 basis 上逐点乘 U(1) 相位，不是含规范链接的协变 momentum smearing。
必须把 `p_basis=(pz,py,px)` 与顶点 Fourier 动量 `q_sink` 分开记录：同一 `E_p` 出现在
VdV 的一条共轭腿和一条非共轭腿时相位抵消；VVV 的三条非共轭腿则产生 `u_p^3`，在
本约定下与 `exp(-i*q_sink*x)` 合并为 `q_sink+3*p_basis`。源顶点的共轭会反转相应相位，
不能直接把 `p_basis` 命名为重子总动量。

相位 basis 的传播对象必须重新绑定为

\[
\tau_{p_{snk},p_{src}}=E_{p_{snk}}^\dagger M^{-1}E_{p_{src}}.
\]

普通 basis 的 perambulator 不能静默复用。交接必须同时给出 `p_basis_sink/source`、
`q_sink`、Fourier 符号、`lattice_shape`、展平顺序、`Nev`、backend/precision 和组态。
`sink2src` 对 VdV 是共轭并交换末两轴，对 VVV 只共轭；固定时间对后的 perambulator
轴序为 `(spin_sink,spin_source,ev_sink,ev_source)`，不得依赖未声明的转置。

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

## 推导例文

以下文件是物理推导例文，不是运行时依赖；新任务仍须按本入口重新核对约定：

| 例文 | 覆盖内容 |
|---|---|
| [`reference/pion_mass.md`](reference/pion_mass.md)、[`reference/rho_mass.md`](reference/rho_mass.md) | 介子双线性、dagger、connected 2pt |
| [`reference/proton_mass.md`](reference/proton_mass.md) | 重子宇称投影与两类轻味缩并 |
| [`reference/Lambda_proton_formfactor.md`](reference/Lambda_proton_formfactor.md) | flavor-changing 三点和顺序源推导 |

## 常见物理边界与自检

- 介子和重子的时间边界不同；重子 backward 是反宇称伙伴，具体模板转
  `pyqcd-physics-spectrum`，不要在本层折叠掉。
- P2 的相位负号或算符整体符号必须保留到分析层；先做独立极限核对，不能凭正值修正。
- einsum 中每个重复指标应恰好收缩两次，输出自由指标必须与关联函数轴序一致；需要
  自动 Wick 时仍要人工核对 flavor、符号、gamma 转置和 connected topology。
- 蒸馏链先分别检查 `E_{-p}(E_p(E))=E`、Gram 保持、同-basis VdV 相位抵消和 VVV
  三倍相位，再进入 perambulator 收缩；只通过局部 basis API 不能称端到端链已闭合。
- 在进入求解器前用极端情形检查：局部算符的零位移极限、简并味替换、$\gamma_5$
  厄米性和已知零动量通道；任何未决符号都在交接表标为假设。

## 交接

完成后只交付“算符定义、关联函数公式、Wick 项、传播子表、缩并字符串及未决假设”，
并附轴序、单位、边界和验证证据。传播子落地用 `pyqcd-propagator`，谱模板用
`pyqcd-physics-spectrum`，重采样和拟合用 `pyqcd-statistics`，数据产品用
`pyqcd-analysis`，纯规范对象用 `pyqcd-gauge`。
