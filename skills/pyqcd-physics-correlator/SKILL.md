---
name: pyqcd-physics-correlator
description: |
  格点 QCD 关联函数推理技能：从物理观测量出发推导完整可计算链条——内插算符构造
  （介子/重子）、关联函数定义（2pt/3pt）、Wick 缩并（γ5 厄米、味对称）、传播子需求
  清单与 einsum 缩并结构。谱分解与拟合模板交给 pyqcd-physics-spectrum。
  触发于：强子质量、形状因子、矩阵元、算符构造、Wick 缩并、非连通图、三点关联函数、
  弱流插入、顺序源（sequential source）、局部/非局部两点函数、
  "我需要哪些关联函数/传播子"。纯规范观测量（Wilson 圈、Polyakov 圈、静态势）
  用 pyqcd-gauge 技能。
metadata:
  openclaw:
    emoji: 🧮
---

# pyqcd-physics-correlator — 观测量 → 关联函数推理

## 目的与边界

给定物理观测量（强子质量、衰变常数、准分布振幅、形状因子……），推导完整链条：

**观测量 → 算符 → 关联函数 → Wick 缩并 → 传播子 → einsum 结构**

使下游工具（pyqcd-propagator / PyQUDA）明确知道要算什么。本技能在
关联函数表达式、传播子清单与缩并结构固定后即停止；欧氏时间依赖、backward 态
结构与拟合模板交给 `pyqcd-physics-spectrum`。

> 记号约定：代码中轻夸克传播子为 `prop_l`、奇异为 `prop_s`，对应公式中 $S_l$、$S_s$。
> einsum 结构仅作**推理**用；PyQCD 仓库内的收缩实现见 `pyqcd/contraction/`
> （宇称投影与边界翻转 `parity_and_boundary`、Wick 缩并图 QC `plot_figure_wick`），
> 实战样例见 `logs/stab1`、`examples/test0`。逐例演示见 `reference/` 目录
> （pion/rho/proton 质量提取、Λ→p 形状因子）。

## 工作流程

### Step 1: 确定内插算符

对目标强子（量子数 $J^{PC}$、味内容）写出内插算符：介子用 Dirac 双线性，
重子用双夸克-夸克结构。例如 $\pi^+$：$\mathcal{O}_{\pi^+} = \bar{d}^a \gamma_5 u^a$；
质子：$\mathcal{O}_p = \epsilon^{abc} (u^a C\gamma_5 d^b) u^c$。
基态质量提取通常最简单局域算符即可；加 γ 矩阵与规范协变导数可访问不同量子数/
激发态/强子结构。非局域算符带 Wilson 线，如
$\mathcal{O}_{\pi^+}(x;z) = \bar{d}^a(0) \gamma_5 W(0,z) u^a(z)$；
矩阵元需在三点函数中插入流 $J_\mu=\bar{u}\gamma_5\gamma_\mu u$。

**γ 矩阵约定**：DeGrand-Rossi 基（PyQUDA 默认，PyQCD 全库统一）：

$$\gamma_1 = \begin{pmatrix} 0 & i \sigma_1 \\ -i \sigma_1 & 0 \end{pmatrix},\quad
\gamma_2 = \begin{pmatrix} 0 & -i \sigma_2 \\ i \sigma_2 & 0 \end{pmatrix},\quad
\gamma_3 = \begin{pmatrix} 0 & i \sigma_3 \\ -i \sigma_3 & 0 \end{pmatrix},\quad
\gamma_4 = \begin{pmatrix} 0 & I \\ I & 0 \end{pmatrix},$$
$$\gamma_5=\gamma_1\gamma_2\gamma_3\gamma_4,\quad \gamma_0=I_{4\times4},\quad C=\gamma_2\gamma_4.$$

$\gamma_\mu^\dagger=\gamma_\mu\ (\mu=1..5)$；$\{\gamma_\mu,\gamma_\nu\}=2\delta_{\mu\nu}$；
仅 $\gamma_1,\gamma_3$ 转置后产生负号。

### Step 2: 写出关联函数

**两点函数**（质量提取）：
$$C_2(\vec{p};t_f,t_i) = \langle \mathcal{O}(\vec{p},t_f) \mathcal{O}^\dagger(\vec{p},t_i) \rangle,
\quad \mathcal{O}(\vec{p},t) = \sum_{\vec{x}} e^{-i \vec{p}\cdot\vec{x}} \mathcal{O}(\vec{x},t).$$

**三点函数**（矩阵元/形状因子）：
$$C_3(\vec{q}; t_f,t_i,\tau) = \langle \mathcal{O}_\text{snk}(\vec{p}_f,t_f) J(\vec{q},\tau)
\mathcal{O}^\dagger_\text{src}(\vec{p}_i,t_i) \rangle, \quad \vec{q} = \vec{p}_i - \vec{p}_f.$$

利用真空时间平移不变性，可对等价时间切片平移平均增强信号。
三点函数的顺序源全链（算符定义→两次 dagger 约定→收缩）见
`reference/Lambda_proton_formfactor.md` 与 pyqcd-propagator 技能。

### Step 3: Wick 缩并与传播子确定

将所有夸克-反夸克对缩并为传播子 $\text{prop}_f(x,y)$，应用：

- **γ5 厄米性**：$\text{prop}_f(x,y) = \gamma_5 S_f^\dagger(y,x)\gamma_5$
  —— backward 传播子转 forward，省求逆；
- **味对称性**：简并 u/d 有 $\text{prop}_u = S_d = S_l$ —— 减少不同传播子数；
- **电荷共轭/同位旋**：可能联系不同图拓扑。

### Step 4: 列出传播子需求清单

典型 $C_\pi(\vec{p};t,0)=\sum_{\vec{x},\vec{y}} e^{-i\vec{p}\cdot(\vec{x}-\vec{y})}
\mathrm{Tr}[ S_l^\dagger S_l ]$ 需对源与汇全体求和，实际用特定源型估计：

- **点源**：固定 $(\vec{y}_0,t_0)$ 加相位，动量相位可置 1 不影响物理；
  多点源平均提升统计。
- **壁源**：全空间切片加相位 $e^{i\vec{p}\cdot\vec{y}}$，相位不可忽略，
  动量均分策略（$p_z=p_1+p_2$）：$p_z{=}1\to(1,0)$；$2\to(1,1)$；$3\to(2,1)$；$4\to(2,2)$…
- **体积源**：随机估计 all-to-all，两点函数少用；非连通图需 Z2/Z4 随机源。
- **移位传播子**：非局域算符需 Wilson 线连接，如
  $\text{prop}_{u,W(\vec{z},t)}(\vec{x},t;\vec{y},0)\equiv W(\vec{z},t;\vec{x},t)S_u(\vec{z},t;\vec{y},0)$。
- **顺序传播子**（三点函数）：由 forward 传播子在汇时刻构造 sequential source
  再求一次逆；每个汇动量与 γ 结构各需额外一次求逆（三点函数主要开销）。

输出清单须指明：味/质量参数、源型（点/高斯 smear 参数）、源位置（时间片、
每组态源数）、是否 APE/HYP smear 连接、汇处理（SS/SP 矩阵）。

## 源策略决策表

| 场景 | 建议 |
|---|---|
| 快速首看/调试 | 点源，每组态 1 个 |
| 介子谱量产 | smear 源，每组态 4 个源时 |
| 重子谱 | 必须 smear 源 |
| 非连通图 | 随机体积源（Z2/Z4） |
| 形状因子/3pt | 顺序源或随机估计 |

多源时（如 `t_src = 0, T/4, T/2, 3T/4`）等效统计倍增，对重子/激发态尤其有效。
最优选择依赖观测量与算力预算，生成代码前向用户确认源配置。

## 常见陷阱

1. **漏掉非连通图**：味单态介子（$\eta$、$\eta'$、$\sigma$）有非连通夸克圈贡献，
   需随机估计技术；非单态（$\pi^+$、$K^+$、$\rho^+$）无非连通图。
   PyQCD 胶子 TMD 链的 disconnected 通道即属此类（见 pyqcd-tmd-chain）。
2. **符号约定错误**：$C(t)$ 整体符号取决于算符归一化与费米子反对易次数；
   用大 t 极限行为校验。PyQCD 实战先例：P2 通道 2pt 带 phase 负号，
   ratio 负/负相消自洽（logs/stab1），拟合前全局符号 sgn·C 消除盆地歧义（dev6）。
3. **周期 vs 反周期边界**：费米子时间方向反周期，**复合态** BC 取决于夸克数：
   介子 $(-1)^2=+1$（cosh 型）；重子 $(-1)^3=-1$（backward 态宇称相反）。
   显式拟合模板见 pyqcd-physics-spectrum；PyQCD 实现：
   `pyqcd/contraction/_baroperator.parity_and_boundary`（P±=½(γ₀±γ₄) 投影 +
   pp/pm 边界符号翻转）。

## 与其他技能配合

- 谱分解/拟合模板 → `pyqcd-physics-spectrum`；传播子求解落地 → `pyqcd-propagator`；
- 纯规范观测量 → `pyqcd-gauge`；数据分析落地 → `pyqcd-analysis`；
- 胶子 TMD 物理链（场强算符级）→ `pyqcd-tmd-chain`。
