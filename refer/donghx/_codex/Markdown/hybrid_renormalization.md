# HYP quark PDF：hybrid 重整化全流程、Wilson-loop 线质量与计算结果

**当前主线更新：2026-11-22。** 三种极化、三种 flavor 的 HYP3D5 connected 矩阵元从 constant 裸量、smooth ZR、hybrid、lambda/Fourier 到 light-cone matching 及外部 PDF 对照，详见[第 15 节](#15-当前-hyp3d5-quark-pdf-全流程代码与结果总账2026-11-22)。当前九种 PDF 主线不含 RGR/LRR；计算完成不等于 ZR/物理质量通过。下文第 1–14 节保留历史设置及测试。

**2026-10-30 更新：**三个 ensemble 的联合分析流程、1000 个 PDF 配对副本、平台/两态检验、UV 系数、参数退化和系统变化详见第 13 节。第 1–12 节保留此前 L32x64 研究及参数说明的历史记录。

## 1. 本次工作的范围与结论

本文第 1–10 节记录使用 L32x64 ensemble 已有 `hyp_3D_5times` 规范链接测量 Wilson loop，并通过长边指数拟合与 Cornell 拟合抽取线质量候选的计算。第 11 节补充当前 hybrid 重整化模型中每个参数的意义、单位、符号、拟合及拼接作用，并记录已完成的约束测试；第 12 节列出参考文献。本文整理已有结果，不因文档更新自动对 PDF 矩阵元施加新的减除。

截至 **2026-10-28**，有效计算目录为：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/hyp_wilson_L32x64_5step_v2_20261027
```

以下将此目录记为 `ROOT`。相邻的 `hyp_wilson_L32x64_5step_20261027` 是失败的 v1 记录，不能作为科学结果来源。

已经确认的结果如下：

- 425 个组态全部完成并通过完成标记和数组回读校验；缺失、无效组态均为 0。
- 公共的 3000 bootstrap 样本已经生成，测量、长边拟合、Cornell 拟合和四份结果 PDF 均已完成，分析完成时间为 **2026-10-27 21:58（Asia/Shanghai）**。
- 纯空间基础表示的一组诊断候选为

  $$
  \boxed{a m_{\mathrm{line,Cornell,eff}}^{F,\mathrm{sp}}
  =0.1874277756\pm0.0021484743,}
  $$

  对应长边窗口 $\ell=8,\ldots,12$、距离窗口 $r=4,\ldots,10$。这是 Cornell 常数项一半，**不是已经可靠确定、可以直接用于正式重整化的最终线性发散系数**。
- 空间—时间基础表示的另一组诊断结果为

  $$
  a m_{\mathrm{line,Cornell,eff}}^{F,\mathrm{temp}}
  =0.4164374769\pm0.0038142814,
  $$

  对应 $\ell=3,\ldots,6$、$r=4,\ldots,10$。这来自时间方向静态能，不能当成同一个 HYP 空间 Wilson line 的减除系数。
- 当前机器可读状态明确为 `final_d_approved=false`、`final_absolute_d=null`；完整分析状态为 `PASS_NUMERICAL_DIAGNOSTIC_ONLY`。该状态不得在未完成额外物理检验前改成“最终系数已确定”。

依据为 `ROOT/analysis/measurement_coverage.json`、`complete.json`、`coefficient_status.json`、`cornell_coefficients.csv` 和 `fit_diagnostics.json`。第 6 节给出具体数值及其限制。

## 2. Ensemble、输入和读取约定

### 2.1 原始数据与旧代码

Ensemble 与格距：

```text
beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64
```

$$
N_x=N_y=N_z=32,\qquad N_t=64,\qquad a=0.0897\ \mathrm{fm}.
$$

HYP 链接输入目录：

```text
/public/group/lqcd/donghx/Hypsmear_beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/hyp_3D_5times
```

与用户原 PDF 计算采用同一套输入定义，参考：

```text
/public/group/lqcd/donghx/Strange_PDF/Conneted_Diagram_final/L32x64/Calc_L32x64_new3.py
Readin_gauge(): 行 1302–1312
输入路径及方向转置: 行 1849–1856
```

实际读取的是每个组态目录下的：

```text
..._cfg_<conf>.lime.contents/msg02.rec04.ildg-binary-data
```

它是已经从 LIME 容器解出的 **raw binary payload**，不能再次按带 LIME record header 的完整容器读取。

### 2.2 字节、张量和方向约定

raw 张量的轴顺序为

$$
(N_t,N_z,N_y,N_x,\mu,c_{\mathrm{row}},c_{\mathrm{col}},r_{\mathrm{Re/Im}}),
\qquad
\mathrm{shape}=(64,32,32,32,4,3,3,2).
$$

- dtype：big-endian float64，代码标记为 `>f8`。
- layout：C-order，实部／虚部相邻。
- site axes：$(t,z,y,x)$，$x$ 为最快的格点坐标。
- link directions：$\mu=(x,y,z,t)$。
- expected payload：$1\,207\,959\,552$ 字节。

输入共有 $N_{\mathrm{cfg}}=425$ 个组态，编号为 $1000,1040,\ldots,17960$。精确排序、路径、输入 stat、XML 哈希及冻结代码哈希见 `ROOT/manifest.json` 和 `configuration_manifest.csv`。

本计算直接使用现有链接：**不再次 HYP、不做 gradient flow、不固定规范、不重投影 SU(3)、不除以 tadpole 因子、不改动输入链接。** 目录标签为 5 次 3D-HYP；生成这些输入时具体的 HYP $\alpha$ 参数在已查 metadata 中未找到，仍为 **待确认**，不得用其他 ensemble 的参数补写。

## 3. Wilson loop 的测量定义

### 3.1 几何、长度和闭合路径

为避免把物理长度和格点整数混淆，本文定义

$$
r=R/a,\qquad \ell=L/a,
$$

其中 $r,\ell$ 为数组中的整数，$R,L$ 为物理长度。令 $U_\mu^{(h)}(x)$ 是输入的 HYP 链接，定义正向 transporter：

$$
S_\mu(x;r)=\prod_{k=0}^{r-1}U_\mu^{(h)}(x+ka\hat\mu),
\qquad
S_\nu(x;\ell)=\prod_{k=0}^{\ell-1}U_\nu^{(h)}(x+ka\hat\nu).
$$

闭合矩形路径为

$$
\begin{aligned}
H_{\mu\nu}(x;r,\ell)
={}&S_\mu(x;r)
S_\nu(x+ra\hat\mu;\ell)\\
&\times S_\mu^\dagger(x+\ell a\hat\nu;r)
S_\nu^\dagger(x;\ell).
\end{aligned}
$$

回程使用对应的 dagger／逆向链接，不能只将数组反序。规范场边界周期，不加费米子反周期负号。零边长矩形按单位圈精确保存为 1。

固定平面顺序为

$$
(\mu,\nu)=(x,y),(y,z),(z,x),(x,t),(y,t),(z,t).
$$

前三个是纯空间圈，后三个是空间—时间圈。测量全部

$$
r=0,\ldots,12,\qquad \ell=0,\ldots,16,
$$

并对 $V_4=64\times32^3=2\,097\,152$ 个起点平均。空间长边最大为格点空间边长的一半。

### 3.2 基础表示与伴随表示

对每个组态 $c$，分别保存

$$
W_F^{(c)}(r,\ell)
=\frac{1}{V_4}\sum_x\frac{\operatorname{Tr}H(x;r,\ell)}{3},
$$

$$
W_A^{(c)}(r,\ell)
=\frac{1}{V_4}\sum_x
\frac{|\operatorname{Tr}H(x;r,\ell)|^2-1}{8}.
$$

$F$、$A$ 分别表示 SU(3) 基础表示与伴随表示。$W_F$ 保留复数；$W_A$ 使用 SU(3) character 恒等式，在**每个起点先计算**后再平均，不能对已经平均的 $W_F$ 取模平方来构造 $W_A$。两者独立拟合，不强加 Casimir scaling。

三个纯空间平面、三个空间—时间平面分别在每个组态内平均后进入统计；原始六平面的结果仍全部保留。起点数和平面数不能当成独立组态数。

## 4. 为什么必须区分纯空间圈与空间—时间圈

conf10000 的全部格点、颜色分量已与同组态薄链作逐元素比较，方向顺序为 $(x,y,z,t)$，最大绝对差为

$$
(1.3781619749,\ 1.3304927544,\ 1.3385501637,\ 0).
$$

因此至少对该组态，时间链接确实未改、空间链接发生了变化。证据为 `ROOT/validation/temporal_link_control.json`。这不等于已经对所有 425 个组态作过同样的薄链对照。

空间—时间圈在长时间方向上拟合的是

$$
\langle W_{it}(r,\ell)\rangle
\simeq A_h(r)\exp[-aV_{\mathrm{temporal}}(r)\ell].
$$

空间 HYP 会影响空间端帽和态重叠；指数长边沿时间方向，不能因为空间边是 HYP，就把拟合常数项解释为 HYP 空间直线的同一自能。

纯空间圈则定义

$$
\langle W_{ij}(r,\ell)\rangle
\simeq A_h^{\mathrm{sp}}(r)\exp[-aE_h^{\mathrm{sp}}(r)\ell].
$$

这直接涉及 HYP 空间长线，但 3D-HYP 不允许未经检查地使用时间—空间交换对称性；有限长边、smear 支撑范围、激发态和有限体积影响都须单独审查。本文称其为“纯空间长边能量／诊断势”，不把它自动等同于已提取基态的物理静态势。

## 5. Bootstrap 与两阶段拟合

### 5.1 3000 样本的共同重采样

固定 $N_B=3000$、随机种子 `20261027`，每个 replica 有放回抽取 425 个组态；所有平面、距离及两个表示共用同一份索引。对组态内方向平均后的圈，记

$$
\overline W=\frac{1}{N_{\mathrm{cfg}}}\sum_c W^{(c)},
\qquad
\overline W^{(b)}=\frac{1}{N_{\mathrm{cfg}}}
\sum_{j=1}^{N_{\mathrm{cfg}}}W^{(I_{bj})}.
$$

拟合用 $\ln\operatorname{Re}\overline W$ 和 $\ln\operatorname{Re}\overline W^{(b)}$：**先求 ensemble／replica 平均，再取实部和对数，不是单组态对数的平均。**

非正的圈均值无合法实对数，保留 NaN／invalid；不取绝对值、不 clip、不补造 bootstrap 拟合。只有 3000 行全部合法时才报告完整 bootstrap 的标准差，采用 `ddof=1`；不足 3000 行时完整误差保留 NaN，并明确报告可用行数。

当前是组态级 IID bootstrap，尚未完成自相关／blocking 研究；统计误差不包含全部系统误差。

### 5.2 第一阶段：长边指数与有效势

对每个 $r$ 拟合

$$
y_r(\ell)=\ln\operatorname{Re}\langle W(r,\ell)\rangle
=b_r-\widehat E(r)\ell,\qquad \widehat E(r)=aE(r).
$$

这里 $b_r=\ln A(r)$ 是该 Wilson loop 的振幅参数，**不是线性发散系数**。

同时保存相邻长边比值对应的有效势：

$$
\widehat E_{\mathrm{eff}}(r,\ell+\tfrac12)
=\ln\frac{\operatorname{Re}\langle W(r,\ell)\rangle}
{\operatorname{Re}\langle W(r,\ell+1)\rangle}.
$$

扫描的长边窗口为

$$
\ell\in[3,6],\ [4,7],\ [5,8],\ [6,10],\ [8,12],\ [10,16],
$$

端点均包含。共保存 $2\times2\times6\times12=288$ 个长边拟合诊断。

使用固定 delete-one jackknife 协方差的相关 GLS；原相关矩阵信息秩相对阈值为 $10^{-12}$，删模阈值为 $10^{-8}$。记录 sample rank、effective rank 和 model rank；正自由度和设计矩阵满秩是硬条件。**不按 $Q$ 或 $\chi^2$ 筛掉可辨识拟合，也不把“保留”解释为“物理模型可信”。**

### 5.3 第二阶段：Cornell 拟合

对每个表示、圈类型和长边窗口，将第一阶段的能量拟合为

$$
\widehat E(r)=c_0+\frac{\alpha}{r}+\widehat\sigma r,
\qquad \widehat\sigma=\sigma a^2.
$$

代码中的 $\alpha$ 是带符号的拟合参数，不能在未改模型时另换成固定的正 Coulomb 系数。距离窗口为

$$
r\in[2,6],\ [3,8],\ [4,10],\ [5,12].
$$

仍采用相关 GLS，对同一 bootstrap 行的势应用固定线性拟合映射，保留距离间及前一阶段的相关性。共保存 96 个 Cornell 拟合。

定义诊断量

$$
\widehat m_{\mathrm{Cornell,eff}}
\equiv a m_{\mathrm{line,Cornell,eff}}=\frac{c_0}{2}.
$$

因圈有两条长边，只有在选定的 Cornell 势零点约定中才如此分配常数项。$c_0$ 同时包含有限的、方案依赖的偏置，不能只凭这个等式宣称已经分离纯 UV 发散。

## 6. 实际拟合结果与可靠性

### 6.1 总体覆盖

| 表示及圈类型 | 完整 3000 样本可辨识 | 部分样本可辨识 | 不可辨识 | 总窗口数 |
|---|---:|---:|---:|---:|
| 基础表示、纯空间 | 21 | 1 | 2 | 24 |
| 基础表示、空间—时间 | 9 | 4 | 11 | 24 |
| 伴随表示、纯空间 | 8 | 4 | 12 | 24 |
| 伴随表示、空间—时间 | 0 | 0 | 24 | 24 |
| 合计 | 38 | 9 | 49 | 96 |

`complete.json` 中的 `identifiable_cornell=47` 包含 38 个完整和 9 个部分可辨识结果，不能读成“47 个都有完整 bootstrap 误差”，更不能读成“47 个都已通过物理质量检验”。

### 6.2 基础表示纯空间候选及长边窗口依赖

固定 $r=4,\ldots,10$，直接摘录 `analysis/cornell_coefficients.csv`：

| 长边窗口 $\ell$ | $\widehat m_{\mathrm{Cornell,eff}}^{F,\mathrm{sp}}$ | bootstrap 标准差 | Cornell $\chi^2/\mathrm{dof}$ | $Q$ |
|---|---:|---:|---:|---:|
| 3–6 | 0.00489849 | 0.00018535 | $96581.263/4$ | 数值为 0 |
| 4–7 | 0.11002832 | 0.00028560 | $6884.781/4$ | 数值为 0 |
| 5–8 | 0.13931882 | 0.00048082 | $6678.845/4$ | 数值为 0 |
| 6–10 | 0.19212689 | 0.00083399 | $248.002/4$ | $1.75\times10^{-52}$ |
| 8–12 | 0.18742778 | 0.00214847 | $10.541/4$ | 0.03224 |
| 10–16 | 无可辨识结果 | NaN | 不适用 | 不适用 |

上述变化远大于部分窗口的统计误差；它本身就是未解决的窗口／模型系统误差，不能把最小统计误差作为选择最终系数的理由。

对 $\ell=8,\ldots,12$ 扩大距离窗口到 $r=5,\ldots,12$ 时，中心候选为 $0.19452684$，Cornell $\chi^2/\mathrm{dof}=9.93682/5$、$Q=0.07705$，但仅 **2963/3000** 个 bootstrap 行合法；因此完整 bootstrap 标准差为 NaN，不能用删掉 37 行后的标准差冒充完整误差。

### 6.3 候选值的第一阶段仍存在显著拟合质量问题

候选 $0.18742778$ 来自 $\ell=8,\ldots,12$。该窗口中每个距离的第一阶段结果为：

| $r$ | $\widehat E(r)$ | bootstrap 标准差 | 长边拟合 $\chi^2/\mathrm{dof}$ | $Q$ |
|---:|---:|---:|---:|---:|
| 4 | 0.31335473 | 0.00012057 | $1069.662/3$ | $1.39\times10^{-231}$ |
| 5 | 0.39290016 | 0.00023479 | $464.659/3$ | $2.17\times10^{-100}$ |
| 6 | 0.45817515 | 0.00044085 | $199.532/3$ | $5.33\times10^{-43}$ |
| 7 | 0.51544332 | 0.00078998 | $79.023/3$ | $4.97\times10^{-17}$ |
| 8 | 0.56863933 | 0.00141037 | $28.034/3$ | $3.57\times10^{-6}$ |
| 9 | 0.61913464 | 0.00262613 | $6.028/3$ | 0.11027 |
| 10 | 0.66771379 | 0.00487608 | $1.203/3$ | 0.75221 |

来源：`ROOT/analysis/fit_diagnostics.json` 中 `long_fits`，标签 `fundamental_spatial_L8_12_R<r>`。这些拟合都满足统计秩条件、都有完整 3000 行，但 $r=4,\ldots,8$ 的单指数模型质量明显不足。具体污染来源仍 **待确认**；目前不能仅凭第二阶段 Cornell 的 $Q=0.03224$ 宣称整条提取流程已经可靠。

### 6.4 空间—时间圈与伴随表示

基础表示空间—时间圈的代表结果：

| 长边窗口 $\ell$ | 距离窗口 $r$ | $\widehat m_{\mathrm{Cornell,eff}}^{F,\mathrm{temp}}$ | bootstrap 标准差 | Cornell $\chi^2/\mathrm{dof}$ | $Q$ |
|---|---|---:|---:|---:|---:|
| 3–6 | 4–10 | 0.41643748 | 0.00381428 | $3.009/4$ | 0.55632 |
| 3–6 | 5–12 | 0.41876089 | 0.00734063 | $5.315/5$ | 0.37871 |
| 4–7 | 4–10 | 0.42843445 | 0.01058249 | $0.612/4$ | 0.96175 |

这些是时间静态能的诊断候选，不是纯空间 HYP 直线候选的替代值；仍需检查第一阶段、平台和窗口系统误差。

伴随表示纯空间圈例如 $\ell=3,\ldots,6$、$r=4,\ldots,10$ 得到

$$
\widehat m_{\mathrm{Cornell,eff}}^{A,\mathrm{sp}}
=0.16799365\pm0.00110674,
\qquad \chi^2/\mathrm{dof}=504.995/4,
\qquad Q=5.57\times10^{-108}.
$$

该数虽有完整 bootstrap，也不能作为可靠伴随减除系数。伴随表示空间—时间圈的 24 个 Cornell 窗口均不可辨识；不得凭基础表示乘 Casimir 比补出结果。

## 7. 这些结果如何联系到线性发散减除

若明确采用直线矩阵元约定

$$
h_{\mathrm{bare}}(z,a)
=e^{-\delta m(a)|z|}\,h_{\mathrm{rest}}(z,a),
$$

则对应的线减除为

$$
h_{\mathrm{sub}}(z,a)
=e^{+\delta m(a)|z|}h_{\mathrm{bare}}(z,a).
$$

在将 $\widehat m=a\delta m$ 作为试验处方时，指数写成 $+\widehat m\,|z|/a$；这是**给定上述裸量衰减约定后**的符号关系，不代表目前已经批准使用 $0.18742778$。反号约定必须连同裸量定义整体转换，不能只改指数符号。

在固定 HYP 处方下可以概念性地写

$$
\delta m_{\mathrm{HYP}}(a)
=\frac{\kappa_{\mathrm{HYP}}}{a}
+m_{\mathrm{finite}}^{\mathrm{scheme}}+\cdots.
$$

单一格距、单一 HYP 次数下的 Cornell 偏置不能唯一分离 $\kappa_{\mathrm{HYP}}$ 与有限残余质量。因此这里可靠保存的是 **Cornell 约定下的有效线质量候选**，不是已隔离的普适 UV 系数。

必须与 gradient-flow 提取的系数区分：此前 flow 方案中的 $d_F$ 或 $d_A$ 乘以 $1/\sqrt{8t_f}$，这里保存的却是无量纲 $a m_{\mathrm{line,Cornell,eff}}$。两套输入和正则化不同，**不能把此前的 $d_F=0.374137$ 移用到 3D-HYP，也不能把这里的 $0.18742778$ 直接命名为同一个 $d_F$**。五次 HYP 也不能未经标定就换成某个确定的 $t_f$。

用于 hybrid 重整化时，还需另外固定短距离方案、有限势零点／残余质量和拼接条件；这些并未由当前 Wilson-loop 拟合完成。已完成的固定候选值参数测试及其限制见第 11 节。

## 8. 代码、数据、图片和任务记录

### 8.1 代码入口

以下路径均相对 `ROOT`：

| 路径 | 用途 |
|---|---|
| `src/hyp_rectangles.cpp` | raw payload 读取、SU(3) 检查、OpenMP 六平面矩形圈测量 |
| `reference/polyakov_base.cpp` | 旧验证实现的冻结基元副本；本程序不调用它的 flow driver |
| `code/prepare.py` | 输入盘点、manifest 和来源冻结 |
| `code/run.py` | 节点本地编译、测试、gate、批量测量、完成校验及断点复用 |
| `code/analyze.py` | 425 conf 汇总、bootstrap、长边及 Cornell 拟合、输出和绘图调用 |
| `code/fit_statistics.py` | 正对数检查、相关 GLS、秩检查、严格 bootstrap 误差 |
| `code/plot_results.py` | 从保存的科学结果生成四份 PDF |
| `code/test_measurement.py` | 独立路径、规范不变性、端序、边界和坏输入测试 |
| `code/test_statistics.py`、`code/test_analysis.py` | 统计及独立 synthetic 全流程测试 |
| `slurm/gate_cpu.slurm`、`measure_cpu.slurm`、`analysis_cpu.slurm` | CPU 验证、测量数组及分析入口 |
| `submission.json` | 已提交作业的回执，避免重复提交 |

### 8.2 数据格式

单组态结果：

```text
ROOT/data/conf<conf>/
  wilson_fundamental.npy    complex，shape (6,13,17)
  wilson_adjoint.npy        real，shape (6,13,17)
  measurement.json         几何、输入、SU(3)、计时等测量信息
  complete.json            来源、manifest 和输出 hash 完成标记

array axes: (plane, r, ell)
```

汇总结果：

```text
ROOT/analysis/
  measurement_coverage.json
  per_configuration_wilson.npz
  bootstrap_indices.npz
  wilson_statistics.npz
  effective_potentials.npz
  long_fits/*.npz
  potentials/*.npz
  cornell_fits/*.npz
  fit_diagnostics.json
  cornell_coefficients.csv
  coefficient_status.json
  complete.json
```

其中 `per_configuration_wilson.npz` 的 `values` 轴为 `(conf,representation,plane,r,ell)`，shape 为 `(425,2,6,13,17)`；`bootstrap_indices.npz` 保存 shape `(3000,425)` 的组态索引及对应 conf 顺序。`wilson_statistics.npz` 中 `central` 轴为 `(representation,kind,r,ell)`，`samples` 再在最前面增加 bootstrap 轴；`kind` 顺序为 `spatial,temporal`，表示顺序为 `fundamental,adjoint`。

有效势数组最后一轴为相邻长边比值，整数 $\ell=0,\ldots,15$ 对应图中的 $\ell+1/2$。各拟合 NPZ 还保留参数、样本、jackknife、拟合映射和 metadata，不能只根据 PDF 图反推数值。

### 8.3 已生成的四份结果图

以下四份是 **425 conf 真实输入的分析结果**，不是 gate 中的 synthetic 测试图：

1. [01_Wilson_loops.pdf](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/hyp_wilson_L32x64_5step_v2_20261027/analysis/plots/01_Wilson_loops.pdf)：Wilson loop 随长边变化，检查实部信号及其衰减。
2. [02_effective_potential_plateaus.pdf](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/hyp_wilson_L32x64_5step_v2_20261027/analysis/plots/02_effective_potential_plateaus.pdf)：相邻长边比值的有效势，检查平台与窗口。
3. [03_static_potential_Cornell.pdf](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/hyp_wilson_L32x64_5step_v2_20261027/analysis/plots/03_static_potential_Cornell.pdf)：各长边窗口提取的能量随距离变化及 Cornell 拟合。
4. [04_line_mass_window_scan.pdf](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/hyp_wilson_L32x64_5step_v2_20261027/analysis/plots/04_line_mass_window_scan.pdf)：$c_0/2$ 对长边和距离窗口的依赖，不能只挑单个最小误差点。

完整数值表：[cornell_coefficients.csv](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/hyp_wilson_L32x64_5step_v2_20261027/analysis/cornell_coefficients.csv)。物理批准状态：[coefficient_status.json](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/hyp_wilson_L32x64_5step_v2_20261027/analysis/coefficient_status.json)。

### 8.4 CPU 作业与安全复查命令

已提交作业为：gate `3876318`、测量数组 `3876319_[0-47]`、分析 `3876320`。CPU 分区为 `a192c1t,cpu6248R,cpueicc,i72c512g`，不使用 GPU。测量每项 24 CPU、32 GiB、24 h；分析 4 CPU、32 GiB、8 h。测量数组覆盖 gate 之外的 424 conf，每项处理 8–9 conf，无 `%N` 并发限制。

只读复查命令：

```bash
cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/hyp_wilson_L32x64_5step_v2_20261027
sed -n '1,120p' analysis/measurement_coverage.json
sed -n '1,120p' analysis/complete.json
sed -n '1,160p' analysis/coefficient_status.json
head -n 8 analysis/cornell_coefficients.csv
ls -lh analysis/plots
sacct -j 3876318,3876319,3876320 -o JobID,State,ExitCode,Elapsed,NodeList
```

数据完成的依据是完成标记和回读校验，不应仅靠历史队列中是否还有 job 判断。`code/submit.py` 已有回执，**不要重复运行**；`code/analyze.py` 也会拒绝覆盖已提交的分析结果。若需改变模型或协方差，另建版本化 CPU 分析任务及输出目录，不修改 manifest 冻结源码或覆盖本批结果。

## 9. 已完成的正确性检查

- 独立 NumPy 真路径与 C++ 基础表示最大差异约 $5.6\times10^{-17}$；局域规范变换前后最大差异约 $1.1\times10^{-16}$。
- 六平面、非立方坐标映射、周期绕回、回程 dagger、单位场及零边、big-endian 实虚成对 I/O、逐起点伴随 character、截断输入拒绝测试均通过。
- 相关 GLS、拟合符号、精确模型、低秩拒绝、非法对数保留、逐观测量 bootstrap 完整性测试通过。
- 独立 synthetic 子目录完成了全分析、严格 JSON 和四份 PDF 的生成测试；其产物不算真实组态结果。
- conf10000 的 SU(3) 最大幺正性误差为 $1.75\times10^{-15}$，行列式误差为 $1.34\times10^{-15}$；矩形圈测量约 20.93 s，含读取／检查的 C++ 总计约 22.77 s。这只是一个节点、一个组态的计时，不是全平台性能保证。
- conf10000 全体积时间链接与薄链精确相同，空间链接不同；解释限制见第 4 节。

证据位置：`ROOT/build/gate_3876318/tests/report.json`、`build/gate_3876318/synthetic_analysis/test_report.json`、`validation/gate_complete.json`、`validation/temporal_link_control.json`、`data/conf10000/measurement.json` 及 `logs/gate_3876318.out`。

v1 的真实组态读取曾因跨节点 `st_dev` 不同被来源 guard 误拒绝；CPU 诊断确认 canonical path、字节数、mtime_ns 和 inode 相同，只有挂载设备号不同。v2 改为不把 `st_dev` 当跨节点全局身份，并保留其余来源和输出 hash 检查。v1 后继任务已取消，不作为重复生产来源。

## 10. 待确认问题与下一步

1. **先审查第一阶段模型。** 以 `02_effective_potential_plateaus.pdf` 和第 6.3 节的逐距离 $\chi^2,Q$ 为起点，检查单指数区间是否存在可信平台；污染来源、3D-HYP 支撑影响与有限空间长边效应均待确认。不能只优化第二阶段 Cornell 的 $Q$。
2. **做窗口和模型系统误差分析。** 同时改变长边、距离窗口；如需增加模型，必须检查可辨识性，并在新的 CPU 分析目录中保留所有结果。当前缺少可信的最终系统误差。
3. **检查组态自相关。** 按实际 Monte Carlo 顺序做 blocking／自相关诊断，再评估 IID bootstrap 误差是否低估。不把 3000 bootstrap 当成 3000 独立组态。
4. **固定有限重整化条件。** 若用于 PDF，先明确矩阵元采用相同的 HYP 链接处方、表示与方向，再决定 Cornell 有限零点和短距离匹配／残余质量的处理。不把 $c_0/2$ 与纯 UV 发散等同。
5. **需要时增加独立信息。** 额外 smear／格距扫描、可靠短距离参照等尚未由本批完成；HYP 的原始 $\alpha$ 参数也仍待确认。

不可随意修改的结论是：两个表示的归一化和逐起点 character、raw payload 的坐标／端序定义、纯空间与时间静态能的区别、公共 bootstrap 索引、非法对数的保留规则，以及 `final_d_approved=false` 的当前状态。未来若经新证据升级结论，应保留旧版本及升级依据，不能静默覆盖。

## 11. 当前 hybrid 重整化模型：每个参数的意义与作用

本节依据当前实际运行的 `nlo_a2` 模型解释参数，主要对应 unpolarized、$u-d$ 通道。文献中的 hybrid 方案、本文使用的 self-renormalization 参数化，以及 Wilson-loop 候选值的约束测试，需要分别辨认。下列符号、正负号与本地代码保持一致；文献若采用反号的质量反项定义，应先转换整个指数约定。[R1–R2]

### 11.1 符号、单位和被拟合的数据

记 $H_B^\Gamma(P^z,z,a)$ 为本流程已经完成局域归一化的无量纲裸矩阵元 shape；$\Gamma$ 表示算符／极化通道。零动量参考态为

$$
H_0^\Gamma(z,a)\equiv H_B^\Gamma(0,z,a).
$$

当前参考态取同一通道的 $u-d$、even-real 数据。联合拟合使用参考态的正值非局域点及其对数，不对移动矩阵元的实部、虚部直接套用对数模型。非法对数必须保留无效状态。

使用自然单位制：

$$
\hbar c=0.1973269804\ \mathrm{GeV\,fm},
\qquad
 a_{\mathrm{GeV}^{-1}}=\frac{a_{\mathrm{fm}}}{\hbar c},
\qquad
 z_{\mathrm{GeV}^{-1}}=\frac{z_{\mathrm{fm}}}{\hbar c}.
$$

下文公式中的 $a$、$z$ 均已换算为 $\mathrm{GeV}^{-1}$，并用 $s\equiv |z|$ 表示 Wilson line 的非负长度。只有 $a\Lambda$、$\mu/\Lambda$ 等无量纲组合可以放入对数。

**大写 $\Lambda$ 是这里的拟合尺度；小写 $\lambda$ 通常表示 Ioffe time：**

$$
\lambda\equiv P^z z.
$$

二者不是同一个参数。前面测试中“重新拟合 lambda”在代码中实际指重新拟合大写的 $\Lambda$。

### 11.2 完整参考态拟合式与重整化因子

先定义

$$
\ell_a\equiv\ln(a\Lambda),
\qquad
\beta_0\equiv 11-\frac{2}{3}n_f,
\qquad
 C_F\equiv\frac{N_c^2-1}{2N_c}=\frac{4}{3},
$$

以及当前代码使用的对数归一化项

$$
\mathcal N(a;\Lambda,d,\mu)
=
\frac{3C_F}{\beta_0}
\ln\!\left[
\frac{-\ell_a}{\ln(\mu/\Lambda)}
\right]
+
\ln\!\left(1+\frac{d}{\ell_a}\right).
$$

当前主分析采用 $n_f=3$。因此，以上第一项的系数为

$$
\frac{3C_F}{\beta_0}=\frac{4}{9}.
$$

这里的系数是本地当前实现的定义；尤其不能把不同极化通道的短距 Wilson 系数直接替换成同一个表达式。

当前 `nlo_a2` 对非零距离的参考态模型是

$$
\boxed{
\ln H_0^\Gamma(s,a)
=
\frac{k s}{a\ell_a}
+\mathcal N(a;\Lambda,d,\mu)
+a^2 f(s)
+
\begin{cases}
\ln C_0^\Gamma(s,\mu)+m_0 s,
& s\le z_1,\\[2pt]
 g(s),
& s>z_1.
\end{cases}
}
$$

此式只在声明的正距离拟合网格上使用。短距段用微扰系数约束有限结构；长距段通过自由函数 $g(s)$ 保留参考态中的非微扰形状。当前两个短距网格点与长距网格点分组不同，不另外强制分段模型在边界处的一阶导数连续。

从这套模型构造的重整化因子为

$$
\boxed{
\ln Z_R^\Gamma(s,a)
=
\frac{k s}{a\ell_a}
+\mathcal N(a;\Lambda,d,\mu)
+m_0 s
+a^2 f(s),
\qquad s>0.
}
$$

其中 $m_0s$ 在所有非零距离都出现，且只出现一次；$g(s)$ 不放入 $Z_R$。在模型成立时，参考态的 self-renormalized 结果满足

$$
\frac{H_0^\Gamma(s,a)}{Z_R^\Gamma(s,a)}
\simeq
\begin{cases}
 C_0^\Gamma(s,\mu), & s\le z_1,\\[2pt]
 \exp\!\bigl[g(s)-m_0s\bigr], & s>z_1.
\end{cases}
$$

这也说明本地的 $g(s)$ 尚包含需要通过 $m_0$ 约定分离的有限线性部分，不能直接把 $\exp[g(s)]$ 当作已经确定目标方案的参考态矩阵元。[R2]

### 11.3 参数总览

| 符号 | 单位 | 当前身份 | 意义和主要作用 |
|---|---|---|---|
| $k$ | 无量纲 | 无约束拟合中自由；固定 UV 质量时导出 | 控制 Wilson-line 线性 UV 项的幅度，通过 $k/[a\ln(a\Lambda)]$ 进入指数 |
| $\Lambda$ | $\mathrm{GeV}$ | 全局拟合参数 | 调节线性项及对数项中的格距依赖；不是本分析独立测定的标准 $\Lambda_{\overline{\mathrm{MS}}}$ |
| $d$ | 无量纲 | 全局拟合参数 | 参数化当前模型中额外的对数／有限归一化修正；不是 Wilson-loop 的线质量 |
| $m_0$ | $\mathrm{GeV}$ | 全局拟合参数 | 有限残余质量参数，通过短距条件连接选定微扰方案；在 $\ln Z_R$ 中乘以 $+s$ |
| $f(s)$ | $\mathrm{GeV}^2$ | 每个网格点一个参数 | 乘以 $a^2$，吸收当前模型保留的格距修正；不是动量空间 PDF |
| $g(s)$ | 无量纲 | 每个长距网格点一个参数 | 保存零动量参考态的有限长距结构；只参与参考态拟合，不直接构成 UV 重整化因子 |
| $\delta m_{\mathrm{UV}}(a)$ | $\mathrm{GeV}$ | 由 $k,\Lambda,a$ 导出 | 本文正衰减约定下的 Wilson-line UV 质量参数；与质量反项的符号约定相关 |
| $C_0^\Gamma(s,\mu)$ | 无量纲 | 按声明阶数计算 | 零阶矩的短距 Wilson 系数，为 $m_0$ 等参数提供微扰匹配条件 |
| $\mu$ | $\mathrm{GeV}$ | 外部固定尺度 | 重整化／匹配尺度，不由本次格距拟合决定 |
| $\alpha_s(\mu)$ | 无量纲 | 外部输入或按声明精度演化 | 微扰系数和 RGR 中的强耦合；不由拟合的 $\Lambda$ 自动重新计算 |
| $z_0,z_1$ | 长度 | 拟合窗口设置 | 分别标记非局域拟合网格的起点和使用短距微扰条件的上限 |
| $z_s$ | 长度 | hybrid 方案设置 | 短距 ratio 与长距 $Z_R$ 的拼接位置；不是 $m_0$ 拟合输出 |
| $\eta(z_s,a)$ | 无量纲 | 拼接条件导出 | 保证 hybrid 矩阵元在 $z_s$ 处连续，不是额外自由幅度 |
| $q_\Gamma^{\mathrm{ren}}$ | 无量纲 | 局域荷的外部／独立输入 | 恢复 normalized shape 缺少的整体归一化；当前流程留到 matching 后恢复 |

### 11.4 $k$：线性 UV 项的幅度

采用第 7 节的正衰减约定，定义

$$
H_B^\Gamma(s,a)
=\exp\!\bigl[-\delta m_{\mathrm{UV}}(a)s\bigr]
H_{\mathrm{rest}}^\Gamma(s,a).
$$

与当前模型比较得

$$
\boxed{
\delta m_{\mathrm{UV}}(a)
=-\frac{k}{a\ln(a\Lambda)},
\qquad
 a\delta m_{\mathrm{UV}}(a)
=-\frac{k}{\ln(a\Lambda)}.
}
$$

当 $a\Lambda<1$ 且 $k>0$ 时，该 UV 项对应裸量的正质量衰减。$k$ 是特定链接、作用量及参数化下的幅度，不是单一格距的无量纲线质量本身。HYP 处方改变后，不能默认保留同一数值；也不能仅凭表示名称把另一正则化的系数搬过来。[R1–R2]

**因此，Wilson-loop 候选所能条件性约束的是质量组合，而不是直接令 $k$ 等于候选数值。** 当前测试把候选解释为纯 UV 项时施加

$$
\boxed{
 a_{\mathrm{L32x64}}\delta m_{\mathrm{UV}}=0.1874
\quad\Longleftrightarrow\quad
 k=-0.1874\ln\!\bigl(a_{\mathrm{L32x64}}\Lambda\bigr).
}
$$

这里“解释为纯 UV 项”是本次参数敏感性测试的假设，第 6–7 节记录的有限方案偏置与窗口问题尚未解决。

### 11.5 $\Lambda$：格距依赖的有效尺度

$\Lambda$ 同时进入线性项的分母和 $\mathcal N$ 的对数结构。在有限格距范围内，改变 $\Lambda$ 可以由 $k$、$d$ 及其他参数部分补偿，因此其拟合值依赖模型和数据约束。

本地优化器使用的变量是

$$
\theta_\Lambda\equiv
\ln\!\left(\frac{\Lambda}{1\ \mathrm{GeV}}\right),
\qquad
 \Lambda=(1\ \mathrm{GeV})e^{\theta_\Lambda}.
$$

代码字段 `logLambda` 对应 $\theta_\Lambda$，不是另一个独立参数；当前边界为

$$
0.001\ \mathrm{GeV}\le\Lambda\le1\ \mathrm{GeV}.
$$

该尺度来自拟合式中的有效参数化，不等于凭本轮数据独立确定 QCD 的标准尺度参数。尤其不能把固定 UV 质量测试得到的很小 $\Lambda$ 直接代入其他物理计算。

### 11.6 $d$：额外对数／归一化修正

$d$ 通过以下因子进入参考态和 $Z_R$：

$$
1+\frac{d}{\ln(a\Lambda)}.
$$

在固定格距、固定参数下，它与 $s$ 无关；因此主要调整不同格距之间的归一化关系，不单独产生随线长增长的指数。它是当前截断模型中吸收高阶及有限修正的拟合参数，不是已计算得到的独立微扰常数。[R2]

必须满足对数定义域：

$$
\ln(a\Lambda)<0,
\qquad
\ln(\mu/\Lambda)>0,
\qquad
1+\frac{d}{\ln(a\Lambda)}>0.
$$

本地采用变量变换保证所有参与拟合格距的最后一个条件：

$$
L_{\min}\equiv
\ln\!\left(\frac{1}{a_{\max}\Lambda}\right),
\qquad
 d=L_{\min}\bigl(1-e^{\theta_d}\bigr).
$$

$\theta_d$ 是优化器变量；最终参数表中的 $d$ 已换回物理参数化。这个 $d$ 也不同于 gradient-flow 文献中的线质量系数 $d_F,d_A$，更不同于短距系数中的常数项。

### 11.7 $m_0$：有限残余质量和短距方案条件

$m_0$ 有质量量纲，负责当前 self-renormalization 参数化中的有限线性部分及其短距方案选择。它不应与 Wilson-line 的纯 UV 幂次发散混为一谈；其数值依赖微扰阶数、拟合窗口、有限方案和参数约定。[R1–R2]

本地定义是

$$
\ln Z_R(s,a)\supset +m_0s.
$$

把它与 UV 项合并，得到

$$
\frac{k s}{a\ln(a\Lambda)}+m_0s
=-\bigl[\delta m_{\mathrm{UV}}(a)-m_0\bigr]s.
$$

因此，在当前代码约定下，排除 $f(s)$ 的显式距离依赖后，净线性衰减系数是

$$
\boxed{
 m_{\mathrm{linear}}(a)
=\delta m_{\mathrm{UV}}(a)-m_0,
\qquad
 a m_{\mathrm{linear}}
=-\frac{k}{\ln(a\Lambda)}-a m_0.
}
$$

这解释了为什么改变 UV 约束后 $m_0$ 可以显著补偿该变化。只看单一格距时，两项都乘以 $s$，不能据此独立确定它们。文献使用的残余质量若放在指数中取相反符号，其数值必须先转换，不能直接套用加号或减号。

$m_{\mathrm{linear}}$ 不是整个 $Z_R$ 的完整局部斜率，因为 $f(s)$ 也随距离变化。

### 11.8 $f(s)$：格距修正函数

当前主模型保留

$$
\Delta_{\mathrm{disc}}(s,a)=a^2 f(s).
$$

这使相同物理线长处、不同格距参考数据之间的离散化差异能够被共同拟合。$f(s)$ 在公共网格的每个点独立参数化，网格间线性插值；网格外沿最靠近边界的一段线性外推，并保存外推标记。外推不代表长距离离散化误差已得到额外数据约束。

采用另一模型 $a^p f_p(s)$ 时，量纲随幂次改变：

$$
[a^p f_p(s)]=1,
\qquad
[f_p(s)]=\mathrm{GeV}^{p}.
$$

所以 `nlo_a` 与 `nlo_a2` 的 $f$ 不能只按数值直接比较。这个函数也不是最终动量空间 PDF $f(x)$。

### 11.9 $g(s)$：参考态中的有限长距结构

$g(s)$ 吸收不能用短距离微扰展开描述的零动量参考态形状。在本地定义中，它还包含尚需由 $m_0$ 条件处理的有限线性部分。因此：

- 它属于参考态数据模型，不能固定成旧解后声称“全部参数已重新拟合”。
- 它不直接进入 $Z_R$，否则会把参考态的有限长距结构一并作为 UV 因子消除。
- 它既不是目标移动矩阵元，也不是傅里叶变换后的 PDF；其通道和参考态依赖需要随输入记录。

只有两个短距点时，本模型用 $\ln C_0^\Gamma+m_0s$ 代替短距的自由 $g$；长距的每个网格点仍独立拟合 $g$。

### 11.10 $C_0^\Gamma$、$\mu$、$\alpha_s$、RGR 和 LRR

短距 Wilson 系数给出选定微扰方案下的零阶矩约束。当前 unpolarized 的 NLO 系数为

$$
C_0^{\mathrm{unpol}}(s,\mu)
=
1+\frac{\alpha_s(\mu)C_F}{4\pi}
\left[
3\ln\!\left(\frac{s^2\mu^2e^{2\gamma_E}}{4}\right)+5
\right].
$$

其中 $\gamma_E$ 是 Euler 常数。当前主分析固定

$$
\mu=2\ \mathrm{GeV},
\qquad
\alpha_s(\mu)=0.296,
\qquad
n_f=3.
$$

此处 $\alpha_s$ 的输入与拟合参数 $\Lambda$ 分开管理；本轮拟合不会自动用后者重新定义强耦合。其他极化必须使用相应的 $C_0^\Gamma$，不能因共享 Wilson line 而忽略局域流和微扰系数的差别。[R2–R3]

RGR 是对声明的重整化群对数进行演化／重求和；可选短距尺度写为

$$
\mu_h(s)=\frac{2e^{-\gamma_E}c_{\mathrm{scale}}}{s},
$$

其中 $c_{\mathrm{scale}}$ 是尺度变化设置。当前另有 LL 演化测试分支，但本轮参数对比的主模型关闭 RGR。

LRR 处理与 leading renormalon 有关的重求和及相应有限方案。拟合出 $m_0$、固定 Wilson-loop 候选，或开启 LL 演化，都不自动等于已经实现 LRR。本轮主模型同样关闭 LRR；恢复更高精度时，应连同 $C_0^\Gamma$、残余质量处方和 matching 一起保持方案一致。

### 11.11 $z_0$、$z_1$、$z_s$ 与实际 hybrid 拼接

当前公共拟合网格是

$$
s_j=(0.15+0.10j)\ \mathrm{fm},
\qquad j=0,1,\ldots,13.
$$

因此 $z_0=0.15\ \mathrm{fm}$，短距模型上限为 $z_1=0.25\ \mathrm{fm}$；共有两个短距点、十二个自由长距 $g$、十四个 $f$。主拼接位置是

$$
z_s=0.25\ \mathrm{fm}.
$$

$z_1$ 决定拟合时哪里使用微扰条件，$z_s$ 决定应用重整化时哪里切换分支；它们在当前主设置中数值相同，但意义不同，不要求任何分析都取同一值。

本地实际输出的 normalized hybrid shape 为

$$
\boxed{
H_{\mathrm{hyb}}^\Gamma(P^z,z;a,z_s)
=
\begin{cases}
1, & z=0,\\[2pt]
\displaystyle
\frac{H_B^\Gamma(P^z,z,a)}{H_0^\Gamma(s,a)},
& 0<s\le z_s,\\[10pt]
\displaystyle
\frac{H_B^\Gamma(P^z,z,a)}{Z_R^\Gamma(s,a)}
\frac{Z_R^\Gamma(z_s,a)}{H_0^\Gamma(z_s,a)},
& s>z_s.
\end{cases}
}
$$

拼接因子定义为

$$
\eta^\Gamma(z_s,a)
\equiv\frac{Z_R^\Gamma(z_s,a)}{H_0^\Gamma(z_s,a)}.
$$

它由连续条件导出，不是新的拟合参数。该条件保证函数值在拼接处相接，不自动保证导数连续，也不消除截断微扰、格距和高扭度误差。[R1]

对同一格距，$\mathcal N$ 与 $s$ 无关，因而在长段比值中精确抵消：

$$
\frac{Z_R^\Gamma(z_s,a)}{Z_R^\Gamma(s,a)}
=
\exp\!\left\{
\left[
\frac{k}{a\ln(a\Lambda)}+m_0
\right](z_s-s)
+a^2\bigl[f(z_s)-f(s)\bigr]
\right\}.
$$

**所以 $d$ 不直接出现在最终长段因子中，但它会通过联合拟合改变其他参数，并间接影响结果。** $g(s)$ 也不直接进入这一步，但释放或固定 $g$ 会改变联合拟合的分解。由于短段直接使用测得的 $H_0$，短段不受这次 $Z_R$ 参数更换影响；误差传播必须保留参考态与移动矩阵元的配对副本。

### 11.12 局域荷、$Z_V$、$Z_A$、$Z_T$ 与原点

normalized shape 在原点满足

$$
H_{\mathrm{hyb}}^\Gamma(P^z,0)=1+0i.
$$

非局域拟合式不直接延伸到原点；本地代码单独设置

$$
Z_R^\Gamma(0,a)=1.
$$

局域荷的整体归一化使用独立的局域流因子，例如

$$
q_V^{\mathrm{ren}}=Z_Vq_V^{\mathrm{bare}},
\qquad
q_A^{\mathrm{ren}}=Z_Aq_A^{\mathrm{bare}},
\qquad
q_T^{\mathrm{ren}}(\mu)=Z_T(\mu)q_T^{\mathrm{bare}}.
$$

它们分别对应向量、轴矢量和张量流；实际 bare charge 还须与相同的运动学及算符约定一致。本流程先保留 normalized hybrid shape，再在 quasi-to-lightcone matching 后恢复所需的重整化局域荷。不能把这三个局域因子当成已完整消除非局域 Wilson-line 发散的 $Z_R$。

### 11.13 先验、自由参数数与误差的含义

这里采用带先验的相关拟合，目标函数写为

$$
\chi^2_{\mathrm{aug}}
=
\bigl[\mathbf F(\boldsymbol\theta)-\mathbf y\bigr]^T
\mathbf C^{-1}
\bigl[\mathbf F(\boldsymbol\theta)-\mathbf y\bigr]
+
\sum_j
\left(
\frac{\theta_j-\overline\theta_j}{\sigma_{\theta_j}}
\right)^2.
$$

$\mathbf y$ 是三个 ensemble 的参考态对数向量；$\mathbf C$ 使用 ensemble 内的冻结收缩协方差、ensemble 间为零的块结构，逆算子按已声明的本征模规则实现。若发生删模，该式中的逆应理解为保留子空间中的逆。$\overline\theta_j$ 与 $\sigma_{\theta_j}$ 是对应物理参数化的先验中心和宽度，不是额外的格点测量数据。

当前主模型先验的数值如下；$\mathcal N(\overline x,\sigma^2)$ 表示均值和方差：

$$
\begin{aligned}
 k&\sim\mathcal N(1,2^2),&
 \theta_\Lambda&\sim\mathcal N(\ln 0.1,1^2),\\
 d&\sim\mathcal N(-0.5,1^2),&
 m_0&\sim\mathcal N\!\left(1\ \mathrm{GeV},(1\ \mathrm{GeV})^2\right),\\
 g(s_j)&\sim\mathcal N(2,2^2),&
 f(s_j)&\sim\mathcal N\!\left(0,(1\ \mathrm{GeV}^2)^2\right).
\end{aligned}
$$

固定 UV 质量时，不再对导出的 $k$ 额外施加独立先验，其余参数先验沿用。无约束方案和约束方案的自由参数数分别为

$$
N_{\mathrm{par}}^{\mathrm{free}}
=4+12+14=30,
\qquad
N_{\mathrm{par}}^{\mathrm{fixed\ UV}}
=3+12+14=29.
$$

这次没有把 Wilson-loop 测定作为带误差的额外似然；两套拟合均只有同一批 $42$ 个参考态数据。两套共有参数沿用相同的辅助 Gaussian prior 随机流，三 ensemble 原 bootstrap 索引逐项保留。

各表参数误差定义为配对 bootstrap 副本的样本标准差：

$$
\sigma_{p,\mathrm{BS}}
=
\sqrt{
\frac{1}{N_B-1}
\sum_{b=1}^{N_B}
\left(p^{(b)}-\overline p_{\mathrm{BS}}\right)^2
},
\qquad N_B=1000.
$$

它不等于 posterior 曲率误差，也不包含尚未传播的 Wilson-loop 质量、窗口和有限方案系统误差。参数变化的误差应由逐副本差值计算：

$$
\sigma_{\Delta p,\mathrm{BS}}
=\operatorname{Std}_{b,\,\mathrm{ddof}=1}
\!\left[p_{\mathrm{with}}^{(b)}-p_{\mathrm{without}}^{(b)}\right].
$$

拟合质量仍须另外判断。当前质量门为

$$
\frac{\chi^2_{\mathrm{data}}}{\mathrm{dof}}<1000,
\qquad Q>10^{-6}.
$$

使用先验的状态为 `prior_constrained`，不把有限误差或优化收敛自动解读为参数已由数据独立确定。

### 11.14 Wilson-loop 质量、有限方案与本轮参数对比

Cornell 拟合给出的量是

$$
a m_{\mathrm{line,Cornell,eff}}=\frac{c_0}{2}.
$$

一般只能在声明的有限势零点约定下，将其概念性分解为

$$
m_{\mathrm{line,Cornell,eff}}(a)
=\delta m_{\mathrm{UV}}(a)
+m_{\mathrm{finite}}^{\mathrm{Cornell}}
+\Delta_{\mathrm{window,model}}(a).
$$

最后一项表示有限长边、激发态、距离窗口和拟合模型等引起的偏差，不是已经被数据唯一分离的参数。相同 HYP 链接、表示和方向有助于使 UV 部分对应，但不会自动令 Cornell 的有效质量等于纯 UV 部分，更不会自动确定目标微扰方案中的 $m_0$。[R1；本地窗口证据见第 6 节]

2026-10-28 的本轮测试按精确条件采用

$$
a_{\mathrm{L32x64}}\delta m_{\mathrm{UV}}=0.1874,
$$

并重新拟合所有其余参数。全局参数对比如下，误差均为 $1000$ 个 bootstrap 副本的标准差：

| 参数 | 不加 Wilson-loop 约束 | 固定 Wilson-loop 候选值 |
|---|---:|---:|
| $k$ | $0.68896\pm0.01147$ | $1.03004\pm0.05269$ |
| $\Lambda\ [\mathrm{GeV}]$ | $0.39867\pm0.02184$ | $0.00902\pm0.00274$ |
| $d$ | $0.24665\pm0.00830$ | $0.02667\pm0.01937$ |
| $m_0\ [\mathrm{GeV}]$ | $0.49046\pm0.02756$ | $0.00380\pm0.00326$ |
| $a\delta m_{\mathrm{UV}}$，L32x64 | $0.40337\pm0.01297$ | $0.18740$，固定 |
| $a m_{\mathrm{linear}}$，L32x64，不含 $f$ | $0.18042\pm0.00234$ | $0.18567\pm0.00148$ |
| $\chi^2_{\mathrm{data}}/\mathrm{dof}$ | $6.81579$ | $15.77633$ |
| $Q$ | $1.88\times10^{-12}$ | $1.23\times10^{-36}$ |

两套拟合均未通过当前 $Q$ 门。参数显著重分配后，净线性组合仍较接近，$f(s)$ 进一步调整，因此最终 hybrid 矩阵元可以接近；这种接近不能证明各个参数稳定，也不能证明 Wilson-loop 候选已经成为正式 UV 质量测定。

全部 $f$、$g$ 数值、误差和比较图见：

- [全部参数比较报告](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_Parameter_Comparison_20261028/REPORT.md)。
- [全部参数 CSV](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_Parameter_Comparison_20261028/artifacts/parameters_comparison.csv)。
- [三页参数比较图](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_Parameter_Comparison_20261028/artifacts/plots/all_parameter_comparisons.pdf)。

### 11.15 公式与本地代码的对应关系

以下目录均相对 `/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm`：

| 实现位置 | 对应内容 |
|---|---|
| `Normalized_Hybrid_20261026/fit_zr.py`，`Model.model` | 参考态分段模型、共享参数与 $f,g$ 网格 |
| `Normalized_Hybrid_20261026/fit_zr.py`，`Model.zr` | 包含一次 $m_0s$、排除 $g(s)$ 的 $Z_R$ |
| `Normalized_Hybrid_20261026/physics.py`，`coefficient` | 各通道 $C_0^\Gamma$ 和可选 LL 演化 |
| `Normalized_Hybrid_20261026/physics.py`，`hybrid` | 短距 ratio、长距拼接和原点约束 |
| `Normalized_Hybrid_20261026/config.json` | 单位约定、尺度、先验、窗口、模型和质量门 |
| `FixedUV_L32x64_am0p1874_AllParams_20261028/analysis.py` | 固定 UV 组合后的全参数联合重拟合 |
| `WilsonLoop_Parameter_Comparison_20261028/compare_parameters.py` | 有无 Wilson-loop 约束的全部参数对照与配对差值误差 |

## 12. 参数说明的参考文献

- **[R1]** X. Ji et al., *A Hybrid Renormalization Scheme for Quasi Light-Front Correlations in Large-Momentum Effective Theory*, arXiv:2008.03886v2，*Nuclear Physics B* **964** (2021) 115311。参见第 3 节关于短长距离分开处理、Wilson-line 质量和连续条件的讨论。文献中质量反项的符号应与本文约定逐式核对。
- **[R2]** Y.-K. Huo et al. (Lattice Parton Collaboration), *Self-Renormalization of Quasi-Light-Front Correlators on the Lattice*, arXiv:2103.02965，*Nuclear Physics B* **969** (2021) 115443。参见 self-renormalization 的格距拟合、有限残余质量和短距方案条件。本文当前采用的 $a^2f(s)$ 离散化项、先验和约束测试以本地代码为准，并非逐项照抄原文。
- **[R3]** Y. Su et al., *Resumming Quark's Longitudinal Momentum Logarithms in LaMET Expansion of Lattice PDFs*, arXiv:2209.01236v2，*Nuclear Physics B* **991** (2023) 116201。用于核对短距系数与重整化群演化的背景；本轮参数对比未开启 RGR 或 LRR。


---

记录日期：**2026-10-28（Asia/Shanghai）**。本次更新：**hybrid 重整化参数含义、LaTeX 公式及 Wilson-loop 约束对照**。

---

## 13. 三 ensemble Wilson-loop 联合抽取：2026-10-30 操作与结果

日期：2026-10-30。工作目录：`Compare_YaoFei/Renorm/WilsonLoop_Joint_20261030`。

### 13.1. 执行范围与结论

已执行 HYP 元数据审计、与 PDF 相同的 1000 副本重建、统一物理长边窗口扫描、两态对照、共同物理距离插值、三个格距 UV 联合拟合、距离/尺度/离散项扫描、分块敏感性和独立数值核验。原始 Wilson-loop 测量与旧分析均保留。

本轮得到可进一步检验的**条件性 UV 参数**，尚未批准为最终重整化系数，也未替换现有 hybrid 矩阵元。固定格距中心值，未传播额外 scale-setting 误差。没有用新 Wilson-loop 模型绕过现有 PDF 的零动量参考态问题。

### 13.2. HYP 处方与样本

- C24P29/L24x72：199 个组态，$a=0.105$ fm；E32P29/L32x64：425 个，$a=0.0897$ fm；F32P30/L32x96：194 个，$a=0.0775$ fm。
- 对三个来源的全部 818 份 `scidac-record-xml` 检索，均未找到生成所用 HYP links 的完整 HYP 参数。XML 主要记录 ensemble 生成配置；其中的 fermion stout 参数不是本轮 Wilson line 的 HYP 配方。
- C24 附近 `Readme_setting.txt` 的 3D 小节名为 `3times`，却写 `num_smear=10`，不能据此确认本轮 `5times` 的具体处方。因此共同 $k$ 的拟合**条件于三个目录实际使用相同 HYP 参数**。三个 gate 组态的 temporal-link 对照均显示时间链接不变；它不等于逐组态重新做了薄链控制。
- 主分析只用基础表示、纯空间三平面。平面先在组态内平均，组态为抽样单位。
- 从逐组态 loop 按 PDF 原组态顺序重新排列，完全复用每套 `reference/*_unpolarized.npz` 的 `boot_indices`，每套 **1000 行**。没有从旧 3000 副本截取，也没有对已有副本再次抽样。indices 原始字节哈希逐项匹配；对一个独立行重新直接平均复核。

证据：`artifacts/smearing_audit.json`、`paired/*.json`、`paired_independent_checks.json`。

### 13.3. 首阶段：统一物理长边窗口

模型为

$$
\ln W_a(R,L)=c_a(R)-E_a(R)L,
$$

这里 $L$ 换成 $\mathrm{GeV}^{-1}$，因此 $E$ 单位是 GeV。对均值/每个副本先平均 loop 再取 log。无效 log 变为 NaN，不 abs/clip。协方差由 delete-one jackknife 传播，相关本征模删模阈值为 $10^{-8}$，未调节样本秩阈值为 $10^{-12}$。每个拟合保留样本秩、有效秩和列归一白化设计矩阵的秩/条件数。

预先声明的窗口和在三套格距上的离散实现为：

| 物理 $L$ 范围 [fm] | C24 的 $L/a$ | E32 的 $L/a$ | F32 的 $L/a$ |
|---|---|---|---|
| 0.70–1.15 | 7–10 | 8–12 | 10–14 |
| 0.80–1.20（注册主窗口） | 8–11 | 9–13 | 11–15 |
| 0.90–1.24（统一较晚对照） | 9–11 | 11–13 | 12–16 |
| 1.00–1.27 | 10–12 | 12–14 | 13–16 |

质量门沿用 $\chi^2/{\rm dof}<1000$、$Q>10^{-6}$，完整统计误差要求 1000 副本全部有效。由于每套只能取整数格点，实际物理边界略有差异，全部实际 $L$ 坐标随结果保存。

注册主窗口仍有上游失败；统一的较晚窗口在共同 $R=0.45,0.55,0.65,0.75,0.85$ fm 所需的全部原始距离点上通过首阶段诊断，且 1000 副本完整。更晚窗口的一部分联合结果只有 998 副本有效，因此不提供完整 bootstrap 误差。未逐距离拼接“最好”窗口，也未将后验选出的较晚对照偷换为注册主结果。

#### 两态对照

在 $R=0.45,0.65,0.85$ fm、$L\simeq0.42\text{--}1.24$ fm 比较

$$
\ln W=c-E_0L+\ln\left(1+B e^{-\Delta E L}\right),\qquad B>0,\quad\Delta E>0.
$$

采用 $\Delta E=0.6\pm0.8$ GeV、$\ln B=0\pm2$ 的先验，另将能隙先验中心改为 0.3 和 1.0 GeV。正重叠与两态截断是这一对照的条件。共同物理 $R$ 的 log-loop 插值也属数值近似。

每组均传播 1000 配对副本，并另保存辅助 Gaussian prior-center 波动与固定 prior-center 的两套误差。共 27 个拟合，状态全部显式为 `prior_constrained`、`data_identifiability=None`。这里的 data $Q$ 使用名义数据自由度；带先验时并非严格校准的普通 p 值。不能把小先验依赖或高 $Q$ 自动称为基态已经确定。中央先验下 C24 的 $R=0.45,0.65$ fm 和 E32 的 $R=0.45$ fm 仍未通过 $Q$ 门，说明较早数据的两态近似也非普遍充分。

### 13.4. 对齐共同物理距离与跨格距能量差

先拟合各原始 $R$ 的能量，再以线性插值在每个 jackknife/bootstrap 上对齐公共 $R$，禁止物理距离外推。主公共网格为 $0.45,0.55,0.65,0.75,0.85$ fm；另统一去掉 $0.45$ fm 作距离范围对照。输入跨 ensemble 协方差块严格为零。

对较晚窗口，共同距离上的能量差近似为常数：

- $E_{C24}-E_{E32}=-0.14084620\pm0.00113974$ GeV，常数拟合 $Q=0.282165$。
- $E_{F32}-E_{E32}=0.12653644\pm0.00219264$ GeV，$Q=0.354386$。

两组差共享 E32，因此它们之间的协方差也被保存。能量差近似常数支持差异的主要部分来自距离无关项，但不能把它全部解释为纯 UV；有限格距、smearing 与海夸克质量差异也可能贡献。

### 13.5. 共同 UV 参数与离散项

采用

$$
E_a(R)=-\frac{2k}{a\ln(a\Lambda_{\rm latt})}+V(R)+a^2F(R),
$$

其中 $a$ 采用自然单位。$V(R)$ 在公共网格上独立，避免固定 Cornell 形状。分别检验不含离散项、共同常数 $F$、逐 $R$ 的 $F(R)$。固定 $\Lambda_{\rm latt}=0.2$ GeV 的 $k$ 结果为：

| $L$ 范围 [fm] | 无离散项 $k$ | 共同 $a^2$ 项 $k$ | 逐 $R$ 的 $a^2$ 项 $k$ | 全部首阶段通过且副本完整 |
|---|---|---|---|---|
| 070_115 | $0.885301\pm0.00274286$ | $0.271152\pm0.0204774$ | $0.318824\pm0.020729$ | 否 |
| 080_120 | $0.887315\pm0.00342825$ | $0.329690\pm0.0286826$ | $0.365816\pm0.0295402$ | 否 |
| 090_124 | $0.853221\pm0.00523039$ | $0.502148\pm0.0428994$ | $0.512962\pm0.0443505$ | 是 |
| 100_127 | $0.859589$（误差不可用） | $0.533098$（误差不可用） | $0.539209$（误差不可用） | 否 |

较晚窗口的无离散项拟合 $\chi^2/\mathrm{dof}=78.023642/9$、$Q=3.99\times10^{-13}$，未过门。共同 $a^2$ 模型为

$$
k=0.50214833\pm0.04289942,\qquad\chi^2/\mathrm{dof}=7.961656/8,\qquad Q=0.437225.
$$

该模型 15 个数据、7 个参数，样本秩/有效秩均为 15，白化归一模型秩为 7，设计矩阵条件数约 2341.0。共同 $a^2$ 系数为 $F=-0.808615\pm0.097732$ GeV$^3$，与 $k$ 的副本相关系数约 $0.99255$，表明 UV 与离散项高度相关。这是给定模型和尺度的统计误差，不包括窗口、HYP、有限质量方案及尺度设定系统误差。

#### $\Lambda_{\rm latt}$ 不能由当前模型同时确定

固定扫描得到：

| $\Lambda_{\rm latt}$ [GeV] | 较晚窗口、共同 $a^2$ 项的 $k$ | $Q$ |
|---|---|---|
| 0.05 | $0.639079\pm0.054598$ | 0.437225 |
| 0.10 | $0.567664\pm0.048497$ | 0.437225 |
| 0.15 | $0.528360\pm0.045139$ | 0.437225 |
| 0.20 | $0.502148\pm0.042899$ | 0.437225 |
| 0.30 | $0.468408\pm0.040017$ | 0.437225 |

所有 $\Lambda$ 的拟合优度相同不是独立交叉验证。在恰好三个格距下，三维向量

$$
u_i(\Lambda)=-\frac{2}{a_i\ln(a_i\Lambda)},\qquad 1,\qquad a_i^2
$$

构成基时，变化 $\Lambda$ 可由重新定义 $k$、共同势的常数偏置和 $a^2$ 系数精确吸收。本轮独立核验的重参数化最大能量残差低于 $3\times10^{-16}$ GeV。故不能把 $\Lambda$ 再放开并声称它被这些数据确定；需额外格距、有限势/短距信息或明确外部约束。

### 13.6. 条件性 UV 线质量

按当前 hybrid 衰减约定，

$$
a\delta m_{\rm UV}=-\frac{k}{\ln(a\Lambda_{\rm latt})}.
$$

较晚窗口、共同 $a^2$ 模型、固定 $\Lambda_{\rm latt}=0.2$ GeV 得到：

| Ensemble | $a\delta m_{\rm UV}$ | 仅本模型统计误差 | $\delta m_{\rm UV}$ [GeV] |
|---|---:|---:|---:|
| L24x72 | 0.22413937 | 0.01914862 | 0.42122614 |
| L32x64 | 0.20941788 | 0.01789094 | 0.46068894 |
| L32x96 | 0.19738363 | 0.01686283 | 0.50256924 |

与旧 Cornell $c_0/2$ 的差不应只按统计误差判定矛盾：势的有限零点、物理窗口和离散模型不同。当前抽取的是受模型约束的 UV 项，不是直接重复读取旧截距。

绝对有限质量仍存在

$$
\delta m\to\delta m+c,\qquad V_{\rm fin}\to V_{\rm fin}-2c
$$

的自由度。后续施加 hybrid 时，需要与 $m_0$、短距匹配和局域归一化约定协调，防止同一有限线质量被减去两次。本轮不由 Wilson-loop 数据测定 hybrid 的 $m_0$。

### 13.7. 窗口、距离与自相关敏感性

- 注册主窗口 0.80–1.20 fm 的共同 $a^2$ 模型 $k=0.329690\pm0.028683$，但部分首阶段不通过。相对统一较晚窗口，配对变化为 $-0.172458\pm0.036107$，约 $-4.78\sigma$。因此仍不能宣称长边污染已经完全消除。
- 更晚窗口给出约 $k=0.53310$，但只有 998/1000 联合副本有效，不给完整统计误差。
- 将统一公共距离下限从 0.45 提到 0.55 fm，得到 $k=0.731027\pm0.127156$，$Q=0.804976$。误差扩大且中心变化，不按哪个结果更好看来选择。

未分块与按相邻存储组态作非重叠 block 的对照（较晚窗口、共同 $a^2$ 模型）为：

| block 大小 | $k$ | 完整统计误差 | 联合 $Q$ | 副本数 |
|---|---:|---:|---:|---:|
| 1 | 0.502148 | 0.0428994 | 0.437225 | 1000/1000 |
| 2 | 0.493647 | 不可用 | 0.35971 | 999/1000 |
| 4 | 0.501330 | 0.0438194 | 0.345932 | 1000/1000 |
| 8 | 0.497969 | 0.0466347 | 0.0192749 | 1000/1000 |
| 16 | 0.474804 | 不可用 | 1.48272e-06 | 999/1000 |

block=2、16 有不完整副本，未删行给出误差。存储组态间隔并非完全均匀；这里是相邻存储记录的分块敏感性，不是已完成的综合自相关时间测定。block 变化会同时改变协方差、有限余数组态的使用以及拟合权重，不将所有变化都归因于单一自相关量。

### 13.8. 输出、验证与后续

- [全部诊断图册](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_Joint_20261030/artifacts/plots/all_diagnostics.pdf)：8 页，含有效能量/窗口、距离能量、跨格距差、$k$ 扫描、分块、两态残差及先验对照、条件性线质量。
- [拟合参数表](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_Joint_20261030/artifacts/tables/joint_parameters.csv)、[首阶段表](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_Joint_20261030/artifacts/tables/first_stage.csv)、[两态表](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_Joint_20261030/artifacts/tables/two_state.csv)、[条件性线质量](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_Joint_20261030/artifacts/tables/conditional_UV_mass.csv)。
- [配对 UV 交接](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_Joint_20261030/artifacts/conditional_UV_handoff.npz)及[状态说明](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_Joint_20261030/artifacts/conditional_UV_handoff.json)：保留同 PDF 的 1000 行，联合拟合输出产生的跨 ensemble 相关也保留。
- 独立 full-covariance 正规方程与主 GLS 的参数相对差约 $1.7\times10^{-10}$；一个副本的独立优化与存储线性传播相对差约 $7.0\times10^{-9}$；共同 $\Lambda$ 重参数化由独立求解验证。

当前可交付的是条件性共同 $k$、相对能量偏置与完全配对副本。尚待解决的项目是确切共同 HYP 处方、残余窗口/距离依赖、有限质量方案以及更多格距/外部短距约束。因此 `physical_validated=False`、`UV_coefficient_approved=False`。

复现时以新输出目录运行 `code/analyze.py`、`code/two_state.py`、`code/sensitivity.py`、`code/report.py`；生产数值脚本要求 CPU Slurm allocation。已完成目录具有完成标记，不应直接覆盖。提交记录和日志保存在本目录。

### 13.9. 复核与原始记录入口

- [独立完整运行报告](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_Joint_20261030/REPORT.md)。
- [数值与文件最终检查](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_Joint_20261030/artifacts/final_checks.json)。
- [HYP XML 元数据审计](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_Joint_20261030/artifacts/smearing_audit.json)。
- [配对索引与重建检查](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_Joint_20261030/artifacts/paired_independent_checks.json)。
- [拟合独立核验及 $\Lambda$ 退化证明](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_Joint_20261030/artifacts/independent_checks.json)。
- [CPU 作业记录](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_Joint_20261030/submissions.json)。

更新日期：**2026-10-30（Asia/Shanghai）**。本节全部数值为当前条件模型的结果，原始测量及已有 hybrid 生产结果保持独立。

---

## 14. Wilson-loop 相对线质量约束与短距有限质量重拟合：unpolarized、u-d（2026-11-01）

日期：2026-11-01。三个 ensemble，1000 个原有配对 bootstrap 副本。

### 14.1. 结果摘要

有限质量参数为 $\bar m=-0.357232\pm0.002868\,\mathrm{GeV}$。

相关拟合 $\chi^2/\mathrm{dof}=61.265386/13$、$Q=3.11663\times10^{-8}$；统计门通过：`False`。

**本轮是诊断性重整化矩阵元测试。原有 P=0 参考的输入质量门未全部通过，Wilson-loop 窗口与共同 HYP 处方也仍有系统性限制；不据此宣称最终物理重整化已经确定。**

图中完整显示数值诊断点；NPZ 另存统计质量掩码及带掩码的产品。局域荷未还原，RGR/LRR 均关闭。

| Ensemble | 保留整数动量 | $z=1\,\mathrm{fm}$ 的新/旧因子 | $z=1.4\,\mathrm{fm}$ 的新/旧因子 |
|---|---|---|---|
| L24x72 | [3, 4] | $0.996927\pm0.002059$ | $0.916507\pm0.032713$ |
| L32x64 | [3, 4, 5] | $1.006939\pm0.001285$ | $0.953788\pm0.024120$ |
| L32x96 | [3, 4] | $0.984660\pm0.003670$ | $0.933412\pm0.018813$ |

该因子同时作用于实部与虚部。不同 ensemble 的相同整数动量不代表相同物理动量。短距段逐副本与原结果在浮点舍入精度内一致。

### 14.2. Wilson-loop 输入与有限质量定义

使用已完成分析的统一较晚长边窗口 $L=0.90\text{--}1.24\,\mathrm{fm}$，共同距离 $R=0.45,0.55,0.65,0.75,0.85\,\mathrm{fm}$。该窗口是后验较晚对照，不替代原注册主窗口；其所需首阶段点质量门和 1000 副本完整性已通过。只读取能量及协方差，不使用该文件里的 $k,\Lambda,F$ 或条件性 UV 质量。

$$
\Delta_i=\frac12\left[E_i(R)-E_{E32}(R)\right]_{\mathrm{const}},\qquad \Delta_{E32}=0.
$$

| Ensemble | $\Delta_i$ [GeV] |
|---|---|
| L24x72 | $-0.070423\pm0.000570$ |
| L32x64 | $0.000000\pm0.000000$ |
| L32x96 | $0.063268\pm0.001096$ |

这是含有限格距等贡献的相对有效线质量，不是已唯一分离的纯 UV 质量。

### 14.3. 短距相关拟合与重整化公式

为消去在 hybrid 比值中抵消的距离无关归一化项，以 $s_0=0.15\,\mathrm{fm}$ 为参考：

$$
Y_i(s)=\ln\frac{H_{0,i}(s)}{H_{0,i}(s_0)}+\Delta_i(s-s_0).
$$

采用与原基准相同的 NLO 系数、$\mu=2\,\mathrm{GeV}$、$\alpha_s=0.296$，在自然单位下拟合

$$
Y_i(s)=\begin{cases}\ln\!\left[C_0(s)/C_0(s_0)\right]+\bar m(s-s_0)+a_i^2F(s),&s\le0.25\,\mathrm{fm},\\G(s)+a_i^2F(s),&s>0.25\,\mathrm{fm}.\end{cases}
$$

其中 $F(s_0)=0$ 固定无关的常数，$G$ 和剩余离散函数 $F$ 在各网格点独立，无参数先验。有限质量实际由 $0.15$ 与 $0.25\,\mathrm{fm}$ 两个短距点的差约束，尚未检查更丰富的短距窗口。

定义仅供比值使用的相对重整化因子

$$
\mathcal L_i(s)\equiv\ln\frac{Z_{R,i}(s)}{Z_{R,i}(s_0)}=(-\Delta_i+\bar m)(s-s_0)+a_i^2F(s).
$$

对归一化裸形状 $S_i(P,s)$，应用

$$
H_i^{\mathrm{hyb}}(P,s)=\begin{cases}S_i(P,s)/H_{0,i}(s),&0<s\le z_s,\\S_i(P,s)\exp[\mathcal L_i(z_s)-\mathcal L_i(s)]/H_{0,i}(z_s),&s>z_s,\end{cases}\qquad z_s=0.25\,\mathrm{fm}.
$$

$s=0$ 固定为 $1+0i$。$\bar m$ 吸收了 E32 参考质量零点，不能与旧模型的 $m_0$ 直接作同方案参数差。$\Lambda,d$ 不作为新模型拟合参数；其距离无关归一化已经消去。

**离散项约定：**$F$ 是完成上述有效 Wilson-loop 扣除后的 PDF 剩余离散项，不额外扣除旧 Wilson-loop 拟合的 $a^2F_{\mathrm{WL}}$。保留其线性部分用于允许两种观测量的有限格距效应不同；数值验证表明把共同 $a^2s$ 项在线质量与该剩余项之间重新分配不会改变 hybrid 因子，故没有对同一项重复扣除。

### 14.4. 误差、协方差与基准比较

每个 ensemble 内严格复用 PDF/Wilson-loop 的相同组态表及 1000 行重采样索引。原始变量为共同网格上的 $[\ln H_0,E_{\mathrm{WL}}]$，每个独立 ensemble 内联合估计 Schaefer–Strimmer 协方差，ensemble 间原始块严格为零，再传播对数比值和共享 E32 的质量差；因此保留 PDF 与 Wilson-loop 的相关及参考 ensemble 引入的跨 ensemble 协方差。

拟合数据 39 个，参数 26 个；协方差秩 39，模型秩 26。无参数先验，不删除失败副本来制造完整误差。

作为协方差敏感性检查，沿用历史 PDF-only 权重但仍传播 Wilson-loop 波动，得到 $\bar m=-0.357652\pm0.002883\,\mathrm{GeV}$、$Q=1.95948\times10^{-62}$。

旧基准为 `Normalized_Hybrid_20261026` 的 `nlo_a2`：$\chi^2/\mathrm{dof}=81.789442/12$，名义 $Q=1.87869\times10^{-12}$，$m_0=0.490461\pm0.027563\,\mathrm{GeV}$。旧基准包含参数先验，其名义 $Q$ 与本轮不能直接用于模型优劣排序。

本轮与旧基准使用相同分子、参考态、NLO 系数、拼接点及 $a^2$ 幂次，但也消去了归一化模型及其先验，所以对比反映完整替代方案的变化，不能全部归因为单个 UV 系数。差值与因子比值的误差逐副本配对计算。

长距 $F(s)$ 使用与旧基准相同的线性插值和边界线段延伸；超过共同网格上限 $1.45\,\mathrm{fm}$ 的数值标记为外推并从统计质量产品中排除。

### 14.5. 图与数据入口

- [全部图册](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_RelativeMass_Hybrid_20261101/artifacts/plots/all_results.pdf)
- [三 ensemble 对比总览](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_RelativeMass_Hybrid_20261101/artifacts/plots/overview_comparison.png)
- [重整化因子比值](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_RelativeMass_Hybrid_20261101/artifacts/plots/hybrid_factor_ratio.png)
- [配对矩阵元差值](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_RelativeMass_Hybrid_20261101/artifacts/plots/paired_matrix_shifts.png)
- [参考态拟合诊断](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_RelativeMass_Hybrid_20261101/artifacts/plots/reference_fit_diagnostics.png)
- [矩阵元比较表](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_RelativeMass_Hybrid_20261101/artifacts/tables/matrix_elements_comparison.csv)
- [参数表](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_RelativeMass_Hybrid_20261101/artifacts/tables/parameters.csv)
- [验证记录](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_RelativeMass_Hybrid_20261101/artifacts/validation.json)

每个 ensemble 的 `artifacts/fits/hybrid_WLrelative_*_unpolarized_u_minus_d.npz` 同时保存新旧矩阵元、1000 配对副本、差值误差、重整化因子及质量掩码。

### 14.6. 复现

CPU Slurm 脚本：`slurm/run.sh`。配置：`config.json`。完成标记：`artifacts/complete.json`。
数值分析脚本禁止覆盖已经完成的输出目录；新测试应使用新目录。

### 14.7. 本次测试的结论与验证

在 $z\simeq1\,\mathrm{fm}$，新旧因子差为约 $0.3\%$–$1.5\%$；在 $z\simeq1.4\,\mathrm{fm}$，差扩大至约 $5\%$–$8\%$。该变化是完整替代方案的结果，不作为纯 UV 函数被唯一测定的证据。

新模型未通过预设 $Q>10^{-6}$ 门。其中心值可用于诊断比较；P=0 参考窗、短距两点近似、Wilson-loop 长边窗与 HYP 共同处方仍需进一步检验。表中误差为给定方案下的统计误差，未包括上述系统误差。

验证包括：三 ensemble 原 1000 行配对索引逐项一致；完整副本及矩阵元配对差误差；参考点从 $0.15$ 改为 $0.25\,\mathrm{fm}$ 后的 hybrid 因子不变；$a^2s$ 项重新分配后因子不变；独立 GLS 求解；原输入文件哈希；13 页图册和报告链接。

- [独立运行报告](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_RelativeMass_Hybrid_20261101/REPORT.md)。
- [最终检查](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_RelativeMass_Hybrid_20261101/artifacts/final_checks.json)。
- [图表浏览入口](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_RelativeMass_Hybrid_20261101/artifacts/index.html)。
- [CPU 作业记录](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoop_RelativeMass_Hybrid_20261101/submissions.json)。

## 15. 当前 HYP3D5 quark PDF 全流程、代码与结果总账（2026-11-22）

### 15.1 本节用途、术语和当前版本

本节按 **2026-11-22 实际保存的代码、配置、NPZ 元数据与完成清单** 更新主线分析。前面第 1–14 节保留 Wilson-loop、线质量约束和历史 hybrid 测试；其中使用的旧 bootstrap 数、裸量拟合模型、动量范围和重整化参数，不能直接替代本节的当前设置。

本批链接为固定的 **5 次 3D-HYP smearing**，数据目录标签为 `hyp_3D_5times`。目前这里汇总的是三种极化、三种 **flavor** 和不同**分析阶段**的结果，未扫描多个 gradient-flow time。HYP 次数不能当作 gradient-flow time；现有主结果也没有环化费米子、有限流时间匹配或小流时间外推。输入链接的完整 HYP 参数仍需以生成端元数据为准。

当前权威汇总目录为：

```text
/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122
```

以下称该目录为 `BASE`，称工作区 `/public/group/lqcd/donghx/Diagram_GluonPDF` 为 `WORK`。

本批覆盖：

$$
\Gamma\in\{\mathrm{unpolarized},\mathrm{helicity},\mathrm{transversity}\},
\qquad f\in\{u,d,u-d\},\qquad P\in\{3,4\}.
$$

三个 ensemble 共 **54 组非零动量结果**，并保存 **27 组零动量辅助裸量**。每组使用全部 **2000 个 bootstrap 样本**。这里的单独 u、d 均为 **connected contribution**；之前的普通或 special disconnected 分析不在本批九种 PDF 主结果中。等质量轻夸克、同位旋对称的条件下，u-d 的 disconnected 项才可相消。

### 15.2 Ensemble、物理动量、时间窗口与 anchor

物理动量由

$$
P^z=\frac{2\pi P}{N_x a}\hbar c,
\qquad \hbar c=0.1973269804\,\mathrm{GeV\,fm}
$$

换算，不能把三个 ensemble 的同一个整数 P 当成相同物理动量。

| ensemble | 格距 [fm] | 配置数 | 原生非负位移 | 本批时间间隔 | anchor 格点 | P3 [GeV] | P4 [GeV] |
|---|---:|---:|---|---|---:|---:|---:|
| L24x72 | 0.1050 | 199 | 0–15 | 8,9,10 | 7 | 1.476002 | 1.968003 |
| L32x64 | 0.0897 | 425 | 0–20 | 9,10 | 8 | 1.295821 | 1.727762 |
| L32x96 | 0.0775 | 194 | 0–20 | 11,12 | 9 | 1.499809 | 1.999745 |

本批裸量统一采用严格窗口和插入时刻 cut：

$$
0.8\,\mathrm{fm}<T<1.1\,\mathrm{fm},
\qquad 2\le\tau/a\le T/a-2.
$$

三个 anchor 对应的实际长度依次是

$$
z_*=0.7350,\ 0.7176,\ 0.6975\ \mathrm{fm}.
$$

L32x64 和 L32x96 在当前窗口内各只有两个可用 T。本批采用常数平台，没有拟合能隙，不能以平台误差小或优化成功来证明激发态污染已消除。此前的 two-state、C2 联合拟合及能隙 prior 扫描属于独立测试。

### 15.3 输入、方向投影与 bootstrap：先逐方向做 ratio

上游输入配置在 `WORK/Fit_SKILL/C3_Ratio_Bootstrap/configs/ensembles.json`，对应的 connected 数据根位于：

```text
/public/group/lqcd/donghx/Strange_PDF/Conneted_Diagram_final/Result/
  beta6.20_mu-0.2770_ms-0.2400_L24x72/hyp_3D_5times/Nev_150
  beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/hyp_3D_5times/Nev_200
  beta6.41_mu-0.2295_ms-0.2050_L32x96/hyp_3D_5times/Nev_200
```

这里的 `Nev` 是输入产物目录标签，不据此重新推断原始 current/perambulator 使用的完整子空间。实际输入文件路径和哈希也保存在本批裸量 NPZ 中。配置排除列表沿用冻结版本，未恢复原来排除的组态。

主要代码与产品：

| 用途 | 工作区相对路径 |
|---|---|
| 上游投影、ratio 与元数据 | `Fit_SKILL/C3_Ratio_Bootstrap/c3ratio/{project,ratio,conventions,config}.py` |
| 投影后的 C3 当前产品 | `Fit_SKILL/artifacts/c3_ratio_v3/<ensemble>/` |
| ratio 当前版本 | `Fit_SKILL/artifacts/c3_ratio_v4/<ensemble>/` |
| 三个 ensemble 的 C3/C2 读取示例 | `Fit_SKILL/C3_Ratio_Bootstrap/examples/L32x96_unpolarized_ud/read_correlators.py` |
| 本批重新组装配对信号和裸量 | `BASE/all_pdf_pipeline/bare.py` |

旧 `c3_ratio_v2` 的 C2 动量轴问题已有修正，不应用它代替当前 ratio_v4 定义。

每个 ensemble 固定配置列表，使用 seed=1115 的共同 bootstrap 索引。同一 ensemble 中，所有极化、flavor、方向、动量、位移、P0 及后续分析保持逐行配对；不同 ensemble 没有共同配置，联合 ZR 中跨 ensemble 数据协方差块设为零。

对方向 d、配置级平均后的关联函数，估计量的顺序是：

$$
R_d^{(b)}(T,\tau,z)=
\frac{\left\langle\overline C_{3,d}(T,\tau,z)\right\rangle_b}
     {\left\langle\overline C_{2,d}(T)\right\rangle_b},
$$

其中横线表示先按上游定义作源点平均，角括号表示在该 bootstrap 索引上作配置平均。之后对各方向的 ratio 作 VDV even/odd 投影和动量方向平均；不是先把不同方向的 C3、C2 分别平均再相除，也不是把单个配置上的 C3/C2 直接平均。

实际拟合的复数信号定义为

$$
R=\operatorname{Re}R_{\mathrm{even}}
+i\operatorname{Im}R_{\mathrm{odd}}.
$$

非零动量使用六个方向，零动量只使用三个独立正轴。u-d 在配置对齐后构造分子差，再进入同一 ratio 和拟合链。无效插入时刻保留 NaN，使用 `valid_tau_mask` 和实际拟合 mask；填充位置不当成零数据。

### 15.4 第 01 阶段：adjacent + 物理长度 anchor 的 constant 裸量

令位移格点为 n，定义

$$
n_*=\max\{n:na<0.75\,\mathrm{fm}\},
\qquad
Q_n(T,\tau)=
\begin{cases}
R_n(T,\tau)/R_{n-1}(T,\tau),&1\le n\le n_*,\\
R_n(T,\tau)/R_{n_*}(T,\tau),&n>n_*.
\end{cases}
$$

局域点直接拟合原始 ratio；其余点分别拟合上述复数比值的常数：

$$
M_0^{(b)}=\operatorname{FitConst}[R_0^{(b)}],
\qquad q_n^{(b)}=\operatorname{FitConst}[Q_n^{(b)}],
$$

$$
M_n^{(b)}=
\begin{cases}
M_{n-1}^{(b)}q_n^{(b)},&1\le n\le n_*,\\
M_{n_*}^{(b)}q_n^{(b)},&n>n_*.
\end{cases}
$$

这是逐样本的复数乘积重建，避免长距离继续逐点递归。anchor 是该 ensemble 低于物理长度阈值的最大格点；本批不使用旧固定 z=7 或 z=8 处方。

拟合操作约定：

- 当前只用一个包含全部可用 T 的窗口，不作 AIC 竞争，也不联合 C2 谱。
- P0 全部位移、以及非零动量的局域点只拟合实部；其余点联合拟合 Re/Im 两个常数，并保留实虚交叉协方差。
- 对全部入选的时间数据作相关 GLS 常数拟合，使用 Schaefer–Strimmer correlation-to-identity shrinkage 和相对阈值为下式的 SVD 处理：

  $$\epsilon_{\mathrm{SVD,bare}}=10^{-8}.$$

- 质量门为

  $$\chi^2/\mathrm{dof}<1000,\qquad Q>10^{-6}.$$

- 分母 SNR 小于 3，或者分母模长的 1% 分位数与模长中位数比值小于 0.05 时标记风险。质量门与弱分母标记独立，且会沿重建依赖链传播；有限诊断数值仍保留，不删除风险 replica。

输出位置与含义：

```text
BASE/01_bare/<channel>/<flavor>/<ensemble>/
  P0.npz, P3.npz, P4.npz
  signals/P0.npz, signals/P3.npz, signals/P4.npz
  fits/P<P>_zNN.npz
```

`M_boot` 是绝对裸矩阵元，轴为 `(bootstrap,z)`。`H0_boot` 在 P3/P4 文件中是该动量的归一化 shape，不能因为名字中有 0 就将其误读成零动量数据：

$$
S_{\Gamma f}^{(b)}(P,z)=\frac{M_{\Gamma f}^{(b)}(P,z)}{M_{\Gamma f}^{(b)}(P,0)}.
$$

只有 P0 文件里的该数组才是零动量参考。信号文件保存 `R_boot`、`Q_boot`，轴为 `(bootstrap,z,T,tau)`；逐位移拟合文件保存常数、协方差、白化映射、谱、秩、卡方与质量门结果。

u-d、u 和 d 是分别构造和拟合的非线性估计量；逐样本拟合后未必严格满足先做差与后做差的代数等式。这种估计器差异不能解释为 flavor 定义发生改变。

### 15.5 第 02 阶段前半：各极化独立拟合 smooth ZR

三个极化分别用本极化 P0 的 u-d shape 拟合一套 ZR；同一极化的 u、d、u-d 共用该套因子及 P0 u-d 参考。没有把 unpolarized 的 ZR 直接用于 helicity 或 transversity，也没有对三个 flavor 分别挑选最好看的参数。

代码、输入及结果均冻结于：

```text
BASE/02_hybrid/ZR/<channel>/
  prepare.py, interpolation.py, model.py, fit_zr.py, physics.py, config.json
  artifacts/p0/<ensemble>.npz
  artifacts/zr_input.npz
  artifacts/zr/hybrid_nlo.npz
  artifacts/<ensemble>/hybrid_factors.npz
```

入口为 `BASE/all_pdf_pipeline/renormalize.py`。对每个 bootstrap 的实 P0 shape 先作 PCHIP 插值，再取对数；共同物理网格为

$$
z=0.15,0.20,\ldots,0.95\ \mathrm{fm}.
$$

插值点不当成新增独立数据。使用原生插值 Jacobian 的切空间投影及 shrinkage，切空间相对阈值为

$$\epsilon_{\mathrm{tangent}}=10^{-7}.$$

当前有效数据秩为 37，模型参数数为 20，名义数据自由度为 17。这里的 Q 使用数据卡方和名义自由度，与含高斯先验罚项的 augmented Q 分开保存。

为明确单位，定义

$$
\bar a=\frac{a_{\mathrm{fm}}}{\hbar c},\qquad
\bar z=\frac{z_{\mathrm{fm}}}{\hbar c},\qquad
\ell_a=\ln(\bar a\Lambda),\qquad
\beta_0=11-\frac{2n_f}{3},\quad C_F=\frac43.
$$

当前代码拟合的函数为

$$
\begin{aligned}
\ln H_{0,\Gamma}(z,a)={}&
\frac{k_\Gamma\bar z}{\bar a\ell_a}
+\frac{3C_F}{\beta_0}\ln\!\left[\frac{-\ell_a}{\ln(\mu/\Lambda_\Gamma)}\right]
+\ln\!\left(1+\frac{d_\Gamma}{\ell_a}\right)\\
&+\bar a^2 f_\Gamma(z)+G_\Gamma(z),
\end{aligned}
$$

$$
G_\Gamma(z)=
\begin{cases}
\ln C_{0,\Gamma}^{\mathrm{NLO}}(z,\mu)+m_{0,\Gamma}\bar z,&z<z_1,\\
g_\Gamma(z),&z\ge z_1,
\end{cases}
\qquad z_1=0.30\,\mathrm{fm}.
$$

**本批离散项采用平方格距，而非把历史章节的一次格距项原样照搬：** `discretization_power=2`。按上述单位，平滑函数 f 的值具有使离散项无量纲的单位。g 使用 degree 8，f 使用 degree 6，二者以移位 Chebyshev 多项式表示。20 个参数依次为 k、logLambda、d、m0、9 个 h 系数、7 个 f 系数。对数参数的约定为

$$\texttt{logLambda}=\ln[\Lambda/(1\,\mathrm{GeV})].$$

`continuous_C0` 在边界施加

$$g_\Gamma(z_1)-m_{0,\Gamma}\bar z_1=\ln C_{0,\Gamma}(z_1,\mu).$$

h 系数描述满足该约束的长距 g；连续条件不等于同时强制导数连续。拟合使用高斯先验，并额外保存固定先验中心的 conditional 样本。主 `param_boot` 包含辅助先验中心涨落，所以不能把其误差称为纯原始数据统计误差。

本批短距系数固定为 NLO：

$$
C_{0,\Gamma}^{\mathrm{NLO}}(z,\mu)=1+
\frac{\alpha_s C_F}{4\pi}
\left[A_\Gamma\ln\!\left(\frac{\bar z^2\mu^2e^{2\gamma_E}}4\right)+B_\Gamma\right],
$$

$$
(A_\Gamma,B_\Gamma)=
\begin{cases}
(3,5),&\mathrm{unpolarized},\\
(3,7),&\mathrm{helicity},\\
(4,4),&\mathrm{transversity}.
\end{cases}
$$

$$\mu=2\,\mathrm{GeV},\qquad \alpha_s(\mu)=0.296,\qquad n_f=3.$$

原点按归一化约束单独处理，不在原点代入上述含对数的短距表达式。

### 15.6 当前 ZR 参数与拟合质量

下表中心为保存的 `param_center`，误差是相应 2000 个 `param_boot` 的 SD；Lambda 是先对每个样本的 logLambda 取指数，再计算 SD，不用对中心线性近似替代。不同参数的相关性应从完整样本读取，不可将下表误差独立传播。

| 极化 | k | Lambda [GeV] | d | m0 [GeV] |
|---|---:|---:|---:|---:|
| unpolarized | 0.559049 ± 0.017144 | 0.473344 ± 0.043090 | 0.280604 ± 0.014179 | 0.363433 ± 0.043279 |
| helicity | 0.705017 ± 0.019435 | 0.508612 ± 0.023716 | 0.284210 ± 0.009611 | 0.679789 ± 0.037267 |
| transversity | 0.578162 ± 0.014289 | 0.595692 ± 0.014302 | 0.164534 ± 0.007356 | 0.549635 ± 0.026661 |

| 极化 | 数据卡方/名义自由度 | 数据 Q | 通过质量门 |
|---|---:|---:|---|
| unpolarized | 5.917504 | 6.89355e-14 | 否 |
| helicity | 9.210160 | 1.25668e-24 | 否 |
| transversity | 8.772873 | 3.60794e-23 | 否 |

三个优化器均成功，全部参数 bootstrap 有限，ZR 参数样本触边比例为零；**这些数值状态不抵消数据 Q 失败的事实**。参数是当前 smooth 模型、先验与参考态定义下的诊断拟合，不代表已经普适测定了 Wilson-line 的绝对线性发散系数。

机器可读摘要为 `BASE/all_pdf_pipeline/ZR_summary.json`。平滑函数全部系数、先验和原始协方差信息以对应 ZR NPZ 为准。

### 15.7 第 02 阶段后半：hybrid 归一化矩阵元

对应模型的 ZR 为

$$
\begin{aligned}
\ln Z_{R,\Gamma}(z,a)={}&
\frac{k_\Gamma\bar z}{\bar a\ell_a}
+\frac{3C_F}{\beta_0}\ln\!\left[\frac{-\ell_a}{\ln(\mu/\Lambda_\Gamma)}\right]
+\ln\!\left(1+\frac{d_\Gamma}{\ell_a}\right)\\
&+\bar a^2 f_\Gamma(z)+m_{0,\Gamma}\bar z.
\end{aligned}
$$

令零动量参考为

$$
H_{0,\Gamma}(z,a)=\frac{M_{\Gamma,u-d}(0,z,a)}{M_{\Gamma,u-d}(0,0,a)},
\qquad z_s=0.25\,\mathrm{fm}.
$$

hybrid 的逐样本乘法因子与归一化矩阵元为

$$
F_\Gamma(z,a)=
\begin{cases}
1/H_{0,\Gamma}(z,a),&|z|\le z_s,\\
\dfrac{Z_{R,\Gamma}(z_s,a)}{H_{0,\Gamma}(z_s,a)Z_{R,\Gamma}(z,a)},&|z|>z_s,
\end{cases}
$$

$$
H_{\Gamma f}^{\mathrm{hyb}}(P,z,a)=S_{\Gamma f}(P,z,a)F_\Gamma(z,a),
\qquad H_{\Gamma f}^{\mathrm{hyb}}(P,0,a)=1.
$$

拼接点参考值采用 PCHIP 插值，每个 bootstrap 独立传播。同一极化共用参考因子，但每个 flavor 的自身局域荷在归一化中先被除去，待 matching 后恢复。

结果在 `BASE/02_hybrid/<channel>/<flavor>/<ensemble>/P{3,4}.npz`。其中 `stage_boot` 的第一轴依次为绝对 bare、归一化 hybrid，轴为 `(stage,bootstrap,z)`。长距点继承整体 ZR 失败标记，超出共同输入区间的点另标记 `ZR_extension_mask`；有数值不等于通过物理质量门。

当前主线**没有**加入 RGR、LRR 或 Wilson-loop 线质量约束。这些操作若改变短距系数或有限质量条件，需要重新拟合配套参数，不能只给主输出再乘一个事后因子。

### 15.8 第 03 阶段：关联复数 lambda 外推与 Fourier

代码入口为 `BASE/all_pdf_pipeline/tail.py`、`tail_core.py`、`config.json`。定义

$$\lambda=\bar zP^z.$$

对同一 bootstrap 的实部、虚部及不同位移建立联合协方差，采用 shrinkage 和相对 SVD cut：

$$\epsilon_{\mathrm{SVD,tail}}=10^{-4}.$$

各 replica 使用由整批样本确定的固定白化映射。不同动量、flavor 保留配对行用于后续比较，但尾部拟合并未强制跨动量或跨 flavor 共用一套参数。

尾部模型为

$$
H_{\mathrm{tail}}(\lambda)=
\left[
\frac{A_1e^{-i\pi a_t/2}}{\lambda^{a_t}}
+\frac{A_2e^{-i\lambda+i\pi b_t/2}}{\lambda^{b_t}}
\right]e^{-\gamma(\lambda-7)}.
$$

这里尾部幂指数用下标区别于格距；保存的参数顺序为 `A1,A2,a,b,gamma`。指数中的 7 是该振幅参数化的参考点，不是所有样本固定的拼接点。

预先注册的窗口为 P3 从 lambda=5 起、P4 从 lambda=6 起，拟合上限取请求的 20 与实测最大 lambda 中的较小值；实际纳入点由 `fit_mask_native` 保存。PCHIP 用于低于第一实际拟合点的实测区域，再用宽度 1.5 的 smoothstep 过渡到尾部。第一实际拟合点才是拼接起点，不向更小 lambda 强行反向使用尾部模型。

$$
\lambda_{\max}=100,\qquad \Delta\lambda=0.05.
$$

Fourier 使用 Hermitian 延拓和本地固定的正指数约定：

$$
H(-\lambda)=H(\lambda)^*,
$$

$$
\widetilde q_{\mathrm{shape}}(x,P)=\frac1{2\pi}
\int_{-\lambda_{\max}}^{\lambda_{\max}}d\lambda\,
 e^{+ix\lambda}H(\lambda)
=\frac1\pi\int_0^{\lambda_{\max}}d\lambda\,
\left[\operatorname{Re}H\cos(x\lambda)-\operatorname{Im}H\sin(x\lambda)\right].
$$

符号与原始矩阵元、核和正负 x 对应须一致；换 Fourier 指数相当于反射 x，不能单独在绘图时翻号。当前保存网格为

$$x=-2,-1.99,\ldots,2.$$

使用 Simpson 积分，保存 cutoff=25、50、100 三套输出，主输出为第三套。它们来自同一组矩阵元样本，不应作为独立统计误差相加。

数据在 `BASE/03_fourier/<channel>/<flavor>/<ensemble>/P{3,4}.npz`：`H_extended_boot` 轴为 `(2000,2001)`，`quasi_boot` 轴为 `(3,2000,401)`。`param_boot`、`replica_bound_hit`、`diagnostics_json`、`fit_mask_native` 记录尾部参数与风险。延伸到 lambda=100 只改变积分截断测试，不能增加实测信息或保证尾部系统误差已消除。

### 15.9 第 04、05 阶段：全阶段展示、NLO matching 与局域荷恢复

第 04 目录是每个 ensemble 的全阶段图汇总，不是又做了一次独立数值变换：

```text
BASE/04_by_ensemble/<channel>/<flavor>/<ensemble>_all_stages.png
BASE/04_by_ensemble/<channel>/<flavor>/all_five_stages.pdf
```

matching 代码为 `BASE/all_pdf_pipeline/{prepare_kernels,matching_physics,match,restore_charges}.py`。三个极化分别使用本通道 NLO hybrid-ratio 核，包括对应的 real、virtual 与 plus-prescription 项；没有把 unpolarized 核通用于三个极化。核的来源与公式对应保存在本批 README 和 `matching_physics.py` 中。

一阶逆 matching 的记号为

$$
q_{\mathrm{shape}}(x,\mu)=\widetilde q_{\mathrm{shape}}(x,P)
-\frac{\alpha_s C_F}{2\pi}\mathcal K_\Gamma
\widetilde q_{\mathrm{shape}}(x,P)+O(\alpha_s^2).
$$

为了覆盖卷积输入，代码对同一份外推后的 H 重新 Fourier 到

$$
x_{\mathrm{quasi}}=-6,-5.99,\ldots,6,
$$

而不是在原输出范围外填充人为 quasi 尾部。匹配输出网格为

$$x=-1,-0.99,\ldots,1.$$

201 个输出格点中 198 个由 `computed_mask` 标为实际计算点；未算的小 x 点保留 NaN。保留积分阶数 16/24、输入支撑 2/4/6/8 以及 lambda cutoff 对照，不对主结果重新强制单位面积归一化。

最后逐 bootstrap 将配套 P0 裸荷与局域重整化常数还回去：

$$
g_{\Gamma f}^{R,(b)}=Z_\Gamma^{(b)}M_{\Gamma f}^{(b)}(0,0),
\qquad
q_{\Gamma f}^{R,(b)}(x,\mu)
=g_{\Gamma f}^{R,(b)}q_{\Gamma f,\mathrm{shape}}^{(b)}(x,\mu),
$$

$$
Z_\Gamma=
\begin{cases}
Z_V,&\mathrm{unpolarized},\\
Z_A,&\mathrm{helicity},\\
Z_T,&\mathrm{transversity}.
\end{cases}
$$

局域 Z 的已登记外部误差通过独立 Gaussian nuisance 传播，同一 ensemble/channel 的 flavor 与动量共用其抽样；固定 Z 的对照另存为 `restored_boot_stat`。来源是本地保存的 arXiv:2310.00814v2 系数记录；L32x64 的旧系数来源仍待确认，不能写成三个 ensemble 均已逐项从该论文直接核实。

结果位于 `BASE/05_matching/<channel>/<flavor>/<ensemble>/P{3,4}.npz`。`normalized_boot` 是不含局域荷的 shape，`restored_boot` 是恢复荷后的分布，轴均为 `(bootstrap,x)`。通道局域荷保存在 `BASE/05_matching/local_charges_<channel>.npz`。当前是 connected diagonal quark matching，未加入单态夸克—胶子混合，不将单独 u、d 宣称为完整物理 PDF。

### 15.10 已生成的图、外部参考与读取入口

[九种 PDF 全阶段总索引](/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/README_9_PDFs.md) 包含每个 channel/flavor 的 bare、hybrid、lambda、quasi 和 matching 图链接。

| 内容 | BASE 下的路径 |
|---|---|
| 全部九种 PDF、108 页 | `All_9_PDFs_AllStages.pdf` |
| 裸量实虚部与完整误差范围 | `01_bare/<channel>/<flavor>/three_ensembles{,_fullrange}.png` |
| hybrid 实虚部与完整误差范围 | `02_hybrid/<channel>/<flavor>/three_ensembles{,_fullrange}.png` |
| lambda 外推、模长与相对误差 | `03_fourier/<channel>/<flavor>/lambda_extension.png`、`lambda_magnitude.png` |
| quasi 动量空间结果 | `03_fourier/<channel>/<flavor>/three_ensembles.png` |
| matching shape、恢复荷结果 | `05_matching/<channel>/<flavor>/matched_normalized.png`、`matched_charge_restored.png` |
| 外部 PDF 对照，9 页 | `05_matching/plots_reference_comparison/all_9_reference_comparisons.pdf` |
| 全部数组格式、读取与复现说明 | `all_pdf_pipeline/README.md`、`read_example.py` |

外部对照沿用本地已保存参考，并用公开 JAM 网格补充 transversity：

| 极化 | 外部分布 | 比较尺度 | 说明 |
|---|---|---|---|
| unpolarized | CT18NLO，LHAPDF | 2 GeV | NLO |
| helicity | NNPDFpol20_lo_as_01180，LHAPDF | 2 GeV | LO 参考，与本次 NLO matching 阶数不同 |
| transversity | JAMDiFF23-transversity_lo 及其 nolat 集合 | 2 GeV | 两套均按 JAMDiFF-23 LO 使用，同时显示含/不含 lattice tensor-charge 约束 |

外部对照脚本为 `BASE/05_matching/compare_references.py`；每个 channel/flavor 提供三列 ensemble、两行分别为恢复荷后的 q 与乘 x 后分布的图，显示区间为

$$0.02\le x\le0.80.$$

它们不是新增的 normalized shape 对照；此前单独 unpolarized u-d 的 normalized 图在旧 `05_matching/plots/` 中保留。JAM 的 u-d 在同一 replica 内先做差，再计算 LHAPDF 不确定度，保留 flavor 相关性。JAM 来源哈希、成员取法和求值尺度见 `05_matching/references/JAMDiFF23_provenance.json`；求值后的网格已写入同目录两个 `JAMDiFF23-transversity*_Q2.npz`。

原九种全阶段绘图与本轮外部参考图有各自的完成清单，不用旧 108 页清单去宣称已经覆盖后加的所有参考图。

### 15.11 计算完成与质量接受：当前应如何解读结果

`BASE/all_pdf_pipeline/complete.json` 保存数值验证为 PASS，包括共同 bootstrap 行对齐、有限覆盖、重建代数、hybrid 乘法、Fourier 符号/系数、局域荷恢复和独立 matching 检验。该 PASS 是计算与数据契约检查通过，不等于物理质量通过。

当前汇总如下；bare/hybrid 的计数按非零动量每组位移计数，每个极化共 348 个位置。风险计数和质量通过数不是互补集合。

| 极化 | bare 质量通过 | bare 弱分母/继承风险标记 | hybrid 质量通过 | 尾部统计门通过 [组] |
|---|---:|---:|---:|---:|
| unpolarized | 348/348 | 72/348 | 60/348 | 12/18 |
| helicity | 306/348 | 110/348 | 56/348 | 13/18 |
| transversity | 64/348 | 104/348 | 22/348 | 8/18 |

所有 54 组尾部均保存 2000 个成功且有限的 replica，但部分组有大量参数触边，最大触边比例按 unpolarized、helicity、transversity 分别为

$$0.999,\quad 0.990,\quad 0.9995.$$

将 Fourier 截断从 50 改到 100 时，各极化跨全体 flavor/ensemble/P/x 的最大 bootstrap 均值变化约为

$$
0.00701,\quad0.03538,\quad0.03684.
$$

这些数值是 cutoff 敏感性诊断，非完整系统误差；单组、单 x 的不确定度应读取配对差值，不能把全局最大值统一添加到每个点。

独立 adaptive 全实轴卷积检查的最大绝对差约为

$$5.94\times10^{-9}.$$

原 unpolarized u-d 裸量与冻结基线的最大差约为

$$3.77\times10^{-15}.$$

因此当前主要问题不是缺少可读文件或数值积分未执行，而是参考态/平台选择、ZR 模型失配、弱分母、尾部模型与触边及系统外推尚需控制。transversity 还保留 `pending_external_physics_confirmation` 的相对算符符号状态。

### 15.12 RGR/LRR、Wilson-loop 测试与当前主结果的关系

`BASE/06_renorm_RGR_LRR/` 已保存 **unpolarized connected u-d** 的 NLO、RGR、RGR+LRR 坐标空间对照，三个 ensemble 为 P3/P4，共用相同裸量和 bootstrap。该测试通过替换短距系数并重新拟合全部配套 ZR 参数实现，不是简单对现成矩阵元加一个曲线修正。三套整体 ZR 同样未通过数据 Q 门。

**当前九种 PDF 的 `01–05` 主线仍是 NLO 基线。** 第 16.6–16.9 节另外记录了 `06_renorm_RGR_LRR` 中仅针对 unpolarized connected $u-d$ 的同窗 RGR+LRR 全链诊断。

Wilson-loop 绝对/相对线质量候选、约束及有限质量重新确定见第 7、11、13、14 节，以及 `/public/home/donghx/.codex/Markdown/linear_devergence.md`。当前 01–05 的 ZR 未输入这套 Wilson-loop 约束。历史 k、质量候选及当前 ZR k 不能不经有限方案对齐直接数值替换，也不应把两个方案下相近的重整化矩阵元解释为已经单独确定了 UV 与有限质量的分解。

### 15.13 常用读取与复现操作

只读取当前结果可执行：

```bash
cd /public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/all_pdf_pipeline
/public/home/donghx/AnaConda/bin/python read_example.py --channel helicity --flavor d --ensemble L32x96 --momentum 4
```

完整重算的原代码顺序为：

```bash
PYTHONDONTWRITEBYTECODE=1 /public/home/donghx/AnaConda/bin/python bare.py --workers 3
PYTHONDONTWRITEBYTECODE=1 /public/home/donghx/AnaConda/bin/python renormalize.py
PYTHONDONTWRITEBYTECODE=1 /public/home/donghx/AnaConda/bin/python tail.py --workers 4
PYTHONDONTWRITEBYTECODE=1 /public/home/donghx/AnaConda/bin/python prepare_kernels.py
PYTHONDONTWRITEBYTECODE=1 /public/home/donghx/AnaConda/bin/python match.py --workers 3
PYTHONDONTWRITEBYTECODE=1 /public/home/donghx/AnaConda/bin/python restore_charges.py
PYTHONDONTWRITEBYTECODE=1 /public/home/donghx/AnaConda/bin/python plots.py
PYTHONDONTWRITEBYTECODE=1 /public/home/donghx/AnaConda/bin/python verify.py
```

命令用于说明生产依赖顺序；本次文档更新没有执行这些重算。外部参考绘图另由 `05_matching/compare_references.py` 完成，重新求值 JAM 需要 LHAPDF 可发现相应网格；已有 reference NPZ 不依赖临时下载目录即可读取。

数组读取示例：

```python
from pathlib import Path
import numpy as np
base = Path('/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/'
            'Constant_Anchor075_P34_AllStages_20261122')
ch, fl, en, p = 'helicity', 'd', 'L32x96', 4
with np.load(base / f'01_bare/{ch}/{fl}/{en}/P{p}.npz') as d:
    bare_boot = d['M_boot']                  # (bootstrap,z), complex
    bare_quality = d['quality_accepted']
with np.load(base / f'02_hybrid/{ch}/{fl}/{en}/P{p}.npz') as d:
    hybrid_boot = d['stage_boot'][1]         # (bootstrap,z), normalized
with np.load(base / f'03_fourier/{ch}/{fl}/{en}/P{p}.npz') as d:
    quasi_boot = d['quasi_boot'][2]          # cutoff=100, (bootstrap,x)
with np.load(base / f'05_matching/{ch}/{fl}/{en}/P{p}.npz') as d:
    x = d['x']
    mask = d['computed_mask']
    matched_boot = d['restored_boot'][:,mask]
    matched_mean = matched_boot.mean(axis=0)
    matched_sd = matched_boot.std(axis=0,ddof=1)
```

所有主图报告逐样本完成非线性操作后的 bootstrap 均值及 SD，不除以 bootstrap 样本数平方根。方案差异使用共同样本逐行相减后取 SD；不要把不同窗口、参考态、重整化处方或尾部选择造成的中心移动隐含混入统计误差。

### 15.14 本次文档更新的依据与核查边界

本节主要核对了以下本地事实来源：

1. `Fit_SKILL/C3_Ratio_Bootstrap/configs/ensembles.json`、ratio 与 conventions 代码。
2. `BASE/all_pdf_pipeline/` 的 `bare.py`、`renormalize.py`、`tail.py`、`tail_core.py`、`match.py`、`restore_charges.py`、`plots.py`、`verify.py` 和配置。
3. 三个极化的 `02_hybrid/ZR/<channel>/model.py`、配置及 `artifacts/zr/hybrid_nlo.npz`，直接读取参数中心和样本 SD。
4. `all_pdf_pipeline/{complete,ZR_summary,summary,tail_summary,independent_matching_validation}.json`。
5. `05_matching/reference_comparison_complete.json`、`references/JAMDiFF23_provenance.json`、图文件以及九种 PDF 总索引。
6. `06_renorm_RGR_LRR/README.md`，核对已完成的测试范围。

复核文件覆盖为：01 裸量共 81 份矩阵元 NPZ；02 hybrid、03 lambda/Fourier、05 matching 各 54 份主 NPZ。04 为图像汇总。上游完成清单已经记录当时的哈希验证；本次为代码、清单及选定参数数组的交叉核对和文档更新，没有重新读取所有原始 C3、重新执行收缩或重算全链。

## 16. RGR/LRR：系数核对、短距窗口扫描与 Wilson loop 约束（2026-11-24）

### 16.1 适用范围与短距系数

`BASE` 沿用第 15 节定义。`BASE/06_renorm_RGR_LRR` 是 HYP3D5 **unpolarized connected $u-d$** 的坐标空间诊断：C24P29/L24x72、E32P29/L32x64、F32P30/L32x96 的归一化 $P=0$ 裸矩阵元用于联合拟合 $Z_R$，再应用于各 ensemble 的 $P=3,4$。三种方案共用原常数平台裸量和同一套 2000 个配对 bootstrap。每种方案更换短距系数后，均重新确定有限质量 $m_0$ 和其余 $Z_R$ 参数；没有对既有 NLO 矩阵元额外乘一次演化因子。九种 PDF 的 `01–05` 主线仍采用固定阶 NLO。第 16.1–16.5 节记录坐标空间诊断；同一短窗的 NLO 与 NLO+RGR+LRR 两套输入已各自继续做 $\lambda$ 外推、Fourier 和 lightcone matching，详见第 16.6–16.9 节。

令 $\mu=2\,\mathrm{GeV}$、$\mu_h=2e^{-\gamma_E}/|z|$、$\alpha_h=\alpha_s(\mu_h)$。当前 RGR 在自然尺度计算系数，再演化到 $\mu$；LRR 在同一自然尺度加入领先 renormalon 的主值（PV）Borel 和：

$$
\begin{aligned}
U_\Gamma(\mu,\mu_h)&=\exp\!\left[\int_{\alpha_h}^{\alpha_s(\mu)}
\frac{\gamma_\Gamma(\alpha)}{\beta(\alpha)}\,d\alpha\right],\\
C_{0,\Gamma}^{\mathrm{NLO+RGR}}(z,\mu)&=U_\Gamma(\mu,\mu_h)C_{0,\Gamma}^{\mathrm{NLO}}(z,\mu_h),\\
C_{0,\Gamma}^{\mathrm{NLO+RGR+LRR}}(z,\mu)&=U_\Gamma(\mu,\mu_h)\left[C_{0,\Gamma}^{\mathrm{NLO}}(z,\mu_h)-\alpha_h r_0(z,\mu_h)+C_{\mathrm{PV}}(z,\mu_h)\right].
\end{aligned}
$$

减去 $\alpha_h r_0$ 是为了避免重复计入 NLO 中已有的领先 renormalon 展开项。对同一 Wilson line，若耦合、活跃味数、尺度与 PV 处方均相同，可共用该领先线质量项及其 NLO 扣除；各极化仍须使用各自的固定阶系数和相应的演化。当前代码的 $C_{0,\Gamma}^{\mathrm{NLO}}=1+\alpha_s C_F(\ell_\Gamma L+b_\Gamma)/(4\pi)$，其中 $L=\ln(z^2\mu^2e^{2\gamma_E}/4)$，vector、axial、tensor 的 $(\ell_\Gamma,b_\Gamma)$ 分别为 $(3,5)$、$(3,7)$、$(4,4)$。

本地 notebook 核对表明，活跃入口使用的是 NLO；其 $\alpha_h r_0$ 扣除、PV 长度因子和 RGR 演化方向与本实现的 NLO 分支相符。在相同耦合端点下，unpolarized 对应的演化因子在四个检查点的最大相对差为 $5.7\times10^{-13}$。当前 running 固定 $n_f=3$，notebook 的耦合采用分段阈值，不能将二者当成同一数值方案。Notebook 的 $+7$ 有限项也不能用于当前 vector 通道。若升级 NNLO，应对应增加 $\alpha_h^2r_1$ 扣除，并统一坐标空间与 matching 的有限质量和 PV 处方。

### 16.2 关联拟合、离散项与 hybrid 拼接

逐样本对 $H_0(z,a)=h_B(0,z,a)/\operatorname{Re}h_B(0,0,a)$ 做 PCHIP，再取对数；共同拟合网格覆盖 $0.15$–$0.95\,\mathrm{fm}$，步长 $0.05\,\mathrm{fm}$。这是**关联拟合**：ensemble 之间协方差设为零；在各 ensemble 内按 PCHIP Jacobian 的原生切空间投影（相对奇异值阈值 $10^{-7}$）去掉插值产生的冗余方向，再对白化所需的投影协方差做 shrinkage。原数据块有效秩为 37，拟合 20 个参数，数据自由度为 17。模型 Jacobian 满列秩与数据拟合通过质量门是两项不同的检查。

$$
\begin{aligned}
\ln Z_R(z,a)&=\ln Z_{\mathrm{UV}}(z,a)+a^2f(|z|)+m_0|z|,\\
\ln H_0(z,a)-\ln Z_R(z,a)&=
\begin{cases}
\ln C_0(z,\mu),& |z|<z_1,\\
g(|z|)-m_0|z|,& |z|\ge z_1,
\end{cases}
\qquad g(z_1)-m_0z_1=\ln C_0(z_1,\mu).
\end{aligned}
$$

$\ln Z_{\mathrm{UV}}$ 含线性斜率 $A(a)=k/[a\ln(a\Lambda)]$ 和只依赖 $a$ 的项，计算时 $a,|z|$ 用 $\mathrm{GeV}^{-1}$。$a^2 f$ 已纳入全部拟合；$f$ 用 6 阶 Chebyshev 基、长距 $g$ 用 8 阶平滑基。同一函数可在每个 $z$ 求值，但没有给每个网格点独立拟合 $f(z)$ 或 $g(z)$。$z_1$ 是短距 OPE 条件的上限，独立于固定的 hybrid 拼接点 $z_s=0.25\,\mathrm{fm}$：

$$
S(P,z)=\frac{h_B(P,z)}{\operatorname{Re}h_B(P,0)},\qquad
H_{\mathrm{hyb}}(P,z)=S(P,z)
\begin{cases}
1/H_0(z,a),& |z|\le z_s,\\
Z_R(z_s,a)/[H_0(z_s,a)Z_R(z,a)],& |z|>z_s.
\end{cases}
$$

短距分支使用相同的 $H_0$，所以不同方案在该分支逐样本一致。图中的 $H_0/Z_R$ 与 $C_0$ 比较用于检查短距匹配模型；$C_0$ 本身已进入该拟合，不能用图上贴近替代独立验证。$|z|>0.95\,\mathrm{fm}$ 属于 $Z_R$ 模型延伸，$f$ 的基函数坐标取拟合右端点，线性指数继续延伸。

### 16.3 原窗口结果、抵消机制与窗口敏感性

原 $z_1=0.300\,\mathrm{fm}$ 且无 Wilson loop 约束时，NLO、NLO+RGR、NLO+RGR+LRR 的全局数据 $\chi^2/\mathrm{dof}$ 分别为 **5.92、34.65、18.50**，三者均未通过 $Q>10^{-6}$。以 F32P30 的 $|z|=0.930\,\mathrm{fm}$ 为例，RGR 和 RGR+LRR 的长距因子相对 NLO 分别为 **3.824** 和 **1.071**。对数因子之比可分解为

$$
\ln\frac{F_i(z,a)}{F_{\mathrm{NLO}}(z,a)}=-\big[\Delta A(a)+\Delta m_0\big](|z|-z_s)-a^2\big[\Delta f(|z|)-\Delta f(z_s)\big],
$$

其中 $F=H_{\mathrm{hyb}}/S$，$\Delta$ 表示与 NLO 的拟合参数或函数值之差。该点 RGR+LRR 的有限质量、UV 斜率、离散三项依次是 $+1.432,-1.393,+0.030$。前两项近乎抵消，解释了该窗口下的接近；这不能证明 LRR 对最终形状普遍很小。同一点将 $\mu_h$ 乘以 $0.8,1,1.2$ 的中心值压力测试，RGR 因子比为 $7.08,3.82,2.86$，RGR+LRR 为 $0.826,1.071,1.150$。此扫描不是系统误差带。

无 Wilson loop 约束时，短距上限扫描结果如下。括号是末位 bootstrap 标准差，单位均为 $\mathrm{GeV}$；最后一列按前三个方案的顺序列出数据 $\chi^2/\mathrm{dof}$。

| $z_1$ [fm] | NLO $m_0$ | NLO+RGR $m_0$ | NLO+RGR+LRR $m_0$ | $\chi^2/\mathrm{dof}$ |
|---:|---:|---:|---:|---:|
| 0.200 | $0.324(66)$ | $0.057(53)$ | $0.242(66)$ | 7.09 / 7.26 / 7.08 |
| 0.225 | $0.322(68)$ | $0.045(62)$ | $0.253(60)$ | 7.02 / 6.98 / 7.49 |
| 0.250 | $0.309(75)$ | $0.039(37)$ | $0.131(92)$ | 6.89 / 15.46 / 11.31 |
| 0.275 | $0.305(77)$ | $0.041(37)$ | $0.112(90)$ | 6.74 / 15.24 / 11.37 |
| 0.300 | $0.363(43)$ | $-0.018(30)$ | $-0.052(35)$ | 5.92 / 34.65 / 18.50 |

15 组均未通过 $Q>10^{-6}$。RGR+LRR 改用 $z_1=0.225\,\mathrm{fm}$ 后，$m_0$ 相对原窗的配对差为 $+0.305\pm0.043\,\mathrm{GeV}$，F32P30 同一长距点相对固定 NLO 基线的因子比从 $1.071$ 升到 $1.299$（相对同窗口 NLO 为 $1.325$）。在 $0.15$ 至 $0.225\,\mathrm{fm}$ 内每个 ensemble 只有一个原生点，插值不会增加独立观测，因此看到短窗口的两个中心值接近，仍不能确认 $m_0$ 已有稳定平台。

### 16.4 Wilson loop 相对线质量约束

采用组态级基本表示纯空间 Wilson loop，与各 ensemble 的 $P=0$ 输入对齐，再用同批 2000 行索引重采样。以 E32P29 作参考，模型约束与数据定义为

$$
\Delta_e^{\mathrm{WL}}=\tfrac12\big[E_e(R)-E_{\mathrm{E32P29}}(R)\big]_{\mathrm{const}},\qquad
\Delta_e^{\mathrm{model}}=\delta_e^{\mathrm{UV}}-\delta_{\mathrm{E32P29}}^{\mathrm{UV}},\qquad
\delta_e^{\mathrm{UV}}=-\frac{k}{a_e\ln(a_e\Lambda)}.
$$

因子 $1/2$ 将矩形的两条长边换成单条线质量。C24P29 和 F32P30 的输入分别为 $-70.42\pm0.56$ 和 $+63.27\pm1.07\,\mathrm{MeV}$，两者相关系数为 $0.3582$。约束不使用绝对质量或先前开放 Wilson line 在 $0.30\,\mathrm{fm}$ 的提取值；共同的 $m_0$ 在质量差中消去。联合拟合保留 Wilson loop 与 PDF 输入的交叉协方差；条件 Wilson loop 协方差正定，未额外截模。联合观测数为 $37+2$，自由度为 19。

| NLO+RGR+LRR | $m_0(0.225)$ [GeV] | $m_0(0.300)$ [GeV] | 配对差：短窗减原窗 [GeV] | $\chi^2/\mathrm{dof}$：短窗 / 原窗 |
|---|---:|---:|---:|---:|
| 无约束 | $0.253\pm0.060$ | $-0.052\pm0.035$ | $+0.305\pm0.043$ | 7.49 / 18.50 |
| 有约束 | $0.322\pm0.009$ | $0.328\pm0.010$ | $-0.006\pm0.004$ | 20.11 / 28.62 |

约束缩小了 $m_0$ 的窗口漂移，但联合拟合的 $Q$ 为 $2.38\times10^{-69}$ 和 $3.64\times10^{-103}$；其余联合方案也未过质量门。两个预测质量差分别偏离输入约 $9$ 和 $5.5$ 个边缘统计标准差。只拟合 Wilson loop 的两个差值，保留现有 UV 形式及 $\Lambda\in[0.001,1]\,\mathrm{GeV}$，最小 $\chi^2_{\mathrm{WL}}$ 也约为 $60.29$。在 F32P30 的 $|z|=0.930\,\mathrm{fm}$，有约束的短窗/原窗长距因子比仍为 $1.189$。因此 $m_0$ 的漂移缩小不足以确认共用 UV 形式或确定稳健的 $Z_R$；下一步应检验相对有效线质量的有限格距与 Wilson loop 窗口系统误差，以及现有斜率模型的格距依赖。

### 16.5 最新 $z_1=0.225\,\mathrm{fm}$ 矩阵元、图和数据入口

三个 ensemble 的 $P=3,4$ 均保存了同一短窗下有、无 Wilson loop 约束的归一化矩阵元。在 $|z|\le0.95\,\mathrm{fm}$ 范围内，有约束/无约束长距因子比偏离 1 的最大值，对 C24P29、E32P29、F32P30 依次为 **0.49%、1.43%、0.45%**。同一 ensemble 的该因子同时作用于两个动量的实部和虚部，因子差只衡量同一窗口下的约束效应。最新 NPZ 的 `stage_boot[scheme,bootstrap,native_z]` 中，`scheme=0` 无约束、`scheme=1` 有约束；局域荷尚未乘回，$H_{\mathrm{hyb}}(P,0)=1$。矩阵元回代与已有诊断文件的最大绝对差为 $6.3\times10^{-16}$。图中有限动量矩阵元的竖虚线是 $z_s$；$H_0/Z_R$ 图的竖虚线是 $z_1$。

- [Notebook 公式核对](/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/RGR_LRR_Notebook_Audit_20261123/README.md)；[基线三方案与协方差模型](/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/06_renorm_RGR_LRR/README.md)；[因子分解图](/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/06_renorm_RGR_LRR/scheme_cancellation/hybrid_factor_cancellation.pdf)。
- [短距窗口扫描与复现](/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/06_renorm_RGR_LRR/m0_window_scan/README.md)；[Wilson loop 联合拟合和残差](/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/06_renorm_RGR_LRR/m0_window_scan_WL/README.md)。
- [两个短距窗口下的 $H_0/Z_R$ 与短距 $C_0$ 比较](/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/06_renorm_RGR_LRR/m0_window_scan_WL/plots_Renorm_matrix/Renorm_matrix_3ens_unpolarized_all_schemes.pdf)；[最新 $P=3,4$ 矩阵元图册](/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/06_renorm_RGR_LRR/RGR_LRR_z1_225_hybrid_P34/RGR_LRR_z1_225_3ens_P34.pdf)；[数据格式与作图脚本](/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/06_renorm_RGR_LRR/RGR_LRR_z1_225_hybrid_P34/README.md)。

### 16.6 两次演化：坐标空间参考尺度与 PDF 因子化尺度

RGR（重整化群重求和）处理固定阶展开中的尺度对数；LRR（领先 renormalon 重求和）规定与 Wilson line 线质量相关的领先 Borel 奇点和有限质量处方。**DGLAP 是动量空间 RGR 的一部分**：一旦选择随 $x$ 变化的 matching 尺度，就需要把结果转换到共同的因子化尺度。加入 LRR 本身不引入另一次 DGLAP 演化。

| 环节 | 演化的对象和方程 | 本次的尺度 |
|---|---|---|
| 坐标空间短距约束 | $C_0$ 由非局域算符的反常维数演化，具体的 $U_\Gamma C_0$ 和 NLO 扣除见第 16.1 节。 | $\mu_{h,z}=2e^{-\gamma_E}/\lvert z\rvert\;\longrightarrow\;\mu_{\mathrm{ref}}=2\,\mathrm{GeV}$。 |
| 动量空间 inverse matching | 在各 $x$ 的自然尺度提取 lightcone PDF，再用非单态 DGLAP 关系求共同尺度的 PDF。 | $\mu_{h,x}=\max(1\,\mathrm{GeV},\,2\lvert x\rvert P_z)\;\longrightarrow\;\mu_F=2\,\mathrm{GeV}$。 |

两个 $2\,\mathrm{GeV}$ 目前取相同数值：$\mu_{\mathrm{ref}}$ 是拟合 $Z_R$ 的短距系数参考尺度，$\mu_F$ 是输出 PDF 的 $\overline{\mathrm{MS}}$ 因子化尺度。坐标空间先在 $\mu_{h,z}$ 计算带 PV（Cauchy 主值）与固定阶扣除的系数，再演化 $C_0$；把它代入 $P_z=0$ 的关联拟合，重拟 $Z_R$ 和 $m_0$，最后按第 16.2 节做 hybrid 拼接。演化通过短距拟合进入 $Z_R$ 模型。长度 $z_1=0.225\,\mathrm{fm}$ 是短距拟合上限，$z_s=0.25\,\mathrm{fm}$ 是 hybrid 拼接点；$P_z$、$\lambda=zP_z$、外推截断和局域荷均不指定最终 PDF 的因子化尺度。

### 16.7 动量空间的 LRR 核和非单态 DGLAP 实现

当前按固定 $n_f=3$、$\alpha_s(2\,\mathrm{GeV})=0.296$ 运行两圈耦合。RGR+LRR 使用在 $\mu_{h,x}$ 求值的一阶逆 hybrid 核。在一阶逆匹配中减去领先线质量修正，修正系数 $\Delta m_{\mathrm{LRR}}(\mu_{h,x})$ 取 PV 和减去其 NLO 展开后的剩余项，并与坐标空间共用 PV 和 NLO 扣除约定。代码用带权坐标矩阵元的 Fourier 变换计算该卷积；全实轴保荷的虚项对应权重中的常数减项：

$$
w_\epsilon(z;z_s)=\theta(|z|-z_s)
\left[|z|e^{-\epsilon|z|}-z_s e^{-\epsilon z_s}\right],
\qquad \epsilon_{\mathrm{central}}=0.06\,\mathrm{GeV}.
$$

式中 $z,z_s$ 用 $\mathrm{GeV}^{-1}$。另存 $\epsilon=0.02,0.10\,\mathrm{GeV}$ 的对照。这里的全实轴虚项与本地 hybrid 核配套；不能把它标成原论文 pion 的有限支撑虚项或 notebook 的完整 NNLO 结果。

对 $0<x<1$，定义 $f_\pm(x,\mu)=q_{u-d}(x,\mu)\pm \bar q_{u-d}(x,\mu)$。使用非单态 NLO splitting 核，求解

$$
f_\pm(x,\mu_{h,x})
=\int_x^1\frac{dy}{y}\,U_\pm^{\mathrm{NS}}\!\left(\frac{x}{y};\mu_{h,x},\mu_F\right) f_\pm(y,\mu_F).
$$

左侧来自自然尺度 inverse matching（含 LRR 修正）。右侧在同一个 $\mu_F$ 定义未知 PDF，用每一行对应的 $\mu_{h,x}$ 构造从共同尺度到自然尺度的演化算子，再利用 $y\ge x$ 的支撑做上三角反解；不能将每个 $x$ 当成独立点乘一个演化因子。得到共同尺度的 $q_{u-d}$ 和 $\bar q_{u-d}$ 后，按 $q_{u-d}(-x,\mu_F)=-\bar q_{u-d}(x,\mu_F)$ 重组负 $x$。固定阶 NLO 对照直接在 $\mu_F=2\,\mathrm{GeV}$ 做 inverse matching，无需另做 DGLAP；RGR+LRR 产物也已经完成这一步，和相同尺度的 LHAPDF 比较前不再演化。

### 16.8 实际操作顺序与输出检查

1. **同窗重整化。**用三个 ensemble 各自的 $P_z=0$ 裸量，两种方案共用同一批 2000 个配对 bootstrap，在 $z_1=0.225\,\mathrm{fm}$ 分别拟合 NLO 和 NLO+RGR+LRR 的 $Z_R$，应用到各 ensemble 的 $P=3,4$。`WL` 对两种方案使用同一 Wilson loop 相对线质量约束；`free` 对两种方案均不加该约束。供下游读取的同窗输入位于 `BASE/06_renorm_RGR_LRR/Lambda_NLO_vs_RGR_LRR_Matched_z1_225_20261124/data/`。
2. **外推和 Fourier。**在每个 bootstrap 副本上，对实、虚部做关联复数尾部拟合，按 $\lambda=zP_z$ 做 Fourier；主结果积分到 $\lambda_{\max}=100$，同时保存 $25$ 和 $50$ 的截断对照。脚本 `fourier.py`，产物目录 `01_tail_fourier/`。
3. **Matching 和局域荷。**用 `matching.py` 和 `rgr.py` 执行第 16.7 节的对应方案；逐样本把同一份局域矢量荷 $Z_V h_B(P_z=0,z=0)$ 乘回 normalized 结果。Matching 输出 `02_matching/<WL或free>/<NLO或RGR_LRR>/<ensemble>/P{3,4}.npz` 的 `normalized_boot` 与 `restored_boot` 轴为 `(bootstrap, x)`。
4. **查输出尺度和参考。**NPZ 的 `mu_GeV` 是最终 $\mu_F$，两套均为 $2\,\mathrm{GeV}$（$Q^2=4\,\mathrm{GeV}^2$）。`scale_mu_h_absx` 以唯一的正 $|x|$ 排列，与含正负两侧的 `x` 数组并非一一等长；RGR+LRR 的 `unregulated_scale_min_mask` 在有符号 `x` 网格标出满足 $2|x|P_z\ge1\,\mathrm{GeV}$ 的点。CT18NLO 比较用 `BASE/05_matching/reference.py` 的 `xfxQ(pid, x, 2.0)`，对应归档的 `Q_GeV=2.0`；如调用 `xfxQ2` 则传入 `4.0`。恢复局域荷和画 $xq(x)$ 都不改变 $\mu_F$。

实测元数据示例：`WL` 的 F32P30（L32x96）$P=4$，$P_z\simeq1.99975\,\mathrm{GeV}$。在 $x=0.20,0.50,0.70$，RGR+LRR 的中间尺度分别约为 $1.000,2.000,2.800\,\mathrm{GeV}$，其中第一点受 $1\,\mathrm{GeV}$ 下限控制；三点的最终 $\mu_F$ 都为 $2\,\mathrm{GeV}$。当前只对 unpolarized connected $u-d$ 跑完这套 RGR+LRR 全链。该比较仍是诊断：两套 $Z_R$ 均未通过质量门，matching 文件中的 `physical_quality_accepted` 为 `False`。

### 16.9 复核入口

- 坐标空间：`BASE/06_renorm_RGR_LRR/physics.py` 的 `running_and_evolution`、`details`，以及 `model.py` 的短距约束。理论对应 arXiv:2305.05212v2 式 (3)–(5) 的坐标空间演化与线质量方案。
- 动量空间：`rgr.py` 的 `prepare`、`evolution_rows`、`invert`、`lrr_mass`，以及 `matching.py` 的 `resummed` 与产物元数据。RGR 参考 arXiv:2209.01236v2 式 (15) 和附录 D；LRR 核参考 arXiv:2305.05212v2 式 (15)–(17)。
- [本次运行说明、数据格式与复现命令](/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/06_renorm_RGR_LRR/Matched_z1_225_Unpolarized_UD_Lightcone_20261124/README.md)；[CT18NLO 对比图册](/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/06_renorm_RGR_LRR/Matched_z1_225_Unpolarized_UD_Lightcone_20261124/plots/CT18NLO_unpolarized_ud_comparisons.pdf)。

## 17. Lightcone matching 的推导、尺度与复现（2026-11-25）

本节对应 HYP3D5 **unpolarized connected $u-d$** 的三 ensemble、$P=3,4$、2000 个配对 bootstrap。`BASE` 沿用第 15 节；以下用 `R = BASE/06_renorm_RGR_LRR` 简写路径。`WL` 和 `free` 分别表示加与不加 Wilson loop 相对线质量约束。这里区分两个独立的扫描：一是固定坐标参考尺度、改变 lightcone 输出尺度；二是重拟坐标参考尺度、固定 lightcone 输出尺度。第 16 节所说的两个 $2\,\mathrm{GeV}$ 是首轮运行的设置。

### 17.1 四类尺度：每个尺度进入哪里？

| 符号 | 当前取法 | 对应对象 |
|---|---|---|
| $\mu_{h,z}$ | $2e^{-\gamma_E}/\lvert z\rvert$，代入时 $z$ 用 $\mathrm{GeV}^{-1}$ | 坐标空间 $C_0$ 的自然尺度，使 NLO 系数中的短距对数消失。 |
| $\mu_{\mathrm{ref}}$ | 基准 $2\,\mathrm{GeV}$；另重拟 $4\,\mathrm{GeV}$ | 将短距系数演化到此处，用于 $P_z=0$ 的 $Z_R$ 拟合与有限质量条件；也出现在拟合模型显式的对数归一化里。 |
| $\mu_{h,x}$ | $\max(1\,\mathrm{GeV},\,2\lvert x\rvert P_z)$ | 动量空间逆 matching 的逐行求值尺度，也是自然尺度 lightcone 结果的尺度标签。$1\,\mathrm{GeV}$ 是本地代码的下限。 |
| $\mu_F$ | 已输出 $1,2,4,10,20\,\mathrm{GeV}$（坐标参考尺度为 $2\,\mathrm{GeV}$） | 最终 $\overline{\mathrm{MS}}$ lightcone PDF 的统一因子化尺度，比较同尺度 LHAPDF 时用 $Q=\mu_F$。 |

坐标空间的演化由非局域算符的反常维数控制；动量空间演化使用 lightcone 非单态 DGLAP。两处的 running 在本代码中都以 $\alpha_s(2\,\mathrm{GeV})=0.296$、固定 $n_f=3$ 为共同定标条件。从 $\mu_{\mathrm{ref}}=2$ 改到 $4\,\mathrm{GeV}$ 时用同一 running 得到 $\alpha_s(4\,\mathrm{GeV})=0.22150969$，没有独立更换耦合定标。$\mu_{h,x}$ 的下限仅用于数值处方，不能修复原本 $2\lvert x\rvert P_z$ 太小的微扰区间，也不能消除有限动量的幂修正。$z_1=0.225\,\mathrm{fm}$ 是短距拟合窗边界，$z_s=0.25\,\mathrm{fm}$ 是 hybrid 拼接点；它们和 $P_z$ 都应随结果记录。

### 17.2 从 hybrid 矩阵元到固定阶逆 matching

记 $H(z,P_z)$ 为在原点归一化的 hybrid 形状，$H(0,P_z)=1$。用 $z$ 的自然单位和 $\lambda=zP_z$，对每个 bootstrap 做尾部外推及 Fourier：

$$
\widetilde q_{\mathrm{hyb}}(x,P_z)=\frac{1}{2\pi}
\int_{-\lambda_{\max}}^{\lambda_{\max}}d\lambda\,e^{+ix\lambda} H(\lambda/P_z,P_z),
\qquad H(-z,P_z)=H(z,P_z)^*.
$$

在 LaMET 的 leading-twist 近似下，固定 quasi 的 hybrid 定义，quasi 和 lightcone shape 满足

$$
\begin{aligned}
\widetilde q_{\mathrm{hyb}}(x,P_z)
&= \int_{-1}^{1}\frac{dy}{\lvert y\rvert}\,
C_{\mathrm{hyb}}\!\left(\xi,\frac{\mu_F}{\lvert y\rvert P_z},\lvert y\rvert P_z z_s\right)
q_{\mathrm{shape}}(y,\mu_F)+\text{幂修正},
\qquad \xi=\frac{x}{y},\\
C_{\mathrm{hyb}}&=\delta(1-\xi)+\frac{\alpha_s(\mu_F)C_F}{2\pi}[R_{\mathrm{hyb}}]_+ +O(\alpha_s^2).
\end{aligned}
$$

上式略去 hybrid 有限质量处方等已固定的参数。光锥端 $q(y,\mu_F)$ 的支撑为 $-1\le y\le1$，负 $y$ 对应 $-\bar q(\lvert y\rvert,\mu_F)$；有限 $P_z$ 的 quasi 可以有 $\lvert x\rvert>1$ 的值。有限质子质量以及更高 twist 会给出幂修正；小 $x$ 或 $x$ 接近 1 时，控制尺度 $\lvert x\rvert P_z$ 或 $(1-\lvert x\rvert)P_z$ 变小，不能只用统计误差评估结果。

例：本地 unpolarized 的一圈实项在 $0<\xi<1$ 包含

$$
R_{\mathrm{bare}}(\xi,p,\mu_F)=\frac{1+\xi^2}{1-\xi}
\left[\ln\!\left(\frac{4\xi(1-\xi)p^2}{\mu_F^2}\right)-1\right]+1,
\qquad p=\lvert y\rvert P_z.
$$

这里 $R_{\mathrm{bare}}$ 是代码中同名的核函数分项。代码在所有三个 $\xi$ 区间计算实项，并在 vector 通道加 $3\operatorname{Si}((1-\xi)p z_s)/(\pi(1-\xi))$ 得到 hybrid 实项；再用实轴上的 plus 处方补齐虚项。作为分布，$[R]_+$ 对光滑测试函数的作用由 $\phi(\xi)-\phi(1)$ 给出，故对常数测试函数为零。数值上按比值变量 $\xi$ 写卷积：实项在 $y=x/\xi$ 取输入值，虚项在 $y=x$ 做局域减项；两者的动量参数分别为 $\lvert x/\xi\rvert P_z$ 和 $\lvert x\rvert P_z$。`matching_physics.py` 在完整实轴做 plus 积分，使用解析的 hybrid 虚项与紧致化积分尾。代码把卷积所需的 quasi Fourier 到 $[-6,6]$（步长 $0.01$），对网格外按零延拓；这仍是有限输入支撑近似。正、负 $x$ 必须与 Fourier 符号、核和 antiquark 约定一起使用。

令 $K_{\mathrm{hyb}}$ 表示上述一圈卷积算符，$a_s=\alpha_s C_F/(2\pi)$。利用

$$
(\mathbf 1-a_sK_{\mathrm{hyb}})(\mathbf 1+a_sK_{\mathrm{hyb}})
=\mathbf 1+O(\alpha_s^2),
$$

得到实际使用的一阶逆核：

$$
q_{\mathrm{shape}}(x,\mu_F)=\widetilde q_{\mathrm{hyb}}(x,P_z)
-a_s(\mu_F)\big[K_{\mathrm{hyb}}(\mu_F)\widetilde q_{\mathrm{hyb}}\big](x)
+O(\alpha_s^2,\text{幂修正}).
$$

固定阶分支直接代入统一的 $\mu_F$（存档主对照取 $2\,\mathrm{GeV}$）。这里截断的是**匹配核**，因此不对一圈核矩阵做精确数值求逆；下节求逆的是不同尺度之间的线性演化关系。

### 17.3 RGR+LRR：先在自然尺度匹配，再恢复统一光锥尺度

**第一步：自然尺度逆匹配。** 在树级 $y=x$，逆核的短距大对数以 $\mu/(2\lvert x\rvert P_z)$ 为特征；用 $\mu_{h,x}$ 求值可避免把从该尺度到共同 $\mu_F$ 的大对数全压在有限阶系数里。实项卷积内部仍使用 $p=\lvert y\rvert P_z$；代码没有将所有积分点的 $p$ 替换成 $\lvert x\rvert P_z$。记 $q_h(x)\equiv q_{\mathrm{shape}}(x,\mu_{h,x})$，本地实现为

$$
\begin{aligned}
q_h(x)
={}&\widetilde q_{\mathrm{hyb}}(x,P_z)
-a_s(\mu_{h,x})[K_{\mathrm{hyb}}(\mu_{h,x})\widetilde q_{\mathrm{hyb}}](x)\\
&-\Delta m_{\mathrm{LRR}}(\mu_{h,x})\,\mathcal F_\lambda[w_\epsilon(z;z_s)H(z,P_z)](x),\\
\Delta m_{\mathrm{LRR}}(\mu)&=\mu\left[S_{\mathrm{PV}}(\alpha_s(\mu))-\alpha_s(\mu)r_0(1,1)\right],\\
w_\epsilon(z;z_s)&=\theta(\lvert z\rvert-z_s)\big[\lvert z\rvert e^{-\epsilon\lvert z\rvert}-z_s e^{-\epsilon z_s}\big].
\end{aligned}
$$

$S_{\mathrm{PV}}$ 是代码的单位 $\mu\lvert z\rvert$ PV 求和，$\alpha_s r_0$ 是已包含在 NLO 内的一阶项。$w_\epsilon$ 的常数减项对应本地全实轴虚项，中央 $\epsilon=0.06\,\mathrm{GeV}$，与第 16.7 节相同。此处 $\Delta m_{\mathrm{LRR}}$ 是 matching 核的 PV 减除系数，区别于 $Z_R$ 拟合得到的有限质量 $m_0$。LRR 本身没有附加一轮 DGLAP。`rgr.prepare` 用两个固定尺度核矩阵提取一圈对数斜率，生成逐行不同 $\mu_{h,x}$ 的矩阵与 PV 系数。修正后的 LRR Fourier 先在均匀有符号 $x$ 网格计算，再取 $x=\pm(0.02,0.03,\ldots,0.99)$。这样不把移除 $x=0,\pm0.01$ 后的非均匀网格直接传给假设均匀间距的 CZT；`weighted_fourier` 还用直接 Simpson 积分在若干点核对。

**第二步：非单态 DGLAP。** 对 $x>0$ 定义 $q=q_{u-d}$、$\bar q=\bar u-\bar d$，$f_\pm=q\pm\bar q$。约定下式中 $P_\pm^{\mathrm{NS}}$ 是 $d/d\ln\mu^2$ 的 splitting 核，依同一固定 $n_f$ 方案截断到 NLO：

$$
\frac{d f_\pm(x,\mu)}{d\ln\mu^2}=
\int_x^1\frac{dy}{y}\,P_\pm^{\mathrm{NS}}(x/y,\alpha_s(\mu)) f_\pm(y,\mu).
$$

记 $U(\mu_b,\mu_a)$ 将统一尺度 $\mu_a$ 的 PDF 演化到统一尺度 $\mu_b$。自然尺度输入在每行有不同的尺度，故选择一个待求的统一尺度 $\mu_F$，逐行构造

$$
f_{\pm,h}(x_i)=\sum_{j\ge i}A_{\pm,ij}(\mu_F)f_\pm(y_j,\mu_F),
\qquad A_{\pm,ij}(\mu_F)=\big[U_\pm(\mu_{h,x_i},\mu_F)\big]_{ij}.
$$

$x$ 网格由小到大排列，DGLAP 的 $y\ge x$ 支撑保证了矩阵上三角，因此可从最大 $x$ 向下代入（`rgr.invert` 使用 `solve_triangular`）。有两个分布、两个非单态核；最后用 $q=(f_++f_-)/2$、$\bar q=(f_+-f_-)/2$ 恢复正负 $x$ 的存储约定。这解释了代码里 $U(\mu_h,\mu_F)$ 的方向：先用共同尺度到自然尺度的关系建方程，反解后得到共同尺度的 PDF。自然尺度匹配结果不是单一尺度上的一条 PDF，不能逐 $x$ 乘一个演化因子。

**第三步：恢复局域荷并比较。** 用同一 bootstrap 的 $g_V^R=Z_V h_B(P_z=0,z=0)$ 乘回统一尺度的 shape，输出 $q^R(x,\mu_F)=g_V^R q_{\mathrm{shape}}(x,\mu_F)$。对这里的非单态矢量荷，恢复局域荷和取 $xq$ 都不改变 $\mu_F$。已匹配到目标尺度的文件无需再次演化来做同尺度比较。用 LHAPDF `xfxQ(pid,x,Q)` 时传 $Q=\mu_F$，`xfxQ2` 则传 $Q^2=\mu_F^2$。上一轮固定 $\mu_{\mathrm{ref}}=2\,\mathrm{GeV}$ 求 $\mu_F=20\,\mathrm{GeV}$，直接反解与从已匹配的 $q(x,2\,\mathrm{GeV})$ 演化的归一化 PDF 逐样本最大差为 $8.21\times10^{-10}$。这是数值闭合检验；CT18NLO 的 $Q=1.3\,\mathrm{GeV}$ 曲线只是跨尺度参照，不参与演化。

### 17.4 为什么改了重整化参考尺度，hybrid 矩阵元却几乎不变？

**短距系数有变化。** 保持 $\mu_{h,z}$、相同 running、PV 处方和减除阶数时，反常维数只依赖于耦合；由演化的组合律，

$$
\begin{aligned}
C_0(z;\mu_b)&=U_\Gamma(\mu_b,\mu_{h,z})C_{0,\mathrm{short}}(z;\mu_{h,z})\\
&=U_\Gamma(\mu_b,\mu_a)C_0(z;\mu_a).
\end{aligned}
$$

$C_{0,\mathrm{short}}$ 包含 NLO 与 PV 减 NLO 重叠。实测 $2\to4\,\mathrm{GeV}$ 时，所有短窗点和边界点都给出 $C_0(z;4)/C_0(z;2)=1.14642108\ldots$（`C0.npz`）。这是与 $z$ 无关的乘法变换；在短距条件 $H_0/Z_R=C_0$ 下，若条件被精确满足，$Z_R$ 会相应反向缩放。实际 $Z_R$ 是带先验和离散效应的关联拟合，不能假设它在所有长度精确按同一常数变换。

**短距分支逐样本严格抵消。** 按第 16.2 节，$H(z,P_z)=S(P_z,z)/H_0(z,a)$（$\lvert z\rvert\le z_s$）。两轮使用相同裸矩阵元和同一归一化零动量矩阵元 $H_0$；该分支不读 $Z_R$，也不直接乘 $C_0$。核对 12 个坐标空间输出，所有短距点逐样本差值均为零。

**长距分支消掉所有与 $z$ 无关的因子。** 写

$$
\begin{aligned}
\ln Z_R(z,a;\mu)&=N(a;\mu)+[A(a;\mu)+m_0(\mu)]\lvert z\rvert+a^2 f(\lvert z\rvert;\mu),\\
A(a;\mu)&=\frac{k(\mu)}{a\ln[a\Lambda(\mu)]},\\
F_\mu(z,a)&=\frac{Z_R(z_s,a;\mu)}{H_0(z_s,a)\, Z_R(z,a;\mu)}.
\end{aligned}
$$

由于两个 $Z_R$ 在同一 ensemble，$N(a;\mu)$（包括显式参考尺度的对数归一化和 $d$ 项）**在每一套参数内部精确抵消**。记 $\Delta B=B(\mu_b)-B(\mu_a)$，两套长距因子的差异只剩

$$
\ln\frac{F_{\mu_b}(z,a)}{F_{\mu_a}(z,a)}=
-(\lvert z\rvert-z_s)\,\Delta[A(a)+m_0]
-a^2\,\Delta[f(\lvert z\rvert)-f(z_s)].
$$

所以 $m_0$ 单项变化不等于实际矩阵元变化。$k,\Lambda,m_0,f$ 的改变共同决定这个差；$g$ 和 $d$ 可以通过共同拟合间接影响它。若只有乘法归一化改变，长距分支也会完全重合。在当前重拟的设置下仍有小的长度依赖残差。

**配对数据核对。** 固定短窗、$z_s$、协方差、光锥输出 $\mu_F=4\,\mathrm{GeV}$ 以及同一批 bootstrap，比较 $\mu_{\mathrm{ref}}=2\to4\,\mathrm{GeV}$：

| 量 | `WL` | `free` |
|---|---:|---:|
| $z_s<\lvert z\rvert\le0.95\,\mathrm{fm}$ 内 $\max\lvert\langle F_4/F_2-1\rangle_b\rvert$ | $0.060\%$ | $0.161\%$ |
| $\max\lvert\langle\widetilde q_4-\widetilde q_2\rangle_b\rvert$（归一化 quasi） | $3.56\times10^{-4}$ | $1.37\times10^{-3}$ |
| $\max\lvert\langle\Delta[xq^R(x,\mu_F=4\,\mathrm{GeV})]\rangle_b\rvert$ | $5.74\times10^{-5}$ | $1.87\times10^{-4}$ |

后两行的区间为 $0.25\le x\le0.8$、$2xP_z\ge1\,\mathrm{GeV}$，下标 $2,4$ 在 quasi 行指**坐标参考尺度**。最大值覆盖三 ensemble、$P=3,4$。例如 F32P30、$P=4$、`WL` 在 $x=0.50$ 的 $xq^R$ 从 $0.327814$ 变为 $0.327802$，配对差为 $(-1.21\pm0.28)\times10^{-5}$，单条光锥曲线的 bootstrap 标准差约为 $0.0066$。同一方案的 $m_0$ 同时由 $0.3218$ 变为 $0.3031\,\mathrm{GeV}$，说明单看 $m_0$ 不能判断最终的尺度敏感性。

这个近似不变是**当前归一化 hybrid 处方下的合理结果**。它检验的是坐标参考尺度的敏感性；与改变 $\mu_F$ 后 lightcone PDF 的 DGLAP 变化可以同时成立。$Z_R$ 的质量门仍未通过，$\lvert z\rvert>0.95\,\mathrm{fm}$ 属模型延伸，表中小差异不能用来估计有限 $P_z$、格距、长距外推或微扰截断等全部系统误差。

### 17.5 quasi PDF 的尺度如何标注？它满足 DGLAP 吗？

**标注方案、参考尺度和动量。** 可以记作 $\widetilde q_{\mathrm{hyb}}(x,P_z;z_s,\mu_{\mathrm{ref}},\mathrm{PV})$，必要时再附格距和尾部截断。对基准数据可以写 “hybrid RGR+LRR quasi shape，$\mu_{\mathrm{ref}}=2\,\mathrm{GeV}$”；$4\,\mathrm{GeV}$ 重拟数据则把参考尺度标签换成 $4\,\mathrm{GeV}$。这两档的区别来自输入矩阵元和其方案标记。Fourier 只是改变表示，不赋予 quasi 一个 lightcone 的 $\mu_F$。$P_z$ 是外部动量，$\mu_{h,x}$ 是 matching 的中间尺度，二者都不能当作 quasi 的统一因子化能标。把同一份 quasi 匹配到 $2$、$10$ 或 $20\,\mathrm{GeV}$，quasi 数据保持同一份。

**标准 DGLAP 适用于匹配后的 leading-twist lightcone PDF。** 有限 $P_z$ 的 quasi 来自等时的空间双局域算符。改变 quasi 自身的重整化处方时，应按对应非局域算符的重整化群方程或方案转换处理，混合归一化方案还会抵消共同的乘法因子；Fourier 变换后即使出现卷积形式，也不等于标准 lightcone splitting 核。改变 $P_z$ 也不是对 quasi 做 DGLAP，因为匹配核和有限动量幂修正随 $P_z$ 改变。

具体地，固定已重整化的 quasi 和 $P_z$，把第 17.2 节的 leading-twist 匹配写成 $\widetilde q=\mathcal C(\mu_F)\otimes q(\mu_F)$。因子化一致性给出（导数包括核内耦合的 running，卷积为算符复合）

$$
\frac{d \mathcal C}{d\ln\mu_F^2}=-\mathcal C\otimes P_{\mathrm{NS}},
\qquad
\frac{d \mathcal C^{-1}}{d\ln\mu_F^2}=P_{\mathrm{NS}}\otimes\mathcal C^{-1}.
$$

因此固定 quasi 输入时，由逆核抽取的 $q(\mu_F)=\mathcal C^{-1}(\mu_F)\otimes\widetilde q$ 随 $\mu_F$ 做 DGLAP。坐标参考尺度的抵消与光锥因子化尺度的演化是两条不同的关系。实际流程应先匹配，再对已匹配的 PDF 做所需的统一尺度演化，或按第 17.3 节一次得到所需的 $\mu_F$。要做与全球拟合的精确高尺度比较，还需核对重味阈值和扰动阶数；目前格点演化固定 $n_f=3$，CT18NLO 参考为可变味数。

### 17.6 操作清单、输出字段与参考文献

1. **建立坐标输入。** 按第 16.2 节在三个 ensemble 的 $P_z=0$ 上拟合 $Z_R$，将同一方案用于各 ensemble 的 $P=3,4$。保持 $z_1$、$z_s$、先验、covariance、PV 处方和 bootstrap 行号一致。改变 $\mu_{\mathrm{ref}}$ 要重做 $C_0$、短距条件和 $Z_R$ 拟合，再更新 hybrid 矩阵元；只改 $\mu_F$ 时直接复用已有 quasi 和自然尺度匹配结果。
2. **外推与 Fourier。** `Matched_z1_225_Unpolarized_UD_Lightcone_20261124/fourier.py` 做关联复数尾部外推，主结果 $\lambda_{\max}=100$、步长 $0.05$，保存 $25,50$ 的截断对照。核入口是 `BASE/all_pdf_pipeline/matching_physics.py`。当前推荐复用 `RGR_LRR_ScaleScan_1_2_4GeV_20261124/compute.py` 生成的 `normalized_at_hard_boot`；该入口修正了归档脚本对 LRR 项使用缺中心点的 Fourier 网格的问题。正文涉及 $2\to10/20\,\mathrm{GeV}$ 的核对均以**修正后的** $\mu_F=2\,\mathrm{GeV}$ 结果为基准。
3. **选择输出能标。** `MuRef2_Lightcone10_vs_CT18NLO1p3_20261125/compute.py` 与 `MuRef2_Lightcone20_vs_CT18NLO1p3_20261125/compute.py` 读入同一份自然尺度结果；`rgr.py` 中 `evolution_rows(..., mu_target=...)` 生成矩阵，`invert` 反解。配套的 `plot.py` 画出同尺度 CT18NLO 及跨尺度参考曲线。坐标尺度扫描的 `MuRef4GeV_20261125/{refit,lightcone,plot}.py` 固定两支的 $\mu_F=4\,\mathrm{GeV}$。各运行目录的 `README.md` 有环境和复现命令。
4. **读取与质量检查。** 自然尺度结果 `normalized_at_hard_boot` 以及最终 `normalized_boot`（shape）、`restored_boot`（恢复荷）都按有符号 `x` 保存。单一输出尺度的形状为 `(2000,196)`；多尺度档在首轴加尺度维。`mu_F_GeV`（或旧文件的 `mu_GeV`）是输出尺度，`short_distance_reference_mu_GeV`（或 `mu_ref_GeV`）是坐标参考尺度。`scale_mu_h_absx` 与正的 `x_abs` 对应，`unregulated_scale_min_mask` 标记 $2\lvert x\rvert P_z\ge1\,\mathrm{GeV}$。`bootstrap_original_row_ids` 用于配对做差，`physical_quality_accepted=False` 表示诊断。作尺度比较前检查上游 $Z_R$ 的 $\chi^2/\mathrm{dof}$、$Q$、尾部触边率和 $\lambda_{\max}$ 敏感性。

例如，从已有坐标输入和已归档的 $10\,\mathrm{GeV}$ 配对结果复现 $20\,\mathrm{GeV}$ 计算和绘图：

```bash
cd /public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/06_renorm_RGR_LRR/MuRef2_Lightcone20_vs_CT18NLO1p3_20261125
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/donghx_mu_F20_mpl
/public/home/donghx/AnaConda/bin/python compute.py
/public/home/donghx/AnaConda/bin/python plot.py
```

本地证据：

- [坐标参考尺度扫描与数表](/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/06_renorm_RGR_LRR/MuRef4GeV_20261125/README.md)；[坐标空间对照图](/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/06_renorm_RGR_LRR/MuRef4GeV_20261125/plots/mu_ref_coordinate.pdf)。quasi 配对差由 `02_lightcone/{WL,free}/<ensemble>/P{3,4}.npz` 的 `quasi_boot` 两个参考尺度槽直接计算；`matrix_summary.csv` 和 `lightcone_summary.csv` 分别记录坐标与光锥的配对差。
- [修正后的自然尺度 matching 与 $\mu_F=1,2,4\,\mathrm{GeV}$ 输出](/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/06_renorm_RGR_LRR/RGR_LRR_ScaleScan_1_2_4GeV_20261124/README.md)；[$\mu_F=20\,\mathrm{GeV}$ 流程与验证](/public/group/lqcd/donghx/Diagram_GluonPDF/Quark_PDF/Constant_Anchor075_P34_AllStages_20261122/06_renorm_RGR_LRR/MuRef2_Lightcone20_vs_CT18NLO1p3_20261125/README.md)。

理论来源：Ji 等 [arXiv:2008.03886](https://arxiv.org/abs/2008.03886)（hybrid 方案）；Chou、Chen [arXiv:2204.08343v4](https://arxiv.org/abs/2204.08343v4) 式 (35)、(38)、(39)（各通道 hybrid-ratio 一圈核）；Su 等 [arXiv:2209.01236v2](https://arxiv.org/abs/2209.01236v2) 式 (12)–(15) 和附录 D（逆核的 RG、自然尺度、从大到小 $x$ 的演化）；Zhang 等 [arXiv:2305.05212v2](https://arxiv.org/abs/2305.05212v2) 式 (3)–(5)、(15)–(17)（坐标系数 RGE 与 LRR 核，本地全实轴虚项的区别见第 16.7 节）；Ji、Zhang、Zhao [arXiv:1706.08962](https://arxiv.org/abs/1706.08962)（空间直 Wilson line 算符的乘法重整化）；Moch、Vermaseren、Vogt [hep-ph/0403192](https://arxiv.org/abs/hep-ph/0403192)（非单态 splitting 核，本地仅取到 NLO）。
