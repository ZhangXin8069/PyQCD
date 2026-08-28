---
name: pyqcd-tmd-algorithm
description: >
  Use when implementing, porting, debugging, or validating PyQCD's
  gradient-flow-renormalized nucleon gluon TMD-PDF calculation, including
  flowed Clover fields, nonzero transverse separation, staple Wilson lines,
  disconnected nucleon matrix elements, soft/rapidity subtraction,
  Collins-Soper evolution, matching, or continuum-limit tests; use when a
  quasi-PDF prototype must be kept distinct from a complete TMD result.
metadata:
  openclaw:
    emoji: 🧬
---

# pyqcd-tmd-algorithm — 物理到实现的算法契约

## 概览与边界

本技能把目标物理量

\[
 f_{g/N}^{[\Gamma]}(x,\boldsymbol b_\perp;\mu,\zeta)
 \quad\longleftarrow\quad
 h_g^{\mathrm{flow}}(z,\boldsymbol b_\perp;P_z,\tau,\ell)
\]

逐层映射为 PyQCD 的数组、接口、统计量和验证门。重点是“方程—离散几何—代码—
可观测证据”闭环，而不是把能运行的 demo 当作物理结果。

本技能补充 `pyqcd-tmd-chain`：后者给出六步物理链和现有入口地图；本技能处理实现
时的坐标契约、有限 staple、核子外态、误差传播、方案分层和失败边界。纯理论推导用
`pyqcd-physics-correlator` / `pyqcd-physics-spectrum`，纯拟合用 `pyqcd-analysis`。

若能访问辅助参考，先读
`/root/MyQCD/docs/report_gluon_tmd_gradient_flow_20260828.tex` 的实现边界和验证矩阵，
再按需读其中列出的 GF-01/GF-02、G-01/G-02、T-01 参考章节。参考代码只能用于理解；
实现必须留在 PyQCD 内，不得 import `refer/` 或 `examples/`。

## 先固定四种状态

在代码、日志和报告中分别使用下列状态，不得混写：

| 状态 | 含义 | 允许的结论 |
|---|---|---|
| 实现存在 | 函数或文件可调用 | 仅说明接口存在 |
| 测试通过 | 受控输入满足形状/数值断言 | 说明该断言通过 |
| 方案闭合 | soft、rapidity、流方案、匹配和尺度约定均明确 | 才能讨论方案内物理量 |
| 真实数据验证 | 组态级核子三点、系统扫描和误差账本完成 | 才能报告物理结果 |

当前 PyQCD 已有 Wilson flow、Clover/对偶场强、直线胶子 OPE、部分 staple/quasi-TMD
接口、质子 2pt 和 Fourier/matching 原型；其中 staple 几何仍未闭合，这不自动等于
“完整核子胶子 TMD-PDF 已完成”。
没有同几何胶子软因子、rapidity subtraction、真实胶子 TMD 三点和完整胶子 TMD matching
证据时，结论必须停留在“原型/接口/待验证”。

## Step 0：算法契约与物理量纲

先写入配置或元数据，再写计算循环：

1. 固定通道（首选非极化 $f_1^g$），Lorentz 投影 $\Gamma$，空间纵向轴和横向轴。当前
   `b_perp`/`b_dir` 接口只覆盖一个横向轴的标量位移；完整二维 TMD 需显式支持
   $\boldsymbol b_\perp=(b_x,b_y)$、任意横向方向和旋转对称性对照。
2. 固定 $z$、$\boldsymbol b_\perp$、有限 staple 长度 $\ell$、流时间 $\tau$、动量
   $P_z$、$\mu$、$\zeta$、Wilson 线表示 $\mathcal R$、匹配阶数和 Fourier 约定。
3. 区分格点单位与物理单位：当前低层接口的 `tau/z/b` 通常是格点整数或格点流时间；
   `quasi_tmd_pdf` 的 `z_grid` 使用 fm、`pz_gev` 使用 GeV，内部转换为 GeV$^{-1}$。
   Ioffe 时间 $\nu=zP_z$ 必须无量纲。
4. 选择中心几何或锚定几何并始终一致。参考报告的中心定义为
   $x_1=+z\hat z/2+\boldsymbol b_\perp$、$x_2=-z\hat z/2$；当前矩阵代码多为
   $x$ 到 $x+z\hat z+\boldsymbol b_\perp$ 的锚定实现。两者不能在同一数据集混用。
5. 复数结果先保留复数 dtype；只在由宇称、时间反演或投影证明后取实部/虚部。正负
   $z$ 的关系必须实测，不能对所有通道强行偶化。

每个产物至少记录：`a_fm`、格点体积、`tau`、流积分步长、`Pz_gev`、`z`、
`b_perp`、`staple_length`、方向、表示、Lorentz 通道、`mu`、`zeta`、匹配阶数、
随机种子、重采样方案、协方差/SVD 参数和代码版本。

若用户未指定初始扫描，可用下列最小冒烟矩阵；它只用于暴露几何和统计问题，不代表
物理结果：$b_\perp=0$ 与一个非零横向向量（如 $(1,0)$ 格点单位）对照，两个非零
纵向动量模（如格点模 $2,4$），两个有限 $\ell$，以及 $\tau=3a^2$ 和一个邻近流时间。
两动量、两 $\ell$ 和两 $\tau$ 必须共享同一组态索引及元数据。

## 物理定义与离散实现

### 1. 梯度流：把 UV 调节器作为正式尺度

Wilson flow 的格点右端写成

\[
 \dot V_\mu(x)=Z_\mu[V](x)V_\mu(x),\qquad
 Z_\mu=P_{\mathrm{ah}}[\Omega_\mu V_\mu^\dagger],
\]

其中 $\Omega_\mu$ 是六个 staple，$P_{\mathrm{ah}}$ 是无迹反厄米投影。当前实现
`wilson_flow` 使用 Lüscher 三阶 RK：

\[
\begin{aligned}
 W_1&=e^{\epsilon Z_0/4}W_0,\\
 W_2&=e^{(8\epsilon Z_1/9-17\epsilon Z_0/36)}W_1,\\
 V_{t+\epsilon}&=e^{(3\epsilon Z_2/4-8\epsilon Z_1/9+17\epsilon Z_0/36)}W_2.
\end{aligned}
\]

执行时：

- 每个所需 $\tau$ 只从同一 $U$ 出发，缓存 $V_\tau$，不要把不同流时间的场强和 Wilson
  线拼在一起；
- 扫描至少两个流时间或固定物理流半径后再做 $a\to0$，$\tau=3a^2$ 只能作为基线点；
- 检查 $V_\tau^\dagger V_\tau\simeq I$、减小 $\epsilon$ 后结果收敛和边界滚动正确；
- 当前 `test9_verify.py` 对 `flow_action_density` 采用 $E(\tau)<E(0)$ 判据。若另
  外分析的是 $\tau^2\langle E(\tau)\rangle$，必须重新声明其预期单调性，不能套用该判据。

### 2. 流化场强、staple 与张量投影

胶子 TMD 的最小非局域结构为

\[
\mathcal O^{g,\tau}_{\mu\nu;\rho\sigma}
 =G^a_{\mu\nu}(x_1;\tau)
 [W_{\rm st}^{\mathcal R}(x_1,x_2;\ell)]^{ab}
   G^b_{\rho\sigma}(x_2;\tau).
\]

这里必须先选定表示：伴随表示使用 $W_{\rm adj}^{ab}$，基本表示则应写成带色迹的
$\operatorname{Tr}[G(x_1)W_{\rm fund}G(x_2)W_{\rm fund}^\dagger]$；只有显式验证
$W_{\rm adj}^{ab}=2\operatorname{Tr}(T^aW_{\rm fund}T^bW_{\rm fund}^\dagger)$ 和
归一化后才可互相对照。

非零 $b_\perp$ 必须由横向路径连接；只有纵向直线的是 quasi-PDF 几何。实现顺序为：

1. 在同一个 $V_\tau$ 上计算 Clover $F_{\mu\nu}$ 和 $\widetilde F_{\mu\nu}$；
2. 以显式路径构造 $W_{\rm st}$，包括 $z$、$b_\perp$、$\ell$、正负方向和周期绕回；
3. 保存各 $M^{\mu\lambda;\nu\rho}$，最后再做投影。基线组合
   $O=M^{tx;tx}+M^{ty;ty}-2M^{xy;xy}$ 可以作为非极化起点，但不能抹掉张量混合矩阵；
4. 检查路径反向/共轭、$W W^\dagger\simeq I$、$b_\perp\to0$ 的直线极限、$z\to0$
   的局部极限，以及规范变换前后的不变量。

当前接口的实际边界：`renorm.staple_wilson_line`/`gluon_tmd_operator` 使用整数
`b_perp` 和 `L`；默认 `L=None` 时通常取 `L=z`。当前 `_path_product` 的横向段调用
是 `start=end`，反向纵向段则以 `forward=False, start=0, end=z+L` 调用；按现有
`range` 逻辑这两段均为空，因此当前函数只执行第一段正向纵向链，不能作为有效的
物理 staple。必须先修复并通过逐段单元测试、起止点/链长计数、常数链接与纯规范场
基准、路径反向 $W(-\mathcal P)=W(\mathcal P)^\dagger$（坐标适配后）和
$b_\perp\to0$ 对照；$W W^\dagger\simeq I$ 只能检查近似幺正性，不能单独证明路径正确。
此外，当前 `M_mu_lambda_nu_rho` 的收缩顺序形成
`Tr(F_nu W† W F_mu_shift)`；若 $W$ 幺正，Wilson 线传输会在代数上抵消，故必须
单独验证端点与颜色收缩，修复前不得把该 API 当作物理 TMD 算符。
`tmd_matrix_elements`/`tmd_matrix_elements_time` 会在高层取实部，`quasi_tmd_pdf` 和
CS 辅助接口还把输入转为 `dtype=float`；复数虚部会被丢弃。复数路径、helicity 或
奇对称通道须在低层和正负 $z$ 对照完成后才能取实部。若物理问题需要独立的 $\ell$，
必须扩展接口并让元数据区分 `z` 与 `staple_length`，不能把 `L=z` 默认为有限 rapidity
极限。当前这些接口按基本表示 $3\times3$ 矩阵和色迹实现；物理定义若采用伴随表示，
必须显式检查 $U_{\rm adj}^{ab}=2\operatorname{Tr}(T^aUT^bU^\dagger)$、颜色归一化和
软因子是否同表示，不能仅凭函数名混用。`operator.staple_operator` 是另一条实现，
应先做几何逐点对照，再选定唯一生产路径。

`gluon_ope_operator_z0` 是直线 OPE（含 $\pm z$ 和交叉 Lorentz 对扩展），
`gluon_ff_operator_z0` 是固定规范、无 Wilson 线的对照通道；二者都不能替代规范不变的
空间 staple TMD。

### 3. 核子外态与 disconnected 矩阵元

对每个组态先构造核子 2pt $C_2$，再以同一组态上的胶子 loop $L_g$ 做真空扣除：

\[
 C_3^g=\langle C_2L_g\rangle-\langle C_2\rangle\langle L_g\rangle,
 \qquad R_g=C_3^g/C_2.
\]

$C_2$ 使用核子宇称投影、动量投影和多个 $t_{\rm sep}$；$R_g$ 经过平台或二态/多态
拟合后才得到 $h_g(z,b_\perp;P_z,\tau,\ell)$。断连误差必须保留规范组态自相关、核子
2pt 噪声、loop 噪声和随机源相关性。

当前 `analysis.run_disconnected_tmd_ratio` 的回归骨架可以接收
`ope_all[cid]['tmd']` 形状 `(nz, nb, Nt)`，但其中若按代码路径先构造
`C3=C2*OPE`，且 `OPE` 确实是同一组态独立测得的胶子 loop $L_g$，那么这正是
断连三点的组态级因子化；真空扣除应是重采样后的
`<C2*OPE>-<C2>*<OPE>`，而不是逐组态再次减去同一个 `C2*OPE`。当前代码在
`Nconf=1` 时会发生后者的恒等抵消；`Nconf>1` 时保留的是组态协方差，但仍需核实
`ope_all` 的来源、归一化和几何是否真为 $L_g$。因此它可以是断连三点的数值骨架，
不能仅凭输入形状就宣称已完成物理测量；只有独立组态级相关数据、真空扣除和多个
$t_{\rm sep}$ 通过后，才可把 `c0` 称为核子胶子矩阵元。

### 4. 软因子、流方案与自重整化分层

可用下式表达待固定的方案模板：

\[
 H_{g,A}^{\rm sub}(z,b;\tau,\ell)
 =\frac{h_{g,A}^{\rm flow}(z,b;\tau,\ell)}
        {\sqrt{S_\tau^g(b,\ell)}}R_\tau^g.
\]

这只是分层契约，不是自动成立的定理。$S_\tau^g$ 必须与主算符使用相同的流时间、
表示、staple 方向、转角、长度和归一化；$R_\tau^g$ 的短距比值、自重整化或混合方案
必须写出实际定义。

可复用的 PyQCD 接口包括 `build_hB_dataset`、`boot_covariance`、`fit_ZR`、
`fit_ZR_samples`、`fit_hR_lambda`；它们能支撑 Z_R/混合方案的数值骨架，但不能在没有
胶子 TMD 软因子和 rapidity 方案的情况下被命名为完整 TMD 重整化。局部
`sftx_gluon_matching_coeff` 只处理流算子到 $\overline{\mathrm{MS}}$ 的局部 1 圈 building
block，不能代替 staple cusp、rapidity 和完整胶子 TMD matching。当前
`soft_function_intrinsic` 主要是 `R/R[0]` 归一化框架，若调用方使用
`tmd_matching_hybrid(..., soft_factor=1.0)`，那仍是匹配 scaffold，不是已测量的软因子。

### 5. Fourier、CS 核与匹配

当前接口的职责要分开：

| 物理步骤 | PyQCD 接口 | 使用边界 |
|---|---|---|
| cos 型准 TMD | `quasi_tmd_pdf` | 对 $h_R(z,b)$ 做纵向 Fourier；不能单独证明 soft/rapidity 已消除 |
| sin 型共线准 PDF | `quasi_pdf_gluon` | 胶子 collinear 交叉检查；$x\to0$ 有保护，不是 TMD 替代物 |
| CS 两动量比值 | `cs_kernel_two_momentum` | 至少两个不同 $P_z$；`z_ref` 与符号/截断要记录 |
| 1 圈混合匹配骨架 | `tmd_matching_hybrid` | `x_tmd` 是输入光锥 TMD，输出是离散准 TMD scaffold；保留 $Z_{ij}$ 矩阵和阶数，不能视为自动完成格点匹配 |
| 流到 MS 的系数 | `sftx_gluon_matching_coeff` | 仅局部流方案 building block |
| 联合外推 | `fit_hR_PDF_extrap_boot` | 需要多 $a/P_z/m_\pi/L$ 数据及协方差 |

匹配至少应显式保留

\[
 \boldsymbol C^g\otimes
 \begin{pmatrix}f_g\\f_q\end{pmatrix},
 \qquad
 Z_{ij}=\delta_{ij}+O\!\left(\frac{\alpha_s C_A}{2\pi}\right),
\]

并记录 $\mu$、$\zeta$、$\alpha_s$、$C_A/C_F$、表示和阶数。$\alpha_s\to0$ 时应回到
单位匹配；匹配矩阵奇异或主值项失稳时应报告条件数和截断，不能静默取伪结果。
`cs_kernel_two_momentum` 直接对两个动量的裸 `c0` 在 `z_ref` 行取对数比，并带
`k_clip` 硬截断；它是工程估计器，不是已经闭合 soft/rapidity 方案的 CS 核。调用
前必须确认两个 $P_z$、同一 $(z,b,\ell,\tau)$ 几何及符号约定，并把截断前后的值同时
归档。

## 推荐算法循环

按以下顺序实现或审查；每一步只消费上一步已通过的对象：

1. **输入**：读取 ILDG/既有规范场，验证形状、SU(3) 近似幺正性、边界和元数据。
2. **流化**：对 $\{\tau\}$ 计算并缓存 $V_\tau$，保存步长收敛和 `flow_action_density`。
3. **算符**：在同一 $V_\tau$ 上缓存 $F^\tau,\widetilde F^\tau$，逐 $(z,b,\ell,\pm)$
   构造 staple 和 Lorentz 分量，先保留复数。
4. **软因子**：独立计算同表示、同几何、同流时间的真空软对象，输出归一化及其协方差。
5. **外态**：复用已验证质子 2pt/蒸馏外态；独立测量胶子 loop 与 2pt 的组态级积，
   构造断连 $C_3^g$、比值和多个 $t_{\rm sep}$ 的平台。
6. **统计**：对 $z,b,\ell,\tau,\pm z,P_z$ 使用同一组态索引做 block-jackknife 或
   bootstrap；奇异协方差使用记录了 cut/条件数的 SVD 或对角回退。
7. **重整化**：先形成 $h^{\rm flow}$，再按明确方案应用 $S_\tau^g$、$R_\tau^g$、
   Z_R/混合和小流时间匹配，保留每层中间量。
8. **提取**：以物理单位构造 $\nu=zP_z$，按已证明的偶奇性质做 cos/sin 或复 Fourier，
   再用两动量提取 CS 核、软函数和匹配矩阵。
9. **外推**：分别检查 $\ell$、$\tau\to0$、$P_z\to\infty$、$a\to0$、有限体积和
   拟合窗；输出中心值、统计误差、系统误差和协方差。
10. **归档**：HDF5/JSON 中保存数组、维度名、单位、参数、seed、命令行、git 版本和
    每个验证门结果。

最小伪代码只表达数据流，不虚构尚不存在的顶层 API：

```python
for conf in configs:
    U = load_gauge(conf)
    for tau in flow_times:
        V = wilson_flow(U, tau=tau, eps=eps)
        F = build_flowed_clover(V)
        for z, b, ell, channel in operator_grid:
            O = build_staple_matrix_element(F, V, z, b, ell, channel)
            save_per_configuration(conf, tau, z, b, ell, O)
    # 概念占位：保存 C2(t_sink,t_src; Pz)、Lg(t_ins; z,b,ell,tau) 和逐组态 C2*Lg。
    measure_nucleon_2pt_and_gluon_loop(
        conf, source_times=source_times, t_seps=t_seps,
        momentum=momentum, loop_estimator=loop_estimator)
assemble_disconnected_ensemble_covariance()
apply_declared_renormalization_and_matching()
run_limits_and_validation_gates()
```

## 验证门与停止条件

| 门 | 必须看到的证据 | 失败时动作 |
|---|---|---|
| 输入/群性质 | 形状正确、$U^\dagger U\simeq I$、dtype/单位明确 | 停在输入层 |
| 流积分 | $V^\dagger V\simeq I$，减小步长结果稳定；按当前定义检查 $E$ | 不进入算符层 |
| 场强/路径 | Clover 反对称、对偶定义一致，路径反向/共轭和 $WW^\dagger$ 通过 | 保留 raw，修几何 |
| 规范性 | 规范变换前后最终标量不变，颜色指标闭合 | 不得用固定规范替代 |
| 几何极限 | $b_\perp\to0$ 回到直线基准，$z\to0$ 和 $\ell$ 扫描可解释 | 不解释横向依赖 |
| 外态 | 真实逐组态 $C_2L_g$、插入时间/源汇动量投影、多个 $t_{\rm sep}$，真空扣除及平台/多态拟合稳定 | 不把单配置 OPE 或未扣除的 raw loop 当矩阵元 |
| 统计 | 同索引重采样，报告 Ncfg、block、cov 条件数、SVD cut 和坏样本 | 降级为未验证 |
| 重整化 | 同几何软因子、流时间/表示/方案闭合，短距和 $\tau$ 窗口可重现 | 不称 renormalized TMD |
| CS/匹配 | 至少两个 $P_z$，核在阶数/尺度变化下稳定，$\alpha_s\to0$ 为单位核 | 只报告 scaffold |
| 连续/系统 | 多 $a$、$P_z$、$\tau$、$\ell$，必要时 $m_\pi,L$；外推协方差和误差带闭合 | 不报最终 PDF 曲线 |

只有在“非零 $b_\perp$ + 有限 $\ell$ + 真实核子胶子三点 + 同几何软/快度处理 +
相应胶子匹配 + 多尺度误差验证”全部具备时，才可以称为第一版完整链。否则使用
“梯度流胶子 quasi-PDF 原型”“带 staple 的裸准关联函数”或“接口/测试骨架”等准确名称。

## 常见错误与反模式

| 错误 | 物理/工程后果 | 修复 |
|---|---|---|
| 只有 $z$ 分离却称 TMD | 没有横向分辨率 | 实现并验证非零 $b_\perp$ staple |
| 非零 `b_perp` 但横向段未走链 | 输出形状看似正确，物理几何仍是错的 | 对每一段逐链计数，并用路径反向/幺正性测试 |
| 用 quasi-PDF 核处理 staple quasi-TMD | 几何和 rapidity 方案不匹配 | 保留胶子 TMD 的矩阵匹配接口 |
| 以梯度流“有限”代替 rapidity subtraction | UV 平滑不等于快度重整化 | 增加同几何 soft/rapidity 层 |
| 把逐组态 `C2*OPE` 与逐组态真空扣除混为一谈 | 单配置时恒等为零，或丢掉 $C_2$–$L_g$ 协方差 | 保存逐组态 $C_2L_g$，再以 ensemble 均值做 $\langle C_2L_g\rangle-\langle C_2\rangle\langle L_g\rangle$ |
| 单点 $\tau=3a^2$ | 无法估计流时间系统误差 | 扫 $\tau$，固定物理窗口 |
| 单个 $P_z$ 拟合 CS 核 | 斜率不可辨识 | 至少两个动量并做比值稳定性检查 |
| 忽略表示、转角、$\ell$ 或正负方向 | soft 因子和主算符不再同方案 | 元数据绑定全部几何参数 |
| 无依据地取实部或偶化 | 丢失 helicity/奇对称通道或掩盖边界效应 | 先做对称性推导与数值对照 |
| 固定规范无 Wilson 线冒充规范不变 TMD | 观测量依赖规范选择 | 仅作对照通道 |
| 随机场/demo 图当物理结果 | 没有核子态、系综和系统误差 | 降级为 smoke/shape 测试 |
| 把“文件存在”当作物理验证 | 只证明 IO，不证明方案闭合 | 按验证矩阵逐门给证据 |

## PyQCD 定向入口与命令

先核实当前 checkout 的签名，再调用，不凭技能文档创造不存在的 API：
以下相对路径命令从仓库根目录 `/root/PyQCD` 执行；若当前 shell 在
`/root/PyQCD/skills`，先切换目录。

- 流与算符：`pyqcd.renorm.wilson_flow`、`flow_action_density`、
  `staple_wilson_line`、`gluon_tmd_operator`、`tmd_matrix_elements_time`；
- 直线/对照算符：`pyqcd.operator.gluon_ope_operator_z0`、
  `gluon_ff_operator_z0`、`get_ope_lorentz_pairs`；
- 统计与外态：`pyqcd.analysis.run_disconnected_tmd_ratio`，以及
  `pyqcd-analysis` 规定的源时刻、重采样、SVD 和拟合流程；
- 提取与匹配：`quasi_tmd_pdf`、`quasi_pdf_gluon`、`cs_kernel_two_momentum`、
  `tmd_matching_hybrid`、`fit_hR_PDF_extrap_boot`；
- 现有 smoke/回归：
  `python examples/pyqcd/tmd_gradient_flow_demo.py`、
  `python examples/pyqcd/test9_gluon_tmd_nucleon.py --smoke`、
  `python examples/pyqcd/test9_verify.py [run_dir]`。

这些命令可以验证原型的入口、形状和数值有限性；它们全部通过仍不能跳过同几何软因子、
rapidity 方案和真实 TMD 三点的物理验收。

## 与其他技能配合

- **REQUIRED BACKGROUND:** `pyqcd-tmd-chain`：确认现有六步链和已验证接口边界。
- **REQUIRED SUB-SKILL:** `pyqcd-physics-correlator`：推导核子 2pt/3pt、Wick 图和传播子清单。
- **REQUIRED SUB-SKILL:** `pyqcd-analysis`：执行源平移、重采样、协方差拟合和系统误差。
- 需要传播子、蒸馏或顺序源时使用 `pyqcd-propagator`；IO、torch 和 MPI 使用
  `pyqcd-infra`；批量管线使用 `pyqcd-pipeline`；报告成文使用 `pyqcd-docs`。
