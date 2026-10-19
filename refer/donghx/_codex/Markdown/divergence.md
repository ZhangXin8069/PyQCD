# Gradient flow 中的线性发散与费米子归一化：闭合圈、规范固定直线、ratio 与 Ward 处方

本文针对本项目的四维 thin-link gradient-flow quark/gluon PDF 计算，解释 Wilson line 的线性发散、其与流时间的关系，以及用 Polyakov loop 或 Wilson loop 抽取线自能的原理和实际操作。公式采用自然单位，$\hbar=c=1$。

**2026-10-18 拟合图阅读导航：**第 25 节逐项解释 `flow_fits.pdf` 中的 `full/low08/low14`、`R/a=6..12/7..12`、红蓝模型、参考流时间以及空白子图。说明以最新已完成的 `line_mass_d_boot3000_20261018`（**1700 conf / 3000 配对 bootstrap**）的冻结代码和机器结果为准，并给出每个示例实际使用的 $\tau$ 列表。`full` 不是全部原始测量点，`low08/low14` 是**下限**，红蓝区分拟合模型而不是基本/伴随表示。第 24 节保留的是 2026-10-17 的历史版本；其先拟合 $-\log\langle W\rangle$ 的时间估计器与本次先拟合原始 $\langle W\rangle$ 的版本不能混用数值。本次只补充文档，不重跑拟合、不选定最终 $d$、不对 PDF 施加减除。

**2026-10-17 方法释义与尺度窗更新导航：**第 24 节详细解释当前图中的 **Wilson Cornell** 与 **Wilson relative**：二者先由 Wilson loop 的时间衰减提取势，前者再拟合空间距离依赖、后者在固定距离作跨 flow-time 差分。尤其注意，本文代码中的 relative **不是** $V(R)-V(R_{\rm ref})$。第 24.7–24.9 节记录最新 `line_mass_scale_windows_20261017` 的尺度条件、候选结果、图表及实现路径；它沿用 **1700 conf / 1000 配对 bootstrap**，没有改成 2000 bootstrap。第 23 节及更早日期的结果、固定 reference 和“下一步”保留为历史记录；当前尺度窗版本的具体定义以第 24 节及其冻结代码为准。本次仅更新说明，没有重跑拟合、改动 matching 或施加 PDF 线质量减除。

**2026-10-17 结果更新导航：**第 23 节记录 L32x64 Wilson loop / Polyakov loop 的实际拟合形式、统计契约、候选系数及拟合图片的绝对路径。区分 **883-conf / 2000-bootstrap 快照版**与 **1700-conf / 1000-bootstrap 全量自动汇总版**；后者并不是 1700-conf / 2000-bootstrap。Wilson 结果仍有有限 $T$、几何及模型系统误差；Polyakov 尚未得到可用于正式减除的可靠系数。当时仅核对已有产物并更新本文，没有重跑拟合或对 PDF 矩阵元施加减除。以下旧日期段落是历史记录：这两份快照以第 23 节为准，后续尺度窗分析见第 24 节。

**2026-10-15 更新导航：**第 16 节补充 Wilson loop、单个/成对 Polyakov loop、规范固定的开口 Wilson line 及跨流时间 double ratio；第 17 节说明用局域矩阵元归一化消去 $Z_\chi$；第 18 节推导矢量 Ward 处方；第 19 节给出 L32x64 分阶段测试计划。新增内容依据 `Pion_DA.pdf`、`Z_chi.pdf` 和 arXiv:2609.23379v1，不更改第 2 节的线质量符号。**本次只更新文档和设计测试，没有提交或执行新的数值任务。**

## 1. 先回答最核心的疑问

“$t_f=0$ 时发散”与“$t_f=0$ 对应目标结果”说的是不同对象：

- 前者指**未减除、未重整化的 Wilson-line 算符**在移去紫外调节器时的奇异行为。
- 后者指**先去除线自能、处理端点和场归一化，并固定重整化方案之后**得到的零流时间极限。

因此不能写成

$$
\text{物理量}=\lim_{t_f\to0}h^{\mathrm{GF,bare}}(t_f,z).
$$

应当理解为，示意地，

$$
\boxed{
h^{\overline{\mathrm{MS}}}(z,\mu)
=\lim_{t_f\to0}\lim_{a\to0}
\left[
C^{-1}(t_f,\mu)\,
e^{m_{\mathrm{line}}(t_f)|z|}\,
h^{\mathrm{GF,norm}}(a,t_f,z)
\right].
}
$$

这里的连续极限在**固定物理** $t_f,z,P_z$ 下进行；$h^{\mathrm{GF,norm}}$ 已包含该通道需要的场归一化，夸克例子为 ringed normalization。若存在算符混合，$C^{-1}$ 应理解为相应的系数矩阵，而不是保证对所有算符都能用一个标量。

这一式子是精确转换或受控极限的目标，不能理解为“乘上一圈公式就已经完全去除了所有阶发散”。此外，$h^{\overline{\mathrm{MS}}}$ 是有尺度、方案依赖的重整化 quasi 矩阵元，**还不是光锥 PDF**。本项目之后仍需要 $\lambda$ 重构、傅里叶变换、LaMET matching 和相应 power-correction 控制。[1,2]

正流时间的观测量也可以是定义良好的连续理论观测量；$t_f=0$ 并不是“物理”的唯一定义。我们取零流时间，是为了转换到所声明的未流化目标算符，而不是因为有限流时间的对象必然没有物理意义。[2,3]

## 2. 记号、单位与符号：不要把不同的“时间”混在一起

本文固定：

$$
\begin{aligned}
a&:\text{格距},\\
t_f&:\text{物理流时间，量纲为长度平方},\\
\tau&\equiv t_f/a^2:\text{无量纲格点流时间},\\
r_f&\equiv\sqrt{8t_f}:\text{流平滑半径},\\
L&\equiv|z|:\text{PDF 算符的直 Wilson-line 长度},\\
L_t&\equiv N_ta:\text{格点的物理时间方向长度},\\
T_{\mathrm{phys}}&\equiv1/L_t:\text{物理温度},\\
T_W&:\text{矩形 Wilson loop 的时间边长}.
\end{aligned}
$$

流时间 $t_f$ 不是算符的欧式时间坐标，也不是源汇间隔 $t_{\mathrm{sep}}$。

采用 SU(3) 生成元与 Casimir 约定

$$
\operatorname{Tr}(T_F^aT_F^b)=\frac12\delta^{ab},\qquad
T_R^aT_R^a=C_R\mathbf1,\qquad
C_F=\frac43,\quad C_A=3,
$$

$$
\alpha_s(\mu)=\frac{g^2(\mu)}{4\pi}.
$$

为了直观描述衰减，本文引入**正的线质量** $m_{\mathrm{line},R}$：

$$
h_R^{\mathrm{GF}}\sim e^{-m_{\mathrm{line},R}L}\times\text{端点系数}\times h_R^{\mathrm{ren}}.
$$

本项目已有 matching 使用**负的** $\delta m_R$，二者关系为

$$
\boxed{\delta m_R=-m_{\mathrm{line},R}.}
$$

因此，原有公式和本文完全一致：

$$
\mathcal O_R^{\mathrm{GF,norm}}(t_f,z)
=C_R^{\mathrm{op}}(t_f,\mu)e^{\delta m_R(t_f)L}
\mathcal O_R^{\overline{\mathrm{MS}}}(z,\mu)+\text{流修正},
$$

$$
\boxed{
\mathcal O_R^{\overline{\mathrm{MS}}}
=\frac{e^{-\delta m_R(t_f)L}}{C_R^{\mathrm{op}}(t_f,\mu)}
\mathcal O_R^{\mathrm{GF,norm}}+\text{流修正}.
}
$$

这里 $C_R^{\mathrm{op}}$ 是算符转换系数，不能与颜色 Casimir $C_R$ 混淆。本项目 quark/gluon 的已有 $\delta m<0$ convention 不应因引入正线质量而被改号。[1；本地依据见第 15 节]

## 3. 线性发散究竟是什么

### 3.1 Wilson line 是沿整条路径累积的紫外自能

取沿欧式单位向量 $v$、长度为 $L$ 的直线，

$$
W_R(L)=\mathcal P\exp\left[
ig\int_0^L ds\,v_\mu B_\mu^a(t_f,sv)T_R^a
\right],\qquad v^2=1.
$$

直 Wilson line 可以看成一个辅助静态颜色源沿路径传播。该颜色源周围有短距离规范场涨落，产生“每单位路径长度的自能”。它不是核子质量，也不是胶子获得了质量，更不是夸克的临界质量。[1]

在一条足够长的线的内部，短距离局域环境近似平移不变，因此每一小段贡献相同。其累积形式为

$$
\ln W_R\big|_{\mathrm{self}}
\sim-m_{\mathrm{line},R}L.
$$

重整化理论将这类贡献组织成一个沿线指数。这里用裸线展开说明局域自能；单独开口线的迹不被当作规范不变物理观测量。真正计算时，开口线与端点场一起构成规范不变算符，或者使用闭合圈。[1]

### 3.2 “线性”首先指对紫外 cutoff 的依赖

令 $\Lambda_{\mathrm{UV}}$ 为高动量 cutoff，则典型的静态自能积分为

$$
m_{\mathrm{line},R}\propto
g^2C_R\int^{\Lambda_{\mathrm{UV}}}
\frac{d^3k}{(2\pi)^3}\frac1{k^2}
\propto g^2C_R\Lambda_{\mathrm{UV}}.
$$

因为

$$
\int^{\Lambda_{\mathrm{UV}}}d^3k\,\frac1{k^2}
\propto\int_0^{\Lambda_{\mathrm{UV}}}dk,
$$

发散是 cutoff 的一次幂，而不是 $\ln\Lambda_{\mathrm{UV}}$，所以叫**线性发散**，也属于 power divergence。

在格点调节下，$\Lambda_{\mathrm{UV}}\sim1/a$，于是

$$
m_{\mathrm{line},R}^{\mathrm{lat}}
\sim\frac{c_R(g_0,\text{格点作用量})}{a},
\qquad
\ln h_R^{\mathrm{bare}}\big|_{\mathrm{line}}
\sim-c_R\frac{L}{a}.
$$

也存在“对长度 $L$ 线性”的累积，但不要误以为它表示“对流时间 $t_f$ 线性”。在 gradient flow 调节下，对应的是 $1/\sqrt{t_f}$ 的奇异项。[1]

## 4. Gradient flow 怎样调节这项发散：一圈积分推导

这一节说明系数、颜色因子和 $\sqrt{8t_f}$ 的来源。采用四维 Euclidean flow；在线性化及方便的规范下，

$$
B_\mu(t_f,k)=e^{-t_fk^2}A_\mu(k)+O(g).
$$

因此两个流场的树级传播子含有

$$
\langle B_\mu^a(t_f,k)B_\nu^b(t_f,-k)\rangle
=\delta^{ab}\delta_{\mu\nu}\frac{e^{-2t_fk^2}}{k^2}
$$

这一 UV 阻尼。更一般的规范项和端点项需要在完整 matching 中处理；下面只抽取长直线内部的 leading line mass。[1,2]

展开 Wilson line，颜色缩并给出 $C_R$，指数中 leading 自能为

$$
\ln W_R\big|_{g^2}
=-\frac{g^2C_R}{2}
\int_0^L ds\int_0^L ds'\,
D_{t_f}((s-s')v).
$$

当 $L\gg r_f$ 时，远离端点的部分占主导：

$$
\int_0^L ds\int_0^L ds'\,D_{t_f}((s-s')v)
=L\int_{-\infty}^{\infty}du\,D_{t_f}(uv)
+\text{非周长项}.
$$

沿线坐标的积分产生

$$
\int_{-\infty}^{\infty}du\,e^{ik_{\parallel}u}
=2\pi\delta(k_{\parallel}),
$$

把四维积分化成垂直于路径的三维积分：

$$
m_{\mathrm{line},R}^{(1)}(t_f)
=\frac{g^2C_R}{2}
\int\frac{d^3k_\perp}{(2\pi)^3}
\frac{e^{-2t_fk_\perp^2}}{k_\perp^2}.
$$

利用球坐标，

$$
\begin{aligned}
I(t_f)
&=\frac{4\pi}{(2\pi)^3}
\int_0^\infty dk\,e^{-2t_fk^2}\\
&=\frac1{2\pi^2}\frac{\sqrt\pi}{2\sqrt{2t_f}}
=\frac1{4\pi^{3/2}\sqrt{2t_f}},
\end{aligned}
$$

于是

$$
\boxed{
m_{\mathrm{line},R}^{(1)}(t_f)
=\frac{\alpha_sC_R}{4\pi}\sqrt{\frac{2\pi}{t_f}}
=\frac{\alpha_sC_R}{\sqrt\pi\,r_f}.
}
$$

对应本项目的符号是

$$
\boxed{
\delta m_R^{(1)}(t_f)
=-\frac{\alpha_sC_R}{4\pi}\sqrt{\frac{2\pi}{t_f}}.
}
$$

这与 Brambilla–Wang 的 Eq. (4.15) 一致：基本表示用 $C_F$，伴随表示用 $C_A$。有限 $t_f$ 把 $1/a$ 型高动量敏感性替换为受控的 $1/r_f$ 型线自能，但**并没有把线自能设为零**。[1]

若定义

$$
m_{\mathrm{line},R}(t_f)=\frac{d_R}{r_f},
$$

则在本流定义与一圈处方下

$$
\boxed{d_R^{(1)}=\frac{\alpha_sC_R}{\sqrt\pi}.}
$$

严格地说，$d_R$ 可随 $r_f$ 通过耦合常数跑动、高阶修正和有限格距效应而变化。“在一段流时间内用常数 $d_R$ 拟合”是需要检验的有效模型，不是所有尺度上的恒等式。

### 4.1 为什么短距离点不能随意使用长线公式

上述热流传播子在位置空间为

$$
D_{t_f}(u)=\frac{1-e^{-u^2/(8t_f)}}{4\pi^2u^2},
\qquad
D_{t_f}(0)=\frac1{32\pi^2t_f}.
$$

直线自能的有限长度表达式为

$$
\ln W_R\big|_{g^2}
=-g^2C_R\int_0^Ldu\,(L-u)D_{t_f}(u).
$$

在 $L\gg r_f$ 时分离出 $-m_{\mathrm{line},R}L$；在 $L\ll r_f$ 时却有

$$
\ln W_R\big|_{g^2}
\simeq-\frac{\alpha_sC_R}{16\pi}\frac{L^2}{t_f}.
$$

因此长线的 $L/r_f$ 渐近项不能代替所有 $L/r_f$ 下的完整转换。在 $z=0$，路径收缩为单位矩阵，没有这一线长度项；局域算符仍可能有自己的端点、混合或短距离奇异性。尤其不能用非局域长线公式替代局域 $z=0$ 的完整匹配。[1；有限长度积分为本文据其树级热核推导]

## 5. 为什么零流时间会“发散”，裸矩阵元却可能趋于零

固定非零 $L$，在连续流调节下

$$
m_{\mathrm{line},R}(t_f)\sim\frac{d_R}{\sqrt{8t_f}}
\xrightarrow[t_f\to0]{}+\infty.
$$

如果只看 line-mass 因子，

$$
h_R^{\mathrm{GF,bare}}(t_f,z)
\sim\exp\left[-\frac{d_RL}{\sqrt{8t_f}}\right]
h_R^{\mathrm{finite}}(z)
\longrightarrow0,
$$

而它的逆转换因子

$$
Z_{\mathrm{line},R}
=\exp\left[+\frac{d_RL}{\sqrt{8t_f}}\right]
\longrightarrow\infty.
$$

所以“有发散”并不要求可见的裸矩阵元一定向 $+\infty$ 增长。发散的可以是线质量、$\ln h$ 的线自能贡献以及重整化因子。裸量的指数衰减同样说明它没有所需要的有限非平凡未调节极限。

把指数在固定阶展开，

$$
e^{-m_{\mathrm{line},R}L}
=1-m_{\mathrm{line},R}L+\cdots,
$$

其中一圈项本身向负无穷增大；这意味着固定阶展开在 $m_{\mathrm{line},R}L\gg1$ 时失效，不能据此推断精确裸量变为负无穷。指数与固定阶展开的适用范围要分开。[1]

### 5.1 有限格距的 $t_f=0$ 并不是无调节器的连续理论

在固定非零格距 $a$ 上，$t_f=0$ 返回原始 thin links，规范场和有限路径乘积仍然是有限的。格点 cutoff 仍在，线质量通常达到 $O(1/a)$ 的有限值。

因此，固定 $a$ 时不能把连续表达式 $d_R/\sqrt{8t_f}$ 一直用到 $t_f=0$。其受控区域需要

$$
a\ll r_f,\qquad \frac{a^2}{t_f}\ll1.
$$

当 $r_f\lesssim a$，高动量阻尼与格点 cutoff 的作用发生交叉，连续小流时间公式不再足以描述数据。精确格点核和线质量在该区域需要另算或做 tree-level improvement。

这就是“先在固定物理正流时间做连续极限，再研究零流时间”与“在一个格距上直接令流时间归零”不能混为一谈的原因。[1–3]

## 6. 正确的重整化、matching 与极限顺序

### 6.1 线发散只是重整化的一部分

夸克双局域算符的示意关系为

$$
h_\Gamma^{\overline{\mathrm{MS}}}(z,\mu)
=C_\psi^{-1}(t_f,\mu)e^{m_{\mathrm{line},F}(t_f)L}
Z_\chi(t_f)h_\Gamma^{\mathrm{GF,bare}}(t_f,z)
+\text{流修正}.
$$

两端 ringed fermion 各含 $Z_\chi^{1/2}$，所以整个 bilocal 乘一份 $Z_\chi$。$Z_\chi$ 处理流费米子场归一化，不替代 Wilson-line mass subtraction。[1,4]

胶子双局域算符使用伴随线质量，两个场强端点还需要依照相对于 Wilson-line 方向的 tensor sector 使用对应转换系数。线自能只依赖路径表示和局域 UV 环境，不依赖它被投影成 unpolarized 还是 helicity；这不代表两通道的完整 matching 相同。[1]

### 6.2 把“去流”与“PDF matching”分开

本项目应明确区分

$$
\begin{aligned}
&\text{有限流时间的裸 C3/C2 与裸矩阵元}\\
&\quad\xrightarrow{\;Z_\chi\text{ 或相应场归一化};\;m_{\mathrm{line}};\;C^{\mathrm{op}}\;}
\text{声明方案中的 quasi bilocal}\\
&\quad\xrightarrow{\;\text{受控小流时间与格距处理}\;}
\text{目标未流化 quasi 矩阵元}\\
&\quad\xrightarrow{\;\lambda\text{ 重构与傅里叶变换}\;}
\text{quasi PDF}\\
&\quad\xrightarrow{\;\text{LaMET kernel、混合及 power corrections}\;}
\text{光锥 PDF}.
\end{aligned}
$$

“$t_f=0$ 后物理”至多是对目标未流化算符的口语表达，并不意味着所有后续 matching 或连续极限已经完成。[1；本地流程见第 15 节]

### 6.3 为什么当前单格距外推必须标为诊断

受控小流时间窗通常同时要求

$$
a\ll r_f\ll L\quad(L\ne0),\qquad
\sqrt{t_f}P_z\ll1,\qquad
t_f\Lambda_{\mathrm{QCD}}^2\ll1.
$$

残余依赖可能包含

$$
\frac{a^2}{t_f},\qquad
\frac{t_f}{L^2},\qquad
t_fP_z^2,\qquad
t_f\Lambda_{\mathrm{QCD}}^2,
$$

其系数与算符、离散化和改善处方有关。单格距可以做 window/model sensitivity 研究，但不能把 $a^2/t_f$ 项的数据不足或模型退化误认为已受控的连续结果。

此外，一圈 line mass 的缺失高阶项一般为

$$
\Delta m_{\mathrm{line},R}
=O\!\left(\frac{\alpha_s^2}{r_f}\right),
$$

其指数误差随 $L/r_f$ 放大。仅以一圈转换后“看上去平”的数据做多项式外推，不能证明发散已在所有阶被去除；需检查尺度、阶数、跑动/对数重求和、非微扰方案和极限控制。[1,5]

## 7. 为什么 Polyakov loop 和 Wilson loop 能测同一个线自能

关键不是“它们的物理内容等于 PDF”，而是**沿线的 leading UV 自能局域且具有可迁移性**。

短距离涨落只看到路径的一小段和颜色表示。在 UV 分离成立时，它不知道线的端点接了核子中的夸克场、胶子场强，还是路径最终闭合。因而对同样的表示和调节方案，leading 自能具有同一个每单位长度系数。[1,5]

需要匹配的条件包括：

- 同一种颜色表示 $R$。
- 同一种 flow 定义、同一物理流时间和 compatible 的原始规范作用量。
- 同样的 thin-link 起点；不能把 HYP+flow 的线质量直接给 thin+flow。
- 对有限格距效应进行检查；连续 Euclidean 的方向等价不能代替有限格距的方向验证。
- 不把端点、拐角/cusp、物理静态势或热自由能也误算为线自能。

### 7.1 Quark 与 gluon 的表示不同

夸克 bilocal 的线是基本表示 $F$。胶子代码常写成

$$
\operatorname{Tr}\!\left[
F(x+z)W_F(x+z,x)F(x)W_F^\dagger(x+z,x)
\right].
$$

利用

$$
W_FT^bW_F^\dagger=T^a(W_A)^{ab},
$$

得到

$$
\operatorname{Tr}[F(x+z)W_FF(x)W_F^\dagger]
=\frac12F^a(x+z)(W_A)^{ab}F^b(x),
$$

即一条伴随 Wilson line。**看到 $W_F$ 和 $W_F^\dagger$ 不能把胶子线质量取为两倍基本表示线质量。** 一圈胶子系数是 $C_A$，不是 $2C_F$。[1]

一圈有

$$
\frac{m_{\mathrm{line},A}^{(1)}}{m_{\mathrm{line},F}^{(1)}}
=\frac{C_A}{C_F}=\frac94.
$$

但在非微扰拟合中，不应未经验证就把精确 Casimir scaling 当作恒等式；基本、伴随表示应各自测量或明确给出近似误差。[1,6]

## 8. Polyakov loop：从静态源自由能中抽取线质量

### 8.1 定义与长度

时间方向 Polyakov holonomy 为

$$
P_R(\boldsymbol x;t_f)
=\prod_{n_t=0}^{N_t-1}
V_4^{(R)}(t_f,\boldsymbol x,n_t),
$$

$$
\mathcal L_R(t_f)
=\left\langle
\frac1{d_R^{\mathrm{rep}}N_s^3}
\sum_{\boldsymbol x}\operatorname{Tr}_R P_R(\boldsymbol x;t_f)
\right\rangle.
$$

这里 $d_R^{\mathrm{rep}}$ 是表示维数，基本为 3、伴随为 8，**不是**线自能拟合参数 $d_R$。整条圈的长度是

$$
L_t=N_ta=\frac1{T_{\mathrm{phys}}}.
$$

对于可定义实自由能的 ensemble 平均，

$$
\mathcal L_R(t_f)
=e^{-m_{\mathrm{line},R}(t_f)L_t}
\mathcal L_R^{\mathrm{sub}}(t_f),
$$

$$
\boxed{
\ln\mathcal L_R(t_f)
=-\frac{d_RL_t}{\sqrt{8t_f}}
\,+\ln\mathcal L_R^{\mathrm{sub}}(t_f).
}
$$

或者用静态源自由能描述，

$$
F_{Q,R}^{\mathrm{flow}}(t_f,T_{\mathrm{phys}})
\equiv-T_{\mathrm{phys}}\ln\mathcal L_R(t_f)
=m_{\mathrm{line},R}(t_f)+F_{Q,R}^{\mathrm{sub}}.
$$

这解释了它为什么能携带 line-mass 信息：闭合圈虽然没有开口端点，但保留了静态源沿整条时间路径的自能。[5,6]

### 8.2 本项目对应的格点单位公式

由于 $t_f=a^2\tau$，

$$
\boxed{
\ln\mathcal L_R(\tau)
=-d_R\frac{N_t}{\sqrt{8\tau}}+\text{有限部分与修正}.
}
$$

若横坐标为

$$
X=\frac{N_t}{\sqrt{8\tau}},
$$

且有限部分在窗口内可近似常数，那么 $\ln\mathcal L_R$ 对 $X$ 的 leading 斜率是 $-d_R$。这不是无需有限部分模型的严格直线拟合。

归一化迹乘一个与 $t_f$ 无关的常数只平移对数的截距，不改变 leading 斜率；但改变局域字符构造、先平均再取模等非线性操作会改变观测量本身。

### 8.3 文献 arXiv:2607.13024v1 是怎样做的

该文研究的是 quenched 理论的 chromoelectric correlator 和 P-wave quarkonium decay，不是本项目的 nucleon PDF。其 Supplemental Material SM1 C，Eqs. (SM17)–(SM19)，用伴随 Polyakov loop 拟合

$$
\mathcal L_8
=\exp\left[-\frac{d}{\sqrt{8t_f}\,T_{\mathrm{phys}}}\right]
\mathcal L_8^{\mathrm{sub}},
$$

$$
\ln\mathcal L_8
=C_0-\frac{d}{\sqrt{8t_f}\,T_{\mathrm{phys}}}
+m_0t_fT_{\mathrm{phys}}^2
+P_n(\sqrt{8t_f}).
$$

其中多项式的常数和二次项由 $C_0$ 与 $m_0$ 承担，其他系数的单位必须随所用的长度单位处理。这个式子中的 $t_f$ 是物理流时间；不是可以不换单位就代入的本项目无量纲 $\tau$。

作者为此额外生成有限温度数据，在多个格距、多个 $N_t$ 上测量，每个参数集合取 1000 个组态，以 Bayesian 模型和多项式阶数/先验/窗口变动处理有限部分，最后做连续外推。其报告 $d=0.315(6)$。[5]

该数值是该文**特定 quenched ensemble、测量和拟合处方的结果**，不应直接替代我们 dynamical-clover ensemble 的 $d_A$。该文的经验窗口也不是普遍定理；我们的 $a/r_f$、流步长、信号质量与有限体积必须单独检验。

### 8.4 为什么单个 flow time 或一个低温 ensemble 不一定足够

在一个流时间上，$-m_{\mathrm{line}}L_t$ 与未知有限自由能无法独立分开。需要依赖于 $1/r_f$ 的形状、多个尺度或额外的重整化条件。

即使用多个 flow time，如果有限部分变化与 $1/r_f$ 项高度相关，也会产生参数退化。低温大 $N_t$ 的圈还可能很小；伴随表示不受基本中心相位影响，却仍可能有严重的信噪比问题。

基本表示的中心对称性也需要单独处理：在纯规范禁闭相中 ensemble 平均可能因中心对称性为零；此时“取模得到非零再拟合”不是原观测量。本项目有动力学夸克，不等于该纯规范极限，但仍须保留并检查复相位。[5,6]

## 9. Wilson loop：从周长项或静态势中抽取线质量

### 9.1 矩形圈的周长规律

取表示 $R$、空间边长 $R_W$、时间边长 $T_W$ 的矩形圈，

$$
W_R(R_W,T_W;t_f)
=\left\langle\frac1{d_R^{\mathrm{rep}}}
\operatorname{ReTr}_R\mathcal P\prod_{C(R_W,T_W)}V(t_f)\right\rangle.
$$

总周长为

$$
\ell_C=2(R_W+T_W).
$$

在流半径小于各边长的区域，leading 线自能具有

$$
\ln W_R
=-m_{\mathrm{line},R}\ell_C
+\ln W_R^{\mathrm{sub}}\,+\text{cusp 与流修正}.
$$

这说明 UV 线自能随**周长**累积；物理禁闭势在适当大圈区间可产生**面积**依赖。两者不同，不能把全部 $-\ln W$ 都称为发散。[1,6]

### 9.2 用静态势分离

在足够大的 $T_W$ 上，

$$
W_R(R_W,T_W;t_f)
\simeq A_R(R_W,t_f)
e^{-V_R^{\mathrm{flow}}(R_W;t_f)T_W},
$$

$$
V_R^{\mathrm{flow}}(R_W;t_f)
=2m_{\mathrm{line},R}(t_f)+V_R^{\mathrm{sub}}(R_W)
+\text{流/格距修正}.
$$

因子 2 来自时间方向上的静态源与反源两条长边。空间短边的周长项在 $T_W\to\infty$ 的势提取中进入振幅，而不是再增加一个势中的线质量。

两个流时间的势差，在 $r_f\ll R_W$ 且长时间平台已验证时，可近似给出

$$
V_R^{\mathrm{flow}}(R_W;t_{f,1})
-V_R^{\mathrm{flow}}(R_W;t_{f,2})
\simeq2\left[m_{\mathrm{line},R}(t_{f,1})
-m_{\mathrm{line},R}(t_{f,2})\right].
$$

该差在合适 $R_W$ 窗口内近似为常数，是很有用的验证。对基本表示势，在受控区间还可采用

$$
V_F^{\mathrm{flow}}(R_W;t_f)
=2m_{\mathrm{line},F}(t_f)+V_0-\frac{\alpha_V}{R_W}+\sigma R_W+\cdots,
$$

但不能把这个模型自动搬给所有表示、所有距离；伴随源可能发生屏蔽/string breaking。

在单个流时间、单个格距上，$2m_{\mathrm{line}}$ 与 $V_0$ 完全退化。要提取绝对线质量，还需要跨流时间的模型和/或明确的有限重整化条件，例如在某个 $R_0$ 规定 $V^{\mathrm{sub}}(R_0)$。[6]

### 9.3 哪些圈组合反而不能提取线质量

Creutz ratio 的周长项相消：

$$
\chi(R_W,T_W)
=-\ln\frac{W(R_W+a,T_W+a)W(R_W,T_W)}
{W(R_W+a,T_W)W(R_W,T_W+a)}.
$$

因此它适合研究力或面积贡献，却不能单独恢复已相消的绝对线自能。同理，静态力

$$
\frac{dV_R^{\mathrm{flow}}}{dR_W}
$$

去掉了 $R_W$ 无关的 $2m_{\mathrm{line},R}$，不能仅从力确定线质量的加性常数。[据第 9.1–9.2 节直接推导]

## 10. 非微扰减除不等于自动完成 $\overline{\mathrm{MS}}$ matching

假设 Polyakov/Wilson-loop 拟合给出了所声明方案中的 $m_R^{\mathrm{NP}}$，则可构造

$$
h_R^{\mathrm{sub,NP}}=e^{m_R^{\mathrm{NP}}L}h_R^{\mathrm{GF,norm}}.
$$

原来的微扰 GF-to-$\overline{\mathrm{MS}}$ 关系为

$$
h_R^{\overline{\mathrm{MS}}}
=C^{-1}e^{m_R^{\mathrm{PT}}L}h_R^{\mathrm{GF,norm}}.
$$

所以二者的正确组合应为

$$
\boxed{
h_R^{\overline{\mathrm{MS}}}
=C^{-1}
e^{(m_R^{\mathrm{PT}}-m_R^{\mathrm{NP}})L}
h_R^{\mathrm{sub,NP}},
}
$$

而不是在 $h_R^{\mathrm{sub,NP}}$ 上再完整乘一次 $e^{m_R^{\mathrm{PT}}L}$。实际有限转换还需与精度、端点定义、线质量的有限部分和方案一致地推导。[1；上述组合为代数推导]

线质量可以有有限方案变化

$$
m_R\longrightarrow m_R+\Delta m_R^{\mathrm{finite}},
\qquad
h_R^{\mathrm{sub}}\longrightarrow
e^{\Delta m_R^{\mathrm{finite}}L}h_R^{\mathrm{sub}}.
$$

只确定 leading $1/r_f$ 系数并不自动固定这个有限常数；辅助静态源还涉及 $O(\Lambda_{\mathrm{QCD}})$ 的 residual-mass/renormalon 处方问题。不同方案可以都去掉 leading 发散，但给出不同的有限矩阵元，需要有限转换才能相互比较。[1]

维数正规化中的 $\overline{\mathrm{MS}}$ 不保留 cutoff 型的线性幂发散，并不意味着格点/flow 方案中没有线自能，也不意味着端点对数或有限 matching 可以不计算。[1]

## 11. 本项目 L32x96 Polyakov 数据：已确认与待完成

以下是读取文件得到的状态，**不是本次重新测量或实时队列审计**。

测量代码：

```
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/polyakov_L32x96/src/polyakov_flow.cpp
```

冻结的 100-conf 数据包：

```
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/polyakov_L32x96/runs/production100_20261011_113518/
```

保存的审计：

```
validation/final_audit.json
```

该审计时间为 2026-10-11 16:57:11（Asia/Shanghai），字段为：

```
status = PASS
validated_conf_count = 100
missing_or_invalid_count = 0
validated_conf_tau_pairs = 2100
```

这是 100 个组态乘 21 个流时间检查点，**包含** $\tau=0$ 的原场诊断点，不能说成 2100 个正流时间样本。

### 11.1 当前测量定义

本测量的格距记录为 $a=0.0775\,\mathrm{fm}$，$N_t=96$，规范场数组轴为 $(t,z,y,x)$，链接方向为 $(x,y,z,t)=(0,1,2,3)$。

时间圈定义为

$$
P(\boldsymbol x)=V_t(\boldsymbol x,0)V_t(\boldsymbol x,1)\cdots V_t(\boldsymbol x,95).
$$

其基本与伴随字符为

$$
p_3(\boldsymbol x)=\frac{\operatorname{Tr}_F P(\boldsymbol x)}3,
$$

$$
\boxed{
p_8(\boldsymbol x)
=\frac{|\operatorname{Tr}_F P(\boldsymbol x)|^2-1}{8}
=\frac{9|p_3(\boldsymbol x)|^2-1}{8}.
}
$$

**先在每个空间位置构造伴随字符，再空间平均。** 不能换成

$$
\frac{9|\overline p_3|^2-1}{8}.
$$

$p_3$ 保存复数，$p_8$ 保存有符号实数；不对 $p_8$ 取绝对值。规范场周期边界，时间圈没有费米子的反周期负号。这些约定已经由现有实现及验证固定，不应在拟合时静默改变。

### 11.2 流时间与格式

生产测量采用四维 Wilson flow、原始 thin gauge、RK3、$\epsilon=0.01$，保存

$$
\tau\in\{0,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,
0.55,0.60,0.65,1.00,1.40,1.80,2.20,2.60,3.00,3.40,3.80\}.
$$

数据布局为

```
data/epsilon0.010/conf<conf>/tau<tag>/p3.npy
data/epsilon0.010/conf<conf>/tau<tag>/p8.npy
data/epsilon0.010/conf<conf>/tau<tag>/metadata.json
data/epsilon0.010/conf<conf>/complete.json
```

局域数组轴为 $(z,y,x)=(32,32,32)$；$p_3$ 为 complex128，$p_8$ 为 float64。metadata 保存空间平均值、步数和输入/源代码身份。

现有代码明确记录：未进行 ensemble 平均、对数变换、$d$ 拟合、line-mass subtraction 或 matching。**数据计算完成不等于 $d$ 已确定。**

### 11.3 当前低温几何的含义

用记录的 $a$，时间长度约为

$$
L_t=96\times0.0775\,\mathrm{fm}=7.44\,\mathrm{fm},
\qquad
T_{\mathrm{phys}}\simeq26.5\,\mathrm{MeV}.
$$

这是低温长时间圈，可能有很强的指数抑制。当前格距属于项目输入记录，本说明未重新做格距尺度设定。是否存在足够精确的 ensemble 信号，需要下一步统计分析；不能仅凭“100 个组态都通过文件检查”断言可以稳定拟合 $d_A$。

## 12. 下一步如何实际拟合：建议操作，不是已完成结果

### 12.1 建立逐组态、跨 flow time 的对齐数据

读取冻结包的组态列表和完成标记，构建

$$
p_{R,c}(\tau)
=\frac1{N_s^3}\sum_{\boldsymbol x}p_{R,c}(\boldsymbol x;\tau),
$$

$$
\widehat{\mathcal L}_R(\tau)
=\frac1{N_{\mathrm{conf}}}\sum_c p_{R,c}(\tau).
$$

同一个组态的不同流时间必须保持配对；空间格点不是 32768 个独立 ensemble 样本。先检查组态序列自相关，必要时 block/bin 后再 bootstrap。

### 12.2 每个 bootstrap 先 ensemble 平均，后取对数

若第 $b$ 个 bootstrap 索引为 $I_b$，则

$$
\widehat{\mathcal L}_R^{(b)}(\tau)
=\frac1{|I_b|}\sum_{c\in I_b}p_{R,c}(\tau),
$$

$$
y_R^{(b)}(\tau)
=\ln\widehat{\mathcal L}_R^{(b)}(\tau).
$$

所有 flow time 共用 $I_b$，以保留跨流时间协方差。

伴随表示采用实对数拟合前，ensemble 平均及其样本分布必须足够远离零。如果存在非正均值或大量样本跨零，**不能**用取绝对值、截成很小的正数或静默丢弃样本解决。

可选的明确策略是：增加统计；在取对数前定义并固定可用窗口；或直接对原始平均量拟合

$$
\mathcal L_R^{\mathrm{model}}(\tau)
=\exp\left[-d_RX(\tau)+\text{有限部分模型}\right]
$$

并检验正均值模型是否被数据支持。原空间拟合可以处理有限统计下的负**测量值**，但不能把缺乏信号的数据变成可靠的 $d$。

基本表示还须检查虚部、中心相位与所用 ensemble 对称性；不允许未经定义直接取模。

### 12.3 先固定全局流时间窗，再比较模型

对 Polyakov 测量，应至少检查

$$
a\ll r_f\ll L_t,\qquad
r_f\ll L_s,
$$

以及用于微扰解释时的 $r_f\Lambda_{\mathrm{QCD}}$、步长误差和 finite-flow distortion。

Polyakov 自身不包含 PDF 的 $P_z$ 或空间分离 $z$，所以测量 $d$ 时不应加入不存在的 $\sqrt{t_f}P_z$ 条件；**把 $d$ 用回 PDF 后**才需要 PDF 的全部尺度窗。

在本单格距数据中，$\tau=0$ 只作原场诊断，不参加 $1/\sqrt{\tau}$ 连续流公式拟合。很小 $\tau$ 的点也可能强受 cutoff 影响；应画出全部点并标记窗口，而不是为了得到某个斜率逐点删选。

一种量纲清楚的试探性参数化是用

$$
x\equiv\frac{r_f}{L_t}=\frac{\sqrt{8\tau}}{N_t},
$$

$$
\ln\mathcal L_R
=-\frac{d_R}{x}+c_0+c_1x+c_2x^2+\cdots.
$$

也可考虑耦合跑动或显式格距修正，例如 line mass 中的 $a^2/t_f$ 效应；但单格距数据往往不能同时稳定确定很多项。以上是**待检验模型**，不是保证正整数幂多项式足够的理论定理。

建议依次做低参数相关拟合、全局窗口变动、模型阶数变动和先验敏感性检查，再选择 AIC/model averaging 或 Bayesian 模型平均。拟合接受条件不能用于掩盖零信号或无法辨识的参数。

### 12.4 需要哪些交叉验证

- 比较 $\epsilon=0.01$ 与 $0.005$ 的 pilot，确认拟合误差没有被积分误差主导。
- 在有限部分减除后检查残余 flow dependence，不只检查 $\chi^2$。
- 有条件时改变 $N_t$，验证 leading $d_R$ 不携带虚假的温度依赖。
- 增加格距，以固定物理流时间比较，检验 cutoff dependence。
- 使用 Wilson-loop 静态势差验证是否得到相同的每单位长度自能。
- 检查基本与伴随表示；不要未经验证用 $9/4$ 强制关联非微扰结果。

**待确认：**当前 100-conf 的低温 $\mathcal L_8$ 是否有适合上述拟合的正信号和稳定窗口；$d_A$ 是否能在单格距上可靠分离；有限 mass scheme 怎样固定；该方案到现有 GF-to-$\overline{\mathrm{MS}}$ 的有限转换；以及方向/离散化迁移误差。

## 13. 将拟合出的线质量用回 PDF，误差如何传播

若得到与 PDF 同组态配对的 bootstrap 线质量样本 $d_R^{(b)}$，可构造

$$
h_R^{\mathrm{sub},(b)}(\tau,z)
=\exp\left[
d_R^{(b)}\frac{|z|/a}{\sqrt{8\tau}}
\right]h_R^{\mathrm{GF,norm},(b)}(\tau,z).
$$

如闭合圈和 PDF 使用不同组态集，不能简单把“第 $b$ 个样本”按编号假配对；要先声明联合采样或独立误差处方。部分重叠的集合还需保持共有组态的相关性。

当只考虑 $d_R$ 的小误差时，

$$
\frac{\sigma_h}{|h|}\bigg|_{d}
\simeq\frac{L}{r_f}\sigma_d.
$$

更一般地，

$$
\operatorname{Var}(\ln h^{\mathrm{sub}})
=\operatorname{Var}(\ln h^{\mathrm{GF,norm}})
+\left(\frac{L}{r_f}\right)^2\operatorname{Var}(d_R)
+2\frac{L}{r_f}\operatorname{Cov}(\ln h^{\mathrm{GF,norm}},d_R),
$$

其中对数形式只用于非零、符号合适的示意；实际复矩阵元或过零点应直接传播 bootstrap 样本。

如果转换因子 $Z$ 被当作确定常数，

$$
\bar h'=Z\bar h,\qquad \sigma_{h'}=|Z|\sigma_h,
$$

因此绝对误差会被放大，但相对误差本身不因这个确定性乘法而改善或恶化。加入 $d_R$ 的测量误差、scale/order uncertainty 后，才会有新的相对不确定度。[据转换公式直接推导]

同一 $d_R$ 作用于很多 $z$、动量和通道，会产生强相关系统误差。不能把每个点的 $d_R$ 误差当作独立误差添加。

### 13.1 C3/C2 为什么不自动取消线质量

本流程的 C2 没有 C3 中的非局域插入 Wilson line，因此取 C3/C2 消掉外态 overlap，不会消掉插入的 $e^{-m_{\mathrm{line}}L}$。同样，除以 $z=0$ 的局域矩阵元通常也不消去非零 $z$ 的这段路径长度因子。

某些 specially defined reduced ratio 使用同样线长度的分子分母，确实可取消共同自能；那是一个**另外声明的重整化/ratio scheme**，其 matching 也要相应改变，不能与现有完整 line-mass subtraction 再叠加。

## 14. 简明核对表：哪些结论不能随意修改

1. 本项目用 $\delta m_R=-m_{\mathrm{line},R}<0$；逆转换乘 $e^{-\delta m_RL}$。
2. “线性发散”是 leading cutoff 幂发散；flow 中为 $1/\sqrt{t_f}$，不是 $t_f$ 的线性项。
3. 固定有限 $a$ 的 $t_f=0$ 原场是有限的；移去调节器后的裸算符与它不是一个极限。
4. 零流时间目标是**重整化后**的算符，不是直接外推裸 C3/C2。
5. quark 用基本线；gluon 色迹形式中的 $W_F, W_F^\dagger$ 合起来是一条伴随线。
6. Polyakov loop 使用 $L_t=N_ta$；不加费米子反周期负号。
7. SU(3) 伴随字符必须局域构造后再平均；保留有符号 $p_8$ 和复数 $p_3$。
8. 先 ensemble 平均再取对数，不是平均逐组态/逐格点对数；跨流时间用共同重采样索引。
9. 不用 $|p_8|$、clipping 或删掉负样本来制造可拟合信号。
10. 从圈中提取 $d_R$ 必须处理有限自由能/势、窗口和有限格距；不能把全部 $-\ln W$ 当发散。
11. 去除 leading line mass 不等于固定有限方案，更不等于自动完成端点、ringed normalization 和 LaMET。
12. 非微扰、ratio/hybrid 与微扰 line-mass subtraction 不能对同一贡献重复施加。
13. 文献的 quenched $d=0.315(6)$ 不是我们的 dynamical ensemble 的实测值。
14. 当前 100-conf Polyakov 文件审计 PASS 不等于已有稳定的 $d$ 拟合。
15. 当前单格距的小流时间外推应保留诊断标签和待确认限制，不升级为受控连续物理 PDF。
16. 单个 Polyakov loop 的路径长度为 $L_t$；成对绕时线的自能总长度为 $2L_t$；静态势含 $2m_{\mathrm{line},R}$，三者不能漏掉或重复因子 2。
17. 开口 Wilson line 的迹一般不是规范不变观测量；指定规范后的有限部分不能直接当作规范不变 PDF 或 $\overline{\mathrm{MS}}$ 矩阵元。
18. $h_\Gamma(z,t_f)/h_\Gamma(0,t_f)$ 消去共同的 $Z_\chi$，不消去非零 $z$ 的线自能，也不保留绝对局域矩阵元归一化。
19. Ward 处方是 $\mathring Z_\chi^V=\zeta_V Z_V/R_V$，不是默认 $Z_V/R_V$，更不是证明 $Z_\chi=1$。
20. $\mathring Z_\chi^V$ 与动能定义在有限 $a,t_f$ 不必相等；论文在精确 matching 下给出的连续极限之比为 $1+O(t_f)$。
21. 同一个费米子场归一化可用于不同极化，但局域 $\zeta_\Gamma$、非局域转换、LaMET kernel 是不同层次；不能相互替代。
22. L32x64 的新 C3 测试沿用用户指定的方案 B；旧 A 结果只能明确标作控制，不能冒充 B。已有 B 窗口验证仅覆盖其测试子集。

## 15. 依据、现有文件与本次工作边界

### 15.1 主要文献

[1] N. Brambilla and X.-P. Wang, *Off-lightcone Wilson-line operators in gradient flow*, JHEP 06 (2024) 210, arXiv:2312.05032v3。重点为 Secs. 2–3 的辅助场重整化、小流时间关系，Sec. 4.1 / Eq. (4.15) 的 $\delta m_R$，及附录的有限流半径讨论。

[2] M. Lüscher and P. Weisz, *Perturbative analysis of the gradient flow in non-abelian gauge theories*, JHEP 02 (2011) 051, arXiv:1101.0963。正流时间的 UV 结构和重整化边界。

[3] M. Lüscher, *Properties and uses of the Wilson flow in lattice QCD*, JHEP 08 (2010) 071, arXiv:1006.4518。平滑半径、正流时间观测量和连续极限背景。

[4] H. Makino and H. Suzuki, *Lattice energy–momentum tensor from the Yang–Mills gradient flow—inclusion of fermion fields*, PTEP 2014 (2014) 063B02, arXiv:1403.4772。Ringed-fermion 场归一化的背景；它不是 line-mass subtraction。

[5] N. Brambilla et al., *Inclusive P-wave Quarkonium Decay Widths from Lattice QCD and pNRQCD*, arXiv:2607.13024v1。Supplemental Material SM1 C，Eqs. (SM17)–(SM19)，为独立伴随 Polyakov-loop 抽取 $d$ 的直接实例；另见 SM1 A 的顺序与 SM1 的减除定义。本次未声称其有更新版本，也未把其结论泛化为完整 PDF matching。

[6] S. Gupta, K. Hübner and O. Kaczmarek, *Renormalized Polyakov loops in many representations*, Phys. Rev. D 77 (2008) 034503, arXiv:0711.2251。不同表示 Polyakov-loop 重整化、静态源自由能和有限重整化方案背景。

[7] Junyuan Huang, *Gradient-Flow Renormalization of Nonlocal Quasi Operators in the Lattice QCD*, CLQCD 2026 presentation, 2026-10-11，用户提供文件 `Pion_DA.pdf`。第 9 页为流半径设置，第 11 页为局域归一化，第 14–17 页为纯规范/矩阵元线自能与 double ratio，第 19–21 页为减除和方案转换，末页为后续极限路线。其示意公式的符号、$\sqrt8$ 和部分算符定义尚不能视为完整代码规范。

[8] M.-Q. Gao, L.-c. Gui, J. Hua, J. Liang and J. Shi, *Flowed quark field renormalization in lattice QCD: A Ward-identity approach and its validation using quark bilinears*, arXiv:2609.23379v1；及 2026-10-11 同题 presentation `Z_chi.pdf`。论文 Eqs. (3)–(9) 定义归一化，Eqs. (10)–(15) 给出 mixed 两点函数和平台提取，Eq. (16) 为顺序双极限。讲义第 4–10 页对应传统 ringed、Ward 路线和跨通道验证。

[9] J. Borgulat, R. V. Harlander, J. T. Kohnen and F. Lange, *Short-flow-time expansion of quark bilinears through next-to-next-to-leading order QCD*, JHEP 05 (2024) 179, arXiv:2311.16799。这是局域双线性 ringed-scheme 系数 $\zeta_\Gamma$ 的依据，不是非局域 PDF 的 LaMET kernel。

[10] Y.-K. Huo et al. (Lattice Parton Collaboration), *Self-renormalization of quasi-light-front correlators on the lattice*, Nucl. Phys. B 969 (2021) 115443, DOI:10.1016/j.nuclphysb.2021.115443。特别见 Sec. 8.3：Landau-gauge-fixed Wilson line 在较短距离与 Wilson-loop 提取的线性项大致相容，但大距离出现明显差异；Landau 大距离矩阵元不具有通常的 transfer-matrix 解释。这提醒我们不能把任意长度的纯规范开口线斜率都视为可直接移植的 PDF 减除因子。

本文的长线高斯积分、周长项相消、逆转换避免重复减除和误差公式已经明确标作从所列定义/关系的直接推导；待验证的拟合模型和本 ensemble 的未测系数不作为文献定论。

### 15.2 本地核对路径

```
/public/home/donghx/.codex/Markdown/gradient_flow.md
/public/home/donghx/.codex/Markdown/gradient_flow_gluon.md
/public/home/donghx/.codex/Markdown/gluon_gradient_flow_matching.md
/public/home/donghx/Note/Gradient_Flow/main.tex
/public/home/donghx/Note/Gradient_Flow/Pion_DA.pdf
/public/home/donghx/Note/Gradient_Flow/Z_chi.pdf
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/polyakov_L32x96/README.md
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/polyakov_L32x96/operations.md
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/polyakov_L32x96/src/polyakov_flow.cpp
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/polyakov_L32x96/runs/production100_20261011_113518/validation/final_audit.json
```

本次对照已有 PDF 文本和 arXiv 原文，生成理论说明并核对保存的测量定义/审计。未重新测量规范场、未拟合 $d_R$、未改动现有 matching 代码或图、未提交新的 GPU/CPU 数值任务。

### 15.3 生成记录

生成日期：**2026-10-14（Asia/Shanghai）**。

本次会话标题：**Gradient-flow 线性发散、零流时间极限与 Polyakov/Wilson-loop 线质量测量**。

## 16. 2026-10-15：三种纯规范测量与跨流时间 double ratio

本节把 [7] 的测量思路统一转换到本项目的记号：$t_f$ 为物理流时间、$\tau=t_f/a^2$、$r_f=\sqrt{8t_f}$，正线质量 $m_R=d_R/r_f$，旧 matching 中 $\delta m_R=-m_R$。以下 $m_R=d_R/r_f$ 只代表需要检验的 leading 模型；耦合跑动、有限项、格距项不可因此删去。

### 16.1 四类观测量中的长度因子

| 观测量 | leading 线自能项 | 不能忽略的其他内容 |
|---|---|---|
| 矩形 Wilson loop | $\ln W_R=-2m_R(R_W+T_W)+\cdots$ | 面积/静态势、四个拐角、有限时间激发态 |
| 单个时间 Polyakov loop | $\ln\mathcal L_R=-m_R L_t+\cdots$ | 单静态源有限自由能、低温信噪比 |
| 两条绕时线的相关函数 | $\ln G_R=-2m_R L_t+\cdots$ | 有限温度的源反源自由能、颜色通道定义 |
| 固定规范的开口直线 | $\ln L_R^G=-m_R L+\cdots$ | 规范和 Gribov-copy 处方、端点与有限长度效应 |

表中省略号不同，**不能通过简单除以路径长度就认为已测出同一个绝对 $m_R$**。需要窗口、有限项和方案对齐后，才能比较 leading UV 系数或线质量差。[1,5–8,10]

尤其第 9 节静态势满足

$$
V_R^{\rm flow}(R_W,t_f)=2m_R(t_f)+V_R^{\rm sub}(R_W)+\cdots.
$$

因此势的流时间变化应除以 2 后与一条开口线或单个 Polyakov loop 的线质量比较。

### 16.2 单个 Polyakov loop 不等于 Polyakov-loop correlator

`Pion_DA.pdf` 第 14 页画了成对绕时线并讨论静态势；现有 `polyakov_L32x96` 程序测量的则是第 8 节的**单个**时间圈及其基本/伴随局域字符。两者不是同一个数据产品。

若新测试采用规范不变的 trace–trace 定义，必须明确写为

$$
p_R(\boldsymbol x;t_f)=\frac{1}{d_R^{\rm rep}}
\operatorname{Tr}_R P_R(\boldsymbol x;t_f),
$$

$$
G_R(R_W;t_f)=
\left\langle\frac1{3N_s^3}
\sum_{\boldsymbol x,i=1}^3
p_R(\boldsymbol x;t_f)
p_R^*(\boldsymbol x+R_W\hat i;t_f)\right\rangle .
$$

两个局域迹分别规范不变，无需规范固定；有限温度下

$$
-\frac1{L_t}\ln G_R(R_W;t_f)
=2m_R(t_f)+F_{Q\bar Q,R}^{\rm sub}(R_W,T_{\rm phys};t_f).
$$

右侧是与这个颜色平均相关函数相应的自由能，不在任意温度自动等于零温静态势。若改用 $\operatorname{Tr}[P(\boldsymbol x)P^\dagger(\boldsymbol y)]$，则必须另外指定规范固定或连接线；若加空间连接线闭合，拐角/交叉点结构也随之改变。

**待确认：**[7] 示意图没有给足颜色迹、连接线和规范处方，不能声称上式逐项复现该组的 Polyakov 相关函数。本项目测试应把自己的 trace–trace 定义写入 metadata。完整相关函数与 connected-subtracted 相关函数分别保存，不为制造信号静默改定义。

### 16.3 规范固定的 Wilson link 是一整条开口直线

这里的“Wilson link”指长度 $L=na$ 的有序链接乘积，而不只是 $n=1$ 的单条格点 link：

$$
U_\mu^G(x,t_f)=G(x)U_\mu(x,t_f)G^\dagger(x+\hat\mu),
$$

$$
W_R^G(x,n\hat\imath;t_f)
=\prod_{k=0}^{n-1}U_i^{G,(R)}(x+k\hat\imath,t_f),
$$

$$
L_R^G(n,t_f)=
\left\langle\frac1{3V_4d_R^{\rm rep}}
\sum_{x,i=1}^3\operatorname{ReTr}_R W_R^G(x,n\hat\imath;t_f)
\right\rangle .
$$

实现应另存每个方向、正反路径及基本表示复迹，再构造所声明的实投影。$n=0$ 时单位线给出 1。未经规范固定时，开口迹因两端局域规范变换而变化，不能当作规范不变观测量。[7,10]

本项目可选 **四维 Landau 规范**作为明确的测试定义：最大化

$$
F[G]=\sum_{x,\mu}\operatorname{ReTr}U_\mu^G(x),
$$

并保存离散散度残差、迭代数、算法、停止条件、随机起点和规范泛函值。这个选择是**我们提出的 pilot 处方**；[7] 未注明其具体规范，不能将其归因于对方。

应区分两条不等同的流程：

1. 对每个 $t_f$ 的 flowed gauge 分别固定到 Landau 规范；
2. 只固定 $t_f=0$ 的场，再将同一个变换随规范协变的流轨迹运输。

第二条一般不保证每个正流时间仍满足 Landau 条件，不能混作第一条。pilot 采用第一条；独立随机起点用于检查 Gribov-copy 敏感性。闭合圈在规范固定前后应不变，但开口线有限部分可随处方变化。

规范固定仅用于独立纯规范测量的场副本；不得把固定后的 Wilson line 单独替换进仍使用原规范 basis/费米子端点的 C3。若实际变换完整算符，所有带颜色的端点也必须一致变换。

### 16.4 从开口线中先测相对线质量

在 $a\ll r_f\ll L$ 且控制其他长程效应的窗口中，考虑

$$
\ln L_R^G(L,t_f)
=-m_R(t_f)L+b_R^G(t_f,L).
$$

不能预先把 $b_R^G$ 的全部 $L$ 依赖设成常数。可先用两个流时间的比值降低共同有限部分的影响：

$$
\ln\frac{L_R^G(L,t_f)}{L_R^G(L,t_{f,\rm ref})}
=-\Delta m_R(t_f,t_{f,\rm ref})L
+\Delta b_R^G(t_f,t_{f,\rm ref},L),
$$

$$
\Delta m_R(t_f,t_{f,\rm ref})
=m_R(t_f)-m_R(t_{f,\rm ref}).
$$

若窗口内 leading 模型可用，则

$$
\Delta m_R\simeq d_R
\left(\frac1{r_f}-\frac1{r_{f,\rm ref}}\right).
$$

只有在端点、有限长度和 IR 项受控后，才可把对 $L$ 的斜率解释为相对线质量。单格距下还应与 Wilson-loop 势差的 $\Delta V_R/2$ 比较。即便比较成功，也只检验相对自能，不自动固定共同的有限 residual mass。

开口线拟合所得有限质量不必与强子矩阵元或 Wilson-loop 处方逐项相同；[10] Sec. 8.3 还展示了大距离差异，并指出 Landau-gauge-fixed 大距离矩阵元没有通常的 transfer-matrix 解释。因此不能以“线越长越容易拟合单指数”作为该观测量的普遍论据；必须同时检查 $L/r_f$ 和 IR/规范处方效应。本项目把它作为**独立测量与交叉验证**，不直接升级为已验证的全阶 PDF 重整化。[10] 使用的是另一套格点/涂抹测量，本文只引用其风险提示，不把它当作本 ensemble 的数值结果。

### 16.5 用强子矩阵元的跨流时间 double ratio 检查 leading 系数

先对同一通道做局域归一化（第 17 节）：

$$
\mathcal M_\Gamma(z,P_z,t_f)
=\frac{h_\Gamma^{\rm GF}(z,P_z,t_f)}
{h_\Gamma^{\rm GF}(0,P_z,t_f)}.
$$

再定义

$$
\mathcal D_\Gamma(z,P_z;t_f,t_{f,\rm ref})
=\frac{\mathcal M_\Gamma(z,P_z,t_f)}
{\mathcal M_\Gamma(z,P_z,t_{f,\rm ref})}.
$$

共同的普通场归一化已消去；若 leading 非局域 SFTE 与局域 SFTE 均适用，则

$$
\ln|\mathcal D_\Gamma|
=-\Delta m_F L
+\ln\left|
\frac{C_\psi(t_f)/\zeta_\Gamma(t_f)}
{C_\psi(t_{f,\rm ref})/\zeta_\Gamma(t_{f,\rm ref})}
\right|
+\Delta_{\rm power}(z,P_z,t_f,t_{f,\rm ref}).
$$

这是依据第 17 节关系的代数推导；$\Delta_{\rm power}$ 不能默认与 $z$ 无关。小流时间对数项可在长线窗口内形成截距，但有限 $r_f/L$、$t_fP_z^2$、cutoff 项会影响斜率。[7] 第 16–17 页使用了此类 double-ratio 思路。

注意：

- 必须使用**相同极化、flavor、动量和运动学归一化**的非零局域分母；不得跨通道随意相除。
- 局域分母或非局域分子接近零时，对数/比值可能失稳，应记录分布和相位；不能通过逐样本丢弃或 clipping 伪造平台。
- $\ln|\mathcal D|$ 只诊断幅度，原复矩阵元及相位必须保留；它不授权对有符号的伴随 Polyakov 均值取绝对值。
- $z=0$ 时精确有 $\mathcal D=1$；但长线渐近拟合不适用于 $z=0$，不能同时用它强迫长线截距为零。
- $t_f=t_{f,\rm ref}$ 时数据恒等为 1，只作实现检查，不当成一个有独立信息的拟合点。
- 只测 $\Delta m$ 并不确定绝对线质量；跨流时间 double ratio 也会消掉与流时间无关的 $m_0L$ 项。

### 16.6 Presentation 的数字与符号不能直接写进现有 matching

[7] 第 17 页报告不同矩阵元与开口线的拟合系数相近，其中 Wilson line 的示意数值为 $-0.1165(1)$。这属于他们自己的 convention/测量和拟合结果，不是本项目 $d_F$ 的输入。

若某页确实定义 $\ln L=\kappa L/\sqrt{t_f}+\cdots$，且长度单位一致，则与本项目 $\ln L=-d_RL/\sqrt{8t_f}+\cdots$ 的映射是

$$
\kappa=-\frac{d_R}{\sqrt8}.
$$

若该页改用 $L/r_f$，则对应系数为 $-d_R$。**只有先核对变量、量纲和指数方向，才能用这个映射。**[7] 第 14、19、末页对 $\sqrt{t_f}$、$\sqrt{8t_f}$ 和减除指数的写法并不足以直接固定完整实现；本项目不照抄其数字或指数符号。静态势的因子 2 也应以明确的两条长边推导，而不是从示意参数 $A_0$ 猜测。

## 17. 用局域归一化消去 $Z_\chi$：Pion DA 路线

### 17.1 消去什么，保留什么

[7] 第 11 页采用同一 flow time 的 $z=0$ 归一化。用普通场的裸矩阵元表示，

$$
h_\Gamma^{\rm ring}(z,P_z,t_f)
=\mathring Z_\chi(t_f)h_\Gamma^{\rm GF}(z,P_z,t_f),
$$

于是

$$
\boxed{
\mathcal M_\Gamma(z,P_z,t_f)
=\frac{h_\Gamma^{\rm ring}(z,P_z,t_f)}
{h_\Gamma^{\rm ring}(0,P_z,t_f)}
=\frac{h_\Gamma^{\rm GF}(z,P_z,t_f)}
{h_\Gamma^{\rm GF}(0,P_z,t_f)}.
}
$$

共同的 $\mathring Z_\chi$ **相消**，不等于它的数值等于 1，也不等于独立确定了它。

分母的线长度为零，因此通常仍有

$$
\mathcal M_\Gamma(z,P_z,t_f)\propto e^{-m_F(t_f)|z|}.
$$

局域归一化不能替代线自能减除。它还把局域电荷/耦合的绝对归一化一并除掉；若需要绝对矩阵元或未归一化 PDF，必须另外恢复目标方案中的局域矩阵元。

### 17.2 归一化后 matching 也随之改变

分别写非局域长线和局域关系：

$$
\mathring Z_\chi h_\Gamma^{\rm GF}(z,t_f)
=C_\psi(t_f,\mu)e^{-m_F(t_f)L}h_\Gamma^{\overline{\rm MS}}(z,\mu)
+\text{流修正},\qquad z\ne0,
$$

$$
\mathring Z_\chi h_\Gamma^{\rm GF}(0,t_f)
=\zeta_\Gamma(t_f,\mu)h_\Gamma^{\overline{\rm MS}}(0,\mu)
+\text{流修正}.
$$

消去 $\mathring Z_\chi$ 得到归一化目标的 leading 转换：

$$
\boxed{
\frac{h_\Gamma^{\overline{\rm MS}}(z,\mu)}
{h_\Gamma^{\overline{\rm MS}}(0,\mu)}
=\frac{\zeta_\Gamma(t_f,\mu)}{C_\psi(t_f,\mu)}
e^{m_F(t_f)L}\mathcal M_\Gamma(z,t_f)
+\text{流/格距修正}.
}
$$

该式是局域与非局域 SFTE 的组合，不把非局域长线公式错误延伸到 $z=0$。局域 $\zeta_\Gamma$ 有通道依赖，所以对归一化后的不同极化，不能因为场因子共用就默认转换系数相同。

[7] 第 20 页给出了其 normalized scheme 的一圈转换示意；它不是本项目未归一化 ringed bilocal 的系数。其有限常数、RG 方向、局域电荷恢复和适用通道应经完整推导后再接入代码；本次不直接使用该页公式替换已有 matching。

### 17.3 共同长度 reduced ratio 是另一种方案

若分子分母具有**相同非零线长度**，某些 reduced ratio 可以同时消去共同场因子与线自能。例如在分母确实非零且运动学定义一致时，

$$
\frac{h_\Gamma(z,P_z,t_f)}{h_\Gamma(z,P_{z,0},t_f)}
$$

可消去共同 $e^{-m_FL}$。但分母也含有真实强子长程信息，得到的是另一个 ratio scheme，不是原来的绝对 quasi 矩阵元。恢复绝对归一化及后续 matching 必须另行定义；不能再完整施加一次已经相消的线质量因子。

## 18. 用矢量 Ward 恒等式确定 $Z_\chi$：保留绝对归一化的路线

### 18.1 核心定义与所需输入

对同一外态矩阵元，[8] 的局域矢量 SFTE 与未流化电荷归一化为

$$
\lim_{a\to0}\left[
\mathring Z_{\chi,\rm kin}(t_f,a)\langle V_\mu^L(t_f,a)\rangle
\right]
=\zeta_V(t_f)\langle V_\mu^R\rangle+O(t_f),
$$

$$
\lim_{a\to0}\left[Z_V(a)\langle V_\mu^L(0,a)\rangle\right]
=\langle V_\mu^R\rangle.
$$

因此在有限 $a,t_f$ 定义（论文 Eq. (8)）

$$
\boxed{
\mathring Z_\chi^V(t_f,a)
=\frac{\zeta_V(t_f) Z_V(a)}{R_V^L(t_f,a)},\qquad
R_V^L=\frac{\langle V_\mu^L(t_f,a)\rangle}
{\langle V_\mu^L(0,a)\rangle}.
}
$$

$Z_V$ 是同一格点作用量与质量/改善处方的未流化局域矢量流电荷归一化；$\zeta_V$ 是 [9] 的 ringed-scheme 小流时间系数，不因连续矢量流受到 Ward 恒等式保护而自动等于 1。测量部分是非微扰的，但全过程仍有 $\zeta_V$ 的微扰截断和尺度误差。

### 18.2 实际用 mixed 介子两点函数提取 $R_V$

用相同、未流化的源 $J_i^\dagger$，只在汇端改变流时间：

$$
C_i(t_f;\Delta t)=\sum_{\boldsymbol x}
\langle V_i^L(t_f;\boldsymbol x,\Delta t)J_i^\dagger(0)\rangle,
\qquad i=1,2,3.
$$

在 $\Delta t$ 足够大、流化区域与源接触可忽略且反向传播已控制的区间，

$$
\frac{C_i(t_f;\Delta t)}{C_i(0;\Delta t)}
\longrightarrow
\frac{\langle0|V_i^L(t_f)|h_0\rangle}
{\langle0|V_i^L(0)|h_0\rangle}=R_{V,i}^L.
$$

共同的源重叠与传播因子消去，**不需要开平方**。如果两端都 flowed，就不是同一个测量定义，不能原样套用该平台公式。[8] 论文 Eqs. (10)–(13)、presentation 第 6–7 页。

论文使用相同 Gaussian-smeared unflowed source、Symanzik gauge flow、四维协变 Laplacian 费米子流和 RK3，步长 $\epsilon_f=0.01$；不同格距采用共同物理 $t_f\simeq0.0012$–$0.026\,\mathrm{fm}^2$ 网格。它们的数值 $\mathring Z_\chi^V$ 不能直接移植到本项目 Wilson-flow ensemble。

### 18.3 与传统动能定义的关系：不能省略有限流时间区别

[8] Eq. (9) 明确指出，在精确 matching 下

$$
\boxed{
\lim_{a\to0}
\frac{\mathring Z_\chi^V(t_f,a)}
{\mathring Z_{\chi,\rm kin}(t_f,a)}
=1+O(t_f).
}
$$

矢量条件吸收了参考矩阵元的部分高维算符贡献，所以有限流时间可有参考外态/分量依赖；有限格距还叠加 cutoff 和改善误差。若用到 $\alpha_s^n$ 的系数，还应有相应缺失高阶误差。不能为了让两条曲线逐点相同而调 $Z_\chi$。

论文的受控目标是

$$
\langle O_\Gamma^R(\mu)\rangle
=\lim_{t_f\to0}\lim_{a\to0}
\frac{\mathring Z_\chi^V(t_f,a)
\langle O_\Gamma^L(t_f,a)\rangle}
{\zeta_\Gamma(t_f,\mu)}.
$$

P/S/A/T 的局域验证支持共同场因子的可用性，但不是对所有非局域 PDF、所有质量和 singlet mixing 的完整验证。[8] presentation 第 8–10、13 页。

### 18.4 如果只用 $Z_V/R_V$，matching 要怎样改变？

定义不含 $\zeta_V$ 的另一种归一化

$$
\widehat Z_\chi^V=\frac{Z_V}{R_V},\qquad
\widehat h_\Gamma=\widehat Z_\chi^V h_\Gamma^{\rm GF}.
$$

由于 $\mathring Z_\chi^V=\zeta_V\widehat Z_\chi^V$，现有非局域转换在 leading SFTE 下应变成

$$
h_\Gamma^{\overline{\rm MS}}
=\frac{\zeta_V}{C_\psi}e^{m_FL}\widehat h_\Gamma
+\text{流/格距修正},
$$

等效非局域系数为 $\widehat C_\psi=C_\psi/\zeta_V$。这是有限归一化的代数转换，不能漏掉 $\zeta_V$ 后还声称使用未改动的 ringed 系数。

### 18.5 对我们三种 quark 极化的使用与统计配对

同一轻夸克作用量、质量和流处方下，可将由一个参考矢量通道固定的同一标量用于

$$
h_\Gamma^{V,(b)}(z,P_z,t_f)
=\mathring Z_\chi^{V,(b)}(t_f,a)
h_\Gamma^{\rm GF,(b)}(z,P_z,t_f).
$$

两端各一个平方根，整体只有一份因子。u、d 简并时共用同一个因子；不要逐极化、逐动量或逐 $z$ 调整归一化以改变曲线形状。helicity、transversity 仍保留自己的 Dirac/相位 convention 和通道相关 LaMET kernel；transversity 的既有外部符号确认状态不因这个测量而自动升级。

每个 bootstrap 从相同组态索引计算 $R_V^{(b)}$、$Z_V^{(b)}$（若同组态实测）、$\mathring Z_\chi^{V,(b)}$ 和矩阵元。独立文献输入的 $Z_V$ 应作为共同 nuisance 传播，不伪造与本组态样本的相关性。归一化因子由系综条件确定；不把逐 conf 的 $\zeta_VZ_V/R_{V,c}$ 逐点乘上该 conf 矩阵元后再平均，作为未经声明的新估计量。

若改用第 17 节的归一化比值，公共 $\mathring Z_\chi^V$ 会再次相消，不需要重复施加；恢复绝对矩阵元时才需要独立局域输入。**胶子算符不乘夸克的 $Z_\chi$**，它的端点和 tensor-sector 转换须另行处理。

### 18.6 把两层减除同时接入时的方案账本

| 分支 | 场归一化层 | 线质量层 | 仍需处理 |
|---|---|---|---|
| 绝对动能 ringed | $\mathring Z_{\chi,\rm kin}$ | 一个明确的 PT 或 NP 处方 | 非局域有限转换、极限、LaMET |
| 绝对 Ward | $\mathring Z_\chi^V=\zeta_VZ_V/R_V$ | 同上 | 有限流时间处方差、非局域有限转换、极限、LaMET |
| $z=0$ normalized | 公共场因子相消 | 同上，长度项仍在 | normalized-scheme 系数与绝对局域输入 |
| 相同非零长度 reduced ratio | 部分或全部公共因子相消 | 共同长度自能可相消 | ratio scheme、IR 分母和绝对归一化恢复 |

若非微扰减除定义 $h^{\rm NP}=e^{m^{\rm NP}L}h^{\rm norm}$，第 10 节中

$$
C^{-1}e^{(m^{\rm PT}-m^{\rm NP})L}h^{\rm NP}
$$

只是避免重复减除的代数恒等式，代回去仍等于原 PT 转换。**单靠这条恒等式不会自动获得超出 PT 的精度。**要真正利用 NP 自能改进结果，必须构造与之配套的有限 NP-to-target 转换/残余质量条件并检验其截断误差，不能将 $C^{\rm PT}$ 原封不动地贴给任意 NP 方案。[1,10；此处为方案组合的代数说明]

## 19. L32x64 测试 Plan：先独立测量，再比较减除，最后接入矩阵元

**状态修订（2026-10-15）：已按用户授权实施并提交 P0/P1，见第21节。**本节是原始分阶段设计，执行范围优先服从第21节：规范固定直线暂缓；费米子分支全部 B 权重、$|n_z|=3,4,5,6$、tsep/a=7..12。完整操作版见：

```
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/plans/L32x64_divergence_Zchi_test_20261015.md
```

### 19.1 固定输入与范围

- ensemble：$32^3\times64$，$a=0.0897\,\mathrm{fm}$。
- 原始 thin LIME：`/public/group/lqcd/configurations/CLOVER/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64`。
- 单组态 smoke：10000；之后固定 10 conf：10000,10040,10080,10120,10160,10200,10240,10280,10320,10360，与既有 B 窗口测试相同；不按测量信号挑 conf。
- $(x,y,z,t)=(0,1,2,3)$，存储轴 $(t,z,y,x)$；四维 Wilson gauge flow；周期规范场；费米子反周期时间边界；不增加 HYP/stout 预流。
- 保留既有 basis；新 vertex/VVV 只在节点 scratch 重构，不长期缓存。不改 frozen A/B 作业文件。
- 纯规范和拟合任务用 CPU；若需要新 Dirac solve/fermion flow，用 `na800-sug` 并包含 `#SBATCH --reservation=donghx_223`。原计划发布时未提交；当前 P0/P1 提交见第21节，不自动启动 P2/P3/P4。

两点长度为 $L_s=2.8704\,\mathrm{fm}$、$L_t=5.7408\,\mathrm{fm}$，温度约 $34.37\,\mathrm{MeV}$；这是由给定格距换算，不是新测量。低温 Polyakov 信号可能不足，不能保证其为首选提取途径。

### 19.2 流时间与尺度检查

纯规范保留小流时间加密和大流时间扫描：

$$
\tau\in\{0,.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,
.80,1.00,1.40,1.80,2.20,2.60,3.00,3.40,3.80\}.
$$

一次读取、累积 RK3 演化；主步长 $.01$，conf10000 另做 $.005$ 的完整步长回归。$\tau=0$ 只作定义检查，不参加 $1/\sqrt\tau$ 连续流拟合。

Ward 先测现有 quark 的八个正流时间 $\{.05,.10,.20,.30,.40,.50,.80,1.00\}$，并配套 $t_f=0$ 的 mixed-correlator 分母；不为第一轮 normalizer 强行补算大流时间的 nucleon C3。

| $\tau$ | $t_f\ (\mathrm{fm}^2)$ | $r_f\ (\mathrm{fm})$ | $r_f/a$ | $\sqrt{t_f}P_z$，$n_z=3/5$ |
|---|---|---|---|---|
| .10 | .00080461 | .08023 | .894 | .186 / .310 |
| .20 | .00160922 | .11346 | 1.265 | .263 / .439 |
| .50 | .00402305 | .17940 | 2.000 | .417 / .694 |
| .80 | .00643687 | .22693 | 2.530 | .527 / .878 |
| 1.00 | .00804609 | .25371 | 2.828 | .589 / .982 |
| 1.80 | .01448296 | .34039 | 3.795 | .790 / 1.317 |
| 3.80 | .03057514 | .49457 | 5.514 | 1.148 / 1.914 |

这里 $P_z=2\pi n_z/(32a)$；数值为量纲换算。小 $\tau$ 的 cutoff 与大 $\tau$ 的 PDF 动量修正有竞争，不能把整个扫描区直接当作一个受控 SFTE 窗。

默认跨流时间参考取已有 $\tau_{\rm ref}=.50$，再用 $.80$ 检查参考敏感性。[7] 的 $r_{f,\rm ref}=.20\,\mathrm{fm}$ 对此格距对应 $\tau\simeq.62142$，不是 $.20$；不混淆物理半径和无量纲流时间。

### 19.3 分阶段工作与放行条件

| 阶段 | 具体工作 | 放行依据 |
|---|---|---|
| P0 定义/输入审计 | 冻结 10 conf、flow 网格、链接表示、规范处方、valence action/mass、$Z_V$ 来源；检查 gauge/perambulator/basis metadata | 不明项标待确认；不把目录存在当完成 |
| P1 纯规范单 conf | 移植现有 Polyakov reader/flow；增加 Wilson rectangles、trace–trace correlator；双步长；规范固定直线暂缓 | 单位场、路径反向、局域伴随字符、规范不变性和 step-size gate |
| P2 纯规范 10 conf | 用同一流轨迹测量，保存有符号/复数原量；先比较 $\Delta V/2$、Polyakov 路线 | 可辨识、窗口/模型/步长敏感性；无信号则保留失败，不造 $d$ |
| P3 Ward 单 conf→10 conf | 独立 full-propagator mixed meson C2；取得 $R_V$；输入/测量同处方 $Z_V$，配套 $\zeta_V$ | 基态/接触/反向项、方向和质量处方检查；缺 $Z_V$ 只发布 $\mathring Z_\chi^V/Z_V$ 诊断 |
| P4 quark 矩阵元对照 | 同一 B estimator 和配对样本，先比较 $\tau=.2,.5,.8$，$|n_z|=3,4,5,6$、tsep/a=7..12、三极化的 normalized 与 absolute-Ward 分支 | 缺 B C3 先单独补其诊断子集和时间窗；不将旧 A 当 B；非局域转换方案对齐 |
| P5 扩大与方案定稿 | 十 conf 足够识别测量/窗口问题后，再提出 30/40 conf 或更多格距、匹配物理流时间的扩展 | 不自动提交；绝对 residual mass 与有限转换未定前只发布诊断 |

更具体的范围、输出 schema、统计估计量、数值阈值和待确认项见操作版 Plan。

### 19.4 不能以“测试成功”替代的理论限制

1. 一个格距上的 $\tau\to0$ 不等于固定物理流时间的连续极限；本测试不宣称解决这一限制。
2. 纯规范提取无需人为加上 PDF 的 $P_z$ 条件；把线质量用于 PDF 时才检查全部尺度比。
3. 即使数值门和拟合可辨识门通过，也可能没有稳定的有限项分离或绝对质量方案。
4. 无可靠 $Z_V$ 不能发布绝对 Ward-normalized 矩阵元；无非零局域分母不能强行归一化。
5. 本 Plan 不把旧的 $Z_\chi=1$ 对照升级成验证结果，不把 [7] 的 normalized coefficient 直接套给绝对 Ward 分支。
6. 动能对照若要在 L32x64 新测，仍需此 ensemble 自己的质量/手征处方；不能复制 L32x96 的临界质量。

## 20. 原始文档更新记录与当时未执行边界（历史）

本轮读取了两份 presentation 的文本和关键图页、arXiv:2609.23379v1 的定义，以及本地 L32x64 生产/B 窗口和 L32x96 Polyakov 的现有代码说明。第 11 节仍是其注明的历史审计，不代表本轮重新检查实时队列；没有因文档更新宣称 L32x64 新测量已完成。

本轮未改变数值代码、frozen 作业、匹配结果或 Note 的 TeX/PDF；没有运行 gauge flow、gauge fixing、Dirac solve、C3 收缩或数值拟合，没有提交任何 CPU/GPU 任务。下一步建议先实现 P0/P1 的独立纯规范单组态测试入口，再按 Plan 放行后续阶段。

原始 `divergence.md` 的本轮备份：

```
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/handoff/divergence_update_20261015/divergence.before.md
```

更新日期：**2026-10-15（Asia/Shanghai）**。

本次会话标题：**闭合圈与规范固定直线的线自能、ratio/Ward 费米子归一化和 L32x64 测试 Plan**。

## 21. P0/P1 实施更新：闭合圈优先、关联费米子 B 分支

更新日期：2026-10-15（Asia/Shanghai）。本节记录之后获得授权的实施，不覆盖第20节的历史。

### 21.1 本轮硬契约

1. 固定规范 Wilson line **暂缓、不测**；先使用 Wilson loop、Polyakov 单圈和双圈构造线自能诊断。
2. 这些纯规范圈本身没有 blending 权重，元数据写 `blending_weight_scheme=not_applicable_pure_gauge`，同时写 `associated_C3_scheme=B`；不能将纯规范圈误称为“加权 B 圈”。
3. 本 Plan 后续全部费米子 C3 统一用 B：

$$
\Omega^B_{(\sigma_a,m),(\sigma_b,n)}=
\begin{cases}
\omega^{(2)}_{mn},&\sigma_a\equiv\sigma_b\pmod{64},\\
w_{\sigma_a,m}w_{\sigma_b,n},&\sigma_a\not\equiv\sigma_b\pmod{64}.
\end{cases}
$$

权重只在未加权几何 vertex 上施加一次；basis、perambulator、源汇 VVV 不再额外加权。$w=d/n$，不是 $\sqrt{\mathrm{diag}\,\omega^{(2)}}$。原始时间取模64、去重，保留全部有序时间对，不加额外 $1/2$ 或时间对数平均。C2 源汇定义不因 B 改动，不对它套 insertion 的二阶权重。B 是已声明的估计量处方，不声称已证明所有与重子共享噪声的高阶联合矩。

4. 费米子固定 $|n_z|=3,4,5,6$，保留正负动量；$t_{\rm sep}/a=7,8,9,10,11,12$，即 $0.6279..1.0764\,\mathrm{fm}$。源/current 全64时间；currentNv400、源汇Nv200；外旋量2×2、开放current4×4；原始 $|z|/a=0..20$。
5. 后续 C3 namespace 为 `C3_schemeB_Pz3456_tsep7to12`。与旧 A、旧 B 窗口诊断分开；缺数据报缺失，禁止从 A fallback。P0/P1 **未提交完整 C3/C2 生产**。

### 21.2 实现、输入审计与已通过回归

隔离包（以下路径相对 `/public/group/lqcd/donghx/Gradient_Flow_GluonPDF`）：

```text
line_mass_L32x64_pilot_20261015/
  src/polyakov_base.cpp        # 原 Polyakov LIME/RK3/SU3 实现的只读副本
  src/closed_loops.cpp         # 新闭合圈入口；不改旧生产
  reference/flow_common_l32x64.py  # B 权重参考的冻结副本
  code/p0_audit.py             # LIME/XML/payload、eigen/peram头、B表和P6
  code/test_polyakov_base.py   # 原独立NumPy流/伴随回归
  code/test_closed.py          # 新逐路径/规范不变性/显式8×8伴随回归
  code/test_analysis.py        # 单conf分析及负号/log保护与出图回归
  code/run.py                 # 输入/源码/binary hash、节点及文件锁
  code/audit.py               # 双步长检查、有效势及非统计诊断图
  code/submit.py              # 冻结源码后提交 CPU 作业链
  slurm/build_local.sh        # 每个实际计算节点编译、回归、binary receipt
  latest.json                 # 当前run与jobID；优先读此文件
```

P0 已在 CPU 计算节点检查固定十conf的 gauge 记录，并读完全部1280个 eigen/peram NPY头，无缺口。几何/precision/payload 和相应 shape/dtype 均通过。此处仅证明输入可读，不证明 valence action、质量、clover 或 $Z_V$ 的物理处方，这些仍待确认。

独立 B 表误差：一阶0，二阶最大绝对差 $2.9103830456733704\times10^{-11}$（大权重下浮点运算顺序差）。模64同帧/跨帧分支、$|n_z|=6$ 均通过。
合成场闭合圈规范变换误差 $1.3392244997425239\times10^{-16}$；逐路径 NumPy 参考最大误差 $3.946495907847236\times10^{-16}$；反向路径误差 $9.104505742017336\times10^{-16}$。这些是实现回归，不是物理 ensemble 信号。

### 21.3 P1 测量定义与实际提交

- conf10000、完整23点 tau 网格（同19.2），epsilon=.01/.005，分别从同一raw累计RK3；最终累计380/760步。
- Wilson rectangles：xt/yt/zt，$R/a=0..12$、$T/a=0..16$；全部 $32^3\times64$ 起点平均。基本复迹/3；伴随 $(|\operatorname{Tr}H|^2-1)/8$ 逐闭合holonomy构造后平均。零边圈为1。
- Polyakov 单圈：完整64条时间链接，逐空间点 p3复数、p8有符号实数；没有费米子反周期负号。双圈为 $\mathrm{mean}_{\boldsymbol x}[p_R(\boldsymbol x)p_R^*(\boldsymbol x+R\hat i)]$，保留三个方向；这是full，不冒充connected-subtracted。
- 输出形状：p3/p8为(32,32,32)；Wilson为(3,13,17)；双圈为(3,13)。只保存小观测量，未新增完整gauge/vertex/VVV缓存，不改旧basis/gauge。
- epsilon门：局域p3/p8、Wilson平均及双圈最大绝对差 $\le2\times10^{-5}$，plaquette差 $\le10^{-6}$；链接SU3误差 $<10^{-10}$。

提交命令（只stage和提交；数值计算在CPU节点）：

```bash
PYTHONDONTWRITEBYTECODE=1 /public/home/donghx/AnaConda/bin/python3 \
  line_mass_L32x64_pilot_20261015/code/submit.py --submit
```

当前冻结run：

```text
line_mass_L32x64_pilot_20261015/runs/P0P1_B_2026-10-15T183837.560011_0800/
```

当前作业链：gate **3853766** → epsilon数组 **3853767_[0-1]** → audit **3853768**。
实时完成状态请读 `latest.json`、Slurm、`validation/P1_epsilon_audit.json`，不要将“提交成功”当“P1已完成”。
2026-10-15 18:47核对：P0/gate已通过；两个真实测量均RUNNING，分别cpu052/epsilon.01与cpu058/epsilon.005，已到tau.10。两节点GLIBC2.17本地编译/全部小型回归通过；tau0的plaquette均为0.5462804251。此时完整双步长audit尚未完成。
CPU partitions为a192c1t,cpu6248R,cpueicc,i72c512g，无数组并发上限；测量初始48cores预算因无空闲整块资源，在任务仍pending时调整为24cores（NumCPUs/CPUSPerTask/MinCPUsNode），64G/48h上限不变，尚非耗时结论。该调度变更记在run的 `resource_update_24cores.json`，未改frozen源码或物理参数。

两次准备过程失败已隔离：3853753是ILDG XML记录末尾NUL兼容性，已只移除尾部NUL后保持严格XML/几何门；3853759是跨节点GLIBC差异，未进行真实测量。其失效依赖任务3853754/3853755/3853760已取消。新版本**在每个任务的实际节点编译并回归**，不跨系统复用gate二进制；源码、输入与验证结果仍全部冻结可追踪。

### 21.4 不得升级的结论与下一步

P1最终会输出 `diagnostics/P1_single_conf_diagnostics.pdf` 和 `line_mass_diagnostic.json`：原始符号、有效势、固定 $T/a=8..16$ 的无权OLS斜率、相对线质量/$d$ 候选。**Ncfg=1，不做bootstrap、不捏造误差；候选不用于正式减除或matching。**符号跨零的单/双Polyakov、Wilson均不取abs、裁剪或选择性删样本强行取log。

先完成步长门，再 P2 十conf统计：同配置/方向相关性、先ensemble均值后log、时间平台、空间/flow窗口与有限流畸变。$\Delta V/2$、单圈 $-\log L/N_t$、双圈 $-\log G/(2N_t)$ 的长度因子不可互换。绝对 residual mass 条件和有限NP-to-MSbar转换仍缺，不能重减PT/NP线质量。

新增 $n_z=6$ 后更需记录尺度张力：$\sqrt{t_f}P_z=(2\pi n_z/32)\sqrt\tau$。若探索门取 $r_f\ge2a$ 与 $\sqrt{t_f}P_z\le.5$，$n_z=6$ 要求同时 $\tau\ge.5$ 和 $\tau\lesssim.180$，无交集；纯规范提取不受Pz门，但用回PDF必须展示此限制。

本次实施会话标题：**L32x64 B方案关联的闭合圈P0/P1测试与Pz3–6/tsep7–12契约**。

## 22. 后续授权：P2 固定十组态闭合圈及相关拟合

更新日期：2026-10-15（Asia/Shanghai）。用户要求按后续Plan继续提交，已提交P2；第21节“P2尚未提交”是该节当时的历史状态。

### 22.1 作业链与冻结路径

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_P2_B_20261015/
  code/submit_p2.py                # P2提交入口，不用继承的旧P1 submit.py
  code/p2_guard.py                 # P1最终步长PASS、源码和复用结果hash
  code/p2_statistics.py            # 配对重采样/显式相关模截断GLS
  code/test_p2_statistics.py       # 计算节点统计/出图回归
  code/analyze_p2.py               # 十conf相关拟合与诊断PDF
  latest.json
  runs/P2_B_2026-10-15T190543.301835_0800/
```

依赖链：P1 audit **3853768** 成功 → P2 gate **3853774** → 九conf数组 **3853775_[0-8]** → 十conf拟合出图 **3853776**。代码内也复核P1报告PASS，不能只绕过Slurm依赖启动。

2026-10-15 19:10核对：P1 audit已完成，23点全部PASS，P2gate已开始在CPU节点运行。
双步长最大绝对差：p3=$9.1595\times10^{-6}$、p8=$9.7946\times10^{-6}$，Wilson基本/伴随=$2.2593\times10^{-7}/4.2813\times10^{-7}$，Polyakov pair基本/伴随=$1.6587\times10^{-8}/3.8911\times10^{-9}$，plaquette=$2.2081\times10^{-7}$。这是积分/测量回归精度，不是$d$的统计精度。

19:12更新：P2gate3853774已COMPLETED/0:0，`validation/P2_statistics_contract.json`为PASS，覆盖配对索引/先均值后log/负值保护/零方差与秩门/显式模删除/参数恢复/实际flow与T轴/全PDF流程。九conf数组3853775已放行，当前因Priority排队；统计3853776等待数组，不宣称十conf测量或d提取已完成。

19:14更新：数组task0..3（conf10040/10080/10120/10160）已分别在cpu052/cpu058/cpu059/cpu054 RUNNING，日志确认真实测量持续推进；task4..8仍因Priority排队。汇总3853776仍Dependency。所有任务24核，当前未发现流化/测量数值错误。

固定conf顺序仍是10000,10040,10080,10120,10160,10200,10240,10280,10320,10360；**只读复用P1 conf10000/epsilon.01，不重复测量**。gate保存该conf全部观测量/元数据hash，统计前复核。其余九conf沿同一23点tau网格、epsilon.01累计到3.8。两份测量C++源码必须与P1 hash相同；不增加HYP/stout/gauge fixing。

CPU四分区、24cores/64GiB、节点本地编译回归、无array并发上限。P1 conf10000主步长24核cpu052实测约821.24s（含全部观测，不含编译回归），不据此保证别的conf/节点耗时相同。纯规范量仍无blending权重，关联B分支定义不变；后续全部C3仍B、$|n_z|=3,4,5,6$、tsep/a7..12。P2没有生成新的C3/C2。

### 22.2 十组态统计与拟合处方

1. 先每conf起点/方向平均（不把方向或空间点当独立conf），再ensemble均值，再log；保留逐conf/方向/表示的原始复数与符号。
2. 1000组bootstrap、seed20261015，每组有放回抽10conf，全部tau/表示/距离/方法使用同一索引。误差std(ddof=0)，不再除 $\sqrt{1000}$。相邻两conf分块（仅5块）另给单圈误差敏感性，不能声称由此已控制自相关。
3. 使用原十conf的delete-one jackknife协方差；相关矩阵本征值小于最大值 $10^{-8}$ 的模式**显式删除**，不抬升零模、不暗改对角拟合。原始样本秩另以相对阈值 $10^{-12}$ 记录；要求sample rank至少Nparam、effective rank大于Nparam、白化列缩放设计满列秩。1000bootstrap不能造出十conf没有的信息。
4. 正信号/有限性门逐窗口覆盖全部replica；失败保留NaN及原因，不取abs/clip，不丢bad样本、不按想要结果删tau、不强制误差最小fallback。
5. Wilson有限T势先以 $-\log W=b+VT$ 做诊断，T窗6..10、8..12、10..16、8..16。相对线质量直接用

$$
-\log\frac{\langle W(R,T;\tau)\rangle}{\langle W(R,T;\tau_{\rm ref})\rangle}
=b(R,\tau,\tau_{\rm ref})+2\,\Delta m_R\,T,
$$

其中 $b$ 吸收不同空间周长/端点幅度项，不能因拟合线性就断言物理有限流势差消失。reference取.5/.8，R/a4..12并要求max(rf,rfref)/R<=.5；平台和窗口漂移仍需审查。
6. 单/双Polyakov分别是 $-\log(L/L_{\rm ref})/N_t$、$-\log(G/G_{\rm ref})/(2N_t)$；正ensemble/replica信号不足就不取log。有限自由能差未知，不能自动将它们等同纯线质量。
7. d候选比较flow窗口.5..1.8、.5..3.8，模型

$$
\Delta m=d\left(\frac1{\sqrt{8\tau}}-\frac1{\sqrt{8\tau_{\rm ref}}}\right),
\qquad
\Delta m=d\left(\frac1{\sqrt{8\tau}}-\frac1{\sqrt{8\tau_{\rm ref}}}\right)+c(\tau-\tau_{\rm ref}).
$$

物理量以格点单位保存；不含tau0，不自动选一个“最终d”。相关拟合保存rank/Q/chi2、每参数bootstrap与jackknife误差、有效率、窗口和模型。Q>=.01仅是预声明的拟合质量诊断，不升级物理状态。

### 22.3 产物与未解问题

所有数值/拟合/绘图均在CPU计算节点。run下 `analysis/` 保存：

- `bootstrap_indices.npz`：索引(1000,10)、conf_ids、seed。
- `per_conf_closed_loops.npz`：Wilson=(conf,tau,rep,dir,R,T)，单圈=(conf,tau,rep)，双圈=(conf,tau,rep,dir,R)，完整方向/复数保留。
- `single_loop_statistics.npz`：原均值、配对样本和block2敏感性。
- `fit_samples.npz`、`report.json`：全部窗口及失败状态，不造有限拟合值。
- `P2_closed_loops_and_relative_mass.pdf`：圈信号、窗口对照、有效势、d候选和失败说明。
- `complete.json`：分析及出图完成marker，不等于物理方案已定稿。

**待确认/未执行：**P3新Ward/full-propagator所需L32x64 valence action/mass/clover/$Z_V$没有可靠输入，本轮未猜参数提交。P4完整B C3未启动；不能把旧A或tau.5/tsep7诊断升级成新全量B数据。绝对residual-mass方案、有限NP-to-MSbar转换、有限T/flow畸变及十conf自相关仍未解决；P2不对已有矩阵元做减除/matching，不宣称完成连续极限或物理PDF。

本次会话标题：**L32x64闭合圈P2十组态missing-only提交及配对线质量诊断**。

## 23. L32x64 Wilson / Polyakov 线性发散拟合：形式、现有结果与图片（2026-10-17）

### 23.1 当前完成了哪两版分析

本节直接核对已有 `report.json`、系数表、完成标记及校验报告，不以“提交成功”代替“拟合完成”。第 22 节的十组态安排保留作历史；目前的闭合圈统计已经扩展为下列两个独立版本。

| 版本 | 实际共同组态数 | bootstrap 次数 / seed | 完成时间（北京时间） | 验证范围 |
|---|---:|---|---|---|
| 2000-bootstrap 快照版 | 883 | 2000 / 20261016 | 2026-10-16 13:52:26 | 全网格数据读回、Wilson 与 log 路线、signed Polyakov/profile、产物校验；`verification.json: PASS` |
| 全量自动汇总版 | 1700 | 1000 / 20261015 | 2026-10-17 04:39:14 | 全量测量及旧版 Wilson / Polyakov-log 分析完成；`analysis/complete.json: complete`，report hash 一致 |

快照版在 2026-10-16 13:15:03–13:15:12 扫描时，有 883 个组态的全部 23 个流时间均已完成，另有 817 个尚不完整；选择依据是完整性，不是圈的符号或信号。CPU 作业链 `3854653 → 3854654 → 3854655` 均为 `COMPLETED/0:0`。全量版测量数组 `3853828_[0-168]` 与分析作业 `3853829` 也均完成。

**不要混淆：**1700-conf 版虽然组态更多，使用的仍是旧的 1000-bootstrap 及较少 flow 窗口；它没有快照版的独立 signed-profile 分析及同样的 `verification.json` 校验包。尚未完成“1700 conf × 2000 bootstrap”的统一新版分析。两版窗口与分析实现有差别，数值变化不能全部归因于增加组态。

两版分析根目录分别为：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_boot2000_20261016

/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_allconf_B_20261015/runs/allconf_B_2026-10-15T205426.224554_0800
```

### 23.2 测量定义、表示与不可改变的符号

ensemble 为 `beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64`，体积 $32^3\times64$，$a=0.0897\,\mathrm{fm}$。从未 smear 的 thin gauge 开始，使用四维 Wilson RK3 flow，$\epsilon=0.01$，累计到 $\tau=3.8$，即 380 步；无 HYP/stout，也没有规范固定。链接方向为 $(x,y,z,t)$，格点坐标顺序为 $(t,z,y,x)$。

完整测量网格为

$$
\tau=\frac{t_f}{a^2}\in
\{0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,
0.50,0.55,0.60,0.65,0.80,1.0,1.4,1.8,2.2,2.6,3.0,3.4,3.8\}.
$$

$\tau=0$ 仅作原始数据与信号对照，不进入 $1/\sqrt{\tau}$ 拟合。以下用 $\mathcal R\in\{F,A\}$ 表示基本/伴随表示，以免和 Wilson loop 的空间边长 $R$ 混淆；令 $n_R=R/a$、$n_T=T_W/a$。

对同一闭合路径 holonomy $H$，每个局部路径先构造

$$
p_F(H)=\frac{\operatorname{Tr}_F H}{3},\qquad
p_A(H)=\frac{|\operatorname{Tr}_F H|^2-1}{8},
$$

再作起点、方向与组态平均。不能用 $|\langle\operatorname{Tr}H\rangle|^2$ 替换 $\langle|\operatorname{Tr}H|^2\rangle$。Wilson 使用 XT/YT/ZT 三平面，$n_R=0,\ldots,12$、$n_T=0,\ldots,16$，平均全部四维起点。单 Polyakov loop 绕完整周期时间方向 $N_t=64$，保留基本表示原始复数及伴随表示的有符号实数；没有费米子反周期负号。还测量三空间方向、$R/a=0,\ldots,12$ 的 Polyakov pair，未作 connected subtraction。

符号沿用第 2 节：

$$
r_f=\sqrt{8t_f},\qquad \frac{r_f}{a}=\sqrt{8\tau},\qquad
h_{\mathcal R}^{\mathrm{GF}}\sim e^{-m_{\mathrm{line},\mathcal R}L},
$$

$$
m_{\mathrm{line},\mathcal R}(t_f)=\frac{d_{\mathcal R}}{\sqrt{8t_f}},\qquad
a\,m_{\mathrm{line},\mathcal R}(\tau)=\frac{d_{\mathcal R}}{\sqrt{8\tau}},\qquad
\delta m_{\mathcal R}=-m_{\mathrm{line},\mathcal R}.
$$

$d_{\mathcal R}$ 是无量纲系数，**已含表示依赖，不再额外乘 Casimir**。基本表示对应 quark unpolarized/helicity/transversity 的 Wilson line；伴随表示对应 gluon unpolarized/helicity。两种表示不能平均，也不能把 $m_A$ 设成 $2m_F$。后续若使用减除因子，应与 $e^{+m_{\mathrm{line},\mathcal R}L}$ 的符号一致，但本节候选值还未获准用于正式减除。

这些是纯规范量，**没有 blending A/B 权重**。目录中的 `B` 只标记相关 quark 工作分支；不能向闭合圈测量中加入 C3 的 A/B 权重表。

### 23.3 Wilson loop 的实际两阶段拟合

**第一阶段：固定流时间对和空间距离，拟合有限 $T_W$ 的相对有效线质量。** 先对每份 ensemble/bootstrap/jackknife 求圈平均，再形成

$$
Y_{\mathcal R}(n_R,n_T;\tau,\tau_{\rm ref})
=-\log\frac{\overline W_{\mathcal R}(n_R,n_T;\tau)}
                 {\overline W_{\mathcal R}(n_R,n_T;\tau_{\rm ref})}
=b_{\mathcal R}+2n_T\,a\Delta m_{\mathcal R}^{\rm eff}.
$$

因此相对势的斜率为 $a\Delta V^{\rm eff}=2a\Delta m^{\rm eff}$；因子 2 对应两个时间方向静态线。$b$ 吸收不随 $n_T$ 变化的空间边、端点与幅度项。另以 $-\log\overline W=b+n_TaV^{\rm eff}$ 保存单流时间有限 $T$ 势诊断。

**这里的 $\Delta m^{\rm eff}$ 仍是 $\Delta V^{\rm eff}/2$，不是已经证明纯净的线自能差。** 若物理有限流势随 $\tau$ 变化，其差的一半也进入该斜率；有限 $T$ 的激发态污染亦可能存在。

实际扫描设置：

- $\tau_{\rm ref}=0.5,0.8$，$n_R=4,\ldots,12$。
- $n_T$ 窗口为 $[6,10]$、$[8,12]$、$[10,16]$、$[8,16]$，端点包含。
- 固定几何筛选为 $\max(\sqrt{8\tau},\sqrt{8\tau_{\rm ref}})/n_R\le0.5$。
- 几何筛选只控制空间边的流半径比例，不自动证明时间边的 flow overlap 或静态平台已受控。

**第二阶段：跨流时间拟合无量纲 $d_{\mathcal R}$。** 定义

$$
f(\tau;\tau_{\rm ref})
=\frac{1}{\sqrt{8\tau}}-\frac{1}{\sqrt{8\tau_{\rm ref}}}.
$$

比较两种模型：

$$
\begin{aligned}
\texttt{d\_only}:\quad
a\Delta m_{\mathcal R}^{\rm eff}
&=d_{\mathcal R}f(\tau;\tau_{\rm ref}),\\
\texttt{d\_plus\_c\_tau\_difference}:\quad
a\Delta m_{\mathcal R}^{\rm eff}
&=d_{\mathcal R}f(\tau;\tau_{\rm ref})
  +c_{\mathcal R}(\tau-\tau_{\rm ref}).
\end{aligned}
$$

$c_{\mathcal R}$ 是本诊断参数化的有限 flow 修正系数，不是已证明完整的高阶展开。两个模型在 reference 处均严格为零，因此拟合排除 reference 自身，避免把精确零当成额外数据。

883-conf 版 nominal flow 窗口为 $[0.5,1.8]$、$[0.5,3.8]$、$[1.0,3.8]$、$[1.4,3.8]$；1700-conf 旧版仅有前两个。**真正参与拟合的点以 `fit_taus` 为准**：几何 mask 会截短窗口，例如 nominal $[1.0,3.8]$ 在 $n_R=8$ 仅保留 $\{1.0,1.4,1.8\}$。不同 nominal 窗口可能落到同一批实际点，不能算作独立的稳定性证据。

### 23.4 Polyakov loop：log 路线与原始 signed-data 路线

单圈 $\overline L_{\mathcal R}(\tau)$ 和未扣连通项的空间 pair $\overline G_{\mathcal R}(R;\tau)$ 的 log 诊断分别定义为

$$
a\Delta m_{P,\mathcal R}^{\rm eff}
=-\frac{1}{N_t}\log\frac{\overline L_{\mathcal R}(\tau)}{\overline L_{\mathcal R}(\tau_{\rm ref})},
\qquad
a\Delta m_{PP,\mathcal R}^{\rm eff}
=-\frac{1}{2N_t}\log\frac{\overline G_{\mathcal R}(R;\tau)}{\overline G_{\mathcal R}(R;\tau_{\rm ref})}.
$$

分母 $N_t$ 与 $2N_t$ 分别来自一条与两条时间绕行线。代码对实投影的 ensemble 及全部所需重采样均执行正信号/有限性检查，随后才取 log，最后使用第 23.3 节的跨 flow 模型。**目前两版、两种表示的单圈与 pair 都没有通过全部所需 replica 正信号守卫的有效 log 系数窗口**；有的窗口还因预定点不足而失败。没有取绝对值、裁剪负值或删除失败 replica。

为避免对接近零且变号的原始信号强行取 log，883-conf / 2000-bootstrap 版另做带符号振幅的原始数据拟合：

$$
\overline L_{\mathcal R}(\tau)
=A_p\exp\{-d_{\mathcal R}[X(\tau)-X_p]\},\qquad
X(\tau)=\frac{N_t}{\sqrt{8\tau}}.
$$

固定 $d$ 时解析消去可正可负的振幅，再计算相关 $\chi^2(d)$ profile。数值扫描 $d\in[-10,10]$，4001 点、步长 0.005，并对每份 bootstrap 作连续细化。指数的数值归一化吸收入振幅，落盘字段为 `A_normalized_shape_center`，不可直接误称其为物理 $A_p$。

`[-10,10]` 是数值搜索范围，**不是先验，也不是置信区间边界**。profile 按最近的 $\Delta\chi^2=1$ 与 $3.8414588$ 交点记录 68%/95% 诊断区间；未闭合端保持 `null/open`，不能用搜索端点代替。局部模型秩、Q 值、profile 闭合和全部 bootstrap 可用率分别记录，互不替代。

### 23.5 bootstrap、协方差和误差解释

先作组态内起点/方向平均，组态才是重采样单位；两版各自对全部 $\tau$、表示、几何和方法共用同一组态顺序及 bootstrap 索引。快照版索引形状为 $(2000,883)$，全量旧版为 $(1000,1700)$，每个样本均有放回抽 $N_{\rm cfg}$ 次。它们不是第 22 节原十组态的配对 bootstrap，也不能直接认为已和 quark 矩阵元样本配对。

中心值取原 ensemble 均值上的估计量；参数统计误差取 bootstrap 标准差 `ddof=0`，不再除 $\sqrt{N_{\rm bootstrap}}$。拟合使用原始组态的固定 delete-one jackknife 协方差，并将 log 与第一阶段拟合的相关性传播到第二阶段。

原始相关矩阵 sample rank 相对阈值为 $10^{-12}$；低于最大相关本征值 $10^{-8}$ 的模**显式删除**，不是抬升零模，也不是切换为对角拟合。要求原始样本秩至少为自由参数数、保留有效秩大于参数数、白化并列缩放的模型设计矩阵满列秩。bootstrap 次数增加不能制造原始样本没有的信息。

任一预定所需样本非正、非有限或拟合失败，都保留失败状态，不按想要的结果删点，不取 abs/clip，不用误差最小的拟合替代。Polyakov 不完整的可用样本集合不能用来生成貌似完整的“2000-bootstrap 误差”。$Q\ge0.01$ 只是此分析预声明的拟合质量诊断，既不等同于物理有效，也不替代 profile/窗口/系统误差检查。

883-conf 版的 circular moving-block 长度 2、4 检查只测试信号误差敏感性，不丢弃奇数余项；1700-conf 旧版使用非重叠 block2，共 850 块。自相关时间尚未确定，快照组态还可能有不均匀间隔；这些检查不能宣称已控制自相关。

### 23.6 Wilson 候选系数：883 conf、2000 bootstrap

下表均取 $\tau_{\rm ref}=0.5$；`d+c` 表示加入 $c(\tau-\tau_{\rm ref})$ 的模型。数值来自独立窗口，误差**仅为统计误差**，不是系统误差包络，也不是 AIC 合并的最终结果。

| 表示 | $n_R$ | $n_T$ 窗 | 实际 $\tau$ 点 | 模型 | $d\pm\sigma_{\rm stat}$ | $\chi^2/\mathrm{dof}$ | $Q$ |
|---|---:|---|---|---|---|---:|---:|
| $F$ | 8 | 10–16 | 1.0, 1.4, 1.8 | d+c | $0.341650\pm0.006203$ | 3.62252 | 0.05700 |
| $F$ | 9 | 10–16 | 1.0, 1.4, 1.8, 2.2 | d+c | $0.343158\pm0.007613$ | 2.37817 | 0.09272 |
| $A$ | 5 | 8–12 | 0.55, 0.60, 0.65 | d only | $0.887313\pm0.008190$ | 0.05219 | 0.94915 |
| $A$ | 5 | 8–12 | 0.55, 0.60, 0.65 | d+c | $0.869996\pm0.054965$ | 0.00725 | 0.93213 |

对应前三个含修正项的窗口，其 $c$ 分别为：$F,n_R=8$：$-0.0102804\pm0.0004010$；$F,n_R=9$：$-0.0094673\pm0.0002978$；$A,n_R=5$：$-0.0045895\pm0.0144461$。

基本表示 576 个 Wilson 跨 flow 候选窗口中，312 个有完整有限统计误差，23 个满足 $Q\ge0.01$，**这 23 个全部需要 d+c 模型**。伴随表示相应为 576 个中 21 个有限、12 个满足 Q 诊断。上述计数含不同 nominal 窗口，不表示彼此独立。

作为反例，同一 $F,n_R=8,n_T=10$–16、实际 $\tau=1.0,1.4,1.8$ 的 d-only 拟合给出 $d=0.488519\pm0.002454$，但 $\chi^2/\mathrm{dof}=317.90$、$Q\simeq8.63\times10^{-139}$，不能因为误差小就采用。伴随的高 Q 只来自很窄的三个流时间点；允许修正项后误差明显增加，尚不能证明 $d_A$ 稳定。

### 23.7 Wilson 候选系数：1700 conf、1000 bootstrap

下表仍取 $\tau_{\rm ref}=0.5$。基本表示的实际流时间集合记为

$$
S_F=\{0.55,0.60,0.65,0.80,1.0,1.4,1.8\}.
$$

| 表示 | $n_R$ | $n_T$ 窗 | 实际 $\tau$ 点 | 模型 | $d\pm\sigma_{\rm stat}$ | $\chi^2/\mathrm{dof}$ | $Q$ |
|---|---:|---|---|---|---|---:|---:|
| $F$ | 9 | 10–16 | $S_F$ | d+c | $0.347805\pm0.006814$ | 1.19109 | 0.31234 |
| $F$ | 8 | 10–16 | $S_F$ | d+c | $0.357274\pm0.003795$ | 3.85958 | 0.00387（未过 0.01） |
| $A$ | 5 | 8–12 | 0.55, 0.60, 0.65 | d only | $0.884488\pm0.005941$ | 0.13535 | 0.87341 |
| $A$ | 5 | 8–12 | 0.55, 0.60, 0.65 | d+c | $0.863442\pm0.038426$ | 0.00193 | 0.96499 |

含修正项的 $c$：$F,n_R=9$ 为 $-0.0098297\pm0.0003488$，$F,n_R=8$ 为 $-0.0093787\pm0.0001921$，$A,n_R=5$ 为 $-0.0055846\pm0.0101155$。

基本表示 288 个候选中 192 个有完整有限统计误差、9 个满足 $Q\ge0.01$，且这 9 个也全部为 d+c 模型；伴随表示为 288 个中 24 个有限、14 个满足 Q 诊断。$F,n_R=9$ 的拟合改善并不等于已经证明 $R$ 平台：相邻 $n_R=8$ 在相同流时间点上未通过 Q 诊断。伴随三点窗口依然太窄，模型修正与窗口系统误差没有消失。

因此可以报告“Wilson 给出 $d_F\sim0.34$、窄窗口 $d_A\sim0.86$–0.89 的候选诊断”，但**不能把上述范围当作最终误差，也不能宣布已获得可直接用于 PDF 的正式非微扰线质量**。

### 23.8 Polyakov signed-profile 的现有结果与失败边界

以下仅来自 **883-conf / 2000-bootstrap** 原始 signed-data 分析。`open` 表示在搜索范围内没有闭合交点，不等于把 $\pm10$ 当作区间端点。$d$ 列只是诊断极小值，不带可用的完整 bootstrap 误差。

| 表示 | nominal $\tau$ 窗 | 诊断 $d$ | $\chi^2/\mathrm{dof}$ | $Q$ | 可用 replica / 2000 | 触边 replica | 68% profile 诊断区间 |
|---|---|---:|---:|---:|---:|---:|---|
| $F$ | 0.5–1.8 | 0.18842 | 1.04424 | 0.39412 | 1842 | 158 | [0.06209, 0.36735] |
| $F$ | 0.5–3.8 | 0.05228 | 0.95428 | 0.48630 | 1848 | 152 | [−0.03207, 0.13087] |
| $F$ | 1.0–3.8 | 0.10940 | 0.83201 | 0.54483 | 1722 | 234 | [0.02423, 0.21552] |
| $F$ | 1.4–3.8 | 0.10838 | 0.99811 | 0.41704 | 1730 | 270 | [0.01744, 0.21894] |
| $A$ | 0.5–1.8 | −10（搜索触边） | 1.13533 | 0.33859 | 986 | 1014 | [open, −0.50402] |
| $A$ | 0.5–3.8 | −10（搜索触边） | 1.96901 | 0.02715 | 1129 | 871 | [open, −0.57042] |
| $A$ | 1.0–3.8 | −10（触边且中央模型秩为 1） | 2.95042 | 0.00702 | 963 | 768 | [open, −0.57736] |
| $A$ | 1.4–3.8 | −0.34347 | 2.92987 | 0.01197 | 1504 | 496 | [open, 0.29230] |

四个基本表示窗口的中心振幅均为负，95% profile 均至少一端未闭合；伴随表示的 68% profile 全部至少一端未闭合。全部八个窗口都缺少完整的 2000-replica 参数误差，落盘 `d_bootstrap_error_ddof0_complete_only=null`。可用数的缺额不全由触边造成，部分窗口还有模型秩或其他失败，不能由触边数推断其余全部成功。

**解释限制：**这里不能把 −10 解读为测得的负线性发散系数，也不能把负振幅拟合的正 $d_F$ 自动解读成物理线质量。某些窗口 Q 看似正常，仅说明该带符号有限参数模型在保留协方差模式上的残差不大，不证明信号、模型或参数误差已受控。全量 1700-conf 版尚未做此新版 signed-profile 分析，目前两版 Polyakov 都没有可用于正式减除的最终 $d_{\mathcal R}\pm\sigma$。

### 23.9 拟合图片的绝对地址与阅读顺序

**A. 883-conf / 2000-bootstrap：**

1. [P2_closed_loops_and_relative_mass.pdf（11 页）](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_boot2000_20261016/analysis/P2_closed_loops_and_relative_mass.pdf)

   绝对地址：`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_boot2000_20261016/analysis/P2_closed_loops_and_relative_mass.pdf`。
   第 1 页单 Polyakov 实/虚部与 block2 信号；第 2 页 pair；第 3–6 页 Wilson 相对有效线质量的四个 T 窗口；第 7/8 页基本/伴随有限 T 势；第 9/10 页基本/伴随 d 候选；第 11 页状态说明。

2. [Polyakov_signed_amplitude_profile.pdf（8 页）](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_boot2000_20261016/analysis/Polyakov_signed_amplitude_profile.pdf)

   绝对地址：`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_boot2000_20261016/analysis/Polyakov_signed_amplitude_profile.pdf`。
   第 1–4 页基本表示，第 5–8 页伴随表示，各按 $[0.5,1.8]$、$[0.5,3.8]$、$[1.0,3.8]$、$[1.4,3.8]$ 排列；左侧原始信号和模型，右侧 $\chi^2(d)$ profile。

3. [line_mass_coefficient_comparison.pdf（2 页）](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_boot2000_20261016/analysis/line_mass_coefficient_comparison.pdf)

   绝对地址：`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_boot2000_20261016/analysis/line_mass_coefficient_comparison.pdf`。
   第 1 页选取 reference=0.5、T 窗 8–12 的 Wilson 候选；这里画的是全部有限候选，**包含低 Q 窗口，不是质量筛选后的最终值**。第 2 页 signed Polyakov 候选的叉号没有完整 bootstrap 误差，不能按普通精确测量点解读。

**B. 1700-conf / 1000-bootstrap：**

4. [全量版 P2_closed_loops_and_relative_mass.pdf（11 页）](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_allconf_B_20261015/runs/allconf_B_2026-10-15T205426.224554_0800/analysis/P2_closed_loops_and_relative_mass.pdf)

   绝对地址：`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_allconf_B_20261015/runs/allconf_B_2026-10-15T205426.224554_0800/analysis/P2_closed_loops_and_relative_mass.pdf`。
   页序同类：信号、四个 T 窗、有效势、d 候选及状态说明。这份与 A.1 同名但输入统计量不同；全量旧版没有独立的 signed-profile PDF。

### 23.10 机器可读结果、代码与后续工作

快照版 `analysis/` 中：

- [d_coefficients.csv](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_boot2000_20261016/analysis/d_coefficients.csv)：全部方法/表示/几何/实际 flow 点、系数、误差及状态；核对时先看 `fit_taus`、`bootstrap_usable`、`Q` 与物理状态，不能只看 d 列。
- [report.json](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_boot2000_20261016/analysis/report.json)：Wilson、log 路线的全部窗口、秩、Q、协方差及失败原因；`fit_samples.npz` 保存配套样本。
- [Polyakov_signed_profile_report.json](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_boot2000_20261016/analysis/Polyakov_signed_profile_report.json)：signed 振幅、profile 区间、每种失败计数；配套 `Polyakov_signed_profile_samples.npz`。
- `bootstrap_indices.npz`、`per_conf_closed_loops.npz`、`single_loop_statistics.npz`、`wilson_statistics.npz`：索引、组态 ID、原始圈与统计量；原始 Wilson 轴为 `(conf,tau,rep,dir,R,T)`，单圈为 `(conf,tau,rep)`，pair 为 `(conf,tau,rep,dir,R)`。
- [verification.json](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_boot2000_20261016/analysis/verification.json)：`status=PASS`、`no_subtraction_applied=true`，两种表示均 `ready_for_PDF_subtraction=false`，保存产物 SHA256。根目录 `snapshot.json`、`manifest.json`、`submission.json`、`complete.json` 固定快照与执行链。

全量版结果为 [report.json](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_allconf_B_20261015/runs/allconf_B_2026-10-15T205426.224554_0800/analysis/report.json)、同目录 `fit_samples.npz`、`bootstrap_indices.npz` 和 [complete.json](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_allconf_B_20261015/runs/allconf_B_2026-10-15T205426.224554_0800/analysis/complete.json)。不要用上游 `latest.json` 单独证明分析完成，也不要声称旧版存在快照版的系数 CSV 或 profile 校验包。

快照版冻结代码在：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_boot2000_20261016/code/analyze_p2.py
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_boot2000_20261016/code/p2_statistics.py
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_boot2000_20261016/code/polyakov_profile.py
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_boot2000_20261016/code/finalize_analysis.py
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_L32x64_boot2000_20261016/code/test_loop_analysis.py
```

这些是本节数值与图表的实现依据。已冻结 request/hash 的目录不应原地修改源码或重复启动；若扩展分析，应新建带日期版本，将全部拟合与绘图提交至 CPU 计算节点。

**下一步（尚未执行/待确认）：**

1. 用已完成的共同 1700 conf 建立独立 **2000-bootstrap** 新版分析，统一窗口与 signed-profile 诊断，保存新索引；这不是把 1000-bootstrap 文件改名。
2. 检查 Wilson 第一阶段的有限 T 平台、相邻 R、reference=0.5/0.8、实际 flow 点及 d-only/d+c 的窗口漂移；分开评估有限流势变化与线自能。空间几何 mask 通过不等于已满足 $a\ll r_f\ll R,T_W$。
3. 在去重后的实际窗口上报告统计和模型/几何系统误差；目前没有最终 AIC 平均或获准用于减除的唯一 $d_F,d_A$。
4. 对 Polyakov 先审查原始信号、振幅符号、profile 开口及全部样本覆盖；不能用取 abs、裁剪或剔除 replica 来制造系数误差。扩大数值扫描本身不能解决缺乏物理约束的问题。
5. 确定自相关处理、绝对 residual-mass 的有限重整化条件及 NP-to-$\overline{\mathrm{MS}}$ 有限转换，再设计与矩阵元配对的误差传播。当前单格距结果不能替代连续极限，也不能声称已经完成正式 PDF 重整化。

本次文档更新日期：**2026-10-17（Asia/Shanghai）**。本次会话标题：**L32x64 Wilson/Polyakov 线性发散拟合双版本结果与图片路径归档**。本次未重新测量、拟合、提交作业或更改 PDF 矩阵元。

## 24. Wilson Cornell、Wilson relative 与静态势：定义、推导和当前代码（2026-10-17）

### 24.1 先给出两条路线的准确含义

**Wilson Cornell 和 Wilson relative 不是两种新的圈测量，而是对同一套 Wilson-loop 数据的两种分析路线。** 共同的第一步都是：固定空间距离与流时间，由圈随欧式时间边长的指数衰减提取一个势。之后才分支：

$$
\begin{aligned}
\text{Cornell:}\quad
\overline W(R,T_W;t_f)
&\ \longrightarrow\ V_{\rm eff}(R;t_f)
\ \longrightarrow\ \{C_0(t_f),\sigma(t_f),E(t_f)\}
\ \longrightarrow\ d,\\
\text{relative:}\quad
\overline W(R,T_W;t_f)
&\ \longrightarrow\ V_{\rm eff}(R;t_f)
\ \longrightarrow\ \frac{V_{\rm eff}(R;t_f)-V_{\rm eff}(R;t_{f,r})}{2}
\ \longrightarrow\ d.
\end{aligned}
$$

| 项目 | Wilson Cornell | Wilson relative（本项目代码定义） |
|---|---|---|
| 第一层输入 | 同一表示、流时间、空间距离下的时间序列 $\overline W$ | 相同 |
| 提取势后扫描什么 | 固定 $t_f$，扫描 $R$ | 固定 $R$，扫描 $t_f$ |
| 拟合对象 | 势的 Cornell 常数 $C_0(t_f)/2$ | 跨流时间的势差 $a\Delta V(R;\tau,\tau_r)/2$ |
| 是否需要空间 Cornell 模型 | 需要 $C_0+\sigma a^2 n+E/n$ | 不需要 |
| reference | 不作跨 flow reference 相减，用截距 $b$ 吸收有限常数 | 每个候选窗取一个 $\tau_r$，逐样本相减 |
| 主要额外系统误差 | 空间模型、$C_0$ 与 $\sigma,E$ 相关、空间窗依赖 | 相互作用本身的 flow 依赖未必抵消，reference/距离依赖 |
| 共同限制 | 有限时间平台、格距伪影、flow 尺度、统计与模型误差 | 相同 |

两条路线共享组态、bootstrap 索引和第一层势样本，**不是统计独立的两次测量**。可以交叉检查，但不能当作独立结果简单加权平均。它们目前都不自动给出可直接用于 PDF 的完整重整化因子。

### 24.2 Wilson loop 为什么与静态势有关？

本节沿用第 23 节的 $\mathcal R\in\{F,A\}$ 表示颜色表示，而 $R$ 表示空间距离。令表示维数 $D_F=3,D_A=8$，以免把维数与待拟合的 $d_{\mathcal R}$ 混用。定义

$$
W_{\mathcal R}(R,T_W;t_f)
=\frac{1}{D_{\mathcal R}}\left\langle
\operatorname{ReTr}_{\mathcal R}\,
\mathcal P\exp\!\left(ig\oint_{\square(R,T_W)}B_\mu(t_f,x)\,dx_\mu\right)
\right\rangle.
$$

下面的谱解释首先针对标准未流化 Wilson loop、适用的转移矩阵设置及足够长的时间方向。考虑空间两点间距为 $R$ 的一对**静态**源和反源：

1. 在欧式时间 0，用空间 Wilson line 把两个源连起来，构成规范不变的静态对态。
2. 令两个源传播 $T_W$。无限重源的空间位置不变，传播正好由两条时间 Wilson line 表示。
3. 在末端用反向空间线闭合并消去静态对。整条路径就是矩形 Wilson loop。

令 $|\Phi_R\rangle$ 是上述空间串构造的试探态。插入静态源扇区的能量本征态，有

$$
\begin{aligned}
W_{\mathcal R}(R,T_W;0)
&\propto\langle\Phi_R|e^{-(H-E_{\rm vac})T_W}|\Phi_R\rangle\\
&=\sum_k A_{k,\mathcal R}(R)\,e^{-E_{k,\mathcal R}(R)T_W}\\
&=A_{0,\mathcal R}(R)e^{-V_{\mathcal R}(R)T_W}
\left[1+\sum_{k>0}\frac{A_{k,\mathcal R}}{A_{0,\mathcal R}}
e^{-\Delta E_{k,\mathcal R}T_W}\right],
\end{aligned}
$$

其中 $V_{\mathcal R}=E_{0,\mathcal R}$ 是相对于真空、包含该调节/重整化处方下静态自能的最低静态对能量；$\Delta E_k=E_k-E_0$。因此

$$
\boxed{
V_{\mathcal R}(R)
=-\lim_{T_W\to\infty}\partial_{T_W}\ln W_{\mathcal R}
=-\lim_{T_W\to\infty}\frac{\ln W_{\mathcal R}}{T_W}.
}
$$

归一化迹的常数、空间串的重叠因子影响振幅 $A_0$，不改变长时间斜率。这里不需要求解 quark propagator；静态源的传播已由时间链接乘积给出。文献常把这个量称为 static energy 或 static potential；本项目的“静态势”指这个从圈衰减抽取的量，并不宣称已分离有效场论中更细的 potential/ultrasoft 分量。[本节补充文献 11，Sec. II A]

有限时间方向上还应检查绕时/热贡献和接近 $N_ta$ 的影响；本项目时间窗远短于完整绕时长度，但这本身不能替代平台与边界效应检验。

**四维 flow 的限定：**本项目的 $B_\mu(t_f,x)$ 由四维热流得到，包含原始欧式时间邻域的信息，并非只对空间串作三维 smearing。因此不能不加条件地把上式的正谱/转移矩阵解释原样赋予任意正 flow、任意小 $T_W$ 的圈。我们在 $T_W\gg r_f$、$R\gg r_f$ 的候选区间用长时间指数来定义/估计 flowed static energy，并检查窗口稳定性；**当前产物仍标为有限窗有效静态势**。经验尺度 mask 通过不证明反射正性，也不证明已到基态平台。文献 11 Appendix A 展示了 flowed Wilson loop 的大时间指数衰减；其数值例子不是本 ensemble 的平台证明。

### 24.3 我们的代码怎样从圈提取势？

定义格点整数边长

$$
n\equiv R/a,\qquad \ell\equiv T_W/a,\qquad
\tau=t_f/a^2,\qquad r_f/a=\sqrt{8\tau}.
$$

若一个指数主导，最近邻有效势为

$$
aV_{\mathcal R,\rm eff}(n,\ell;\tau)
=\ln\frac{\overline W_{\mathcal R}(n,\ell;\tau)}
{\overline W_{\mathcal R}(n,\ell+1;\tau)}.
$$

它随 $\ell$ 出现稳定平台才支持单指数近似。现在尺度窗代码复用的实际估计量是**对整个时间窗作相关直线拟合**，不是把相邻点有效势作无权平均：

$$
\boxed{
-\ln\overline W_{\mathcal R}(n,\ell;\tau)
=b_{W,\mathcal R}(n,\tau)
+\ell\,[aV_{\mathcal R,\rm eff}(n;\tau)] .
}
$$

$b_W=-\ln A_0$ 对应时间无关的振幅项。第一层斜率是 $aV$，尚未除以 2。最新尺度窗分析采用 $\ell\in[10,16]$ 或 $[8,12]$（均包含端点），分别保存，不混成一个窗口；没有在本版做多态拟合。L32x64 的 $a=0.0897\,\mathrm{fm}$，以上物理时间窗分别约为 $[0.897,1.4352]\,\mathrm{fm}$ 和 $[0.7176,1.0764]\,\mathrm{fm}$。

统计顺序为：同一组态内完成起点/方向平均、使用声明的实部投影，再对 ensemble 或对应 bootstrap 的组态求平均，**最后取平均值的对数**。不是先对每个组态取 $\log W$ 再平均。若所需平均值不正，正实 log 链不可构造，保留失败状态；不能取绝对值、强行改号或删掉坏 replica 来得到“完整误差”。

目前所谓 $V_{\rm eff}$ 还包含有限时间窗、激发态和有限 flow 的影响，不能因为时间拟合有一个有限斜率，就把它称为已验证的物理基态势。完整 bootstrap 误差与 $Q$、残差、时间窗稳定性也是不同检查。

### 24.4 为什么势里是两条线的自能，为什么要除以 2？

在流半径小于路径边长的长线区间，周长自能给出示意分解

$$
W_{\mathcal R}(R,T_W;t_f)
\simeq e^{-2m_{\mathrm{line},\mathcal R}(t_f)(R+T_W)}
\mathcal A_{\mathcal R}(R,t_f)
e^{-V_{\rm int,\mathcal R}(R;t_f)T_W}\times\cdots .
$$

省略号包括端点/拐角、对数和非渐近流修正；这不是对任意小圈成立的精确分解。取负对数并按时间边长求斜率，有

$$
\begin{aligned}
-\ln W_{\mathcal R}
&\simeq
\underbrace{2m_{\mathrm{line},\mathcal R}R-\ln\mathcal A_{\mathcal R}}
_{\text{不随 }T_W\text{ 增长，进入振幅/截距}}
+\left[2m_{\mathrm{line},\mathcal R}+V_{\rm int,\mathcal R}(R;t_f)\right]T_W.
\end{aligned}
$$

即在所声明的分离处方及近似下，

$$
\boxed{V_{\mathcal R}(R;t_f)
=2m_{\mathrm{line},\mathcal R}(t_f)+V_{\rm int,\mathcal R}(R;t_f).}
$$

两个时间长边分别对应静态源与反源，所以自能是 $2m$；两条空间边的 $2mR$ 不再额外加到势的时间斜率中。面积项若表现为 $\sigma RT_W$，提取势后则是 $\sigma R$，是相互作用的一部分，**不是同一个 UV 周长自能**。

沿用本文符号，定义

$$
q(\tau)\equiv\frac{1}{\sqrt{8\tau}},\qquad
m_{\mathrm{line},\mathcal R}(t_f)\simeq\frac{d_{\mathcal R}}{r_f},\qquad
a m_{\mathrm{line},\mathcal R}\simeq d_{\mathcal R}q(\tau),\qquad
\delta m_{\mathcal R}=-m_{\mathrm{line},\mathcal R}.
$$

这里的常数 $d_{\mathcal R}$ 是窗口有效模型；跑动、对数及格距修正可能使它依赖尺度。$d_{\mathcal R}$ 已含表示因子，不再乘 $C_F$ 或 $C_A$。若将来获准作 line subtraction，裸线的 $e^{-mL}$ 应乘 $e^{+mL}$，不能改变第 2 节的负 $\delta m$ convention。

**一个可直接理解 flow 修正的最低阶例子。** 连续、固定耦合的高斯流近似下，两条时间长线的交换项为

$$
V^{(1)}_{\rm int,\mathcal R}(R;t_f)
=-C_{\mathcal R}\alpha_s\int\frac{d^3\mathbf k}{(2\pi)^3}
\frac{4\pi e^{-2t_f\mathbf k^2+i\mathbf k\cdot\mathbf R}}{\mathbf k^2}
=-\frac{C_{\mathcal R}\alpha_s}{R}\,
\operatorname{erf}\!\left(\frac{R}{r_f}\right).
$$

结合第 4 节的单线自能，得到

$$
V^{(1)}_{\mathcal R}(R;t_f)
=\frac{2C_{\mathcal R}\alpha_s}{\sqrt\pi\,r_f}
-\frac{C_{\mathcal R}\alpha_s}{R}\operatorname{erf}(R/r_f).
$$

这仅是帮助理解的最低阶连续推导，不是给当前全套数据强加的拟合式。它明确说明：改变 flow 不仅改变第一项自能，也改变第二项相互作用；只有 $R/r_f$ 足够大时，后者才近似回到不随 flow 变化的 $-C_{\mathcal R}\alpha_s/R$。不能在 $R\sim r_f$ 时，把所有 flow 依赖都归给线自能。[由文献 1、3 的流核和第 4 节积分直接推导；文献 11 讨论了静态相互作用的有限流修正。]

### 24.5 Wilson Cornell：先拟合空间形状，再拟合常数的 flow 依赖

**步骤一：**用第 24.3 节时间拟合得到每个 $n,\tau$ 的 $aV_{\rm eff}(n;\tau)$。

**步骤二：**在同一个 $\tau$ 上，选多个空间距离，用 Cornell 型模型

$$
V_{\mathcal R}(R;t_f)
=C_{0,\mathcal R}^{\rm phys}(t_f)
+\sigma_{\mathcal R}(t_f)R+\frac{E_{\mathcal R}(t_f)}{R}
$$

拟合。代码实际使用的无量纲形式是

$$
\boxed{
aV_{\mathcal R,\rm eff}(n;\tau)
=C_{0,\mathcal R}(\tau)
+[\sigma_{\mathcal R}(\tau)a^2]\,n
+\frac{E_{\mathcal R}(\tau)}{n},\qquad
C_{0,\mathcal R}=a C_{0,\mathcal R}^{\rm phys}.
}
$$

这对应三列设计矩阵 $(1,n,1/n)$ 和参数名 `C0,sigma_a2,E`。常见写法是 $-\alpha_V/R+\sigma R+C_0^{\rm phys}$；代码用 $+E/n$，所以吸引的 Coulomb 项通常对应 $E<0$，不能因为采用加号就手动翻转数据。当前代码对每个 flow 分别拟合三参数，没有预设不同 flow 共用同一个 $\sigma$ 或 $E$，也没有施加 $\sigma>0$ 的 prior。

$C_0$ 是拟合模型中的空间常数，**不是 $V(0)$**：Cornell 的 $1/R$ 项在 $R=0$ 奇异，而实际空间窗不包含 0。有限且较窄的空间窗中，$C_0,\sigma,E$ 可能强相关，常数的提取可以明显依赖空间模型和窗口。

**步骤三：**因为时间斜率中含 $2m$，若空间有限常数近似 flow 独立，可写

$$
C_{0,\mathcal R}(\tau)
\simeq2d_{\mathcal R}q(\tau)+2b_{\mathcal R}.
$$

最新代码实际比较

$$
\begin{aligned}
\texttt{d\_b}:\quad
\frac{C_{0,\mathcal R}(\tau)}2
&=d_{\mathcal R}q(\tau)+b_{\mathcal R},\\
\texttt{d\_b\_c\_tau}:\quad
\frac{C_{0,\mathcal R}(\tau)}2
&=d_{\mathcal R}q(\tau)+b_{\mathcal R}+c_{\mathcal R}\tau.
\end{aligned}
$$

所以图名 **Wilson Cornell** 可能指“固定 flow 的空间 Cornell 拟合图”，也可能指“从 Cornell 常数得到 $d$ 的跨 flow 图”；必须同时看文件编号与纵轴。不是把原始 $W(R,T_W)$ 直接对 Cornell 函数拟合。

单个 flow 的 $C_0/2$ 仍包含有限加性常数，**不能直接设成单条线的纯 UV 质量**。跨 flow 的 $d$ 也依赖上述有效模型；要唯一指定有限 residual-mass 部分，还需给出有限重整化条件。基本表示的 Cornell 模型也不能无检验推广到所有距离/伴随表示：动力学夸克的 string breaking、伴随源屏蔽及短距 flow 变形均可能破坏它。

### 24.6 Wilson relative：固定同一个距离，对两个 flow 的势作差

“relative”在本项目新代码中严格指

$$
\boxed{
Y_{\mathcal R}(n;\tau,\tau_r)
\equiv\frac{aV_{\mathcal R,\rm eff}(n;\tau)
-aV_{\mathcal R,\rm eff}(n;\tau_r)}2.
}
$$

这里 $n$、表示和时间拟合窗保持相同，只改变 flow。由第 24.4 节的分解，

$$
Y_{\mathcal R}
=a[m_{\mathrm{line},\mathcal R}(\tau)-m_{\mathrm{line},\mathcal R}(\tau_r)]
+\frac a2[V_{\rm int,\mathcal R}(R;\tau)-V_{\rm int,\mathcal R}(R;\tau_r)]
+\text{有限窗修正}.
$$

如果相互作用的 flow 变化足够小，才可以近似把第一项作为主导。实际比较的模型是

$$
\begin{aligned}
\texttt{d\_only}:\quad
Y_{\mathcal R}
&=d_{\mathcal R}[q(\tau)-q(\tau_r)],\\
\texttt{d\_c\_tau\_difference}:\quad
Y_{\mathcal R}
&=d_{\mathcal R}[q(\tau)-q(\tau_r)]
+c_{\mathcal R}(\tau-\tau_r).
\end{aligned}
$$

其中 $c(\tau-\tau_r)$ 是诊断性修正模型，不是完整高阶展开的证明。reference 处 $Y=0$ 为构造恒等式，不能作为一个有独立误差的额外数据点；代码从拟合向量和协方差中排除它。

**为什么它可以少拟合一个常数？** 若同一距离的势具有不随 flow 变化的有限部分，它在相减中抵消，不需要先用多个距离估计 Cornell 常数。因此 relative 不依赖空间 Cornell 参数化；但残留的相互作用 flow 依赖、有限时间污染仍然存在，通常要看不同 $R$ 的候选 $d$ 是否相容。

**它不能确定什么？** 势差不受 $m(\tau)\mapsto m(\tau)+m_{\rm fin}$ 影响，所以只能确定跨 flow 的变化以及给定模型中的系数，不能单凭它恢复被消掉的绝对有限质量常数。拟合出 $d$ 不等于已经固定完整重整化方案。

**reference 的版本区别：**

- 第 23 节旧 P2 流程对圈的跨 flow log-ratio 拟合时间斜率，扫描固定 $\tau_r=0.5,0.8$；该节描述仍对应旧版本。
- 当前 `line_mass_scale_windows_20261017` 则先分别拟合各 flow 的时间斜率，再在势样本层面相减，且 $\tau_r$ 是**每个尺度允许候选窗中的最低 flow time**。例如 `low08` 在有 $0.8$ 点时取 $\tau_r=0.8$，而 `low14` 取自己的最低合格点；不能把它们全都标成 reference=0.5。

对同一 replica $b$，代码计算

$$
Y^{(b)}(\tau;\tau_r)
=\tfrac12[aV^{(b)}(\tau)-aV^{(b)}(\tau_r)],
$$

再用这些共同 reference 的配对样本建立跨 flow 的协方差。误差关系为

$$
\operatorname{Var}(Y)
=\frac14\left[\operatorname{Var}(aV_\tau)+\operatorname{Var}(aV_{\tau_r})
-2\operatorname{Cov}(aV_\tau,aV_{\tau_r})\right].
$$

不能把两个 flow 独立重采样，也不能忽略协方差而简单误差平方相加。

从大时间形式也可推导

$$
-\ln\frac{\overline W_{\mathcal R}(n,\ell;\tau)}
{\overline W_{\mathcal R}(n,\ell;\tau_r)}
=\Delta b_W+\ell\,a\Delta V_{\mathcal R},\qquad
Y_{\mathcal R}=-\frac12\lim_{\ell\to\infty}\frac1\ell
\ln\frac{\overline W_{\mathcal R}(n,\ell;\tau)}
{\overline W_{\mathcal R}(n,\ell;\tau_r)}.
$$

这是旧的直接 loop-ratio 时间斜率路线与新的势差路线的物理联系。**有限样本下，两者的相关最小二乘估计量不保证数值相同**：当前新代码是“分别用各 flow 的时间协方差拟合后相减”，不是对 loop-ratio 统一作一次联合时间拟合。只有在相应线性估计/权重一致等额外条件下，才可能严格等价。

**不要混淆另一种常见的 relative potential：**

$$
V(R;t_f)-V(R_0;t_f)
=V_{\rm int}(R;t_f)-V_{\rm int}(R_0;t_f).
$$

它在**同一个 flow、不同距离**之间作差，会把想测的 $2m(t_f)$ 直接消掉；静态力 $\partial_R V$ 和 Creutz ratio 也会消掉该空间常数。这些量适合研究相互作用、设置有限势的零点，不能单独恢复绝对线自能。文献 11 Eq. (3) 采用这种空间 reference 重整化势，**不等于本项目图里的 Wilson relative**。

### 24.7 最新尺度约束版本：窗口、统计与候选结果

冻结版本根目录为

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017
```

仍为 L32x64、1700 conf（`1000,1010,...,17990`）、1000 组配对 bootstrap、seed `20261015`，每组有放回抽取 1700 次。所有表示、flow、距离和时间共享索引。代码使用 delete-one jackknife 的相关矩阵构建固定白化，删除低于最大相关本征值 $10^{-8}$ 的模式；样本信息秩的相对 cut 为 $10^{-12}$，无 prior。沿用完整样本政策：只有全部 1000 replica 参数有限时才给完整 bootstrap 标准差（`ddof=0`），不会把剔除坏样本后的 spread 当作完整误差。组态自相关/分块效应仍待确认。

主尺度处方及三个对照如下；数值是本次**经验测试阈值**，不是文献给出的普遍定理：

| 尺度处方 | $r_f/a$ 下限 | $r_f/R_{\min}$ 上限 | $r_f/T_{\min}$ 上限（Wilson） |
|---|---:|---:|---:|
| `baseline` | 2 | 0.5 | 0.5 |
| `tighter_R` | 2 | 0.35 | 0.5 |
| `finer_resolution` | 3 | 0.5 | 0.5 |
| `a2_over_tf_le05` | 4 | 0.5 | 0.5 |

所有处方还要求 $r_f/(32a)\le0.2$、$r_f/(64a)\le0.2$。单 Polyakov loop 没有空间分离，因此不施加虚构的 $R_{\min}$；也没有 Wilson 时间窗的 $T_{\min}$。$r_f\ge2a$ 只意味着 $\tau\ge0.5$，最低点仍有 $a^2/t_f=2$，**不等于**强层级 $a^2/t_f\ll1$；最严格对照为 $\tau\ge2$，也不等于已完成连续极限。

Cornell 的空间窗为 $n=6,\ldots,12$、$7,\ldots,12$、$8,\ldots,12$、$9,\ldots,12$；相应 $R_{\min}$ 是这些窗的下端。relative 固定 $n=6,8,10,12$。二者时间窗均为 $[10,16]$ 或 $[8,12]$。先由尺度条件确定 eligible flow，再比较 `full`、`low08`、`low14`、`low22`、`trim_high`（删除允许窗中最高一个点）；完全相同的实际点集去重记录。

例如 baseline、$n=8$、时间窗 $[10,16]$，$r_f/R\le0.5$ 给出 $\tau\le2$，在原有 checkpoint 中最高为 $1.8$；`low08` 因而采用 $\{0.8,1.0,1.4,1.8\}$，relative 的 reference 为 $0.8$。不同距离的最高允许 flow 可能不同，因此比较 $d(R)$ 时必须同时看每个点自己的 `fit_taus`，不能把不同 flow 窗口造成的漂移全归因于距离效应。

本版本**没有按 $\chi^2$、$Q$、最小误差或 AIC 筛选/平均结果**。秩不足、log 不可构造或部分 bootstrap 失败仍保留状态和图；“每个候选都有记录”不意味着可以伪造一个有限结果。下列是预先固定、用于说明问题的基本表示例子，不是最终推荐值：

| 路线/模型 | $R/a$ 或空间窗 | 实际 flow 集合 | $d_F$ | 完整 bootstrap | $\chi^2/{\rm dof}$；$Q$ |
|---|---|---|---|---|---|
| relative，`d_only` | 固定 8 | $0.8,1,1.4,1.8$ | $0.445035\pm0.001809$ | 1000/1000 | 163.003；$1.62\times10^{-71}$ |
| relative，`d_c_tau_difference` | 固定 8 | 同上 | $0.346301\pm0.006043$ | 1000/1000 | 3.215；0.07296 |
| relative，`d_only` | 固定 10 | $0.8,1,1.4,1.8,2.2,2.6,3$ | $0.598020\pm0.002444$ | 1000/1000 | 467.576；数值下溢为 0 |
| Cornell，`d_b` | 8–12 | $0.8,1,1.4,1.8$ | 中心 0.584582，**无完整误差** | 990/1000 | 0.00689；0.99314 |

以上时间窗均为 $[10,16]$、尺度处方均为 baseline。第二行同时得到 $c_F=-0.00790094\pm0.00046070$，只有一个剩余自由度，不能因 $Q$ 较大就确认为正确物理模型。第一行说明**小统计误差不等于模型能描述数据**；最后一行说明**高 $Q$ 不会补齐缺失的 bootstrap 误差**。两种方法中心不同也不能仅凭名称判断谁“更正确”，应联合审查时间窗、空间模型、实际 flow 集合、修正项和样本覆盖。

收紧尺度后，可用点会变少，部分模型不能辨识；这正是单格距数据目前的限制。Polyakov 在当前尺度窗下尚无完整 1000-bootstrap 误差、可用于正式减除的系数；signed 原始圈诊断与正实 log 势链保持隔离。没有获得唯一正式的 $d_F,d_A$，没有把候选值用于现有 PDF。

已完成的 CPU 作业为拟合 `3856230`、绘图 `3856233`、独立核验 `3856234`，均在 `a192c1t` 分区退出码 0；本次文档更新没有重新执行这些作业。1548 个候选均有索引图页，333 个有限中心、146 个完整 bootstrap 误差（包含各阶段及隔离 signed 诊断，**不是**333 个已验证物理 $d$）。来源 hash、尺度门、样本轴、误差政策与页码核验通过，也不能替代物理验证。

### 24.8 图片怎样看：Cornell 的两个层级与 relative 对照

以下都在本版根目录的 `figures/` 中，绝对地址如下：

1. [00_scale_window_examples.pdf](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/figures/00_scale_window_examples.pdf)：先看固定示例，理解被尺度排除的点与参与拟合的点。
2. [01_Wilson_Cornell.pdf](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/figures/01_Wilson_Cornell.pdf)：**固定 flow，横轴 $R$、纵轴 $aV_{\rm eff}$**，展示空间 Cornell 模型；对应第 24.5 节步骤二，不是 $d$ 的最终拟合。
3. [02_Wilson_Cornell.pdf](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/figures/02_Wilson_Cornell.pdf)：**横轴 $\tau$、纵轴 $C_0/2$**，展示 Cornell 常数的 $dq+b$ 或 $dq+b+c\tau$ 拟合。
4. [03_Wilson_relative.pdf](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/figures/03_Wilson_relative.pdf)：**固定 $R$，横轴 $\tau$、纵轴 $a\Delta V/2$**，查看相应 reference 和两个差分模型。
5. [05_scale_admissibility.pdf](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/figures/05_scale_admissibility.pdf)：只由几何决定的可用窗口。
6. [06_d_window_stability.pdf](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/figures/06_d_window_stability.pdf)：不同尺度/窗口的 $d$ 候选与失败/不完整误差状态。
7. [08_d_vs_distance.pdf](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/figures/08_d_vs_distance.pdf)：不同距离或 Cornell 空间窗下的候选；每个点同时注明自身 flow 区间。

图上的有限原始点不以实线连接；实线是模型。浅灰空心点是有可用误差但未参与当前候选拟合的上下文点，不应被当作已用于拟合。逐候选页码用 [figure_index.csv](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/analysis/figure_index.csv) 查找。

### 24.9 实现与依据：如何复查，不应随意改变什么

实现入口及机器结果：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_static_allfits_20261017/code/allfits.py
    fit_all(): Wtime_*，对 -log(mean W) 作 [1,ell] 相关时间拟合。
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/code/scale_fits.py
    scale_mask(): 先验尺度筛选；windows(): 扫描和去重。
    fit() 中 spatial_potential 段：[1,n,1/n] -> C0,sigma_a2,E。
    design_flow(): q,1,tau 或 q-q_ref,tau-tau_ref。
    run_flows(): reference=候选窗最低 tau，relative 在势样本层相减/2。
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/code/frozen_allfits.py
    whiten()/linear(): 固定 JK 白化、QR、秩门和完整 bootstrap 误差。
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/analysis/reused_time_fits.csv
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/analysis/all_fit_results.csv
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/analysis/report.json
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/analysis/fit_samples.npz
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/analysis/scale_selection.csv
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/analysis/window_deduplication.csv
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/analysis/validation.json
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/RESULTS.md
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_scale_windows_20261017/VALIDATION.md
```

本版 `fit_samples.npz` 对新拟合保存 `<label>__jk`、`<label>__boot`，而上游时间拟合文件使用 `<label>_jackknife`、`<label>_bootstrap`；读样本时不能混用键名。第一层参数顺序为 `[b,aV]`，Cornell 层为 `[C0,sigma_a2,E]`，flow 层按 `parameter_names` 读取。结果表中的 `R_a=8` 对 Cornell 标签表示空间窗下端、对 relative 标签表示固定距离，须结合 `R_window` 与方法区分。

**不能静默改变的结论与约定：**

- 当前 Wilson relative 的 reference 在 **flow time** 上，不在空间距离上；不同版本的直接 loop-ratio 与势差估计量应分别标记。
- 势的时间斜率包含两条长线的自能；从 $C_0$ 或 $\Delta V$ 估计单线量时除以 2。单 Polyakov loop 只有一条绕时线，pair 才有两条，不能互套因子。
- $m_{\rm line}>0$ 表示裸线衰减，$\delta m=-m_{\rm line}$；$d$ 已含颜色表示依赖。
- 本项目的 Polyakov pair 是有限温度下未作 connected subtraction 的完整 color-averaged correlator；不能无说明地称为零温 singlet potential，即使也用了 Cornell 型诊断。
- 正实 log 失败、部分 bootstrap、不可辨识和拟合质量差分别记录；不能以高 $Q$ 或收敛掩盖它们。
- 当前只有一个格距，经验尺度窗及 $d+c\tau$ 模型的改善不证明连续可控，也不固定有限重整化常数。时间平台、跑动/对数、模型和自相关误差仍**待确认**。

**补充原始文献与本地核对：**

[11] N. Brambilla, V. Leino, J. Mayer-Steudte and A. Vairo, *The static force from generalized Wilson loops on the lattice using gradient flow*, arXiv:2312.17231v2。Sec. II A 的 Eqs. (1)–(4) 给出 Wilson loop/静态能、空间 reference 的有限重整化以及静态力；Appendix A 展示 flowed Wilson loop 的长时间衰减。文献讨论静态力和不同格距的受控分析，不是本项目单格距候选 $d$ 的物理验证。本次核对了 arXiv PDF 原文。

同时核对本文文献 [1]、[3] 的流核/线自能，以及 [7] `Pion_DA.pdf` 第 15 页的静态势 Cornell 型表达。该 presentation 明确表示其当时势信号尚不足以作稳健线性发散拟合，转而使用 Wilson line；**不能把它的示意 $A_0(\tau)$ 系数直接当作本项目已包含两条静态线、已固定 $\sqrt8$ convention 的单线 $d$**。第 24.4 节的有限流 Coulomb 例子是由流核直接推导的最低阶说明，第 24.5–24.7 节的具体模型、尺度阈值和数值均来自本地实现/结果，不冒充文献通用结论。

本次文档更新日期：**2026-10-17（Asia/Shanghai）**。

本次会话标题：**Wilson Cornell 与 Wilson relative 的定义、Wilson loop 到静态势的推导及尺度窗结果解释**。

本次仅更新文档并核对已有代码、原始文献和产物；未重跑测量/拟合、未提交新的 CPU/GPU 作业、未改动旧的冻结目录、matching 系数或 PDF 矩阵元。

## 25. 如何读 `flow_fits.pdf`：窗口名、空间距离与红蓝拟合（2026-10-18）

### 25.1 适用版本与最简对照

本节直接核对以下已完成版本，不以文件名猜测拟合定义：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_d_boot3000_20261018/
    code/fit_d.py
    analysis/report.json
    analysis/d_results.csv
    analysis/window_deduplication.csv
    figures/01_Wilson_relative_flow_fits.pdf
    figures/01_Wilson_Cornell_flow_fits.pdf
```

|图中的标记|准确含义|不表示什么|
|---|---|---|
|`full`|在该表示、几何和尺度处方下，使用全部允许的已测量 flow 点|不是所有原始 $\tau$，也不自动从 $\tau=0$ 开始|
|`low08`|在上述允许点中再取 $\tau\ge0.8$|不是 $\tau\le0.8$，也不是只取 $\tau=0.8$|
|`low14`|在上述允许点中再取 $\tau\ge1.4$|不是 $\tau\le1.4$|
|`Cornell R/a=6..12`|固定 flow time 下，用 $R/a=6,7,8,9,10,11,12$ 的势拟合空间依赖|不是 flow 窗、Wilson 时间窗或组态编号|
|`Cornell R/a=7..12`|空间拟合改用 $R/a=7,8,9,10,11,12$，剔除距离 6|不是只拟合两个端点|
|蓝色曲线|leading 模型：只含 $1/\sqrt{8\tau}$ 的线质量主项，以及该方法允许的常数|不是“基本表示”的颜色|
|红色曲线|在蓝色模型基础上增加有限 flow 修正 $c\tau$（relative 中为 $c(\tau-\tau_r)$）|不是“伴随表示”的颜色，也不是通过/未通过的标记|

`fundamental`（基本表示）与 `adjoint`（伴随表示）由**整页标题**区分；两种表示各自都有蓝、红两种拟合。这里的颜色定义只适用于本节列出的 Wilson flow 拟合图，不能套到其他 PDF：例如 `07_bootstrap1000_vs_3000.pdf` 中蓝/红表示 1000/3000 样本版本，而不是上述模型。

### 25.2 先分清三个独立的拟合变量

记

$$
n\equiv R/a,\qquad \ell\equiv T/a,\qquad
\tau\equiv t_f/a^2,\qquad
q(\tau)\equiv\frac{1}{\sqrt{8\tau}},\qquad
\frac{r_f}{a}=\sqrt{8\tau}.
$$

三种窗口位于不同步骤：

1. **Wilson 时间窗**，如 `T/a=10..16`：固定 $n,\tau$，对各整数 $\ell=10,\ldots,16$ 的 Wilson loop 时间衰减作相关拟合，提取 $aV_{\rm eff}(n;\tau)$。当前 3000-bootstrap 版本在原始圈上拟合

   $$
   \langle W(n,\ell;\tau)\rangle
   =A(n;\tau)\exp[-\ell\,aV_{\rm eff}(n;\tau)].
   $$

   另一个时间窗为 `T/a=8..12`。$T$ 是矩形圈的时间边长，不是流时间 $t_f$，也不是核子三点函数的 $t_{\rm sep}$。全四维 flow 与有限时间窗下，这里先称**有效静态能量**，不能只因拟合收敛就宣布已达到 $T\to\infty$ 静态势。
2. **Cornell 空间窗**，如 `R/a=6..12`：固定 $\tau$，再拟合不同距离的有效势；该步骤不存在于固定 $R$ 的 Wilson relative 分支。
3. **flow-time 窗**，如 `full/low08/low14`：最后拟合上述势的自能部分如何随 $\tau$ 变化，抽取候选 $d$。`flow_fits.pdf` 的横坐标是 $\tau$，不是 $R/a$。

因此，标题同时有 `T/a=[10,16]`、子图有 `Cornell R/a=7..12`、页标题有 `low08`，并不矛盾：它们依次指定时间、空间、流时间三层的数据选择。

### 25.3 `full/low08/low14` 的尺度筛选与实际流时间

代码先运行 `mask()`，再运行 `windows()`。这里两份 Wilson flow 拟合 PDF 展示的是 `baseline`，条件为

$$
\tau>0,\qquad
\frac{r_f}{a}\ge2,\qquad
\frac{r_f}{R_{\min}}\le0.5,\qquad
\frac{r_f}{T_{\min}}\le0.5,\qquad
\frac{r_f}{32a}\le0.2,\qquad
\frac{r_f}{64a}\le0.2.
$$

对于 relative，$R_{\min}$ 就是固定的 $R$；对于 Cornell，$R_{\min}$ 是空间拟合窗的最短距离。$T_{\min}$ 是时间窗下端。于是，以 $n_{\min}=R_{\min}/a$、$\ell_{\min}=T_{\min}/a$ 表示，候选点必须满足

$$
\boxed{
0.5\le\tau\le
\min\left\{
\frac{n_{\min}^{2}}{32},\,
\frac{\ell_{\min}^{2}}{32},\,
5.12,\,20.48
\right\}.
}
$$

还必须与**确实完成测量的 checkpoint 集合**取交集；没有测量的点不能补入。例如该版本原始 checkpoint 为

$$
\begin{aligned}
\tau\in\{&0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,\\
&0.50,0.55,0.60,0.65,0.80,1.00,1.40,1.80,\\
&2.20,2.60,3.00,3.40,3.80\}.
\end{aligned}
$$

其中没有 $0.70$、$0.90$ 等点。虽然测了 $\tau=0$ 和 $0.05$ 等小流时间，它们不属于本次 `baseline/full`，因为不满足 $r_f/a\ge2$。这里的 `baseline` 仅为经验尺度诊断，**不等于**已实现严格的 $a^2/t_f\ll1$；例如 $\tau=0.5$ 时 $a^2/t_f=2$。不能把本项目这组阈值写成文献中普遍成立的受控窗口。

设通过筛选的集合为 $S$，则

$$
\begin{aligned}
S_{\tt full}&=S,\\
S_{\tt low08}&=S\cap[0.8,\infty),\\
S_{\tt low14}&=S\cap[1.4,\infty),\\
S_{\tt low22}&=S\cap[2.2,\infty).
\end{aligned}
$$

另有 `trim_high`，含义是从 **`full`** 的有序集合中删去最高的一个点，不是在 `low08` 上继续删点。标准 Wilson flow 拟合 PDF 的页循环是 `full/low08/low14/low22`；`trim_high` 和其他尺度处方请查 CSV、报告及稳定性图。相同实际点集会记录在 `window_deduplication.csv` 中而不重复拟合。

例如 `baseline`、$T/a=10..16$、固定 $R/a=8$ 或 Cornell $R/a=8..12$ 时：

|窗口名|尺度筛选与窗口下限共同决定的实际 $\tau$ 集合|
|---|---|
|`full`|$0.50,0.55,0.60,0.65,0.80,1.00,1.40,1.80$|
|`low08`|$0.80,1.00,1.40,1.80$|
|`low14`|$1.40,1.80$|
|`low22`|空集|
|`trim_high`|$0.50,0.55,0.60,0.65,0.80,1.00,1.40$|

此表是**选中的 checkpoint**，relative 还要扣除参考点，详见第 25.6 节。它不是所有子图通用的 flow 范围：例如 $T/a=10..16$ 时 relative 的 $R/a=10,12$ 可保留到 $\tau=3.0$，而换成 $T/a=8..12$ 后因时间门限制最多保留到已测的 $1.8$。

### 25.4 `R/a=6..12` 与 `7..12`：空间 Cornell 拟合及其连带影响

每个固定 $\tau$ 上使用

$$
\boxed{
aV_{\rm eff}(n;\tau)
=C_0(\tau)+\bigl[\sigma(\tau)a^2\bigr]n
+\frac{E(\tau)}{n}.
}
$$

空间拟合的三列是 $[1,n,1/n]$，三个参数依次为 $[C_0,\sigma a^2,E]$。$E$ 含 Coulomb 型项的符号，不另外默认正号。$C_0$ 是 Cornell 模型中的常数参数，**不是**在 $R=0$ 处测得的势，模型本身的 $1/n$ 也不允许直接代 $n=0$。

|子图标题|空间拟合的整数距离|距离点数|
|---|---|---|
|`Cornell R/a=6..12`|$6,7,8,9,10,11,12$|7|
|`Cornell R/a=7..12`|$7,8,9,10,11,12$|6|
|`Cornell R/a=8..12`|$8,9,10,11,12$|5|
|`Cornell R/a=9..12`|$9,10,11,12$|4|

在 L32x64 的 $a=0.0897\,\mathrm{fm}$ 下，$6a=0.5382\,\mathrm{fm}$、$7a=0.6279\,\mathrm{fm}$、$12a=1.0764\,\mathrm{fm}$。所以前两个子图使用的物理距离分别为 $0.5382$–$1.0764\,\mathrm{fm}$ 与 $0.6279$–$1.0764\,\mathrm{fm}$，且包含中间的每个整数格点距离。

改变空间下端用于检查短距离 flow 重叠及 Cornell 模型依赖；**它也会改变允许的 flow 范围**。以 `baseline`、$T/a=10..16$ 为例：

|Cornell 空间窗|`full` 中最大已测 $\tau$|`low08` 中全部 $\tau$|`low14` 中全部 $\tau$|
|---|---|---|---|
|$6..12$|$1.0$|$0.8,1.0$|空集|
|$7..12$|$1.4$|$0.8,1.0,1.4$|$1.4$|
|$8..12$|$1.8$|$0.8,1.0,1.4,1.8$|$1.4,1.8$|
|$9..12$|$2.2$|$0.8,1.0,1.4,1.8,2.2$|$1.4,1.8,2.2$|

这四个 `full` 都从 $0.5$ 起，并含 $0.55,0.6,0.65,0.8,1.0$。例如 $n_{\min}=6$ 给出 $\tau\le36/32=1.125$，故已测点只到 $1.0$；$n_{\min}=7$ 给出 $\tau\le49/32=1.53125$，故只到 $1.4$。不能把这些子图的差异全归因于去掉一个空间点，因为 flow 数据集合也变了。

与此不同，`Wilson_relative` 页上的 **`R/a=8` 只表示固定空间距离**，不做 $8..12$ 的空间 Cornell 拟合。机器结果中两种方法都可能写 `R_a=8`：relative 是固定距离，Cornell 是空间下端，必须结合 `method` 和空间层的 `R_window=[8,12]` 解读。

### 25.5 蓝色与红色到底拟合什么：两条分支的完整公式

保持本文既定的单线正自能约定，记颜色表示为 $s=F,A$：

$$
am_s(\tau)=d_s\,q(\tau)+\cdots,
\qquad q(\tau)=\frac1{\sqrt{8\tau}},
\qquad \delta m_s=-m_s.
$$

$d_s$ 已含表示依赖，不应再无说明地乘一次 Casimir。Wilson 静态能量中有**两条**时间方向长线的自能，所以两种分支都要除以 2 才与上述单线 $d_s$ 对应。

**Cornell（absolute）分支：**先在每个 $\tau$ 抽取 $C_0(\tau)$，图上纵坐标为

$$
y_C(\tau)=\frac{C_0(\tau)}2.
$$

$$
\begin{array}{lll}
\text{蓝色，代码 absolute}:&y_C=d_s q+b,&2\text{ 个参数 }[d_s,b],\\[2pt]
\text{红色，代码 absolute\_c}:&y_C=d_s q+b+c\tau,&3\text{ 个参数 }[d_s,b,c].
\end{array}
$$

$b$ 是该方案下允许的有限常数项；不把它叫作线性发散系数。Cornell 分支不减参考 flow 点。

**固定 $R$ 的 relative 分支：**选择同一候选窗口的最低流时间 $\tau_r$，在中心值、每个 JK 和每个 bootstrap 样本中使用相同参考作差：

$$
y_R(\tau;\tau_r)
=\frac{aV_{\rm eff}(R;\tau)-aV_{\rm eff}(R;\tau_r)}2,
\qquad q_r=q(\tau_r).
$$

$$
\begin{array}{lll}
\text{蓝色，代码 relative}:&y_R=d_s(q-q_r),&1\text{ 个参数 }[d_s],\\[2pt]
\text{红色，代码 relative\_c}:&y_R=d_s(q-q_r)+c(\tau-\tau_r),&2\text{ 个参数 }[d_s,c].
\end{array}
$$

不再另加独立截距，因为参考差分把流时间无关的常数消去了。这里的差分是**同一 $R$ 的不同 flow time**，不是不同 $R$ 之间的差分。

红色的 $c\tau$ 是有限-flow 修正的**经验模型项**，不是另一个线性发散：发散项是 $d_s/\sqrt{8\tau}$。它也可能吸收相互作用势的 flow 依赖、有限 Wilson 时间效应等，因此“加 $c\tau$ 后 $Q$ 变大”不等于证明已得到普适的线质量或受控的零流时间展开。两条曲线各自独立拟合 $d_s$，不是先固定蓝色的 $d_s$ 再只增加一个 $c$。

### 25.6 为什么 `low08` 的 relative 点从 1.0 开始，以及为什么有的窗口无曲线

仍取 $R/a=8$、$T/a=10..16$、`baseline`：

|候选窗口|选中的 checkpoint|relative 的 $\tau_r$|relative 实际拟合/画点的横坐标|
|---|---|---|---|
|`full`|$0.5,0.55,0.6,0.65,0.8,1.0,1.4,1.8$|$0.5$|$0.55,0.6,0.65,0.8,1.0,1.4,1.8$|
|`low08`|$0.8,1.0,1.4,1.8$|$0.8$|$1.0,1.4,1.8$|
|`low14`|$1.4,1.8$|$1.4$|仅 $1.8$|

在 $\tau=\tau_r$ 处 $y_R\equiv0$ 对所有样本恒成立，不能把这个零方差参考点再当成一个独立测量约束。所以代码删除该点，同时传播共享参考造成的全部相关性。`low08` 从 1.0 开始画相对差分点，**不代表漏算了 0.8**；0.8 已进入每一个差分的参考。换成 `full` 后参考变为 0.5，因此两页纵坐标的定义也不同，不是简单删掉部分横坐标。

若选中的 checkpoint 有 $K$ 个，则 relative 只有 $K-1$ 个数据点；absolute/Cornell 有 $K$ 个。当前代码要求正自由度，且协方差和设计矩阵的有效秩足够。满秩时可直观理解为

$$
N_{\rm data}>N_{\rm par},
\qquad
\mathrm{dof}=\mathrm{rank}(C_{\rm retained})-N_{\rm par}>0.
$$

上表 `low14` 的 relative 只有一个非参考点：蓝色 1 参数也只有 0 自由度，红色 2 参数更不够，故两者都记录 `insufficient_points_for_positive_dof`。同一几何的 Cornell `low14` 有两个点，但蓝色需要 2 参数、红色需要 3 参数，也不能得到正自由度拟合。这是**数据约束不足**，不是被 $\chi^2$ 或 $Q$ 门槛删除。另一方面，`R/a=9..12` 的 `low14` 有三个点，蓝色 absolute 可以在秩足够时拟合，红色仍没有正自由度。

其他无曲线原因包括空集合、重复点集未重拟合、上游参数不可辨识或 Polyakov 实对数不成立；要查 `status`，不能统称“拟合失败”或偷偷删除失败样本。本版没有用 $\chi^2$、$Q$ 或 AIC 把结果筛走，也不为满足“每处有结果”而把精确插值伪装成有统计自由度的拟合。

### 25.7 数据点、误差棒、拟合带与拟合质量

在本版两份 Wilson flow 拟合图中：

- **黑色实心点与误差棒**：上游势或 $C_0/2$ 的中心估计和完整 bootstrap 误差；误差采用全部 3000 样本的标准差（`ddof=0`）。数据点不连线。
- **灰色空心点**：中心值有限，但不能给出完整的 bootstrap 误差。这里不能照搬旧尺度窗图的“未参与拟合上下文点”解释；应结合当前图标题与数据状态判断。
- **蓝/红实线**：两个模型各自的中心参数预测，不是把数据点连起来。
- **同色半透明带**：每个横坐标上，全部 3000 个可用 bootstrap 拟合预测的逐点标准差带。若没有完整可用样本，则不画完整误差带；可能仍保留中心曲线。
- **图内 $d,\chi^2/\mathrm{dof},Q$ 与可用样本数**：分别用于看参数、模型描述程度与统计完整性。没有设置 quality cuts 不等于所有模型都合理；很小的参数误差也不代表模型正确。

相关拟合使用原始组态 JK 构造并固定的协方差及既定秩处理，3000 次 bootstrap 沿链配对传播。误差带未自动合并空间窗、时间窗、flow 窗、蓝红模型、离散化或自相关系统误差；它不是“最终物理 $d$ 的总置信带”。只有一个格距，也没有因此完成连续极限验证。

例如，最新结果中基本表示、relative、$R/a=8$、$T/a=10..16$、`baseline/low08` 的两个模型为：

|模型|候选 $d_F$（bootstrap 统计误差）|$\chi^2/\mathrm{dof}$|$Q$|可用 bootstrap|
|---|---|---|---|---|
|蓝色 `relative`|$0.44220\pm0.00184$|$143.076$|$7.29\times10^{-63}$|3000/3000|
|红色 `relative_c`|$0.34807\pm0.00611$|$2.82188$|$0.09299$|3000/3000|

此例说明：蓝色虽有很小的统计误差，却不能描述该窗口的 flow 依赖；红色增加修正后改善，但这只是一个候选窗口的比较，**没有据此指定最终 $d$，也没有将任何一个值应用到 quark/gluon PDF 的线性发散减除**。这里的数值与 1000-bootstrap 历史快照区分保存。

### 25.8 读图操作顺序与复查入口

建议按照“**表示 → 时间窗 → 方法与空间窗 → flow 窗 → 模型 → 状态/质量**”读一张图。例如：

> `Wilson_Cornell: fundamental; T/a=[10,16]; low08`，子图 `Cornell R/a=7..12`：先用基本表示 Wilson loop 的时间边长 10–16 提取各距离的有效势；每个 flow time 用空间距离 7–12 做三参数 Cornell 拟合；保留通过尺度门且 $\tau\ge0.8$ 的已测 flow 点，即 $0.8,1.0,1.4$；蓝色用三点拟合 $dq+b$，满秩时有 1 自由度；红色 $dq+b+c\tau$ 有 3 参数，没有正自由度，因而不能给出该窗口的受约束红色拟合。这不是漏画，也不是红色代表另一种颜色表示。

图和表的可点击入口：

- [Wilson relative 的 flow 拟合图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_d_boot3000_20261018/figures/01_Wilson_relative_flow_fits.pdf)
- [Wilson Cornell 的 flow 拟合图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_d_boot3000_20261018/figures/01_Wilson_Cornell_flow_fits.pdf)
- [窗口稳定性图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_d_boot3000_20261018/figures/02_d_window_stability.pdf)
- [每个窗口的系数、误差和状态表](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_d_boot3000_20261018/analysis/d_results.csv)
- [实际 flow 点集及去重表](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_d_boot3000_20261018/analysis/window_deduplication.csv)
- [完整机器报告](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_d_boot3000_20261018/analysis/report.json)
- [冻结实现](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/line_mass_d_boot3000_20261018/code/fit_d.py)：`mask()` 为尺度门，`windows()` 为窗口和去重，`design()` 为红蓝模型，`flows()` 为参考差分和模型输入，`plot_results()` 为颜色、图例和绘制。

复查具体窗口时联合使用 `method/representation/prescription/R_a/T_window/window_variant/model`，读取 `fit_taus`、`fit_taus_without_reference`、`reference_tau`、`status`、`bootstrap_complete`，不要只凭 `full` 或 `R_a` 一个字段判断。Cornell 的 `reference_tau` 为 `null`；虽然通用字段名叫 `fit_taus_without_reference`，对 absolute 分支它仍等于完整输入 flow 点集。

Polyakov 的 flow 图沿用相同的窗口下限命名，但其尺度门不能机械照搬 Wilson：单圈没有 $R$、没有 Wilson 时间窗；pair 有分离距离 $R$，没有 Wilson 时间窗。Polyakov 信号图、signed 原始圈诊断及 1000/3000-bootstrap 比较图应各读自己的图例。当前 Polyakov 仍未给出可批准用于 PDF 减除的完整可靠系数，详见对应结果报告；不能以这份标签说明替代信号和系统误差检查。

本次文档更新日期：**2026-10-18（Asia/Shanghai）**。

本次会话标题：**flow_fits 图中 flow 窗、Cornell 距离窗与红蓝模型的逐项解释**。

本次仅核对已有冻结代码/机器结果并补充文档；未重新拟合、未重画已有 PDF、未提交 CPU/GPU 作业、未改变任何拟合窗口或统计处方。第 24 节及更早记录作为有日期的历史版本保留。
