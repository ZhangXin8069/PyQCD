# Gradient flow 中的 Wilson-line 线性发散：理论理解、零流时间极限与闭合圈测量

本文针对本项目的四维 thin-link gradient-flow quark/gluon PDF 计算，解释 Wilson line 的线性发散、其与流时间的关系，以及用 Polyakov loop 或 Wilson loop 抽取线自能的原理和实际操作。公式采用自然单位，$\hbar=c=1$。

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

## 15. 依据、现有文件与本次工作边界

### 15.1 主要文献

[1] N. Brambilla and X.-P. Wang, *Off-lightcone Wilson-line operators in gradient flow*, JHEP 06 (2024) 210, arXiv:2312.05032v3。重点为 Secs. 2–3 的辅助场重整化、小流时间关系，Sec. 4.1 / Eq. (4.15) 的 $\delta m_R$，及附录的有限流半径讨论。

[2] M. Lüscher and P. Weisz, *Perturbative analysis of the gradient flow in non-abelian gauge theories*, JHEP 02 (2011) 051, arXiv:1101.0963。正流时间的 UV 结构和重整化边界。

[3] M. Lüscher, *Properties and uses of the Wilson flow in lattice QCD*, JHEP 08 (2010) 071, arXiv:1006.4518。平滑半径、正流时间观测量和连续极限背景。

[4] H. Makino and H. Suzuki, *Lattice energy–momentum tensor from the Yang–Mills gradient flow—inclusion of fermion fields*, PTEP 2014 (2014) 063B02, arXiv:1403.4772。Ringed-fermion 场归一化的背景；它不是 line-mass subtraction。

[5] N. Brambilla et al., *Inclusive P-wave Quarkonium Decay Widths from Lattice QCD and pNRQCD*, arXiv:2607.13024v1。Supplemental Material SM1 C，Eqs. (SM17)–(SM19)，为独立伴随 Polyakov-loop 抽取 $d$ 的直接实例；另见 SM1 A 的顺序与 SM1 的减除定义。本次未声称其有更新版本，也未把其结论泛化为完整 PDF matching。

[6] S. Gupta, K. Hübner and O. Kaczmarek, *Renormalized Polyakov loops in many representations*, Phys. Rev. D 77 (2008) 034503, arXiv:0711.2251。不同表示 Polyakov-loop 重整化、静态源自由能和有限重整化方案背景。

本文的长线高斯积分、周长项相消、逆转换避免重复减除和误差公式已经明确标作从所列定义/关系的直接推导；待验证的拟合模型和本 ensemble 的未测系数不作为文献定论。

### 15.2 本地核对路径

```
/public/home/donghx/.codex/Markdown/gradient_flow.md
/public/home/donghx/.codex/Markdown/gradient_flow_gluon.md
/public/home/donghx/.codex/Markdown/gluon_gradient_flow_matching.md
/public/home/donghx/Note/Gradient_Flow/main.tex
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/polyakov_L32x96/README.md
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/polyakov_L32x96/operations.md
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/polyakov_L32x96/src/polyakov_flow.cpp
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/polyakov_L32x96/runs/production100_20261011_113518/validation/final_audit.json
```

本次对照已有 PDF 文本和 arXiv 原文，生成理论说明并核对保存的测量定义/审计。未重新测量规范场、未拟合 $d_R$、未改动现有 matching 代码或图、未提交新的 GPU/CPU 数值任务。

### 15.3 生成记录

生成日期：**2026-10-14（Asia/Shanghai）**。

本次会话标题：**Gradient-flow 线性发散、零流时间极限与 Polyakov/Wilson-loop 线质量测量**。
