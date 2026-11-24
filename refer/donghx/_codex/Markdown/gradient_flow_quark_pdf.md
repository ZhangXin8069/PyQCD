# Quark PDF gradient flow：计算流程、推导与结果交接

**更新日期：2026-10-25；本轮主线：L32x64、Blending B 方案、20 conf、100 paired bootstrap、01–06 分步分析。**

本文以实际源码、冻结的参数清单、CSV 结果和独立回读记录为依据。历史 L32x96/A 方案、绝对 ringed 归一化和旧的 $\lambda$/Fourier/LaMET 结果仍保留在原位置；它们不会因为本文更新而自动成为这批 B 方案的后续结果。

当前已补齐 $P_z=3,4,5,6$ 的 06 数据。其准确状态是“固定单格距、中间方案 $S$ 下的条件性诊断矩阵元”。全部数值与配对传播完成，但 $m_0$ 拟合的 $\chi^2$ 极大、$Q$ 近零，完整有限方案关系仍未确认，不能称为已经验收的 $\overline{\mathrm{MS}}$ 矩阵元或最终 PDF。

## 1. 项目目标与当前操作链

目标是使用 thin-link 四维 gauge/fermion gradient flow，计算核子中 connected quark unpolarized、helicity、transversity 的空间非局域矩阵元，最终经过重整化/方案转换、受控的 flow/格距分析、有限 Ioffe-time 重构和 LaMET matching，研究光锥 PDF。

当前数据链按用户选定顺序实施：

$$
\boxed{
h_{\rm bare}^{\rm GF}
\xrightarrow[\text{同 }P_z,\Gamma,f,\tau]{z=0\text{ 局域归一化}}
\mathcal M
\xrightarrow{e^{+d_F|z|/\sqrt{8t_f}}}
H_{03}
\xrightarrow{F_\Gamma^{\rm RGR}}
H_{04}
\xrightarrow[\text{固定 }a]{c_0+c_1\tau}
H_{05}
\xrightarrow[\text{同 bootstrap 行}]{e^{+m_0(P_z=0)|z|}}
H_{06}.
}
$$

这里指数内全部长度、质量必须使用同一单位；后文给出代码中 fm/GeV 的具体转换。编号依次对应：

| 步骤 | 含义 | 本轮状态 |
|---|---|---|
| 01 | B 方案 $C_3/C_2$ 的 direct Ratio plateau + AIC 裸拟合 | 复用已有结果和 fallback 标记 |
| 02 | 同通道、flavor、动量、flow 的 even-real $z=0$ 局域归一化 | 已完成、逐样本核验 |
| 03 | 固定候选 $d_F=0.374137$ 的基础表示线减除 | 已完成，仅施加一次 |
| 04 | 局域归一化后的 GF 逆转换；一圈边界 + LL RGR | 已完成；完整有限方案关系待确认 |
| 05 | $H=c_0+c_1\tau$ 纯线性 flow-time 截距 | 已完成 all-flow 与 D_loose 两套 |
| 06 | 共享 $P_z=0$ 的 $z/a=1,2$、$b=0$ 诊断 $m_0$，配对应用到 moving | 四个动量均补齐；质量拟合未通过物理验收 |

本轮没有重新做 $C_3$ GPU 测量，也没有重新拟合 01 裸矩阵元；没有为每个 moving 动量重新调 $m_0$。这批最新 B 方案尚未接入新的 $\lambda$ 外推、Fourier 或 LaMET。旧方案存在这些下游图，不代表新的 B 方案也已经完成。

## 2. 参数、样本及不可混淆的坐标

### 2.1 Ensemble 与输入

- ensemble：beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64；
- 体积：$32^3\times64$，$a=0.0897\,\mathrm{fm}$，$L=32a$；
- gauge：未 HYP/stout smear 的 thin-link；
- configurations：$10000,10040,\ldots,10760$，共 20 个；
- 当前分析通道：unpolarized、helicity、transversity；
- flavor：connected $u,d,u-d$。

规范场原始路径：

~~~
/public/group/lqcd/configurations/CLOVER/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64
~~~

connected $u,d$ 不包括 disconnected 项。在简并轻夸克、同一作用量和等同测量处方的同位旋极限下，$u-d$ 的相应 disconnected 项才按非 singlet 论证相消。不能把单独 connected $u,d$ 标成完整 flavor/singlet PDF。

### 2.2 符号字典

| 符号 | 本文含义 |
|---|---|
| $t_f$ | 有量纲 flow time，维度长度平方 |
| $\tau=t_f/a^2$ | 无量纲 flow time |
| $\epsilon=\Delta\tau$ | 数值积分步长 |
| $N_{\rm step}$ | 到 checkpoint 的累计 flow 步数 |
| $t_J$ | 流后 bilocal 所在的格点 current time |
| $\sigma_a,\sigma_b$ | 热流两端各自的原始 basis 时间 |
| $T=t_{\rm sep}$ | source–sink separation，拟合代码中通常以格点单位记 |
| $d_t=t_J-t_{\rm src}$ | 插入点相对 source 的时间间隔，不是 flow time |
| $n_P$ | 动量整数 mode，文件中仍命名 Pz |
| $n=z/a$ | 空间线长的格点数，非动量 mode |
| $b$（截距） | 短距离 log-ratio 模型的独立常数项，本主窗固定为 0 |
| $b$ 或上标 $(b)$（样本） | bootstrap 行号；与自由截距无关 |

使用自然单位时

$$
z_{\mathrm{GeV}^{-1}}=\frac{z_{\mathrm{fm}}}{\hbar c},\qquad
t_{f,\mathrm{GeV}^{-2}}=\tau\left(\frac{a_{\rm fm}}{\hbar c}\right)^2,\qquad
\hbar c=0.1973269804\,\mathrm{GeV\,fm}.
$$

动量为

$$
P_z^{\rm phys}=\frac{2\pi n_P\hbar c}{32a},\qquad
n_P=3,4,5,6.
$$

约为 $1.30,1.73,2.16,2.59\,\mathrm{GeV}$。文件中 $P_z=3$ 不是 3 GeV。
空间分离 $z/a=0,\ldots,20$，对应最大 $1.794\,\mathrm{fm}$。

### 2.3 Flow checkpoints、半径和时间窗

上游 B 方案 moving $C_3/C_2$ 测量含八个 $\tau=0.1,0.2,\ldots,0.8$。
当前局域归一化/线减除/RGR/05–06 主链只用后七个：

| $\tau$ | $\epsilon$ | 累计 $N_{\rm step}$ | $r_f=\sqrt{8t_f}$ [fm] | 原始时间半窗 $w$ | $\mu_f$ [GeV] |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.01 | 20 | 0.113463 | 5 | 2.60629 |
| 0.3 | 0.01 | 30 | 0.138963 | 5 | 2.12803 |
| 0.4 | 0.01 | 40 | 0.160460 | 6 | 1.84293 |
| 0.5 | 0.01 | 50 | 0.179400 | 6 | 1.64836 |
| 0.6 | 0.01 | 60 | 0.196523 | 7 | 1.50474 |
| 0.7 | 0.01 | 70 | 0.212269 | 8 | 1.39312 |
| 0.8 | 0.01 | 80 | 0.226925 | 8 | 1.30315 |

$$
\mu_f=\frac{e^{-\gamma_E/2}}{\sqrt{2t_f}}.
$$

先前选定的弱物理半径窗为

$$
a<r_f<0.3\,\mathrm{fm}
\ \Longleftrightarrow\
\frac18<\tau<
\frac{(0.3/a)^2}{8}=1.3981946511\ldots.
$$

$\tau=0.1$ 的 $r_f<a$，因此没有进入这版七点后处理。这个半径窗只筛选实际已测 checkpoints，不意味着区间内所有实数 $\tau$ 都已有数据，也不等于 $a^2/t_f\ll1$ 已成立。

### 2.4 Bootstrap 的唯一索引

100 份 iid configuration bootstrap；每行从这 20 个组态有放回抽取 20 次，seed=1115。
flow、channel、flavor、动量、$z,T,d_t$、$P_z=0$/moving 全部共享原来的索引行。

索引文件：

~~~
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_B_pionDA_workflow_20261022/00_audit/bootstrap_indices_Nboot100_seed1115.npy
~~~

shape 为 $(100,20)$，SHA256：

~~~
86c034214702c29af6671956f4ff23c7749c4e34af103a9173d03da455b6056b
~~~

中心值从原 ensemble 中心及其拟合计算，不替换为 bootstrap 均值。
统计误差统一

$$
\sigma_X=\sqrt{\frac1{B-1}\sum_{b=1}^{B}
(X^{(b)}-\overline X_{\rm boot})^2},\qquad B=100.
$$

不再除以 $\sqrt{100}$。100 次重采样不是 100 个独立 gauge configurations。
当前未通过 bootstrap 数量消除组态自相关，也未补入所有系统误差。

## 3. 四维 flow 与 distillation 的完整关系

### 3.1 算符和热流方程

quark flowed bilocal 定义为

$$
O_\Gamma^{\rm GF}(z,t_f;x)=
\bar\chi(t_f,x)\Gamma
W_{B(t_f)}(x,x+z\hat z)\chi(t_f,x+z\hat z).
$$

两个 flowed endpoint 和 Wilson line 必须使用同一条由 thin links 开始的 gauge-flow history。$C_2$ 和核子 source/sink 保持未流动；不是把整个核子也 flow 一遍。

用反 Hermitian 矩阵连接的连续记号（耦合已吸收入连接），

$$
\partial_t B_\mu=D_\nu G_{\nu\mu},\qquad
G_{\mu\nu}=\partial_\mu B_\nu-\partial_\nu B_\mu+[B_\mu,B_\nu],
$$

$$
\partial_t\chi=\Delta[B(t)]\chi,\qquad
\partial_t\bar\chi=\bar\chi\,\overleftarrow\Delta[B(t)].
$$

若改用 Hermitian 分量 gauge potential，必须同时恢复对应的 $ig$；不能只改 commutator 符号。数值实现直接使用 SU(3) links，link 方向按 $(x,y,z,t)$，坐标数组按 $(t,z,y,x)$；自旋/颜色轴另列。

形式解是

$$
\chi(t_f)=K(t_f,0)\psi,\qquad
\bar\chi(t_f)=\bar\psi K^\dagger(t_f,0).
$$

因 $B(t)$ 随 flow 演化，

$$
K(t_f,0)=\mathcal T_t\exp\left[\int_0^{t_f}dt\,\Delta[B(t)]\right].
$$

它不是一般意义的幺正矩阵。自由场有 $K(p)=e^{-t_fp^2}$，这是扩散/衰减而非幺正旋转。流后 basis 的 Gram 矩阵不必为 1；不能在流后未经推导重新正交化并继续使用未转换的 perambulator。

### 3.2 原始 basis、四维嵌入和 perambulator

原始 time-slice basis

$$
V_n(\sigma,\boldsymbol x,c),\qquad
V^\dagger(\sigma)V(\sigma)=I_{N_v}
$$

有 $(N_t,N_v,L_z,L_y,L_x,N_c)$ 个数组轴。把每个 $(\sigma,n)$ 视为一列，

$$
\mathbb V_{x,(\sigma,n)}=
\delta_{x_4,\sigma}V_n(\sigma,\boldsymbol x,c).
$$

因此矩阵 view 为

$$
\mathbb V:
(N_tL_zL_yL_xN_c)\times(N_tN_v),
\qquad \Phi=K\mathbb V.
$$

列数仍是 $N_tN_v$，因为仅对现有输入列施加 $K$，并没有求出整个时空颜色空间上的密集 $K$。

生产中 current $N_v=400$，source/sink $N_v=200$。这些列含 exact、compressed 和高模 noise 部分，不能把 400 全部称作 400 个最低精确本征模。已有 light perambulator

$$
\mathcal P_f(\sigma_a,\sigma_b)
=V^\dagger(\sigma_a)D_f^{-1}(\sigma_a,\sigma_b)V(\sigma_b)
$$

保持不变。operator-only flow 通过 $\Phi$ 和 vertex 进入 $C_3$，不需为每个 bilocal 重新求 Dirac 逆。

### 3.3 两个原始时间求和与 vertex

$$
\Phi_{\sigma n}(t_J,\boldsymbol x)
=[K(t_f,0)\,\delta_{t,\sigma}V_n(\sigma)](t_J,\boldsymbol x).
$$

vertex 为

$$
J_{mn}(t_J;\sigma_a,\sigma_b;z)=
\sum_{\boldsymbol x}
\Phi_{\sigma_a m}^\dagger(t_J,\boldsymbol x)
W_{B(t_f)}(\boldsymbol x,\boldsymbol x+z\hat z;t_J)
\Phi_{\sigma_b n}(t_J,\boldsymbol x+z\hat z).
$$

$C_3$ 中必须有两个原始时间的独立求和：

$$
C_{3,\Gamma}=
\sum_{\sigma_a,\sigma_b}\sum_{m,n}
\Omega^B_{(\sigma_a,m),(\sigma_b,n)}
J_{mn}(t_J;\sigma_a,\sigma_b;z)\,
\mathcal C_{\Gamma;mn}(\sigma_a,\sigma_b;\text{perambulators, spectators}).
$$

有限半窗只保留 $|\sigma_{a,b}-t_J|_{\rm periodic}\le w$ 的块。这个聚合截断与 gauge flow 积分步数是两个不同近似；已有居中点 window 测试不能自动证明所有非居中插入点也同样收敛。
费米子时间 APBC 与 baryon wrapping 需沿用原始 receipt；后处理不得重复乘 wrap sign。

### 3.4 B 方案权重

设 sector 的完整维数为 $d_r$，保存 sampled columns 为 $n_r$，一阶权重 $w_r=d_r/n_r$。
同一原始时间的二阶表为

$$
\omega^{(2)}_{mn}=
\begin{cases}
d_r/n_r,&m=n\text{ 且同 sector},\\
\dfrac{d_r(d_r-1)}{n_r(n_r-1)},&m\ne n\text{ 且同 sector},\\
(d_r/n_r)(d_s/n_s),&m\in r,\ n\in s,\ r\ne s.
\end{cases}
$$

当前 B 方案：

$$
\boxed{\Omega^B_{(\sigma_a,m),(\sigma_b,n)}
=\begin{cases}
\omega^{(2)}_{mn},&\sigma_a=\sigma_b\pmod {N_t},\\
w_{\sigma_a m}w_{\sigma_b n},&\sigma_a\ne\sigma_b\pmod {N_t}.
\end{cases}}
$$

相等与否判断的是原始 $\sigma_a,\sigma_b$，不是流后 $t_J$；同 slot number 但不同原始时间不是同一个输入向量。flow 不重新抽 noise，因此权重不随 flow 后的 norm/Gram 重新估算。代码标记

~~~
weight_scheme = same_time_second_cross_time_independent_product
weight_applied_count = 1
~~~

完整高阶共享噪声矩、generator 跨原始时间的独立性和 baryon 完全无偏性仍待确认。块/Wick 回归通过属于实现验证，不自动构成这个统计定理。

详细历史分析另见：

- [basis_vertex.md](/public/home/donghx/.codex/Markdown/basis_vertex.md)
- [flow_equation.md](/public/home/donghx/.codex/Markdown/flow_equation.md)

## 4. 实际 $C_3/C_2$ 的轴、Gamma 和 bootstrap 顺序

### 4.1 固定 current anchors 与时间映射

current anchors 为 $t_J=0,16,32,63$。对每个 $T=7,\ldots,12$、$d_t=0,\ldots,T$，

$$
t_{\rm src}=(t_J-d_t)\bmod64,\qquad
t_{\rm sink}=(t_{\rm src}+T)\bmod64.
$$

这只是四个固定 current 的采样，不是完整 64 source 平均。
因为固定 current 下 source 随 $d_t$ 变化，$C_2$ 必须保留 $d_t$ 轴并逐项匹配 $C_3$。

moving raw $C_3$ shape：

~~~
(flavor_u_d, sign, pol, signed_momentum, z, current_m, current_o)
= (2,2,4,8,21,4,4)
~~~

$C_2$ shape 为 $(pol,signed\_momentum)=(4,8)$。这两个开放 current spin 的 $4\times4$ 与外态旋量降维的 $2\times2$ 是不同对象，不能只按维数相同/不同混读。

有符号动量顺序是

$$
(+3,-3,+4,-4,+5,-5,+6,-6),
$$

仅 z 方向；这批不是六空间方向平均。

### 4.2 DR Gamma 与通道相位

可执行 convention 来自冻结的 reference/conventions.py；$\gamma_5=\gamma_1\gamma_2\gamma_3\gamma_4$。代码做当前自旋 contraction 后：

| channel | raw current matrix | $C_3$ 使用的极化 | 额外相位/组合 |
|---|---|---|---|
| unpolarized | $\gamma_4$ | nopol | $1$ |
| helicity | $\gamma_5\gamma_3$ | pol35 | $-i$ |
| transversity | $\gamma_4\gamma_1\gamma_5$、$\gamma_4\gamma_2\gamma_5$ | pol15、pol25 | 两个横向结果平均后乘 $+i$ |

矩阵 contraction 用保存的 current 轴约定，代码为 $\sum_{m,o}C_{mo}\Gamma_{mo}$。
不应仅凭“trace”一词擅自额外转置 Gamma。transversity 的代数 gamma 检查与外部算符相对符号确认是两层工作；后者状态仍为 pending_external_physics_confirmation。

### 4.3 当前 quark 估计量：每个 bootstrap 的 correlator 均值之比

先在每个配置内平均四个 current anchors，形成 $C_{3,k}$、逐 $d_t$ 匹配的复数 nopol $C_{2,k}$。
配置级先构造 $C_3^{u-d}=C_3^u-C_3^d$。

对中心及每一 bootstrap 行 $I_{b\ell}$，先平均配置：

$$
\overline C_{3}^{(b)}=\frac1{20}\sum_{\ell=1}^{20}C_{3,I_{b\ell}},
\qquad
\overline C_{2}^{(b)}=\frac1{20}\sum_{\ell=1}^{20}C_{2,I_{b\ell}},
\qquad
R^{(b)}_{\pm P,s}=\frac{\overline C_{3,\pm P,s}^{(b)}}
{\overline C_{2,\pm P,{\rm nopol}}^{(b)}}.
$$

**这是当前 B quark 代码已实现的 complex ratio-of-means。**不是先逐配置取 $C_3/C_2$ 再平均，不把 $C_2$ 换为实部或绝对值，也不先把 helicity $C_3$ 扔掉实部。先保留完整复数，随后做对称投影并分别拟合两分量。
不要把历史 gluon ratio 的不同估计量要求直接套到此产品上；如要改估计量，必须另建版本重算 ratio/拟合。

代码 sign 0 为 minus、sign 1 为 plus，因此准确的 even/odd convention 为

$$
R_e(P)=\frac{R_-(P)+R_+(P)}2,\qquad
\boxed{R_o(P)=\frac{R_-(P)-R_+(P)}2}.
$$

同 $|P|$ 的正负动量在完成各自相除后组合：

$$
R_e(|P|)=\frac{R_e(+P)+R_e(-P)}2,\qquad
R_o(|P|)=\frac{R_o(+P)-R_o(-P)}2.
$$

即负动量 odd 先乘 $-1$，不能误写成 plus-minus 后还使用旧样本而不改符号。
最终分析分量为 $\operatorname{Re}R_e$、$\operatorname{Im}R_o$，三个 channel 都保留两分量。

无效 common $d_t$ padding 保留 NaN/mask；接触点 $d_t=0,T$ 已保存，但拟合 cut 排除。

## 5. 01：direct Ratio plateau、AIC 与 fallback

上游目录：

~~~
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_B_ratio20_boot100_T7_Q1em3_chi1e3_20261020_v2
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_B_Pz0_ratio20_boot100_T7_Q1em3_chi1e3_20261021
~~~

唯一实际模型为 constant plateau：

$$
R(T,d_t)=M.
$$

这不是自由双态拟合。物理上 excited-state 项可以写为

$$
R=M+A e^{-\Delta E d_t}+B e^{-\Delta E(T-d_t)}
+C e^{-\Delta E T}+\cdots;
$$

当前靠固定 $T_{\min}$ 和 endpoint cuts 降低污染，未由这个常数模型证明消除 excited states。

$$
T_{\min}=\lfloor0.6\,{\rm fm}/a\rfloor+1=7,\quad
7a=0.6279\,{\rm fm},\quad
T_{\max}=8,9,10,11,12,\quad c=1,2,3.
$$

共有 15 个预定候选：每个使用 $T=7,\ldots,T_{\max}$，$d_t=c,\ldots,T-c$。
接受门是 $Q\ge10^{-3}$、$\chi^2/{\rm dof}\le10^3$，不是 raw $\chi^2<10^3$。

01 的 covariance 处方与 05 不同：使用 Schaefer–Strimmer correlation-to-identity shrinkage，
SVD cut $10^{-12}$，relative/absolute jitter 为 $10^{-12}/10^{-18}$。100 个 bootstrap 的有限信息秩与正则后有效秩分开保存。

每个 cut 内

$$
{\rm AIC}_j=\chi_j^2+2k+2N_{{\rm omit},j},\quad k=1,\qquad
w_j=\frac{e^{-({\rm AIC}_j-{\rm AIC}_{\min})/2}}
{\sum_{j'\in{\cal A}}e^{-({\rm AIC}_{j'}-{\rm AIC}_{\min})/2}},
$$

$$
M_c=\sum_{j\in{\cal A}}w_jM_j.
$$

中心和每一 bootstrap 独立计算质量门及 AIC 权重；各 cut 内混合后，再对可用 cut 等权平均。
不跨不同 cut 直接归一化 AIC。

某 cut 无通过质量门的候选时，使用候选在完整 100 样本上预计算的标准差最小者；raw $\chi^2$ 与预定候选顺序破同分。该条被标为 fallback，不能称为质量通过。
若连有限候选都没有，则保留缺失，不能人为造一个 fit。
quality-only 和 selected、每行 fallback、window/cut 漂移分别保存；进入 02–06 的是已有 selected 样本，窗/cut 系统误差没有自动加入后续统计 error。
$z=0$ 的 odd 为代数零，不伪造一个 $Q=1$ 的拟合。

## 6. 02：局域归一化与 $Z_\chi$

### 6.1 环形归一化和局域比值不是同一产物

普通 flowed fermion 仍需共同场归一化。ringed 定义可写

$$
\mathring\chi=Z_\chi^{1/2}\chi,\qquad
Z_\chi(t_f)=\frac{-2N_c}
{(4\pi)^2t_f^2
\langle\bar\chi\overleftrightarrow{\slashed D}\chi\rangle_{m_R=0}}.
$$

一条 bilocal 有两个 endpoint，整体因子是一个 $Z_\chi$，不是平方、也不是平方根。
一味做 $C_3/C_2$ 并不消除此场因子；$C_2$ 没有 flowed insertion。
动能分母是单 flavor 真空 trace，不是 $u-d$ 真空差。

本轮 L32x64 没有声称完成绝对 ringed 动能测量，而是用同通道局域比值：

$$
D_{\Gamma f}(P,\tau)
=h_{\Gamma f,{\rm even,real}}^{\rm GF}(0,P,\tau),\qquad
\boxed{\mathcal M_{\Gamma f}(z,P,\tau)=
\frac{h_{\Gamma f}^{\rm GF}(z,P,\tau)}{D_{\Gamma f}(P,\tau)}}.
$$

分母与分子同 $\Gamma,f,P,\tau$，$P_z=0$ 用自己的局域分母。不是除以 $h(0,P=0)$，也不是除以 $h(z,0)$。
两个分量均除以同一个 even-real 局域分母，不除以 odd 的零点。

若规范化是同一 ensemble/flow 处方下的公共标量，

$$
\frac{Z_\chi h^{\rm GF}(z)}{Z_\chi h^{\rm GF}(0)}
=\frac{h^{\rm GF}(z)}{h^{\rm GF}(0)}.
$$

这是因子相消，不是测出 $Z_\chi=1$，也不是测得绝对 ringed 矩阵元。
若另设计逐配置随机归一化并在 ensemble 平均内相乘，必须重新声明估计量和统计传播，不能仅凭上式证明该新估计量与本轮相同。

### 6.2 逐 bootstrap 分母与电荷

$$
\mathcal M_c^{(b)}(z)=h_c^{(b)}(z)/D^{(b)}(0).
$$

同一行同时生成分子、分母，不用中心分母代替 bootstrap 分母。
局域点满足精确约束

$$
\mathcal M_{\rm even,real}(0)=1,\qquad
\mathcal M_{\rm odd,imag}(0)=0.
$$

其零误差来自定义，不是额外高精度测量。局域电荷也被除掉，后续得到的是归一化形状；
如需绝对 PDF 归一化，必须另外恢复同方案 charge，tensor/axial charge 不能随意设为 1。
各 flavor 独立归一化后通常有

$$
\mathcal M^{u-d}\ne\mathcal M^u-\mathcal M^d.
$$

## 7. 03：线性发散、$d_F$ 和一次 NP 减除

### 7.1 符号和量纲

直线的正 self-energy convention 为

$$
h^{\rm GF}(z)\sim e^{-m_F(t_f)|z|},\qquad
m_F^{\rm NP}(t_f)=\frac{d_F}{r_f},\qquad r_f=\sqrt{8t_f}.
$$

quark Wilson line 属基础表示；$d_F$ 已含表示依赖，不额外乘 $C_F$。
已有 Wilson relative 诊断给出 $d_F=0.374137$ 的候选：来自
$R/a=6$、$T_w/a=8\ldots12$、参考 $\tau_r=0.15$、上界 1.0、含 $c(\tau-\tau_r)$ 修正的 flow fit。
该上游 flow fit 的 $\chi^2/{\rm dof}=62.06299$、$Q=1.55\times10^{-52}$，不是批准的普适最终系数。
这次按用户要求固定其中心做试算，没有把上游质量失败隐藏。

相关来源：

~~~
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_relative_d_from_Tw_20261020/analysis/final_d_status.json
/public/home/donghx/.codex/Markdown/divergence.md
~~~

### 7.2 实际减除

$$
\boxed{H_{03,c}=
\exp\left[\frac{d_F|z/a|}{\sqrt{8\tau}}\right]\mathcal M_c,\qquad
d_F=0.374137.}
$$

因为 $|z|/r_f=|z/a|/\sqrt{8\tau}$，这里指数无量纲。两个实分量乘同一实指数。
$z=0$ 指数为 1，局域约束保持。

基础表示/伴随表示不能互换，不能再乘 factor 2，也不能在 04 再叠加完整 PT line exponential。

若写纯 PT 转换

$$
\mathcal R^{\overline{\rm MS}}_\Gamma
=\frac{\zeta_\Gamma}{C_\psi}
e^{m_F^{\rm PT}|z|}\mathcal M_\Gamma,
$$

代入 $H_{03}=e^{m_F^{\rm NP}|z|}\mathcal M$ 后是

$$
\mathcal R^{\overline{\rm MS}}_\Gamma
=\frac{\zeta_\Gamma}{C_\psi}
e^{(m_F^{\rm PT}-m_F^{\rm NP})|z|}H_{03}.
$$

当前 04 只乘端点因子、将 NP 线方案视作待定的中间方案 $S$。
不能未经证明假设上式两个 line mass 的差已是有限常数；leading 差、running 和 flow 幂修正仍可能存在。
因此下面的 $m_0$ 是检验中的有限条件，不是自动完成了这项严格方案闭合。

## 8. 04：为什么局域归一化要改 GF matching 系数

### 8.1 两个 SFTE 关系、方向和颜色因子

本节讨论的是四维 flowed **quark** 双线性在小流时间下到
\(\overline{\rm MS}\) 的局域转换。设

$$
\mathring\chi(t,x)=\mathring Z_\chi^{1/2}(t)\,\chi(t,x),
\qquad
\mathring O_\Gamma(t,x)=\bar{\mathring\chi}(t,x)\Gamma
\mathring\chi(t,x).
$$

局域短流时间展开（SFTE）定义 \(\zeta_\Gamma\) 的方向为

$$
\boxed{
\mathring O_\Gamma(t,x)=\zeta_\Gamma(t,\mu)
O_\Gamma^{\overline{\rm MS}}(\mu,x)+O(t\Lambda_{\rm QCD}^2,t m_q^2).}
$$

因此 \(\zeta_\Gamma\) 是 flowed 局域算符到
\(\overline{\rm MS}\) 局域算符的正向系数；真正从 GF 数据恢复
\(\overline{\rm MS}\) 数据时要使用其逆关系。这里的 ringed 场因子、局域
双线性系数和非局部长 Wilson line 端点系数是三个不同对象，不能互换。

在一圈计算中，对相同的外部夸克态计算 flowed 和
\(\overline{\rm MS}\) 的 amputated vertex，并使用 \(\Gamma\) 投影：

$$
\zeta_\Gamma(t,\mu)=
\frac{\operatorname{Tr}\!\left[\Gamma^\dagger
\Lambda_\Gamma^{\rm GF,R}(p,t)\right]}
{\operatorname{Tr}\!\left[\Gamma^\dagger
\Lambda_\Gamma^{\overline{\rm MS},R}(p,\mu)\right]}
\quad\text{(同一投影归一化)}.
$$

可以在共享红外处方下取外部动量为零；GF 与
\(\overline{\rm MS}\) 的红外部分相同，差值留下有限的短流时间匹配系数。流方程给一圈积分带来
\(e^{-t k^2}\) 或 \(e^{-2t k^2}\) 的热核因子，典型积分产生

$$
\mu^{2\epsilon}\!\int\!\frac{d^{4-2\epsilon}k}{(2\pi)^{4-2\epsilon}}
\frac{e^{-2t k^2}}{(k^2)^2}
\ \sim\ \frac{1}{16\pi^2}
\left[\frac{1}{\epsilon_{\rm UV}}+
\ln\!\left(2\mu^2t e^{\gamma_E}\right)+\text{finite}\right].
$$

ringed-fermion 的真空动能归一化去掉流动夸克场的 UV 发散；把 flowed 自能、流动顶角、ringed 场有限项和
\(\overline{\rm MS}\) counterterm 相加后，所有 scheme 依赖的有限项固定下来。定义

$$
\ell_s=\frac{\alpha_s C_F}{4\pi},
\qquad
L_f=\ln\!\left(2\mu^2t_f e^{\gamma_E}\right).
$$

这里必须使用 quark 基础表示的 Casimir

$$
C_F=\frac{N_c^2-1}{2N_c}=\frac43\quad(SU(3)).
$$

用户有时写成 \(C_A\)。如果 \(C_A\) 是标准的 adjoint Casimir
\(C_A=N_c=3\)，它不适用于本节的 quark vector/axial/tensor
bilinear；quark 一圈交换的颜色结构是
\(T^aT^a=C_F\mathbf 1\)。标准 \(C_A\) 应用于 gluon field、三胶子顶角或 adjoint Wilson line。若某篇资料把一般表示 Casimir 另记为 \(C_A\)，必须先声明该记号，不能直接按标准 QCD 记号代入。

### 8.2 一圈系数及其通道依赖

在连续、无质量、nonsinglet、相同 ringed 处方以及声明的有限
\(\gamma_5\)/tensor 处方下，局域 bilinear 的一圈结果为

$$
\boxed{
\zeta_{V/A}(t_f,\mu)=
1+\ell_s\left(\frac12-\ln432\right)
+O(\alpha_s^2),}
$$

$$
\boxed{
\zeta_T(t_f,\mu)=
1+\ell_s\left(L_f-\ln432\right)
+O(\alpha_s^2).}
$$

这里 \(V\) 和 \(A\) 分别对应 \(\Gamma_V=\gamma_z\) 与
\(\Gamma_A=\gamma_z\gamma_5\)，即当前的 unpolarized 和 helicity；
\(T\) 对应 \(\Gamma_T=\sigma_{z\perp}\)（或按 Euclidean convention
带 \(\gamma_5\) 的等价 tensor 投影），即 transversity。
\(\ln432\) 是热核积分与 ringed 归一化在该有限方案中的共同有限常数，
不是拟合得到的线性发散参数。改变有限 ringed 方案时，有限常数会随之改变，
但只要所有步骤使用同一方案，物理转换不变。

为什么 \(V\) 和 \(A\) 相同而 \(T\) 不同？首先，质量为零的
nonsinglet vector/axial current 由手征对称性联系；在连续理论中
\(\gamma_V=\gamma_A=0\)，并且相应的 Dirac 投影满足同样的
一圈代数结构，例如

$$
\gamma_\rho\gamma_\mu\gamma_\rho\ \sim\ \gamma_\mu,
\qquad
\gamma_\rho\gamma_\mu\gamma_5\gamma_\rho\ \sim\ \gamma_\mu\gamma_5.
$$

因此 \(L_f\) 项在 vector/axial 通道中相消，只留下有限常数
\(1/2-\ln432\)。这句话只对 nonsinglet axial current 直接成立；singlet
helicity 还可能受 axial anomaly 和 quark--gluon mixing 影响。

tensor current 是 chiral-odd 且不受 vector/axial Ward 恒等式保护。其
Dirac 投影不同，例如在 \(d\) 维正则化中

$$
\gamma_\rho\sigma_{\mu\nu}\gamma_\rho
\ \sim\ (d-4)\sigma_{\mu\nu},
$$

并且 tensor current 有非零 anomalous dimension。以
\(\alpha_s/(4\pi)\) 为约定，一圈
\(\gamma_T^{(0)}=2C_F\)。因此 \(\zeta_T\) 必须保留
\(L_f\)：

$$
\frac{\partial L_f}{\partial\ln\mu}=2,
\qquad
\mu\frac{d}{d\mu}\zeta_T
=2\ell_s+O(\alpha_s^2),
$$

其尺度依赖正是 tensor current 的 RG 依赖。故 transversity 的差别不是
后续 LaMET kernel 造成的，而是在 GF 局域算符转换到
\(\overline{\rm MS}\) 局域算符时已经出现。

### 8.3 局域归一化后为什么出现 \(\zeta_\Gamma/C_\psi\)

非局部 flowed bilocal 的小流时间关系写成

$$
Z_\chi h_\Gamma^{\rm GF}(z)
=C_\psi(t_f,\mu)e^{-m_F^{\rm PT}|z|}
 h_\Gamma^{\overline{\rm MS}}(z)+\Delta_z,
$$

而局域点满足

$$
Z_\chi h_\Gamma^{\rm GF}(0)
=\zeta_\Gamma(t_f,\mu)h_\Gamma^{\overline{\rm MS}}(0)+\Delta_0.
$$

对代码使用的局域比值

$$
\mathcal M_\Gamma^{\rm GF}(z)=
\frac{h_\Gamma^{\rm GF}(z)}{h_\Gamma^{\rm GF}(0)},
\qquad
\mathcal R_\Gamma^{\overline{\rm MS}}(z)=
\frac{h_\Gamma^{\overline{\rm MS}}(z)}
     {h_\Gamma^{\overline{\rm MS}}(0)},
$$

忽略受控的 SFTE 余项后，有

$$
\mathcal M_\Gamma^{\rm GF}(z)=
\frac{C_\psi}{\zeta_\Gamma}
 e^{-m_F^{\rm PT}|z|}\mathcal R_\Gamma^{\overline{\rm MS}}(z),
$$

所以乘在局域归一化 GF 数据上的逆转换为

$$
\boxed{
\mathcal R_\Gamma^{\overline{\rm MS}}(z)=
\frac{\zeta_\Gamma}{C_\psi}
 e^{+m_F^{\rm PT}|z|}\mathcal M_\Gamma^{\rm GF}(z).}
$$

长线端点系数为

$$
C_\psi=1-\ell_s(3L_f+2+\ln432)+O(\alpha_s^2).
$$

因此在一圈展开而不是直接使用未经展开的有理比时，
\(\ln432\) 抵消并得到

$$
\boxed{
F_{V/A}^{\rm FO}=\frac{\zeta_{V/A}}{C_\psi}
=1+\ell_s\left(3L_f+\frac52\right)+O(\alpha_s^2),}
$$

$$
\boxed{
F_T^{\rm FO}=\frac{\zeta_T}{C_\psi}
=1+\ell_s(4L_f+2)+O(\alpha_s^2).}
$$

当前代码使用这两个一圈逆转换多项式。若使用未归一化的绝对 ringed
bilocal，则先使用 \(C_\psi^{-1}e^{+m_F|z|}\)；只有在分母也采用同一
局域 bilinear 时才出现额外的 \(\zeta_\Gamma\)。局域比值还除去了局域荷，
因此若最终需要绝对 PDF，仍须另行恢复同一方案下的 vector、axial 或 tensor
charge。不能把 \(C_\psi\)、\(\zeta_\Gamma\) 和局域电荷混为一个因子。

先前 presentation 可能使用正向系数 \(C_\psi/\zeta_\Gamma\)；
必须先确认等式方向，再决定代码中乘逆系数还是正向系数。
Pion_DA.pdf p20 的符号也不能未经方向核对直接移植。

### 8.4 LL RGR 的推导与实现

参数固定

$$
\mu=2\,\mathrm{GeV},\quad
\alpha_s(2\,\mathrm{GeV})=0.3,\quad
N_f=3,\quad C_F=4/3,\quad \beta_0=11-2N_f/3=9.
$$

一圈 running：

$$
\alpha_s(\mu)=\frac{\alpha_s(\mu_{\rm ref})}
{1+\frac{\beta_0\alpha_s(\mu_{\rm ref})}{2\pi}
\ln(\mu/\mu_{\rm ref})}.
$$

写 $p_\Gamma=(3,3,4)$、$c_{f,\Gamma}=(5/2,5/2,2)$。
从对数系数求 RG：

$$
\mu\frac{d\ln F_\Gamma}{d\mu}
=2p_\Gamma C_F\frac{\alpha_s}{4\pi}+O(\alpha_s^2),
\qquad
\mu\frac{d\alpha_s}{d\mu}
=-\frac{\beta_0}{2\pi}\alpha_s^2+\cdots.
$$

相除并积分，

$$
d\ln F_\Gamma=-\frac{p_\Gamma C_F}{\beta_0}d\ln\alpha_s,
\qquad
\boxed{U_\Gamma(\mu,\mu_b)=
\left[\frac{\alpha_s(\mu)}{\alpha_s(\mu_b)}\right]^{-p_\Gamma C_F/\beta_0}}.
$$

选 $\mu_b=\mu_f=e^{-\gamma_E/2}/\sqrt{2t_f}$，使边界 $L_f(\mu_f)=0$：

$$
F_\Gamma^{\rm RGR}(t_f,\mu)=U_\Gamma(\mu,\mu_f)
\left\{1+\frac{\alpha_s(\mu_f)C_F}{4\pi}
\left[p_\Gamma\ln(2\mu_f^2t_fe^{\gamma_E})+c_{f,\Gamma}\right]\right\}.
$$

04 实际逐点

$$
H_{04,c}(z>0,P,\tau)=F_\Gamma^{\rm RGR}H_{03,c}(z,P,\tau).
$$

$z=0$ 不套长线系数，保留 $(1,0)$。
nonlocal long-line 与 local limit 的极限次序不同，不能把局域约束当作估计 $F_\Gamma$ 的数据。

这只是“一圈 boundary + LL”，不是 NLL、LRR 或 PV。没有导入
Renorm_3ens_nloRGR.py 的 RAW/多格距 self-renormalization 质量符号或 LRR 参数。
代码对 $\mu_b<0.7\,\mathrm{GeV}$ 或 $\alpha_s>0.6$ 报错/缺失，
$\alpha_s>0.4$ 留警告，不冻结或剪裁 coupling。
$P_z=0$ predecessor 还保存 $k_f$ 或 $k_z=0.75,1.5$ 的独立尺度探针，不等于完整微扰系统误差。
moving 主链仅用 central RGR 和 FO 控制分支。

## 9. 05：小 flow-time 纯线性外推及尺度限制

### 9.1 当前实际模型和 covariance

每个 component、$P,z,\Gamma,f$ 使用

$$
H_{04}(\tau)=c_0+c_1\tau,\qquad H_{05}=c_0.
$$

至少 4 个 flow 点，使用允许集合中最宽连续窗。没有二次模型、没有 AIC 模型平均、没有 prior，
不因单点看起来更好而改窗。01 的 AIC 与本步不同，不能混称为“全流程 AIC”。

设 $B_{bi}$ 为第 $i$ 个 flow 的 bootstrap：

$$
C_{ij}=\frac1{B-1}\sum_b
(B_{bi}-\overline B_i)(B_{bj}-\overline B_j).
$$

用完整 cross-flow covariance，`ddof=1`。先取 $D=\operatorname{diag}(\sqrt{C_{ii}})$ 和相关矩阵
$R=D^{-1}CD^{-1}=U\Lambda U^T$，删除
$\lambda_k/\lambda_{\max}<10^{-8}$ 的模式；原始信息秩记录门为 $10^{-12}$。
不是把 eigenvalue 抬高，不是改 diagonal covariance。

保留模式的 whitening 为

$$
W=\Lambda_{\rm kept}^{-1/2}U_{\rm kept}^TD^{-1},\qquad
K_{\rm GLS}=[(WX)^T(WX)]^{-1}(WX)^TW,
\quad X_i=(1,\tau_i).
$$

$$
\widehat\beta=K_{\rm GLS}y,\qquad
\widehat\beta^{(b)}=K_{\rm GLS}B_b,\qquad
\widehat\beta=(c_0,c_1).
$$

实际用 whitened/column-scaled least squares；满秩时独立检查普通 full-covariance inverse。
${\rm dof}=r_{\rm eff}-2>0$。
finite、rank、positive-dof 是硬门；不能通过 prior、删失败样本或对角 fallback 造拟合。
$\chi^2,Q$ 只记录，不作为删点门：
$Q\ge0.05$ 且 $\chi^2/{\rm dof}\le3$ 是单独的诊断 flag，不是最终物理验收。

### 9.2 两套窗口

allflow_display_only：全部 $\tau=0.2,\ldots,0.8$，无尺度筛选，仅展示/诊断。

D_loose：同时要求

$$
\frac{a}{r_f}<1,\qquad \frac{r_f}{|z|}<1,\qquad
\sqrt{t_f}|P_z|=\sqrt\tau\,\frac{2\pi|n_P|}{32}<1.
$$

$z/a=1$ 时 $r_f/a\ge\sqrt{1.6}>1$，全部不合格；
$z/a=2$ 时 $\tau<0.5$，仅 $0.2,0.3,0.4$ 三点，不足 4 点。
因此两点的 D_loose 截距缺失，不得用 all-flow 数值补成“受控”结果。
$P_z=6,\tau=0.8$ 还有 $\sqrt{t_f}|P_z|=1.05372>1$，D_loose 会排除该点。

### 9.3 约束的来源

流后典型 UV 动量为 $k\sim1/\sqrt{t_f}$。
格点色散 $\widehat k^2=k^2+O(a^2k^4)$ 带来相对 $a^2/t_f$ artefact，
故受控 continuum flow 分析要求 $a^2/t_f\ll1$；不是任意正 $t_f$ 都满足。
长线展开还要求 $t_f/z^2\ll1$，外态热流修正含 $t_fP_z^2$，长程含
$t_f\Lambda_{\rm QCD}^2$。这些给出典型 hierarchy：

$$
a\ll r_f\ll|z|,\qquad \sqrt{t_f}|P_z|\ll1,\qquad
t_f\Lambda_{\rm QCD}^2\ll1.
$$

当前七点 $a^2/t_f=1/\tau=1.25\ldots5$，没有参数性很小的 cutoff 比；
“$a<r_f$”只对应 $\tau>1/8$，与这个更强条件不同。

单格距下取 $\tau\to0$ 会走入未受控 cutoff 区，因此 05 是固定 $a$ 截距诊断。
受控做法需固定物理 $t_f,z,P$ 做连续极限后研究小流时间，或有论证的 joint fit。
RGR 只整理 logarithms，不修复缺少尺度层级、cutoff 或 flow 幂修正。

## 10. 06：短距离 $C_0$、$m_0$ 与配对 moving 修正

### 10.1 为什么减掉 $d_F/r_f$ 后还讨论残余质量

减去沿线长度累积的 UV self energy，并没有自动固定有限、与长度成正比的 scheme freedom。
在已声明的中间方案 $S$ 下可写

$$
H_{S,\Gamma}(z,P)=e^{-m_{0,\Gamma}|z|}
\mathcal R_\Gamma^{\overline{\rm MS}}(z,P)+\text{余项}.
$$

短距离和归一化处方的有限差、line-mass scheme 以及漏掉的幂修正都可能影响拟合出的斜率。
当前分通道/flavor 诊断 $m_0$ 不能自动解释为已证实的普适线质量；
很强的 channel/flavor 依赖是需要检查的证据。

### 10.2 为什么能尝试 $P_z=0$，目标又为什么不是常数 1

对短距离归一化 nonsinglet 矩阵元，局域 OPE 含

$$
\mathcal R_\Gamma(z,P,\mu)=
\sum_n C_{\Gamma n}(z^2\mu^2)
\frac{(izP_z)^n}{n!}
\frac{\langle x^n\rangle_\Gamma}{\langle x^0\rangle_\Gamma}
+O(z^2\Lambda^2,z^2M_N^2,\text{mixing/other amplitudes}).
$$

在适用的不变振幅和投影条件下，$P_z=0$ 消去 $n\ge1$ 的 Ioffe-time 幂项；
leading 目标仍是 $C_{0,\Gamma}(z,\mu)$，而不是把 radiative coefficient 也设为 1。
当前将此应用到三通道/connected flavors 是声明的短距离诊断；完整 trace、target-mass、有限格距/端点方案关系仍需检查。

用户指定的一圈目标：

$$
L_z=\ln\frac{z_{\mathrm{GeV}^{-1}}^2\mu^2}{4e^{-2\gamma_E}},\qquad
C_{0,\Gamma}^{\rm FO}=1+\frac{\alpha_s(\mu)C_F}{4\pi}
(p_\Gamma L_z+c_{0,\Gamma}),
$$

| 通道 | $p_\Gamma$ | $c_{0,\Gamma}$ |
|---|---:|---:|
| unpolarized | 3 | 5 |
| helicity | 3 | 7 |
| transversity | 4 | 4 |

不要将目标常数 $(5,7,4)$ 与 flow 端点常数 $(2.5,2.5,2)$ 混用。

自然尺度 $\mu_z=2e^{-\gamma_E}/|z|_{\mathrm{GeV}^{-1}}$，RGR 目标为

$$
C_{0,\Gamma}^{\rm RGR}(z,\mu)=U_\Gamma(\mu,\mu_z)
\left[1+\frac{\alpha_s(\mu_z)C_F}{4\pi}
\left(p_\Gamma\ln\frac{z^2\mu_z^2}{4e^{-2\gamma_E}}
+c_{0,\Gamma}\right)\right].
$$

边界对数在中心自然尺度为 0。FO 的 $H$ 必须配 FO 的 $C_0,m_0$，RGR 也必须配同分支；
不能从另一分支抽一个更小误差质量。

### 10.3 固定 $z/a=1,2$、$b=0$ 的拟合

$P_z=0$ 主窗为

$$
z_1=0.0897\,{\rm fm},\qquad z_2=0.1794\,{\rm fm}.
$$

所以这是用户后来选定的两点窗，不再声称严格 $z<0.15\,{\rm fm}$；它满足 $z<0.3\,{\rm fm}$。
定义

$$
Y_i=\ln\frac{H_{05,\Gamma}^{P=0}(z_i)}
{C_{0,\Gamma}^{\rm RGR}(z_i,\mu)}
=-\frac{m_{0,\Gamma}|z_i|_{\rm fm}}{\hbar c},\qquad b=0.
$$

两点、一个参数；满秩时 $\mathrm{dof}=1$。$z=0$ 不参与对数，不增加一个“无误差拟合点”。
不加第二个线质量参数，不再拟合 $d_F/r_f$。
此前 allflow_display_only: b free 的 b 是 log-ratio 的截距诊断，不是 Wilson self energy；本主链不使用它。

每行必须 $H_i>0,C_{0i}>0,H_i^{(b)}>0$，否则如实缺失。
不取绝对值、不剪裁、不剔除过零 replicas 来制造 real log。
log-ratio 的完整 covariance 用配对 $Y^{(b)}$ 构造，GLS 约定与 05 相同。

$$
m_0=-(\ell^TC_Y^+\ell)^{-1}\ell^TC_Y^+Y,
\qquad \ell_i=|z_i|_{\rm fm}/(\hbar c).
$$

原 $P_z=0$ 代码还保存 $z/a=1,2,3$ 和 $2,3$ 的诊断窗以及独立尺度探针；不自动替代主窗。
D_loose 在 $z/a=1,2$ 没有合格 $\tau=0$ 截距，因此没有该窗口下的合格主质量。

### 10.4 当前 RGR 质量结果

以下来自 m0_sources.csv，100/100 paired rows；统计误差固定 $d_F,\alpha_s$ 和方案。

| channel | flavor | $m_0$ [GeV] | $\chi^2/{\rm dof}$ | $Q$ |
|---|---|---:|---:|---:|
| unpolarized | u | $-0.157170\pm0.000481$ | 142910 | 近 0 |
| unpolarized | d | $-0.245370\pm0.000763$ | 95363 | 近 0 |
| unpolarized | u-d | $-0.201458\pm0.001118$ | 24274.1 | 近 0 |
| helicity | u | $-0.135751\pm0.002177$ | 3098.54 | 近 0 |
| helicity | d | $-0.149782\pm0.003941$ | 1592.97 | 近 0 |
| helicity | u-d | $-0.354539\pm0.000242$ | 264818 | 近 0 |
| transversity | u | $-0.198098\pm0.001499$ | 18967.6 | 近 0 |
| transversity | d | $-0.217740\pm0.002752$ | 7579.74 | 近 0 |
| transversity | u-d | $-0.184031\pm0.002029$ | 7628.22 | 近 0 |

极大的 $\chi^2$ 不能由“bootstrap 齐全”或“误差很小”修复。不得把这些数字宣称为通过拟合的物理质量。
$Q=0$ 是 CSV 浮点输出，理解为极小/数值下溢，不是数学上严格概率为零。

### 10.5 moving 06：使用原行配对，不重拟合

对每个 channel/flavor 读取已有 $P_z=0$ central $m_0$ 和 100 行 $m_0^{(b)}$，
四动量、两分量共同乘实指数：

$$
\boxed{H_{06,c}^{(b)}(P,z)=H_{05,c}^{(b)}(P,z)
\exp\left[\frac{+m_0^{(b)}(P=0)|z|_{\rm fm}}{\hbar c}\right].}
$$

中心也用原中心 $H_{05}$ 与中心 $m_0$。相同 b 是原 bootstrap 行，不重新抽取或打乱质量样本。
误差由 corrected samples 的标准差取得；包括本处方下 $m_0$ 的统计涨落和与 $H_{05}$ 的相关性。
当前质量为负，因此 06 的振幅通常比 05 小；这来自已定义的正指数，不应为图像美观改号。

不能逐动量再调质量使曲线对齐，也不能给虚部取独立质量。
符号 $m_0$ 在不同文章中可指正斜率或负斜率；与 arXiv:2607.13024v1/旧 Renorm 代码比较前须先对齐定义。

## 11. 数值结果、覆盖和检查

### 11.1 完成范围

最新 moving CPU Slurm job：3869986，完成标记

~~~
PASS_MOVING06_PAIRED_RGR_READBACK
physics_validated = false
~~~

主 RGR all-flow 分支：

- $9$ 个 channel/flavor；
- 每个 $2\times4\times21=168$ 个 component/momentum/z entries；
- 共 $1512/1512$ finite entries、$151200/151200$ bootstrap entries；
- $z>0$ 的非零 odd-imag 有 720 个；
- 72 个 $z=0$ entries 是 $(1,0)$ exact constraints，不是新拟合；
- 每个可用点完整保留 100 行。

D_loose 每个 channel/flavor 有 152/168 finite，缺的 16 个是 $z/a=1,2$ × 2 components × 4 momenta。
九种组合合计 1368/1512 finite。缺失点保持 NaN/status，图上的灰字说明不应消掉。

05 的中心质量 flag 统计（由 flow_fit_diagnostics.csv 汇总，不重新拟合）：

| policy，RGR | 真正执行的 nonlocal fits | $Q\ge0.05,\chi^2/{\rm dof}\le3$ | poor_fit_record_only | local constraints |
|---|---:|---:|---:|---:|
| allflow_display_only | 1440 | 949 | 491 | 72 |
| D_loose | 1296 | 861 | 435 | 72 |

这些 Q 值是条件于当前 covariance/处方的诊断；保留数值不代表质量通过。

### 11.2 一组可用于继续工作的数值校验点

例：$P_z$ mode=3、$u-d$、$z/a=4$、RGR/allflow_display_only，读 points.csv：

| channel | component | 05 $c_0$ | 06 配对质量修正后 |
|---|---|---:|---:|
| unpolarized | even-real | $1.377476\pm0.005465$ | $0.954986\pm0.004043$ |
| unpolarized | odd-imag | $-0.646597\pm0.004929$ | $-0.448278\pm0.003523$ |
| helicity | even-real | $1.445078\pm0.004081$ | $0.758438\pm0.002115$ |
| helicity | odd-imag | $-0.868185\pm0.008006$ | $-0.455660\pm0.004183$ |
| transversity | even-real | $1.422187\pm0.004291$ | $1.017727\pm0.004420$ |
| transversity | odd-imag | $-0.903775\pm0.003047$ | $-0.646747\pm0.003168$ |

只是 baseline readback 数字，不是物理结论。后续若读取结果得到全零虚部或整体反号，应先核对字段、odd 方向、相位和质量指数。

### 11.3 已完成的验证

依据 verification.json、regression.json：

1. 20 conf 顺序与索引 SHA256 对齐；
2. 01–03 逐数组复用、02 分母和 03 指数逐 bootstrap 恒等式；
3. B 权重仅一次，无额外 PT line exponent；
4. 9 个 $P_z=0$ RGR endpoint、实虚部纯线性和 100 样本回归；
5. FO 分支全部中心、100 样本与已有 FO 输出逐项一致；
6. 5472 次满秩 full-covariance inverse 独立对照；
7. 06 的质量来源、同 b 配对、GeV/fm、指数正号和两分量乘同一因子；
8. 回读验证保留非零虚部；
9. 25 个结果 PDF（各 9 页）的页数、marker-only、图例/坐标范围和部分页面渲染检查。

上述只证明数值链按声明执行；不会使坏的 $m_0$ 质量或缺失的 hierarchy 自动成立。

## 12. 所有关键路径与数据格式

### 12.1 缩写根路径

以下路径均为绝对路径，不能把 A/B 或旧/新 moving 分支混读。

| 缩写 | 根目录 |
|---|---|
| ROOT | /public/group/lqcd/donghx/Gradient_Flow_GluonPDF |
| RAW_B | ROOT/quark_l32x64_B_insertions_20conf_8flow_20261017 |
| FIT | ROOT/quark_l32x64_B_ratio20_boot100_T7_Q1em3_chi1e3_20261020_v2 |
| FIT_P0 | ROOT/quark_l32x64_B_Pz0_ratio20_boot100_T7_Q1em3_chi1e3_20261021 |
| LOCAL | ROOT/quark_l32x64_B_localnorm20_boot100_20261021_v2 |
| LINE | ROOT/quark_l32x64_B_localnorm_line_dF0p374137_20261021 |
| SOURCE | ROOT/quark_l32x64_B_pionDA_workflow_20261022 |
| PARENT | ROOT/quark_l32x64_B_m0_RGR_z12_20261025 |
| CURRENT | ROOT/quark_l32x64_B_moving_RGR_m0_20261025 |

RAW_B 主产品位于 C3_C2_schemeB/；居中点只读复用路径由 manifest 和 receipt 指出。
01–03 文件位于 SOURCE/{01_bare,02_localnorm,03_line}/data/{moving,Pz0}/。
$P_z=0$ 质量位于

~~~
PARENT/04_m0/RGR/allflow_display_only/{channel}_{flavor}_primary_z12.npz
~~~

最新主 moving 数据：

~~~
CURRENT/04_flow_conversion/RGR/moving/tau0p20_{channel}_{flavor}.npz
CURRENT/05_tau0_linear/RGR/allflow_display_only/moving/tau0_{channel}_{flavor}.npz
CURRENT/06_m0_corrected/RGR/allflow_display_only/moving/{channel}_{flavor}.npz
~~~

flow 文件 tau0p20,...,tau0p80；flavor 文件名用 u、d、u_minus_d。
FO 和 D_loose 分别在相应的 variant/policy 子目录，不与主分支拼接。

### 12.2 分步骤 NPZ

~~~
value                 (component=2, p=4, z=21)
samples               (component=2, boot=100, p=4, z=21)
error                 std(samples, axis=1, ddof=1)
value_complex         (p=4,z=21), where present
samples_complex       (boot=100,p=4,z=21), where present
component_labels      ['even_real','odd_imag']
pz_values             [3,4,5,6]
delta_z_values        [0,...,20]
boot_indices          (100,20)
confs                 (20,)
~~~

仅 stage-04/05/06 等实际添加 projection 的文件保证有 complex 字段；
不要假定旧 SOURCE/01–03 也包含相同复数字段。
value/samples 为实数两分量表示，component=1 已是 $\operatorname{Im}H_o$，禁止再次取 imag。

05 另保存 covariance、reference_indices、reference_beta/boot、sample/effective/model ranks、
dof、chi2、Q、fit_executed、constraint_mask、bootstrap_usable、inherited_fallback/boot。
06 附 mass_GeV/boot、factor/boot、mass_source、mass_source_sha256 和未验证普适性标记。
字段精确拼写以 run_cpu.py 和 NPZ metadata 为准；不凭宽泛“同格式”猜轴。

### 12.3 六步骤汇总 NPZ 的不同 schema

~~~
CURRENT/data/RGR/{channel}_{flavor}.npz
~~~

这里不是单个通用 value/samples，而是带步骤前缀：

~~~
01_bare_value, 02_localnorm_value, 03_line_value, 04_RGR_conversion_value
  axes: (component, flow, p, z) = (2,7,4,21)

01_bare_samples, ..., 04_RGR_conversion_samples
  axes: (component, flow, boot, p, z) = (2,7,100,4,21)

05_tau0_linear_value, 06_m0_corrected_value
  axes: (component,p,z) = (2,4,21)

05_tau0_linear_samples, 06_m0_corrected_samples
  axes: (component,boot,p,z) = (2,100,4,21)
~~~

还有 steps、flows、mass_source、mass_GeV、mass_boot_GeV 及 provenance。
flow 轴与 bootstrap 轴顺序不能互换；$P_z=0$ all-z export 的单动量/flow axes 也要单独读取。

## 13. 结果图片和查看顺序

所有主图根：

~~~
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_B_moving_RGR_m0_20261025/plots
~~~

| 文件 | 内容 |
|---|---|
| 01_bare.pdf | 七 flow 裸矩阵元，四动量×实/虚两列 |
| 02_localnorm.pdf | 局域归一化后的相同布局 |
| 03_line.pdf | 一次线减除后 |
| 04_RGR_conversion.pdf | 有限 flow 的 RGR conversion 后 |
| 05_RGR_tau0_linear.pdf | 纯线性 flow 截距 |
| 05_06_RGR_moving_comparison.pdf | 05/06 同 subplot 比较，推荐首先查看 |
| 06_RGR_m0_corrected.pdf | 仅 06 四动量结果 |
| 05_06_RGR_D_loose.pdf | 弱尺度窗口；缺失点如实保留 |
| 05_06_FO_moving_comparison.pdf | FO 一致处方控制分支 |

每份 9 页，顺序 channel 外循环 unpolarized/helicity/transversity，
flavor 内循环 u/d/u_minus_d。因此第 3、6、9 页分别是三通道的 $u-d$。
01–04 另有 plots/01_bare/Pz3.pdf 等按动量拆分的 16 个文件。
05–06 将四个动量放同一页的不同子图便于比较。

画图约定：

- 横轴 $z/a=0,\ldots,20$，显示范围 $[-0.5,20.5]$；
- 实部 y limits $[-1,2]$，虚部 $[-2,1]$；
- finite-flow 点横向小偏移，offset 不代表改变物理 $z$；
- 不同 $z$ 的数据不连实线，error bars 使用 markers；
- $z=0$ exact constraints 单独标记；
- 超出统一 y limits 的点没有从 NPZ 删掉，clipping 记录在 points.csv/axis_audit.json；
- D_loose missing 不用零点或 allflow 值填充。

直接链接：

- [05–06 RGR 比较图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_B_moving_RGR_m0_20261025/plots/05_06_RGR_moving_comparison.pdf)
- [06 RGR 四动量结果](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_B_moving_RGR_m0_20261025/plots/06_RGR_m0_corrected.pdf)
- [D_loose 对照](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_B_moving_RGR_m0_20261025/plots/05_06_RGR_D_loose.pdf)
- [完整图片索引](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_B_moving_RGR_m0_20261025/plots/index.csv)

Pz0 质量与短距离目标比较图位于 PARENT/05_plots/；对应 index.csv 列出全部 PDF，
全 $z$ 且保留虚部的步骤导出在

~~~
PARENT/step_matrix_elements_all_z_fixed_re_im_20261025/
~~~

旧 quark_l32x64_B_momentum_step_plots_20261025 的 moving 04/05 曾读取 20261022 FO/二次 AIC 分支。
最新 CURRENT 已统一为 RGR+纯线性；如需当前主结果，不再把旧图混作最新 RGR。

## 14. 代码入口、提交命令和运行边界

最新代码：

| 文件（相对 CURRENT） | 职责 |
|---|---|
| code/moving_common.py | 源路径/哈希、axes、原索引、端点函数导入、scale mask、投影规则 |
| code/run_cpu.py | 04 conversion → 05 full-covariance 线性 → 06 paired mass；只读 01–03 |
| code/plot_cpu.py | 25 个 PDF、图例布局、轴和 clipping 账本 |
| code/verify_cpu.py | 独立回读、配对、质量符号、统计误差及结果状态 |
| submit.py | stdlib-only 语法/哈希冻结与 sbatch，拒绝重复提交 |
| slurm/run_cpu.slurm | CPU 数值处理/画图/验证顺序 |

复用 PARENT/code/coefficients.py、fit_core.py；后者定义相关矩阵小模删除 GLS。
上游原始 C3→ratio 的精确实现应看 FIT/raw_B.py、ratios_cpu.py、fit_cpu.py、reference/conventions.py。

已有运行命令（记录，不在已完成目录重复执行）：

~~~bash
/public/home/donghx/AnaConda/bin/python -B /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_B_moving_RGR_m0_20261025/submit.py
~~~

包装器无须 --submit 参数；它已有 submission.json 时会拒绝重复提交。
CPU 脚本使用 a192c1t、cpu6248R、cpueicc、i72c512g 四个 partitions，1 task、1 CPU、8 GB、45 min，
清理 PYTHONHOME/PYTHONPATH、禁用 GPU 并固定 BLAS 线程后依次运行 run_cpu.py → plot_cpu.py → verify_cpu.py。
数值投影、bootstrap、拟合、回读测试和作图应在 CPU allocation 中执行，不在登录节点运行。

future GPU $C_3$ 生产沿用用户指定 na800-sug 与 reservation donghx_223；
本文更新不代表已经提交新的 GPU 或 CPU 分析任务。
不能修改冻结源码后在原目录“继续”：读取的 source_manifest 包括代码/README/配置和上游文件哈希，
变更窗口、估计量、质量或系数须新建 isolated successor。

可只读查看：

~~~bash
sacct -j 3869986 --format=JobID,State,ExitCode,Elapsed,NodeList
head -20 /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_B_moving_RGR_m0_20261025/coverage.csv
~~~

## 15. 不可无记录改变的约定

1. thin-link、四维 flow、相同 gauge history 和 fermion APBC；
2. L32x64 $a=0.0897$ fm、这 20 conf、100 行 seed1115 索引；
3. 七 flow、$\epsilon=0.01$、$w$ 半窗、四个 anchors 及 dt-matched $C_2$；
4. B 权重按原始 $\sigma_a,\sigma_b$ 判断、只施加一次；
5. $C_2$ 取复数 nopol 的 ratio-of-bootstrap-means，先逐 signed-P 相除再配对；
6. odd 为 minus-plus，负 P 的 odd 反号，channel phases 为 $1,-i,+i$；
7. 三通道都保留 even-real/odd-imag，第二分量不再取 imag；
8. direct Ratio 仅 constant 模型，$T_{\min}=7$，15 个窗/cut，AIC/fallback 分开；
9. 局域分母同 channel/flavor/P/tau 的 even-real $z=0$，电荷未恢复；
10. 固定 $d_F$ 仅一次正指数，不加 Casimir/2 或额外 PT line exponential；
11. GF 逆端点因子 $\zeta_\Gamma/C_\psi$，一圈 boundary + LL，不冒称 NLL/LRR；
12. 05 纯线性，完整 covariance、小模删除和 rank/finite 硬门；
13. 06 使用同分支 $P_z=0$ $z/a=1,2,b=0$ 原质量，不在 moving 上重拟合，严格同 b 配对；
14. D_loose 缺失、poor-fit/fallback 状态和 transversity 待外部确认必须保留；
15. 受控 continuum、最终 $\overline{\rm MS}$、LaMET/full singlet 尚未完成的边界。

这些是冻结 baseline 的约定；不是禁止未来改进。修改时须另建版本、保存比较与哈希，不能静默改写旧结果。

## 16. 待确认项及下一步

1. **首先审视 $m_0$ 失配。**查看 $P_z=0$ 的 $H_{05}$、RGR/FO $C_0$、两点 log ratio 与诊断三点/尺度窗；检查短距离 $r_f/z$、cutoff、局域电荷约定和 finite-scheme closure。当前 $Q$ 不支持把 06 当作已验收结果。
2. **传播 line-mass 系统误差。**固定候选不是最终 $d_F$；重新看 Wilson-time、$R$、flow reference/window 稳定性和 interaction 污染。1700-conf/3000-bootstrap loop 样本不能按行号冒充与 20-conf/100-bootstrap quark 样本配对；需联合/外部 nuisance 处方。
3. **补足 B 的统计与时间支持论证。**保留已做块/Wick/window 测试，重点检查非居中点、boundary anchors、跨时间 generator/noise 高阶矩，不能将实现一致当完全无偏证明。
4. **检查两种 normalization 目的。**若只做局域归一化形状，继续用公共 $Z_\chi$ 相消和对应 $\zeta_\Gamma/C_\psi$；若要绝对 ringed 矩阵元/电荷，才补同处方 kinetic/Ward/charge 及配对误差，不能把旧 L32x96 的 $Z_\chi$ 数字直接套到 L32x64。
5. **完整 flow/spacing 研究。**当前一格距截距不受控；L32x96 后续应按相同物理 $r_f$ 而不是相同 $\tau$ 配对，状态需重新审计 scheduler/receipt，不依据旧文档推断已完成。
6. **下游 $\lambda$→Fourier→LaMET。**在 normalization、有限 scheme 和质量处方固定后，保留两复分量和缺失 mask，处理有限 $\lambda=zP_z$ tail 后 Fourier，再用相同归一化 scheme 的通道 kernel。已有历史 A/$Z_\chi$=1 downstream 只能作参考，不能混入当前 B 链。
7. **transversity 及 singlet。**外部相对算符符号待确认；unpolarized/helicity singlet 需要 disconnected 和 quark–gluon mixing，transversity 无 leading-twist gluon mixing 不等于本地符号已验证。

## 17. 来源、文献和 Note 对应

本轮直接核对的证据入口：

- CURRENT/README.md、RESULTS.md、config.json、source_manifest.json；
- CURRENT/verification.json、regression.json、complete.json；
- CURRENT/m0_sources.csv、flow_fit_diagnostics.csv、coverage.csv、axis_audit.json、points.csv；
- PARENT/code/coefficients.py、fit_core.py、run_cpu.py、README.md、RESULTS.md；
- SOURCE/code/audit_and_prepare_cpu.py、common.py、convert_cpu.py；
- FIT/common.py、raw_B.py、ratios_cpu.py、fit_cpu.py、reference/conventions.py、README.md。

理论来源分层：

- Lüscher gauge/fermion flow 与 Makino–Suzuki ringed 归一化；
- Brambilla–Wang，arXiv:2312.05032v3，特别是 Appendix A 的更正后端点/长线公式；
- arXiv:2311.16799 的局域双线性 flowed coefficients；具体 inverse ratio 和有限处方见本文第 8 节及 divergence.md §27；
- 本地 [Pion_DA.pdf](/public/home/donghx/Note/Gradient_Flow/Pion_DA.pdf) p20 的 flow→MS/RGR 流程作为比较依据，非直接照抄其符号；
- arXiv:2607.13024v1 的 residual mass 思路与本地 [divergence.md](/public/home/donghx/.codex/Markdown/divergence.md)；不同质量斜率 convention 需显式换号；
- [tau.md](/public/home/donghx/.codex/Markdown/tau.md) 记录尺度条件的独立推导；
- [gradient_flow.md](/public/home/donghx/.codex/Markdown/gradient_flow.md) 保留历史 quark/gluon 链和结果；不把其旧日期状态覆盖本轮核验。

Note 同步源为 [main.tex](/public/home/donghx/Note/Gradient_Flow/main.tex)，
最新独立章节题为 “L32x64 scheme-B moving quark PDF: complete 01–06 analysis”，
label 为 sec:l32x64-moving-20261025；编译后 [main.pdf](/public/home/donghx/Note/Gradient_Flow/main.pdf) 的目录可直达。
新章节明确覆盖本轮 01–06、推导、质量表、实虚部/axis 保留与所有限制；旧历史章不删除。

---

生成/更新日期：**2026-10-25（Asia/Shanghai）**。  
本次会话标题：**L32x64 B 方案 quark PDF 六步流程、局域归一化 RGR 推导及 paired $P_z=0$ 残余质量结果交接**。  
本文不记录 API key、token、密码或其他凭据。
