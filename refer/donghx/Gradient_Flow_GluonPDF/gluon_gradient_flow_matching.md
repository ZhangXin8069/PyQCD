# Gluon unpolarized/helicity gradient-flow matching：理论推导与当前实现

本文整理本项目中 thin-link、四维 Wilson gradient flow 胶子算符的
matching 约定。目标是把格点上有限流时间的 bare OPE/C3/C2，依次转换为
重整化的空间类 gluon bilocal（quasi/pseudo 矩阵元），再与 light-cone
unpolarized gluon PDF $g(x,\mu)$ 和 helicity gluon PDF
$\Delta g(x,\mu)$ 联系起来。

这里要严格区分三个层次：

1. **lattice finite-flow bare observable**：本项目 `C3/C2` 和 Sumratio
   fit 得到的 $M^{\rm GF,bare}$；
2. **gradient-flow scheme ($\to\overline{\rm MS}$) quasi/pseudo bilocal**：
   小流时间展开（small-flow-time expansion, SFTE）和 Wilson-line line-mass
   conversion；
3. **quasi/pseudo $\to$ light-cone PDF**：LaMET 或 pseudo-ITD matching，
   胶子 singlet 通道还包括 quark–gluon mixing。

只有完成三层以及连续极限、$P_z$ 极限和 power-correction 控制后，才可以
称为物理的 gluon PDF。本文件中的一圈系数主要来自
Brambilla--Wang, arXiv:2312.05032（JHEP 06 (2024) 210）；unpolarized
和 helicity 的 quasi/pseudo-to-light-cone kernels 分别见
arXiv:2107.08960、arXiv:2207.08733 和 arXiv:2112.02011。后几篇并不是
本项目 finite-flow 算符的 GF-to-$\overline{\mathrm{MS}}$ 系数，不能直接替代 SFTE。

---

## 1. 记号、流方程和本项目的 scheme

### 1.1 四维 Wilson flow

连续 Euclidean flow field $B_\mu(\tau,x)$ 满足

$$
  \partial_\tau B_\mu(\tau,x)=D_\nu G_{\nu\mu}(\tau,x),
  \qquad B_\mu(0,x)=A_\mu(x),
$$

$$
  G_{\mu\nu}=\partial_\mu B_\nu-\partial_\nu B_\mu+[B_\mu,B_\nu],
  \qquad D_\mu=\partial_\mu+[B_\mu,\,\cdot\,].
$$

本项目的初始场 $A_\mu$ 是**未做 HYP 或其他 smear 的 thin links**。在格点
上使用三阶 Runge--Kutta Wilson-flow，步长

$$
  \epsilon=0.01,
  \qquad \widehat\tau\equiv\frac{\tau}{a^2}
       =N_{\rm step}\,\epsilon .
$$

正流时间的平滑半径为

$$
  r_{\rm flow}=\sqrt{8\tau}=a\sqrt{8\widehat\tau}.
$$

自由场极限中一个 flowed gauge field 带来 $e^{-\tau k^2}$，两端的
field-strength bilocal 因而带来 $e^{-2\tau k^2}$。这是**四维**动量的
阻尼；若只在三维空间 flow，matching 系数必须重新计算，不能使用四维
flow 的系数。

### 1.2 Euclidean/Minkowski 约定

Minkowski 采用

$$
  \eta_{\mu\nu}=\operatorname{diag}(1,-1,-1,-1),
  \qquad \epsilon_{\rm M}^{0123}=+1,
  \qquad \widetilde F^{\rm M}_{\mu\nu}
  =\frac12\epsilon^{\rm M}_{\mu\nu\rho\sigma}F_{\rm M}^{\rho\sigma}.
$$

代码坐标顺序为

$$
  (x,y,z,t)=(0,1,2,3),
  \qquad \epsilon_{\rm E}^{xyzt}=+1,
  \qquad \widetilde F^{\rm E}_{\mu\nu}
  =\frac12\epsilon^{\rm E}_{\mu\nu\rho\sigma}F^{\rm E}_{\rho\sigma}.
$$

取 Wick rotation

$$
  x^0_{\rm M}=-i x^t_{\rm E},\qquad A_t^{\rm E}=-iA_0^{\rm M},
$$

则

$$
  F^{\rm E}_{ti}=-iF^{\rm M}_{0i},
  \qquad F^{\rm E}_{ij}=F^{\rm M}_{ij},
$$

$$
  \widetilde F^{\rm M}_{0i}=\widetilde F^{\rm E}_{ti},
  \qquad \widetilde F^{\rm M}_{ij}=-i\widetilde F^{\rm E}_{ij}.
$$

因此 temporal-electric 和 spatial-magnetic 项在 helicity channel 中的
Wick phase 不同。代码中的 dual tensor 直接使用 Euclidean epsilon；不能
在 OPE 输出之后再额外乘一个 temporal-link phase。

### 1.3 Clover field strength 和 bilocal

代码构造 flowed four-leaf Clover tensor

$$
  F^{\rm clov}_{\mu\nu}(\tau,x)
  =-\frac{i}{8}\left[Q_{\mu\nu}(\tau,x)-Q_{\mu\nu}^\dagger(\tau,x)\right],
$$

并作 color-traceless projection

$$
  F_{\mu\nu}\longrightarrow F_{\mu\nu}
    -\frac13\operatorname{Tr}(F_{\mu\nu})\mathbf 1.
$$

对于 $z$-方向的空间分离，定义

$$
 M_{\mu\nu;\rho\sigma}^{\rm GF}(z,\tau)
 =\sum_{\boldsymbol x}\operatorname{Tr}\!\left[
 F_{\mu\nu}(\tau,x+z\hat z)W_\tau(x+z,x)
 F_{\rho\sigma}(\tau,x)W_\tau(x,x+z)\right].
$$

两端 field strength、直 Wilson line 和 flow time 必须相同。代码保存
(+z)、(-z) 两个方向以及

$$
 M^{\rm even}(z)=M(+z)+M(-z),\qquad
 M^{\rm odd}(z)=M(+z)-M(-z),
$$

且数组中没有除以 2。

注意 Brambilla--Wang 的连续算符通常写成 $g^2 F W F$。代码中的 Clover
归一化是格点约定；在使用其系数前，应先确定一个 tree-level normalization
$\kappa_F(a,\tau)$，使

$$
  (gF)_{\rm cont}=\kappa_F(a,\tau)\,F_{\rm code}.
$$

若代码输出未显式包含 $g^2$，则 bilocal 需要相应乘以
$\kappa_F^2$（或将它吸收到定义的 tree-level conversion 中）。
GF-to-$\overline{\mathrm{MS}}$ 系数不能在 field normalization 尚未对齐时直接套用。

---

## 2. Unpolarized 和 helicity 算符的 Euclidean 构造

定义两个基本 Euclidean primitive：

$$
 U^E_{\mu\nu}(z)=\sum_{\boldsymbol x}\operatorname{Tr}
 [F^E_{\mu\nu}(x+z)W^\dagger F^E_{\mu\nu}(x)W],
$$

$$
 H^E_{\mu\nu}(z)=\sum_{\boldsymbol x}\operatorname{Tr}
 [F^E_{\mu\nu}(x+z)W^\dagger \widetilde F^E_{\mu\nu}(x)W].
$$

在下式中 $i=x,y$，不包括 longitudinal $z$ 方向。

### 2.1 Unpolarized

Minkowski pseudo-distribution 的结构可写为

$$
  \mathcal O_U^{\rm M}
   =\sum_{i=x,y}M^{\rm M}_{0i;i0}+2M^{\rm M}_{xy;yx}.
$$

利用 $F_{i0}=-F_{0i}$、$F_{yx}=-F_{xy}$ 和 Wick dictionary：

$$
  M^{\rm M}_{0i;i0}=U^E_{ti},\qquad
  2M^{\rm M}_{xy;yx}=-2U^E_{xy}.
$$

所以代码约定下

$$
  \boxed{\mathcal O_U^{\rm GF}(z,\tau)
   =T_U^{\rm GF}(z,\tau)-S_U^{\rm GF}(z,\tau)},
$$

$$
  T_U=U^E_{tx}+U^E_{ty},\qquad S_U=2U^E_{xy}.
$$

物理 unpolarized projection 是

$$
  \mathcal O_{U,\rm even}^{\rm GF}(z,\tau)
  =\mathcal O_U^{\rm GF}(+z,\tau)+
    \mathcal O_U^{\rm GF}(-z,\tau).
$$

这就是当前 production selector

```text
channel = unpolarized
orientation = even_sum
component = combined = Mtiti - Mijij
projection = Re
```

### 2.2 Helicity

Minkowski helicity tensor bilocal取

$$
  \mathcal O_{\Delta g}^{\rm M}
  =\sum_{i=x,y}\widetilde M^{\rm M}_{0i;0i}
    +2\widetilde M^{\rm M}_{xy;xy},
$$

其中第二个 field strength 做 dual。由 electric/magnetic decomposition，

$$
 E_i=F^{\rm M}_{0i},\qquad F^{\rm M}_{ij}=-\epsilon_{ijk}B_k,
$$

目标结构为

$$
  E_xB_x+E_yB_y-2B_zE_z.
$$

逐项 Wick rotation 给出

$$
  \widetilde M^{\rm M}_{0i;0i}=+iH^E_{ti},
  \qquad 2\widetilde M^{\rm M}_{xy;xy}=-2iH^E_{xy},
$$

故去掉共同的 $i$ 后，代码基底中的理论目标是

$$
  \boxed{\mathcal O_H^{\rm GF}(z,\tau)
   =T_H^{\rm GF}(z,\tau)-S_H^{\rm GF}(z,\tau)},
$$

$$
  T_H=H^E_{tx}+H^E_{ty},\qquad S_H=2H^E_{xy}.
$$

helicity 的自旋相关部分取 odd separation：

$$
  \mathcal O_{H,\rm odd}^{\rm GF}(z,\tau)
  =\mathcal O_H^{\rm GF}(+z,\tau)-
    \mathcal O_H^{\rm GF}(-z,\tau).
$$

当前分析中保存的三个诊断为

$$
  (T_H-S_H)_{\rm odd},\qquad
  (T_H+S_H)_{\rm odd},\qquad
  (T_H)_{\rm odd}.
$$

只有第一个由 Wick rotation 和 $(E-B)$ 结构确定为 production target；
后两个用于比较旧 HYP note 的符号和统计信号。GF matching 不会把旧的
plus-sign 诊断变成物理 helicity 算符。

代码的 helicity numerator 使用完整复数 `pol35` 两点函数，先完成 vacuum
subtraction、$C_3/C_2$ 构造和 bootstrap/jackknife，再取

$$
  \operatorname{Im}R_H
  =\operatorname{Re}[-iR_H].
$$

不能在协方差构造之前先丢掉 `pol35` 的实部或虚部。

---

## 3. 为什么 gradient flow 需要 matching

正流时间使裸 flowed composite operator 在连续极限具有良好 UV 性质，
但它定义的是一个带尺度 $\sqrt{\tau}$ 的新 scheme，而不是
$\overline{\mathrm{MS}}$ light-cone operator。小流时间展开为

$$
  \mathcal O^{\rm GF}(\tau,z)
  =\sum_k C_k(\tau,\mu,z)\,\mathcal O_k^{\rm R}(z,\mu)
   +O(\tau\Lambda_{\rm QCD}^2,\tau/z^2).
$$

对非局部直线算符，Wilson-line 的线性自能表现为

$$
  \exp[\delta m_A(\tau)|z|],
$$

其中下标 $A$ 表示 adjoint Wilson line。若不做这一步 conversion，改变
flow time 会同时改变 UV scheme、line self-energy 和真实的短距离 OPE
系数，不能简单解释为“降噪”。

适用的尺度窗口至少要求

$$
  a^2/\tau\ll1,\qquad
  \sqrt{8\tau}\ll |z|,\qquad
  \tau P_z^2\ll1,\qquad
  \tau\Lambda_{\rm QCD}^2\ll1,
$$

以及 $P_z^2\gg M_N^2$。在本项目 $a\simeq0.0775\,\mathrm{fm}$ 且
$z/a=2$ 时，

$$
  \widehat\tau=0.5\Longrightarrow r_{\rm flow}/a=2,
$$

已经不满足简单的 $r_{\rm flow}\ll |z|$ 条件。因此现有
$\widehat\tau=0.1\sim3.8$ 是 finite-flow scan，不应全部当作
asymptotic-SFTE window。

---

## 4. 辅助场推导：把 Wilson line matching 化为 local-current matching

### 4.1 一维 adjoint auxiliary field

将直 Wilson line 写成一维 auxiliary field $h_v$ 的传播子。对空间类
方向 $v$，$v^2=-1$（Euclidean 表述中方向规范依赖可等价处理）。
积分掉 $h_v$ 后得到 adjoint Wilson line：

$$
  \langle h_v(x)\bar h_v(0)\rangle
  \propto W_v(x,0)\,\theta(v\!\cdot\!x).
$$

相应的局部重夸克到胶子 currents 为

$$
  J_{\parallel\perp}^{\mu\nu}(x)
   =gF_{\parallel\perp}^{\mu\nu}(x)h_v(x),
  \qquad
  J_{\perp\perp}^{\mu\nu}(x)
   =gF_{\perp\perp}^{\mu\nu}(x)h_v(x).
$$

这里以 $v$ 分解 field strength：

$$
  F_{\parallel\perp}^{\mu\nu}
  =g_{\parallel}^{\mu\alpha}g_{\perp}^{\nu\beta}F_{\alpha\beta},
  \qquad
  F_{\perp\perp}^{\mu\nu}
  =g_{\perp}^{\mu\alpha}g_{\perp}^{\nu\beta}F_{\alpha\beta}.
$$

对 `v∥z` 的空间直线，`F_{∥⊥}` 是含 `z` 的分量，`F_{⊥⊥}` 是两个指标
都垂直于 Wilson-line 方向的分量。本项目的普通 unpolarized pieces
`F_{tx}`、`F_{ty}`、`F_{xy}` 都属于 `F_{⊥⊥}`；而 helicity 中的 dual
tensor 会把它们映射到含 `z` 指标的 `F_{∥⊥}` 分量。这一点决定了两种
channel 的 matching 因子不同。

### 4.2 局部 current 的 SFTE

在小流时间，逐个 local current 展开为

$$
  J_i^{\rm GF,R}(\tau,x)
   =c_i(\tau,\mu)J_i^{\overline{\rm MS}}(x,\mu)+O(\tau),
  \qquad i=\parallel\perp,\perp\perp.
$$

一圈计算的逻辑是：

1. 在 GF scheme 中计算 flowed partonic vertex/self-energy；
2. 在 $\overline{\mathrm{MS}}$ scheme 中用同一 IR regulator 计算 unflowed current；
3. dimensional regularization 下 $\tau=0$ 的无尺度积分只含
   $1/\epsilon_{\rm UV}-1/\epsilon_{\rm IR}$；
4. GF 和 unflowed amplitudes 的 IR 部分相同，差值留下 UV matching
   coefficient；
5. auxiliary-field self-energy 给 Wilson-line 的 line mass，vertex 和
   field-strength current 给 (c_i)。

adjoint 表示下 $C_R=C_A$（本项目 SU(3) 中 $C_A=3$）。定义

$$
  L_\tau(\mu)\equiv
  \ln\!\left(2\mu^2\tau e^{\gamma_E}\right).
$$

Brambilla--Wang 的一圈结果为

$$
  \boxed{c_{\parallel\perp}(\tau,\mu)=1+O(\alpha_s^2)},
$$

$$
  \boxed{c_{\perp\perp}(\tau,\mu)
   =1+\frac{\alpha_s(\mu)C_A}{4\pi}L_\tau(\mu)
    +O(\alpha_s^2)},
$$

以及 adjoint Wilson-line 的 line-mass

$$
  \boxed{\delta m_A(\tau)
   =-\frac{\alpha_s(\mu)C_A}{4\pi}
     \frac{\sqrt{2\pi}}{\sqrt{\tau}}+O(\alpha_s^2)}.
$$

$(\delta m_A)$ 的整体符号随 Wilson-line 和 Euclidean 约定改变；本文采用
上式以及
$\mathcal O^{\rm GF,R}=e^{+\delta m_A|z|}\mathcal C\mathcal O^{\overline{\rm MS}}$
的约定。实际代码应在 `+z`、`-z` 上使用同一个 $|z|$ 因子。

### 4.3 从 local current 到 bilocal

由于 auxiliary-field formulation 下不同位置的 renormalized currents
之间不会再产生新的 UV divergence（除了已减掉的 line mass），两端的
matching coefficient 直接相乘：

$$
  \mathcal O_{ij}^{\rm GF,R}(\tau,z)
  =e^{\delta m_A(\tau)|z|}
   c_i(\tau,\mu)c_j(\tau,\mu)
   \mathcal O_{ij}^{\overline{\rm MS}}(\mu,z)+O(\tau).
$$

因此

$$
  \boxed{\mathcal O_{ij}^{\overline{\rm MS}}(z,\mu)
  =\frac{e^{-\delta m_A|z|}}{c_i c_j}
   \mathcal O_{ij}^{\rm GF,R}(\tau,z)+O(\tau)}.
$$

即使最后在本项目的 unpolarized channel 中两项使用相同的
`c_{⊥⊥}^2`，仍建议在 matching 前保留 `T_U` 和 `S_U` 两项：
这样可以逐项检查 tensor identity、tree-level normalization 和可能的
高阶/晶格对称性混合。

---

## 5. Unpolarized GF-to-$\overline{\mathrm{MS}}$ matching

对 $v\parallel z$，unpolarized 的 temporal 项和 spatial 项都由两个
`F_{⊥⊥}` 构成：

$$
  T_U^{\rm GF}=U_{tx}^E+U_{ty}^E,
  \qquad
  S_U^{\rm GF}=2U_{xy}^E.
$$

因此两部分在 Brambilla--Wang 的一圈 SFTE 中具有相同的 conversion：

$$
  T_U^{\overline{\rm MS}}
  =\frac{e^{-\delta m_A|z|}}{c_{\perp\perp}^2}
    T_U^{\rm GF,R},
$$

$$
  S_U^{\overline{\rm MS}}
  =\frac{e^{-\delta m_A|z|}}{c_{\perp\perp}^2}
    S_U^{\rm GF,R}.
$$

最终先用共同的 conversion，再作相对负号：

$$
  \boxed{
  \mathcal O_U^{\overline{\rm MS}}
  =\frac{e^{-\delta m_A|z|}}{c_{\perp\perp}^2}
   \left[T_U^{\rm GF,R}-S_U^{\rm GF,R}\right].}
$$

这里的 `parallel/perp` 是相对于 Wilson-line 方向 `z`，不是相对于
Euclidean 时间方向 `t`。因此对于当前 `z`-方向 OPE，可以把同一个 scalar
factor 乘在 `T_U^{GF}-S_U^{GF}` 外。在一圈精度，

$$
  \frac1{c_{\perp\perp}^2}
  =1-2\frac{\alpha_s C_A}{4\pi}L_\tau+O(\alpha_s^2).
$$

若选自然 scale

$$
  \mu_\tau=\frac{e^{-\gamma_E/2}}{\sqrt{2\tau}},
  \qquad L_\tau(\mu_\tau)=0,
$$

则显式 logarithm 在一圈消失，但 line-mass 因子仍然存在，并且需要把
$\mu_\tau$ 演化到共同分析 scale。自然 scale 不是免除 matching 的理由。

对 even separation，

$$
  \mathcal O_{U,\rm even}^{\overline{\rm MS}}(z)
  =\mathcal O_U^{\overline{\rm MS}}(+z)
   +\mathcal O_U^{\overline{\rm MS}}(-z).
$$

由于 line-mass 只依赖 $|z|$，两方向使用同一 conversion factor；但原始
(+z/-z) 数据仍应保留到最后，以检查 Hermiticity 和 parity。

---

## 6. Helicity GF-to-$\overline{\mathrm{MS}}$ matching

helicity primitive 是 $F W\widetilde F$。对于 $v\parallel z$：

- $H_{ti}=F_{ti}W\widetilde F_{ti}$：第一端是
  $F_{\perp\perp}$，dual 后的第二端是含 `z` 的
  $F_{\parallel\perp}$；
- $H_{xy}=F_{xy}W\widetilde F_{xy}$：第一端也是
  $F_{\perp\perp}$，dual 后的第二端同样是
  $F_{\parallel\perp}$。

Euclidean epsilon tensor 本身是一个固定的线性张量，不引入新的独立
one-loop renormalization coefficient。因此两项都得到同一个混合系数：

$$
  H_{ti}^{\overline{\rm MS}}
  =\frac{e^{-\delta m_A|z|}}
          {c_{\parallel\perp}c_{\perp\perp}}
    H_{ti}^{\rm GF,R},
$$

$$
  H_{xy}^{\overline{\rm MS}}
  =\frac{e^{-\delta m_A|z|}}
          {c_{\perp\perp}c_{\parallel\perp}}
    H_{xy}^{\rm GF,R}.
$$

所以 $T_H$、$S_H$ 以及任意线性组合在此一圈 SFTE 下可使用共同因子

$$
  \mathcal Z_H(\tau,z;\mu)
  \equiv\frac{e^{-\delta m_A(\tau)|z|}}
               {c_{\parallel\perp}(\tau,\mu)c_{\perp\perp}(\tau,\mu)}.
$$

物理 helicity target 为

$$
  \boxed{
  \mathcal O_{H,\rm odd}^{\overline{\rm MS}}(z)
  =\mathcal Z_H(\tau,z;\mu)
   \left[(T_H-S_H)^{\rm GF}(+z)
        -(T_H-S_H)^{\rm GF}(-z)\right].}
$$

同样，诊断组合可以写成

$$
  (T_H+S_H)_{\rm odd}^{\overline{\rm MS}}
  =\mathcal Z_H(T_H+S_H)_{\rm odd}^{\rm GF},
  \qquad
  (T_H)_{\rm odd}^{\overline{\rm MS}}
  =\mathcal Z_H(T_H)_{\rm odd}^{\rm GF},
$$

但这只是 conversion 关系，不能改变 production helicity 算符应为
$(T_H-S_H)_{\rm odd}$ 的结论。

代码中 helicity $C_3/C_2$ 的复数结构应先完成

$$
  C_3^H=\langle O_H C_2^{\rm pol35}\rangle
       -\langle O_H\rangle\langle C_2^{\rm pol35}\rangle,
  \qquad
  R_H=\frac{C_3^H}{\operatorname{Re}C_2^{\rm nopol}},
$$

再取物理 Euclidean phase projection

$$
  \operatorname{Im}R_H=\operatorname{Re}[-iR_H].
$$

SFTE scalar coefficient 应作用在完整复数 $R_H$ 或完整复数拟合矩阵元上；
只有当系数是本节的实数 scalar 时，才可在最后等价地乘到
$\operatorname{Im}R_H$ 上。不能先对 `pol35` 或 C3 做 imaginary projection
再进行 vacuum subtraction 和 bootstrap。

---

## 7. 从重整化 bilocal 到 PDF：LaMET/pseudo-ITD 层

GF-to-$\overline{\mathrm{MS}}$ conversion 之后仍不是 light-cone PDF。定义重整化的
coordinate-space matrix elements

$$
  h_U^{\overline{\rm MS}}(z,P_z,\mu),
  \qquad
  h_H^{\overline{\rm MS}}(z,P_z,\mu),
  \qquad \nu=P_z z.
$$

### 7.1 Light-cone 定义与 invariant amplitudes

取 light-cone (n^2=0)、(P^+=P\cdot n)，adjoint Wilson line

$$
  W^{ab}(\xi^-,0)=\mathcal P\exp\!\left[
    ig\int_0^{\xi^-}\!d\eta^-\,A^{+c}(\eta^-)\,(T^c_{\rm adj})^{ab}
  \right].
$$

作为一个明确的归一化约定，可以定义

$$
  xg(x,\mu)=\frac{1}{2\pi P^+}\int d\xi^-\,e^{-ixP^+\xi^-}
  \langle P|F^{+i,a}(\xi^-)W^{ab}F^{+i,b}(0)|P\rangle_\mu,
$$

$$
  x\Delta g(x,\mu)=\frac{i}{2\pi P^+}\int d\xi^-\,e^{-ixP^+\xi^-}
  \langle P,S_L|F^{+i,a}(\xi^-)W^{ab}\widetilde F^{+i,b}(0)|P,S_L\rangle_\mu,
$$

其中 $i=1,2$ 是 transverse index，且
$\widetilde F^{\mu\nu}=\epsilon^{\mu\nu\rho\sigma}F_{\rho\sigma}/2$。
文献可能把 $x$、$2P^+$ 或 helicity 的整体 $i$ 因子放到 PDF 或
matrix element 的定义中；这只改变整体归一化/符号，不改变本文的
relative (T-S) tensor structure。实际代码、SFTE 系数和最终 PDF 必须
使用同一套约定。

将

$$
  M^{\mu\alpha;\nu\beta}(z,P)
  =\langle P|F^{\mu\alpha}(z)W(z,0)F^{\nu\beta}(0)|P\rangle
$$

分解成 Lorentz invariant amplitudes 后，unpolarized 取 spin-independent
amplitude，helicity 取含 (S_L) 的 antisymmetric amplitude。空间类 lattice
矩阵元在大 $P_z$ 和小 $z^2$ 时由同一组 twist-2 light-cone amplitudes
控制；有限 $P_z$、$z^2$ 和 flow radius 产生 target-mass、higher-twist
和 flow-power corrections。

### 7.2 quasi-PDF 写法

Fourier transform 采用

$$
  \widetilde g^{\rm GF}(x,P_z,\mu,\tau)
  =\int\frac{dz}{4\pi}
    e^{-ixP_z z}h_U^{\rm GF}(z,P_z,\tau),
$$

$$
  \Delta\widetilde g^{\rm GF}(x,P_z,\mu,\tau)
  =\int\frac{dz}{4\pi}
    e^{-ixP_z z}h_H^{\rm GF}(z,P_z,\tau).
$$

在 singlet sector，GF-scheme quasi distributions 的 factorization 是一个
矩阵卷积：

$$
  \begin{pmatrix}
  \widetilde g^{\rm GF}\\[2pt]
  \widetilde\Sigma^{\rm GF}
  \end{pmatrix}(x,P_z,\tau)
  =\int_{-1}^{1}\frac{dy}{|y|}
  \begin{pmatrix}
   C_{gg}^{\rm GF} & C_{gq}^{\rm GF}\\
   C_{qg}^{\rm GF} & C_{qq}^{\rm GF}
  \end{pmatrix}
  \!\left(\frac{x}{y},\frac{\mu}{|y|P_z},
        \sqrt{\tau}|y|P_z,\sqrt{\tau}\mu\right)
  \begin{pmatrix}g\\\Sigma\end{pmatrix}(y,\mu)
  +\delta_{\rm power}.
$$

helicity 同样是

$$
  \begin{pmatrix}
  \Delta\widetilde g^{\rm GF}\\[2pt]
  \Delta\widetilde\Sigma^{\rm GF}
  \end{pmatrix}
  =\int_{-1}^{1}\frac{dy}{|y|}
  \begin{pmatrix}
   C_{\Delta g\Delta g}^{\rm GF} & C_{\Delta g\Delta q}^{\rm GF}\\
   C_{\Delta q\Delta g}^{\rm GF} & C_{\Delta q\Delta q}^{\rm GF}
  \end{pmatrix}
  \!\left(\frac{x}{y},\ldots\right)
  \begin{pmatrix}\Delta g\\\Delta\Sigma\end{pmatrix}(y,\mu)
  +\delta_{\rm power}.
$$

这里 $C^{\rm GF}$ 可以在一致 scheme 下拆成

$$
  C^{\rm GF\to PDF}
  =C^{\rm quasi/pseudo\to PDF}
   \otimes C^{\rm GF\to\overline{\rm MS}\,quasi}.
$$

如果直接使用一个已经包含 flow conversion 的 combined kernel，就不能再
额外乘一次 SFTE coefficient。

### 7.3 pseudo-ITD 写法

也可以先形成 reduced pseudo-ITD。例如 unpolarized：

$$
  \mathfrak M_U(\nu,z^2,\mu)
  =\frac{\mathcal M_U(\nu,z^2,\mu)}
         {\mathcal M_U(0,z^2,\mu)},
$$

helicity 则使用 odd-in-(z) 的 spin-dependent invariant amplitude，并作
相应的 zero-Ioffe-time normalization。短距离 OPE 具有形式

$$
  \mathfrak M_U(\nu,z^2,\mu)
  =\int_0^1du\,C_U(u,z^2\mu^2,\alpha_s)
    \mathfrak I_g(u\nu,\mu)+O(z^2\Lambda^2),
$$

$$
  \mathfrak M_H(\nu,z^2,\mu)
  =\int_0^1du\,C_H(u,z^2\mu^2,\alpha_s)
    \Delta\mathfrak I_g(u\nu,\mu)+O(z^2\Lambda^2).
$$

$C_U$ 和 $C_H$ 是不同的 channel-specific kernels。unpolarized 的一圈
gluon pseudo-distribution kernels 见 arXiv:2107.08960、arXiv:2112.02011
及 arXiv:2510.26425 的对应 scheme；helicity 的 invariant-amplitude 和
polarized pseudo-ITD matching 见 arXiv:2207.08733、arXiv:2112.02011。
这些文献中的 HYP、ratio 或 hybrid renormalization 不能不加修改地用于
本项目 thin-link GF scheme。

---

## 8. 当前代码对应的可执行 matching 流程

对每个 $(\tau,z,P_z)$ 建议按以下顺序实施：

1. **流场和 OPE**：从 thin link 做四维 Wilson flow；保存 Clover
   $F_{\mu\nu}$、$T_U,S_U,T_H,S_H$ 以及 (+z/-z) raw components。
2. **C3/C2 和 state isolation**：完整使用 `nopol` denominator；helicity
   numerator 使用完整复数 `pol35`；在每个 bootstrap replica 内先作
   vacuum subtraction、
   $\operatorname{Re}C_3/\operatorname{Re}C_2$ 或
   $\operatorname{Im}C_3/\operatorname{Re}C_2$，最后重采样和拟合。
3. **算符选择**：
   $$
   U=(T_U-S_U)_{\rm even},\qquad
   H=(T_H-S_H)_{\rm odd}.
   $$
4. **代码归一化**：确认 `F_clov` 与连续 $gF$ 的 tree-level factor
   $\kappa_F(a,\tau)$，并检查 color-traceless projection、Wilson-line
   representation 和 $g^2$ 因子。
5. **SFTE window mask**：只在
   $a^2/\tau\ll1$、$r_{\rm flow}\ll|z|$、
   $\tau P_z^2\ll1$、$\tau\Lambda^2\ll1$ 的公共窗口使用一圈系数；
   窗口选择应是全局 systematic choice，不能逐点挑误差最小的点。
6. **GF-to-$\overline{\mathrm{MS}}$ conversion**：对 unpolarized 的 $T_U$、$S_U$
   分别乘共同的 $e^{-\delta m|z|}/c_{\perp\perp}^2$，再作减法；helicity 的两个 raw
   pieces 乘共同的 $e^{-\delta m|z|}/(c_\parallel c_\perp)$，再作
   $T_H-S_H$ 和 odd difference。
7. **scale evolution**：可在
   $\mu_\tau=e^{-\gamma_E/2}/\sqrt{2\tau}$ 附近评价系数，再演化到
   统一的 $\mu$。
8. **quasi/pseudo matching**：使用与 operator、ratio/hybrid scheme、
   $\Gamma$/tensor basis 和 perturbative order 一致的 unpolarized 或
   helicity kernel；singlet 情形保留 quark--gluon mixing matrix。
9. **系统学**：改变 $P_z$、$z_{\max}$、flow-time window、
   $\mu$、matching order、fit ansatz，并检查 target-mass、higher-twist、
   finite-volume、discretization 和 continuum limit。

### 8.1 本次生成的 matching 对比图

本目录 `matching/plot_matched_bare_matrix.py` 从每个流时间的
`bare_sumdiff_aic_*_N231.npz` 读取同一组 $1500$ 个 configuration-bootstrap
样本。由于在当前 $z$-方向 OPE 中一圈 SFTE 因子是依赖 $(\tau,z)$ 的实数
scalar，它可以逐 replica 乘到拟合矩阵元上；生产分析仍建议在逐组态 OPE、
vacuum subtraction 和 $C_3/C_2$ 之前应用 `gluon_flow_to_quasi.py`。

对比图使用 $a=0.0775\,\mathrm{fm}$、$\mu=2\,\mathrm{GeV}$、
$\alpha_s=0.25$、$\kappa_F=1$，固定流时间图取 $\tau/a^2=0.5$，横坐标为
格点 Ioffe time

$$
  \nu=zP_z=2\pi(z/a)n_z/L_z,
  \qquad L_z=32.
$$

图中同时显示 unpolarized $(T_U-S_U)_{\rm even}$、helicity $T_H{}_{\rm odd}$、
存储的 $(T_H+S_H)_{\rm odd}$，以及由共享 bootstrap 样本构造的
$(T_H-S_H)_{\rm odd}$ 约定检查项。空心圆是 GF bare，实心方块是乘上
一圈 GF-to-$\overline{\mathrm{MS}}$ 因子后的结果。另给出 $z/a=2$ 的流时间
扫描和 $\tau/a^2\geq1$ 的线性 $M(\tau)=M_0+c\tau$ AIC 加权外推。

这些图是 matching 的数值诊断，不是最终 PDF：当前尚未确定 Clover
tree-level $\kappa_F$，也没有加入 singlet gluon--quark mixing、LaMET/
pseudo-ITD kernel、target-mass/higher-twist 修正或连续极限。

当前输出的 `ratio_v3` 和 `bare_matrix_latest` 仍属于
**finite-flow bare disconnected observables**。它们可以作为上述第 1--3
步的输入，但尚未完成第 4--9 步，不能直接解释为 $g(x)$ 或
$\Delta g(x)$。

---

## 9. 参考文献

- M. Lüscher, *Properties and Uses of the Wilson Flow in Lattice QCD*,
  arXiv:1006.4518；M. Lüscher, arXiv:1302.5246。
- C. Monahan and K. Orginos, *Quasi Parton Distributions and the Gradient
  Flow*, arXiv:1612.01584。
- N. Brambilla and X.-P. Wang, *Off-Lightcone Wilson-Line Operators in
  Gradient Flow*, arXiv:2312.05032, JHEP 06 (2024) 210。
- T. Khan et al., *Unpolarized Gluon Distribution in the Nucleon from
  Lattice QCD*, arXiv:2107.08960, Phys. Rev. D 104, 094516 (2021)。
- C. Egerer et al., *Towards the Determination of the Gluon Helicity
  Distribution in the Nucleon from Lattice QCD*, arXiv:2207.08733v2,
  Phys. Rev. D 106, 094511 (2022)。
- I. Balitsky, W. Morris and A. Radyushkin, *Polarized Gluon
  Pseudodistributions at Short Distances*, arXiv:2112.02011,
  JHEP 02 (2022) 193。
- C. Chen et al., *Unpolarized Gluon PDF of the Nucleon from Lattice QCD at
  Physical Point in the Continuum Limit*, arXiv:2510.26425v2。
