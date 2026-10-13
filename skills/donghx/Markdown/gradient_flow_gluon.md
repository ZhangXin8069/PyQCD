# Gradient Flow for Gluon Unpolarized and Helicity Operators

## 0. Scope and notation

本文档整理目前对话中关于 gluon unpolarized/helicity 算符的理论推导，主要依据：

- [Off-lightcone Wilson-line operators in gradient flow](https://arxiv.org/html/2312.05032v1)
- `Gradient_flow.pdf` 中的 Sections 4、8、11-15

约定如下：

- $t_f$ 表示 gradient-flow time；它不是 Euclidean 时间坐标 $t$。
- flow radius 为
  \[
  r_F=\sqrt{8t_f}.
  \]
- Wilson line 沿空间 $z$ 方向。
- Euclidean 坐标顺序为 $(x,y,z,t)$，并取
  \[
  \epsilon_{xyzt}^{E}=+1.
  \]
- 这里的 GF operator 表示有限流时间的 flowed operator；它不是物理的 light-cone PDF。

理论上需要区分三个层次：

\[
\text{finite-flow GF operator}
\longrightarrow
\overline{\rm MS}\text{ quasi/pseudo operator}
\longrightarrow
\text{light-cone PDF}.
\]

Gradient-flow matching 只负责第一步。LaMET 或 pseudo-ITD matching 是后续独立步骤。

---

## 1. Wilson-line gluon operator and auxiliary-field formalism

一般的 gluon Wilson-line operator 写成

\[
\mathcal O^{\mu\nu\alpha\beta}(zv)
=g^2 F^{\mu\nu}(zv)W(zv,0)F^{\alpha\beta}(0),
\]

其中 $v$ 是 Wilson-line 方向，$W(zv,0)$ 是 straight Wilson line。

文章引入一维 auxiliary adjoint field $h_v$：

\[
\mathcal L_{h_v}=\bar h_v(iv\cdot D)h_v.
\]

积分掉 $h_v$ 后，其 propagator 等价于 Wilson line：

\[
\langle h_v(x)\bar h_v(0)\rangle_{h_v}
\propto W(x,0).
\]

因此，非局域 Wilson-line 算符的重整化可以转化为局域 current 的重整化。

这一点的作用是：先计算局域 heavy-to-gluon current 的 flow matching，再得到两个不同空间点之间 Wilson-line operator 的 matching。

---

## 2. 为什么要分解为 \(\parallel\perp\) 和 \(\perp\perp\)

Wilson line 引入了一个特殊方向 $v^\mu$，所以 gluon field strength 不再只有一个统一的 renormalization factor。

定义

\[
g_{\parallel}^{\mu\nu}=\frac{v^\mu v^\nu}{v^2},
\qquad
g_{\perp}^{\mu\nu}=g^{\mu\nu}-g_{\parallel}^{\mu\nu}.
\]

场强分解为

\[
F_{\parallel\perp}^{\mu\nu}
=g_{\parallel}^{\mu\alpha}g_{\perp}^{\nu\beta}F_{\alpha\beta},
\]

\[
F_{\perp\perp}^{\mu\nu}
=g_{\perp}^{\mu\alpha}g_{\perp}^{\nu\beta}F_{\alpha\beta}.
\]

对应的局域 currents 为

\[
J_{\parallel\perp}^{\mu\nu}=gF_{\parallel\perp}^{\mu\nu}h_v,
\qquad
J_{\perp\perp}^{\mu\nu}=gF_{\perp\perp}^{\mu\nu}h_v.
\]

当 $v=(1,0,0,0)$ 时，$F_{\parallel\perp}$ 与 chromoelectric field 对应，而 $F_{\perp\perp}$ 与 chromomagnetic field 对应。

这两个 current 有不同的 UV vertex correction，因此一般有不同的 matching coefficient：

\[
J_i^{\rm GF}(t_f)
=c_i(t_f,\mu)J_i^{\overline{\rm MS}}(\mu)+O(t_f),
\qquad i=\parallel\perp,\perp\perp.
\]

这里的“两个方向”不是两次不同的 gradient flow。实际只需对同一个 flowed gauge field 构造两个独立的 tensor contraction，并分别匹配。

如果在形成 unpolarized combination 之前不保留两个部分，就无法使用不同的 $c_i$，也无法正确处理 flow-scheme conversion。

---

## 3. 为什么需要小流时间展开

正流时间 $t_f>0$ 给场引入了有限的 UV resolution scale $r_F=\sqrt{8t_f}$。因此 flowed operator 是一个短距离 scheme，而不是普通的 $\overline{\rm MS}$ operator。

在 $t_f$ 足够小、但仍满足 $a^2/t_f\ll1$ 的区域，可以进行 short-distance expansion：

\[
J_i^{\rm GF}(t_f)
=c_i(t_f,\mu)J_i^{\overline{\rm MS}}(\mu)+O(t_f).
\]

对于带 Wilson line 的 bilocal operator，small-flow-time expansion 为

\[
\mathcal O_i^{\rm GF}(t_f,z)
=C_i(t_f,\mu)e^{\delta m(t_f)|z|}
\mathcal O_i^{\overline{\rm MS}}(z,\mu)
+O(t_f).
\]

这里：

- $C_i$ 是 flow-scheme 到 $\overline{\rm MS}$ 的短距离 matching coefficient；
- $e^{\delta m|z|}$ 表示 Wilson-line self-energy 的 line-mass contribution；
- $O(t_f)$ 表示高维算符和有限流时间修正，实际分析中还包括 $t_f/z^2$ 与 $t_f\Lambda_{\rm QCD}^2$ 等修正。

小流时间展开的主要作用有三点：

1. 把有限流时间 scheme 转换到可用于 LaMET/pseudo-PDF 的 $\overline{\rm MS}$ scheme；
2. 将非局域 Wilson-line matching 化成局域 current matching；
3. 在相同 IR regulator 下比较 GF 和未流 perturbative matrix elements，使 IR poles 相消，只留下短距离 matching coefficient。

它不意味着在单个 lattice spacing 上直接令 $t_f=0$。正确顺序是先满足

\[
a\ll r_F\ll |z|,
\qquad
r_FP_z\ll1,
\qquad
t_f\Lambda_{\rm QCD}^2\ll1,
\]

再做连续极限和 flow-scheme conversion。

---

## 4. Unpolarized gluon operator

### 4.1 Minkowski definition

Note 采用的 Minkowski unpolarized operator 为

\[
\mathcal M_U^M
=\sum_{i=x,y}M_{0i;i0}^M+2M_{xy;yx}^M,
\]

其中

\[
M_{\mu\nu;\rho\sigma}^M
=\operatorname{Tr}
\left[
F_{\mu\nu}^M(z)W^\dagger F_{\rho\sigma}^M(0)W
\right].
\]

注意第二个 field strength 的指标是反向的：

\[
0i;i0,
\qquad
xy;yx.
\]

由场强反对称性

\[
F_{i0}=-F_{0i},
\qquad
F_{yx}=-F_{xy},
\]

Minkowski operator 中两个 transverse structures 都带有相应的反向指标符号。

### 4.2 Wick rotation

采用

\[
F_{0i}^M=+iF_{ti}^E,
\qquad
F_{ij}^M=F_{ij}^E.
\]

对于 temporal term：

\[
\begin{aligned}
M_{0i;i0}^M
&=\operatorname{Tr}
\left[(+iF_{ti}^E)W^\dagger(-iF_{ti}^E)W\right]\\
&=U_{ti}^E.
\end{aligned}
\]

对于 spatial term：

\[
\begin{aligned}
M_{xy;yx}^M
&=\operatorname{Tr}
\left[F_{xy}^EW^\dagger(-F_{xy}^E)W\right]\\
&=-U_{xy}^E.
\end{aligned}
\]

因此

\[
\boxed{
\mathcal O_U^E(z)
=U_{tx}^E(z)+U_{ty}^E(z)-2U_{xy}^E(z)
}
\]

定义

\[
T_U=U_{tx}+U_{ty},
\qquad
S_U=2U_{xy},
\]

则

\[
\boxed{
\mathcal O_U^E=T_U-S_U.
}
\]

unpolarized 使用 $z\to-z$ 的偶部分：

\[
\mathcal O_U^{E,\mathrm{even}}(z)
=\mathcal O_U^E(+z)+\mathcal O_U^E(-z).
\]

---

## 5. Helicity gluon operator

### 5.1 Minkowski definition

定义 dual tensor

\[
\widetilde F^{M,\mu\nu}
=\frac12\epsilon_M^{\mu\nu\rho\sigma}F^M_{\rho\sigma}.
\]

Minkowski helicity operator 为

\[
\mathcal M_{\Delta g}^M
=\sum_{i=x,y}\widetilde M_{0i;0i}^M
+2\widetilde M_{xy;xy}^M,
\]

其中第二个 field strength 被 dualized。

用

\[
F_{0i}^M=E_i,
\qquad
F_{ij}^M=-\epsilon_{ijk}B_k,
\]

以及

\[
\widetilde F_{0i}^M=B_i,
\qquad
\widetilde F_{ij}^M=\epsilon_{ijk}E_k,
\]

可得

\[
\mathcal M_{\Delta g}^M
\sim E_xB_x+E_yB_y-2B_zE_z.
\]

纵向项的负号来自

\[
F_{xy}^M=-B_z,
\qquad
\widetilde F_{xy}^M=E_z.
\]

### 5.2 Euclidean continuation

除了

\[
F_{0i}^M=+iF_{ti}^E,
\]

还需要使用 dual tensor 的 Wick dictionary：

\[
\widetilde F_{0i}^M=\widetilde F_{ti}^E,
\qquad
\widetilde F_{ij}^M=-i\widetilde F_{ij}^E.
\]

因此 temporal part 为

\[
\widetilde M_{0i;0i}^M=+iH_{ti}^E,
\]

而 spatial part 为

\[
2\widetilde M_{xy;xy}^M=-2iH_{xy}^E.
\]

于是

\[
\mathcal M_{\Delta g}^M
=i\left(H_{tx}^E+H_{ty}^E-2H_{xy}^E\right).
\]

去掉整个 Euclidean helicity operator 的共同 phase $i$，得到

\[
\boxed{
\mathcal O_{\Delta g}^E
=H_{tx}^E+H_{ty}^E-2H_{xy}^E
}
\]

定义

\[
T_H=H_{tx}+H_{ty},
\qquad
S_H=2H_{xy},
\]

则

\[
\boxed{
\mathcal O_{\Delta g}^E=T_H-S_H.
}
\]

helicity 的自旋相关部分在 $z\to-z$ 下取奇部分：

\[
\boxed{
\mathcal O_{\Delta g}^{E,\mathrm{odd}}(z)
=\mathcal O_{\Delta g}^E(+z)-\mathcal O_{\Delta g}^E(-z).
}
\]

因此旧的 $T_H+S_H$ 改变了 longitudinal $B_zE_z$ 项的相对符号，不是整体 convention 的改变。

---

## 6. Gradient-flow matching coefficients

令

\[
L_t=\ln\left(2\mu^2t_f e^{\gamma_E}\right).
\]

文章给出的局域 current coefficients 为

\[
c_{\parallel\perp}(t_f,\mu)
=1+O(\alpha_s^2),
\]

\[
c_{\perp\perp}(t_f,\mu)
=1+\frac{\alpha_s C_A}{4\pi}L_t
+O(\alpha_s^2).
\]

对同类型 field-strength bilocal，两个 endpoint 的系数相乘：

\[
C_{\parallel\perp}=c_{\parallel\perp}^2,
\qquad
C_{\perp\perp}=c_{\perp\perp}^2.
\]

因此一圈时

\[
C_{\parallel\perp}=1+O(\alpha_s^2),
\]

\[
C_{\perp\perp}
=1+\frac{\alpha_s C_A}{2\pi}L_t
+O(\alpha_s^2).
\]

adjoint Wilson line 的 line-mass 为

\[
\delta m(t_f)
=-\frac{\alpha_s C_A}{4\pi}
\frac{\sqrt{2\pi}}{\sqrt{t_f}}
+O(\alpha_s^2).
\]

符号约定取为

\[
\mathcal O_i^{\rm GF}(t_f,z)
=C_i(t_f,\mu)e^{\delta m(t_f)|z|}
\mathcal O_i^{\overline{\rm MS}}(z,\mu)+O(t_f).
\]

反向转换为

\[
\mathcal O_i^{\overline{\rm MS}}(z,\mu)
=\frac{e^{-\delta m(t_f)|z|}}{C_i(t_f,\mu)}
\mathcal O_i^{\rm GF}(t_f,z)+O(t_f).
\]

---

## 7. Unpolarized gluon matching

当前 Wilson line 沿 $z$ 方向，因此 parallel/transverse 必须相对于
$v\parallel z$ 判断。$F_{tx}$、$F_{ty}$ 和 $F_{xy}$ 都不含 $z$ 指标，
所以 $T_U$ 与 $S_U$ 的两个 endpoint 都属于 $F_{\perp\perp}$，两项使用
同一个 bilocal matching coefficient：

\[
T_U^{\rm GF}(t_f,z)
=C_{\perp\perp}(t_f,\mu)e^{\delta m|z|}
T_U^{\overline{\rm MS}}(z,\mu)+O(t_f),
\]

\[
S_U^{\rm GF}(t_f,z)
=C_{\perp\perp}(t_f,\mu)e^{\delta m|z|}
S_U^{\overline{\rm MS}}(z,\mu)+O(t_f).
\]

因此

\[
\boxed{
\mathcal O_U^{\overline{\rm MS}}
=e^{-\delta m|z|}
\left[
\frac{T_U^{\rm GF}}{C_{\perp\perp}}
-
\frac{S_U^{\rm GF}}{C_{\perp\perp}}
\right]+O(t_f)
=\frac{e^{-\delta m|z|}}{C_{\perp\perp}}
\left(T_U^{\rm GF}-S_U^{\rm GF}\right)+O(t_f).
}
\]

一圈显式写为

\[
\mathcal O_U^{\overline{\rm MS}}
=e^{-\delta m|z|}
\left(1-\frac{\alpha_s C_A}{2\pi}L_t\right)
\left(T_U^{\rm GF}-S_U^{\rm GF}\right)
+O(\alpha_s^2,t_f).
\]

这里的 tensor-basis assignment 必须与 Wilson-line 方向、Minkowski-to-Euclidean dictionary 和 Note 的 convention 一致；不能仅凭代码中的字母 $(t,x,y,z)$ 自动判断 parallel/transverse。

如果使用 nonperturbative hybrid/ratio renormalization 已经减去了 Wilson-line mass，则不能再次乘上 $e^{-\delta m|z|}$。

---

## 8. Helicity gluon matching 的推导

helicity operator 在每一项中包含 $F$ 和 dual field，因此需要 endpoint-by-endpoint matching。

### 8.1 Dual field 的 endpoint coefficients

因为 Hodge dual 交换两个 tensor sectors：

\[
\widetilde F_{\parallel\perp}\sim F_{\perp\perp},
\qquad
\widetilde F_{\perp\perp}\sim F_{\parallel\perp},
\]

所以

\[
F_{\parallel\perp}^{\rm GF}
=c_{\parallel\perp}F_{\parallel\perp}^{\overline{\rm MS}},
\]

\[
\widetilde F_{\parallel\perp}^{\rm GF}
=c_{\perp\perp}\widetilde F_{\parallel\perp}^{\overline{\rm MS}},
\]

以及

\[
F_{\perp\perp}^{\rm GF}
=c_{\perp\perp}F_{\perp\perp}^{\overline{\rm MS}},
\]

\[
\widetilde F_{\perp\perp}^{\rm GF}
=c_{\parallel\perp}\widetilde F_{\perp\perp}^{\overline{\rm MS}}.
\]

### 8.2 两个 helicity pieces

对于 temporal-like piece：

\[
F_{\parallel\perp}(z)W\widetilde F_{\parallel\perp}(0),
\]

匹配系数为

\[
c_{\parallel\perp}c_{\perp\perp}.
\]

对于 spatial-like piece：

\[
F_{\perp\perp}(z)W\widetilde F_{\perp\perp}(0),
\]

匹配系数为

\[
c_{\perp\perp}c_{\parallel\perp}.
\]

因此两项相同，定义

\[
\boxed{
C_{\Delta g}(t_f,\mu)
=c_{\parallel\perp}(t_f,\mu)c_{\perp\perp}(t_f,\mu)
}
\]

也就是

\[
C_{\Delta g}
=\sqrt{C_{\parallel\perp}C_{\perp\perp}}.
\]

一圈时

\[
\boxed{
C_{\Delta g}(t_f,\mu)
=1+\frac{\alpha_s C_A}{4\pi}L_t
+O(\alpha_s^2).
}
\]

### 8.3 Euclidean helicity matching

由于 Minkowski helicity operator 与 Euclidean operator 的关系为

\[
\mathcal O_{\Delta g}^{M}
=i\left(T_H^E-S_H^E\right),
\]

共同的 $i$ phase 在两边相同，不影响 flow matching。因此

\[
T_H^{\rm GF}(t_f,z)
=C_{\Delta g}(t_f,\mu)e^{\delta m|z|}
T_H^{\overline{\rm MS}}(z,\mu)+O(t_f),
\]

\[
S_H^{\rm GF}(t_f,z)
=C_{\Delta g}(t_f,\mu)e^{\delta m|z|}
S_H^{\overline{\rm MS}}(z,\mu)+O(t_f).
\]

所以

\[
\boxed{
\mathcal O_{\Delta g}^{E,\overline{\rm MS}}
=
\frac{e^{-\delta m|z|}}{C_{\Delta g}}
\left(T_H^{\rm GF}-S_H^{\rm GF}\right)
+O(t_f).
}
\]

取 helicity 的 odd-$z$ projection：

\[
\boxed{
\mathcal O_{\Delta g}^{E,\overline{\rm MS},\mathrm{odd}}(z)
=
\frac{e^{-\delta m|z|}}{C_{\Delta g}}
\left\{
[T_H-S_H](+z)-[T_H-S_H](-z)
\right\}
+O(t_f).
}
\]

一圈展开为

\[
\mathcal O_{\Delta g}^{E,\overline{\rm MS},\mathrm{odd}}
=e^{-\delta m|z|}
\left[1-\frac{\alpha_s C_A}{4\pi}L_t\right]
\mathcal O_{\Delta g}^{E,\rm GF,odd}
+O(\alpha_s^2,t_f).
\]

---

## 9. 对 helicity matching 结果的严格限定

上面的

\[
C_{\Delta g}=c_{\parallel\perp}c_{\perp\perp}
\]

是从文章给出的 endpoint current matching 和四维 dual tensor 的 sector exchange 推导出的自然结果。

文章直接计算并明确给出的，是

\[
c_{\parallel\perp},
\qquad
c_{\perp\perp},
\qquad
C_{\parallel\perp}=c_{\parallel\perp}^2,
\qquad
C_{\perp\perp}=c_{\perp\perp}^2.
\]

它没有单独发表你当前 Euclidean helicity bilocal 的完整 partonic matching calculation。因此，在正式物理结果中仍应验证：

1. dual tensor 与 flow matching 是否严格可交换；
2. Euclidean phase 和 $\epsilon_{xyzt}$ convention 是否与 perturbative calculation 一致；
3. 是否存在额外 tensor mixing；
4. 后续 helicity LaMET/pseudo-ITD matching 中的 singlet-quark mixing。

所以，现阶段可以把

\[
C_{\Delta g}=c_{\parallel\perp}c_{\perp\perp}
\]

作为 helicity GF-to-$\overline{\rm MS}$ 的 endpoint-factorized one-loop matching，但应明确标注其推导性质，并通过独立的一圈 helicity partonic calculation 验证。

---

## 10. 当前数据分析中的推荐顺序

对每个 $(t_f,z,P_z)$：

1. 从 raw OPE 构造
   \[
   T_U=M_{tx;tx}+M_{ty;ty},
   \qquad S_U=2M_{xy;xy};
   \]
2. 构造 unpolarized
   \[
   O_U=T_U-S_U,
   \]
   并取 even-$z$；
3. 从 dual raw OPE 构造
   \[
   T_H=H_{tx}+H_{ty},
   \qquad S_H=2H_{xy};
   \]
4. 构造 helicity
   \[
   O_{\Delta g}=T_H-S_H,
   \]
   并取 odd-$z$；
5. 先做 disconnected vacuum subtraction、完整复数 ratio 和 bootstrap 重采样；
6. 做 ground-state extraction；
7. 对当前 $v\parallel z$ 的 unpolarized 两个部分共同应用
   $C_{\perp\perp}=c_{\perp\perp}^2$，并应用 helicity 的候选
   $C_{\Delta g}=c_{\parallel\perp}c_{\perp\perp}$；
8. 最后再进行 LaMET 或 pseudo-ITD matching，以及 gluon-singlet quark mixing。

在这些步骤完成前，结果应称为 finite-flow bare gluon matrix element 或 finite-flow disconnected gluon ratio，而不是 gluon PDF。

---

## 11. 当前项目状态与本轮对话汇总（2026-09-08）

本节记录当前代码实际生成的 finite-flow gluon OPE 数据、统计拟合和图表。
它是对前面理论约定的实现状态补充，不把裸矩阵元写成重整化或物理 PDF。

### 11.1 已确认的理论与算符结论

在当前 Euclidean convention 下：


\[
T_U=U_{tx}+U_{ty},\qquad S_U=2U_{xy},\qquad
\mathcal O_U^E=T_U-S_U,
\]

\[
T_H=H_{tx}+H_{ty},\qquad S_H=2H_{xy},\qquad
\mathcal O_{\Delta g}^E=T_H-S_H.
\]

unpolarized 使用 even-$z$ convention，helicity 使用 odd-$z$ convention。因而当前数据层同时保留三种统计算符变体：

| channel | no S | +S | -S（当前 convention 的物理候选） |
|---|---|---|---|
| unpolarized | $T_U$ | $T_U+S_U$ | $T_U-S_U$ |
| helicity | $T_H$ | $T_H+S_H$ | $T_H-S_H$ |

matching 层仍采用前文的 sector assignment：

\[
C_{\parallel\perp}=c_{\parallel\perp}^2,\qquad
C_{\perp\perp}=c_{\perp\perp}^2,
\]

\[
C_{\Delta g}=c_{\parallel\perp}c_{\perp\perp}
 =\sqrt{C_{\parallel\perp}C_{\perp\perp}},
\]

并保留 adjoint Wilson-line line-mass 因子 $\delta m(t_f)$。$C_{\Delta g}$ 目前是由 endpoint matching 和 dual-sector exchange 推导的候选结果，尚未由独立的完整一圈 helicity partonic calculation 直接确认；正式物理结果前仍需检查 dual tensor、Euclidean phase、tensor mixing 及 singlet-quark mixing。

### 11.2 C3/C2 all-operator 产品

已从 raw flowed OPE 和同一份 C2 配置构造全部六个算符的 disconnected C3/C2：

- flow time $t_f/a^2=0.1,0.2,0.3,0.4,0.5,0.6,1.0,1.4,1.8,2.2,2.6,3.0,3.4,3.8$；
- $N_{\rm cfg}=406$，$N_{\rm boot}=1000$，seed $=1115$；
- $|P_z|=3,4,5$，$z/a=0,\ldots,24$，$T=5,\ldots,15$，symmetric cuts $c=1,2,3$；
- 使用 traceless flowed field，unpolarized 为 real projection，helicity 为完整 complex `pol35` numerator 的 imaginary projection；denominator 为 real `nopol` C2；
- 每个 bootstrap replica 重新计算 disconnected vacuum subtraction、C3/C2 ratio 和 summed ratio；
- 所有 flow time 使用同一份 $N=406$ configuration manifest、C2 轴和 bootstrap index。

主要数组形状为：

```text
c3_original, ratio_original:       (6, 25, 11, 16, 3)
c2_original:                       (2, 11, 3)
ratio_sum_bootstrap:                (1000, 6, 3, 25, 11, 3)
c2_bootstrap:                       (1000, 2, 11, 3)
```

权威输入目录为：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/ratio_bootstrap_all_operators_v1/
```

冻结 manifest 的 SHA256 为
`1a2161d754493f03a4c5e26d5fe20a0a83446f762f3a8bc159cc282f866b53e4`。该 C3/C2 产品是 finite-flow、bare、disconnected gluon-OPE estimator，不包含 renormalization、matching、continuum extrapolation 或 zero-flow conversion。

### 11.3 `$fit-bare` 默认 forward Sumratio-difference AIC 拟合

裸矩阵元使用

\[
S_c(T)=\sum_{\tau=c}^{T-c}R_c(T,\tau),\qquad
D_c(T)=S_c(T+1)-S_c(T),\qquad D_c(T)=M.
\]

对于本项目 $a=0.0775\,\mathrm{fm}$，严格阈值给出

\[
T_{\min}=\lfloor 0.6/a\rfloor+1=8,
\qquad T_{\min}a=0.6200\,\mathrm{fm}>0.6\,\mathrm{fm}.
\]

每个固定 cut 只扫描从 $T_{\min}=8$ 开始的窗口，$T_{\max}=9,\ldots,15$。候选窗口使用 Schaefer--Strimmer correlation-to-identity covariance 和 registered SVD cut $10^{-12}$。质量门为：中心 $Q\geq0.01$、correlated $\chi^2/\mathrm{dof}\leq3$、正自由度、bootstrap finite fraction $\geq0.90$。

在同一固定 cut 内，逐 replica 使用

\[
\mathrm{AIC}_{i,b}=\chi^2_{i,b}+2k+2N_{\mathrm{omit},i},
\qquad k=1,
\]

并重新归一化 AIC 权重。不同 cut 改变了 summed observable 和 likelihood，因此不做跨 cut 的单一 AIC 比较；先在各 cut 内平均窗口，再对可用 cuts 等权混合。若 replica 没有通过质量门的候选，则依次使用正自由度 minimum-$\chi^2$ fallback，最后才使用单个 forward-difference 的 zero-dof diagnostic。accepted 与 filled 结果严格分开：

- `M`, `M_boot`, `fit_accepted`：质量控制后的 accepted 结果；不通过的位置保持 NaN；
- `M_filled`, `M_boot_filled`：为保持完整数组而生成的 diagnostic fallback；不能描述为 accepted fit；
- statistical error 为 `std(ddof=1)`，不除以 $\sqrt{N_{\rm boot}}$。

本次使用的 all-operator AIC 拟合目录为：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/fit_all_operators_aic_v1/
```

### 11.4 拟合验证结果

14 个 flow time、6 个算符、3 个动量和 25 个 $z$ 位置共 $6300$ 个拟合位置。collection validator 已确认所有输出的形状、AIC normalization、固定 $T_{\min}$、共享 configuration/bootstrap 轴、输出 hash 和 filled replica coverage。

| $t_f/a^2$ | accepted / 450 | central fallback / 450 | mean replica fallback fraction |
|---:|---:|---:|---:|
| 0.1 | 381 | 69 | 0.06583 |
| 0.2 | 379 | 71 | 0.06603 |
| 0.3 | 379 | 71 | 0.06511 |
| 0.4 | 359 | 91 | 0.06652 |
| 0.5 | 358 | 92 | 0.07258 |
| 0.6 | 348 | 102 | 0.07292 |
| 1.0 | 366 | 84 | 0.06957 |
| 1.4 | 378 | 72 | 0.06642 |
| 1.8 | 386 | 64 | 0.06677 |
| 2.2 | 389 | 61 | 0.07281 |
| 2.6 | 379 | 71 | 0.07828 |
| 3.0 | 374 | 76 | 0.08118 |
| 3.4 | 359 | 91 | 0.08288 |
| 3.8 | 361 | 89 | 0.08492 |
| **合计** | **5196 / 6300** | **1104 / 6300** | — |

### 11.5 裸矩阵元图

已绘制全部 14 个 flow time 的裸矩阵元随正 $z$ 的变化：

- 横轴严格为 $[-0.5,15.5]$，数据点为 $z/a=0,\ldots,15$；
- unpolarized 和 helicity 各包含 no S、+S、-S 三个算符；
- 每个算符展示 $P_z=3,4,5$ 三个面板；
- filled circle 表示至少两个 cuts 可用的 accepted fit，open circle 表示只有一个 cut 的 accepted fit，`X` 表示 central fallback；
- 图使用 `M_filled` 和 `filled_statistical_error` 以展示完整诊断覆盖；正式质量控制分析应优先读取 `M` 和 `statistical_error`。

图表文件为：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/fit_all_operators_aic_v1/plots_v1/unpolarized_bare_matrix_elements_vs_z_equal_cut_mixture.pdf
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/fit_all_operators_aic_v1/plots_v1/helicity_bare_matrix_elements_vs_z_equal_cut_mixture.pdf
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/fit_all_operators_aic_v1/plots_v1/bare_matrix_elements_all_operators_vs_z_equal_cut_mixture.pdf
```

collection 验证报告为：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/fit_all_operators_aic_v1/fit_collection_validation.json
```

### 11.6 当前未解决问题与下一步计划

当前还不能把这些结果称为物理 gluon PDF。待解决事项为：

1. 根据 $a^2/t_f$、$t_f/z^2$、$t_f\Lambda_{\rm QCD}^2$ 和 $r_FP_z$ 选择并检验可靠的 small-flow-time window；
2. 分别应用 unpolarized 的 $C_{\parallel\perp},C_{\perp\perp}$ 和 helicity 的候选 $C_{\Delta g}$，并显式处理 line-mass convention，避免重复 renormalization；
3. 对 $C_{\Delta g}$ 做独立的一圈 helicity partonic calculation，检查 dual tensor continuation、Euclidean phase、tensor mixing 和 singlet-quark mixing；
4. 对 AIC window、三个 cut、flow time 和 $P_z$ 做稳定性分析，并将 accepted 与 fallback 的影响分开报告；
5. 只有在 GF-to-$\overline{\mathrm{MS}}$ conversion、LaMET/pseudo-ITD matching、singlet mixing 和 continuum strategy 明确后，才进入物理 PDF 分析。

本轮新建的 fit/plot 层是版本化目录，不覆盖原始 C3/C2 产品、旧版 ratio 产品或 `Note/Gradient_Flow` 的既有源文件；`Note/Gradient_Flow/main.tex` 与 `main.pdf` 已在前一步同步理论和数据契约。
