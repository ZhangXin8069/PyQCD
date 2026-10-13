# Blending estimator、极限与 MPI 契约

本文件是 `pyqcd-blending` 的按需参考。读取时机：定义 stochastic estimator、
重权、dilution、elemental/perambulator 轴序，或设计 unbiasedness 验收。

## 1. 空间、basis 与 ownership

固定时间片上的颜色-空间空间：

$$
\mathcal L=\mathbb C^{N_cV_3},\qquad D=N_cV_3=3V_3.
$$

低模空间 $\mathcal L_1$ 由 $N_{\rm ev}$ 个 orthonormal gauge-covariant Laplacian
eigenvectors 张成：

$$
P_1=\sum_{i=1}^{N_{\rm ev}}|v_i\rangle\langle v_i|,
\qquad
\mathcal L_2=(I-P_1)\mathcal L,
\qquad
D_2=D-N_{\rm ev}.
$$

stochastic vectors $\eta_r$ 必须在 $\mathcal L_2$ 中归一化且等向采样。组合 basis：

$$
\phi_i=
\begin{cases}
v_i,&0\le i<N_{\rm ev},\\
\eta_{i-N_{\rm ev}},&N_{\rm ev}\le i<N_{\rm ev}+N_{\rm st}.
\end{cases}
$$

阵列输入不证明 $P_1\eta_r=0$、$\eta_r^\dagger\eta_s=\delta_{rs}$、isotropy 或
seed independence。生产记录必须保存 noise ensemble、normalization、
orthogonalization tolerance 与 projection residual。

completed quark line 与 basis vector 是不同 artifact type。若对 basis vector
应用 phased-link smearing，必须建立新的 basis artifact ID 并重新验证正交性与
projection；不能把普通 momentum-smeared propagator 直接当作 `eigvecs`。

## 2. Reweighting estimator

定义 successive factor：

$$
\omega_n=\frac{D_2-n}{N_{\rm st}-n},
\qquad 0\le n<N_{\rm st}.
$$

对 mode tuple $I=(i_1,\ldots,i_r)$，$d(I)$ 是 tuple 中不同 stochastic high-mode
labels 的数量：

$$
W(I)=\prod_{n=0}^{d(I)-1}\omega_n.
$$

分类：

| tuple | weight |
|---|---|
| 全为 low modes | $1$ |
| 两个不同 high labels | $\omega_0\omega_1$ |
| 两个相同 high labels | $\omega_0$ |
| 一个 low、一个 high | $\omega_0$ |
| 三个不同 high labels | $\omega_0\omega_1\omega_2$ |
| 三个 high labels，恰有两个不同 | $\omega_0\omega_1$ |
| 三个 high labels，全部相同 | $\omega_0$ |

distillation limit $N_{\rm st}=0$ 仅允许 low indices，且 $W=1$。
$N_{\rm st}\le D_2$，其中 $D_2=3V_3-N_{\rm ev}$ 使用 global spatial volume。

完整 unbiased statement 是 noise-ensemble expectation，例如

$$
\mathbb E_\eta\left[
\sum_{ij}\Omega_{ij}|\phi_i\rangle|\phi_j\rangle
\langle\phi_i|\langle\phi_j|
\right]=I\otimes I,
$$

前提是已声明的 orthonormal/isotropic sampling。单次 realization 或单个 seed
不证明该等式。

## 3. Dilution 与精确极限

若使用 dilution projectors $P^{[d]}$，必须满足

$$
P^{[d]}P^{[d']}=\delta_{dd'}P^{[d]},
\qquad
\sum_dP^{[d]}=I_{\mathcal L_2}.
$$

每个 diluted source、solve、perambulator 和 contraction 保存
`(seed,time,noise_label,dilution_label)`。必须明确 reweighting 在 dilution
之前还是之后，并独立证明 expectation。

极限：

| 条件 | 结论 |
|---|---|
| $N_{\rm st}=0$ | retained $N_{\rm ev}$ subspace 的 standard distillation |
| $N_{\rm ev}=D$ | 已是 exact full space，不需要 stochastic complement |
| $N_{\rm st}=D_2$ 且 complement 完整 | $\omega_n=1$，combined basis 精确张成 $\mathcal L$ |
| dilution 完整且所有 solves 齐全 | stochastic subspace 的 diluted identity 精确 |

这些是代数或完整采样极限，不自动证明 correlated/incomplete ensemble 的无偏性。

## 4. 数据布局与 global phase

basis：

```text
(mode, t, z, y, x, color) = (N, Lt, Lz, Ly, Lx, 3)
```

momentum：`(px,py,pz)`；offset：`(z0,y0,x0)`。

$$
\phi_{\mathbf p}(z,y,x)=
\exp\left[-2\pi i\left(
\frac{p_xx_{\rm global}}{L_x^{\rm global}}+
\frac{p_yy_{\rm global}}{L_y^{\rm global}}+
\frac{p_zz_{\rm global}}{L_z^{\rm global}}
\right)\right],
$$

$$
(z_{\rm global},y_{\rm global},x_{\rm global})
=(z_0+z,y_0+y,x_0+x).
$$

多 spatial rank 缺少 global shape 或 rank offset 时必须拒绝。Fourier momentum
是精确投影标签，不能与 optional basis-smearing mode 混写。

## 5. Elementals

meson：

$$
\Phi_{ij}(t,\mathbf p)=W_{ij}
\sum_{\mathbf x,c}\phi_i^*(\mathbf x,t,c)
e^{-i\mathbf p\cdot\mathbf x}\phi_j(\mathbf x,t,c),
$$

输出 `(t,sink_mode,source_mode)=(Lt,N,N)`。

baryon：

$$
B_{ijk}(t,\mathbf p)=W_{ijk}\sum_{\mathbf x}\epsilon_{abc}
\phi_i^a(\mathbf x,t)\phi_j^b(\mathbf x,t)\phi_k^c(\mathbf x,t)
e^{-i\mathbf p\cdot\mathbf x},
$$

输出 `(t,N,N,N)`，颜色维严格为 3。

## 6. Projected perambulator

目标对象：

$$
\tau_{ij}^{\alpha\beta}(t,t_0)
=\langle\phi_i(t)|D^{-1}_{\alpha\beta}(t,t_0)|\phi_j(t_0)\rangle.
$$

轴序：

```text
(t, sink_spin, source_spin, sink_mode, source_mode)
```

source construction、Dirac action、boundary、solver、sink projection、MRHS 与
true residual 归 callback 所有。串行 callback orchestration 不是 MRHS
implementation，也不证明求解收敛或成本可接受。

## 7. Two-point contraction 边界

Wilson-like backward line 使用

$$
\tau_{\rm back}(t)=
(\gamma_5\otimes I_N)\tau^\dagger(t)(\gamma_5\otimes I_N).
$$

meson contraction 至少需要固定 interpolaor 的 spin kernel、source-bar kernel 和
Fourier convention。baryon $uud$ 使用 direct-minus-exchange：

$$
C_N(t)=C_{\rm direct}(t)-C_{\rm exchange}(t).
$$

sink diquark 不一定是 source diquark 的普通 dagger；共轭由所选 baryon
interpolator 决定。局部代数测试不证明完整 flavor topology、disconnected term 或
ensemble normalization。

## 8. MPI 与 GPU 条件

1. communicator 只连接拥有同一 global time slices 的 ranks。
2. time decomposed 时禁止 world reduction。
3. phase 使用 global extents 和 rank global offset。
4. elemental reduction 使用 same-timeslice spatial pieces，每 piece 只加一次。
5. CuPy collective 只有在目标 stack 已验证 CUDA-aware MPI 时启用；
   无隐式 host fallback。
6. 保存 rank/grid、local/global shape、precision、backend、communicator ID 与
   collective 次数。

two-rank stub、single-rank early return 或 CPU collective 不能证明真实 MPI
transport。

## 9. 验收 recipe

1. 对同一小系统实现 independent weight oracle；覆盖 low-only、重复/不同
   stochastic labels 和非法索引。
2. 验证 $N_{\rm st}=0$ 与 standard distillation 一致。
3. 构造显式 orthonormal complement，验证 projection residual、Gram matrix 和
   $N_{\rm st}=D_2$ exact limit。
4. 对 shifted rank offset 构造 Fourier phase oracle，并检查每次 local piece 的
   contribution 恰好求和一次。
5. 验证 perambulator 轴序、spin orientation、source/sink mode 和 true residual。
6. 用 isospin/asymmetric complex test matrix 验证 $\gamma_5$ Hermiticity；裸
   transpose-conjugate 必须在该 test 中失败。
7. 预声明 exact/all-to-all reference、若干独立 seeds、bias/variance tolerance，
   报告 $\Delta$、$z$、置信区间、失败率和成本分解。

这些 recipe 是验收计划。`no-runtime-claim`：除非目标环境确有记录，否则不得声称
GPU、CUDA-aware MPI、QUDA solve 或 production estimator 已验证。
