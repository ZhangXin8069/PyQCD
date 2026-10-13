---
name: pyqcd-blending
description: Use when a PyQCD task needs to design, implement, or audit blended all-to-all estimators with retained Laplacian modes plus a stochastic orthogonal complement, including blending tuple weights, meson or baryon elementals, projected perambulators, global-coordinate Fourier phases, or gamma5-Hermitian contractions.
metadata:
  short-description: Blended all-to-all estimator contracts
---

# pyqcd-blending — Blended all-to-all

## 目的与边界

本技能定义 PyQCD 中 blending 的估计器与数据契约：在固定时间片上组合 $N_{\rm ev}$
个低 Laplacian modes 和 $N_{\rm st}$ 个已投影到低模正交补的 stochastic vectors，
构造 blending weights、meson/baryon elementals、projected perambulators 和两点
contraction。它不负责生成 noise、投影低模、选择 dilution、执行 Dirac solve、
实现 I/O 或把真实数据结果解释为物理结论；这些工作按交接路由到对应技能。

本技能也不把 generic momentum-smeared quark source 当成 blending basis。详细
估计器、极限、轴序和 MPI 边界见
[`references/estimator-contract.md`](references/estimator-contract.md)。在提出
unbiasedness、distillation limit、dilution 或 production claim 前必须读取该文件。

## 核心物理与数据契约

### Basis 与随机空间

固定时间片的颜色-空间空间为

$$
\mathcal L=\mathbb C^{3V_3},\qquad D=3V_3,\qquad
\mathcal L_2=(I-P_1)\mathcal L,\qquad D_2=D-N_{\rm ev}.
$$

$P_1$ 由 $N_{\rm ev}$ 个 gauge-covariant Laplacian eigenvectors 张成。
stochastic vectors $\eta_r$ 必须归一化、等向采样，并已经投影到
$\mathcal L_1^\perp$。组合 basis 为

$$
\phi=(v_1,\ldots,v_{N_{\rm ev}},\eta_1,\ldots,\eta_{N_{\rm st}}).
$$

API 接收数组不等于证明投影、正交性、isotropy 或 seed independence。数组或单次
noise realization 不能声称 expectation-level unbiasedness。

### Stochastic weights

令

$$
\omega_n=\frac{D_2-n}{N_{\rm st}-n},\qquad 0\le n<N_{\rm st}.
$$

对 mode tuple $I$，令 $d(I)$ 为其中不同 stochastic high-mode labels 的数量：

$$
W(I)=\prod_{n=0}^{d(I)-1}\omega_n,
\qquad W(\varnothing)=1.
$$

重复 stochastic label 只计一次。$N_{\rm st}=0$ 时只允许低模索引且 $W=1$，这是
standard distillation limit，不是一般情况下的 full-space identity。权重使用
global $D_2=3V_3-N_{\rm ev}$，不能用 rank-local volume。

### 轴序、全局坐标与 Fourier phase

basis 固定为

```text
(mode, t, z, y, x, color) = (N, Lt, Lz, Ly, Lx, 3)
```

公开 momentum 为 `(px,py,pz)`，spatial shape/offset 为 `(z,y,x)`。rank-local
phase 必须使用 global coordinate：

$$
\phi_{\mathbf p}(z,y,x)=
\exp\left[-2\pi i\left(
\frac{p_xx_{\rm global}}{L_x^{\rm global}}+
\frac{p_yy_{\rm global}}{L_y^{\rm global}}+
\frac{p_zz_{\rm global}}{L_z^{\rm global}}
\right)\right].
$$

其中 $(z_{\rm global},y_{\rm global},x_{\rm global})
=(z_0+z,y_0+y,x_0+x)$。多 spatial rank 时必须同时提供 global shape 与本 rank
offset。

### Elementals 与 perambulator

meson elemental 为

$$
\Phi_{ij}(t,\mathbf p)=W_{ij}
\sum_{\mathbf x,c}\phi_i^*(\mathbf x,t,c)
e^{-i\mathbf p\cdot\mathbf x}\phi_j(\mathbf x,t,c),
$$

baryon elemental 为

$$
B_{ijk}(t,\mathbf p)=W_{ijk}\sum_{\mathbf x}\epsilon_{abc}
\phi_i^a\phi_j^b\phi_k^c
e^{-i\mathbf p\cdot\mathbf x}.
$$

projected perambulator 为

$$
\tau_{ij}^{\alpha\beta}(t_1,t_2)
=\langle\phi_i(t_1)|D^{-1}_{\alpha\beta}(t_1,t_2)|\phi_j(t_2)\rangle.
$$

其固定轴序为
`(t,sink_spin,source_spin,sink_mode,source_mode)`。callback 可以串行给出
source/solve/project 接口，但 production 若使用 MRHS/batching，必须记录实际
solve ownership、true residual 和成本。

### $\gamma_5$ Hermiticity

Wilson-like propagator 的 backward line 使用

$$
\tau_{\rm back}(t)=
(\gamma_5\otimes I_N)\tau^\dagger(t)(\gamma_5\otimes I_N),
$$

不能替换成不带 spin kernels 的裸 `conj().T`。meson 与 baryon 的 overall sign、
source-bar、diquark 共轭和 Fourier convention 必须由
`pyqcd-physics-correlator` 的 operator definition 固定；局部代数通过不等于完整
Wick、flavor 或 disconnected 结构已验证。

### Global-coordinate MPI 边界

多 spatial-rank 运行时：

1. communicator 只能连接同一组 global time slices 的 ranks；time 被分解时禁止
   world reduction。
2. phase 使用 global shape 和每个 rank 的 global offset。
3. $N_{\rm st}\le D_2$ 使用 global $3V_3$。
4. collective 只对同一 global time slice 的 spatial pieces 求和，且每个 piece
   只计一次。
5. CuPy 输入只有在目标 MPI stack 已验证 CUDA-aware 时才允许显式 opt-in；无隐式
   host fallback，单 rank stub 不能证明真实 transport。

### Artifact ownership

低模/stochastic basis、projected perambulator、solver source 与 completed
momentum-smeared propagator 是不同 artifact。若 blending 内部需要对 basis vector
应用 phased-link transformation，该操作属于 basis construction 并必须记录为
新的 basis ID、参数和 checksum；不能继承 generic source-smearing 的身份。

## 最小工作流

1. 冻结 global $V_3$、$N_{\rm ev}$、$N_{\rm st}$、noise ensemble、dilution
   projectors、seed 策略、basis artifact 和 configuration ID。
2. 验证低模 orthonormality、stochastic vectors 的归一化/isotropy，以及
   re-orthogonalization residual；保存未投影与投影后的 provenance。
3. 由 global $D_2$ 计算 tuple weights，分别覆盖 low-only、single/double/triple
   high label 和重复 label 分类。
4. 使用 global coordinates 构造 meson/baryon elementals；记录 momentum order 和
   local offset。
5. 通过显式 source/solve/project callbacks 构造 projected perambulators，并保存
   solver action、boundary、tolerance、true residual 与 MRHS 策略。
6. 在 `pyqcd-physics-correlator` 契约下完成 contraction；backward line 必须用
   $\gamma_5$ Hermiticity，并显式保留 source-bar、diquark 和 overall sign。
7. 先用 small-system 或 complete-distillation independent reference 验证代数
   极限，再用多个独立 seeds 和预声明 tolerance 检查 expectation 与 variance。

## 验收门

- Weight 门：$\omega_n$ 的域与 tuple classification 正确，重复 high label 只计
  一次；$N_{\rm st}=0$ 严格退化为 weight 1 的 distillation。
- Basis 门：低模正交性、stochastic normalization、low-mode projection 和
  dilution projector completeness 均有数值证据。
- Limit 门：$N_{\rm st}=0$ 对照 retained-subspace distillation；
  $N_{\rm st}=D_2$ 且 complement 完整时对照 exact full-space algebra。
- Elemental 门：global-coordinate phase 在 rank offset 下有独立 oracle，local
  pieces 被恰好求和一次。
- Perambulator 门：输出轴序、spin orientation、source/sink mode 标签和 true
  residual 可追溯；串行 callback 不被宣称为 MRHS 实现。
- Contraction 门：$\gamma_5$ Hermiticity 与 direct-minus-exchange 由独立小系统
  reference 验证，overall sign 与 flavor convention 明确。
- Statistical 门：使用多个独立 seeds，报告 $\Delta=\bar O-O_{\rm ref}$、
  $z=\Delta/\sigma_{\bar O}$、置信区间、失败率和随
  $N_{\rm st},N_{\rm ev}$、dilution 的 variance；单 seed 或未标注 tolerance
  不能证明 unbiasedness。
- 无运行时证据时只能给出权重、shape 或 synthetic algebra 结论。
  `no-runtime-claim`：不得声称已在 GPU、CUDA-aware MPI、QUDA 或 production
  ensemble 上验证。

## 与 PyQCD 技能的交接

| 交接对象 | 必须交付 |
|---|---|
| `pyqcd-propagator` | basis/source layout、Dirac action、solve callback、MRHS、residual、raw/smeared ownership |
| `pyqcd-physics-correlator` | meson/baryon interpolator、Wick topology、$\gamma_5$/diquark 约定、输出轴序 |
| `pyqcd-infra` | backend/precision、global/local layout、same-timeslice communicator、HDF5 与 round-trip |
| `pyqcd-pipeline` | basis/perambulator/configuration ID、checksum、cache key 和失败重跑边界 |
| `pyqcd-statistics` | seed ensemble、estimator、reference、error bar、bias/variance 验收 |
| `pyqcd-momentum-smear` | 仅在显式 basis callback 处会合；保持独立 artifact ID，禁止 field 直接传递 |

## 来源与证据边界

改写来源：
`skills/LQCD_Skills/skills/pyquda_blending/`，来源快照提交
`90bcbb8739c80b41a3965f41dab50bdfc9841b16`。本文只改写 estimator 公式、数据契约
和验证边界，不复制外部 scripts，也不把来源中的局部 kernel 当作完整生产流水线。

主要物理依据为 arXiv:2505.01719v2 的 Eqs. (1)-(4)、(31)-(32)、(37) 及
supplemental unbiasedness proof；arXiv:2009.10691v1 仅用于 momentum-smeared
distillation 背景。

来源工作树中的 deterministic CPU/single-DCU checks 只覆盖被明确声明的数组代数；
它们不证明 noise generation、orthogonal projection、dilution、ensemble
unbiasedness、真实 CUDA-aware MPI、QUDA solves 或 production correlator。
