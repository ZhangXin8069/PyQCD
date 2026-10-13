---
name: pyqcd-gluon-renorm
description: |
  Use when reviewing PyQCD bare or gauge-fixed gluon observables, g0
  normalization, Clover field strength and dual tensors, Fourier half-link
  phases, time-decomposed reductions, or EMT/current renormalization claims.
---

# pyqcd-gluon-renorm — bare、gauge-fixed 与 renormalized 分界

## 目的与边界

本技能负责纯胶子观测量从规范链接到可用矩阵元的数值契约，并严格区分：

```text
bare lattice observable
gauge-fixed candidate
renormalized physical operator
```

输入可以包括 PyQCD 规范场、Wilson flow/涂抹后的链接、Clover field strength、
Wilson-line products 和 Fourier modes。输出若缺少 operator basis、mixing、scheme、
scale 与 continuum matching，只能称 `bare_*` 或 `diagnostic_*`。

## 适用与不适用

| 适用 | 不适用或必须转交 |
|---|---|
| `g0`、plaquette、Clover `F_munu` 与 dual tensor | quark propagator、RI vertex 或 quark bilinear conversion |
| Wilson-line `F-W-F-W†` 类矩阵元 | 把局部 bare contraction 称为最终 EMT/current |
| Fourier 单模、全模与 half-link 相位 | 用公共相位替代方向特异 `exp(-i p_mu/2)` |
| 时间分解、MPI ownership 和归约设计 | 在世界归约中混合不同 global time slices |
| 建立 EMT/current 的 provenance blocker | 缺 mixing/renormalization 时发布 physical result |

## 核心公式与归一

方向与 PyQCD gauge 布局以 `pyqcd-conventions` 为准：

```text
Lorentz labels: 0=x, 1=y, 2=z, 3=t
pyqcd gauge:    (Nt,Nz,Ny,Nx,4,3,3)
```

`g0` 由 gauge action 决定，不由 fermion action 名称决定：

$$
\beta=\frac{6}{g_0^2}\quad\text{(Wilson)},
\qquad
\beta=\frac{10}{g_0^2u_0^4}\quad
\text{(explicit legacy tadpole)}.
$$

Wilson 分支不得传 `u_0`；legacy 分支必须显式命名 normalization 与 `u_0`。

link-centered gauge potential 构造成本为

$$
A_\mu^{\rm code}(x+\tfrac12\hat\mu)
=\frac1{g_0}
\left[\frac{U_\mu-U_\mu^\dagger}{2i}\right]_{\rm traceless},
$$

Clover 场强使用四个 oriented plaquette leaves；dual tensor 使用

$$
\widetilde F_{\rho\sigma}
=\frac12\epsilon_{\rho\sigma\mu\nu}F_{\mu\nu}.
$$

`1/2` 防止对 antisymmetric pairs 重复计数。Wilson-line product 的核心归一为

$$
\mathcal O=-\frac12\operatorname{Tr}
\left[F_1(x)W_d(x,x+\ell\hat d)
F_2(x+\ell\hat d)W_d^\dagger\right].
$$

对 link-centered `A_mu` 做 Fourier transform 时，方向 `mu` 单独乘
`exp(-i p_mu/2)`；不能使用依赖 `p_x+p_y+p_z+p_t` 的 common phase。

## 工作流

1. 冻结 ensemble、gauge action、`beta`、格距、体积、边界、链接方向和 trace
   normalization；先由 gauge action 计算 `g0`。
2. 决定输入层级：raw links、smearing、Wilson flow 或已规范固定链接，并保存源 hash
   与变换参数。
3. 构造 plaquette/Clover/dual 或 Wilson-line product；记录 Lorentz pair、路径方向、
   长度、前进/回退 orientation 和 `1/2` normalization。
4. Fourier 前将 checkerboard/目标轴序显式转换；对偶数 extent 的 Nyquist mode
   采用统一 canonical representative，并应用方向特异 half-link phase。
5. 明确 MPI ownership：先做局域空间收缩，只归约 compact result；time-resolved
   observable 必须保证全球时间切片不混合。
6. 把 bare matrix element 与 gauge-fixed candidate 分开保存；EMT/current 继续保留
   provenance blocker，不进入未建立的 renormalization。
7. 交给统计、分析和 TMD 层时携带单位、轴、归一化、ownership 与状态字段。

## 验收门

- [ ] `g0` normalization 与生成 ensemble 的 gauge action 一致；fermion action 未参与推断。
- [ ] direction labels、array axes、checkerboard parity 和 PyQCD gauge layout 未混用。
- [ ] full `F_munu` 的反对称性、dual `1/2` 和 magnetic pair map 经独立检查。
- [ ] Wilson line 的方向、长度、orientation、端点和 dagger 顺序逐项核对。
- [ ] link-centered Fourier 的每个方向使用自己的 half-link phase。
- [ ] time-decomposed reduction 不跨 global time；`Gt`/time ownership 明确。
- [ ] 单 rank 或 fake communicator 测试不被表述为多 rank 证据。
- [ ] 与独立 Clover/显式 DFT/reference contraction 比较 real 与 imaginary parts 均通过。
- [ ] 未提供 operator definition、mixing matrix、scheme、scale 和 continuum matching
  时，EMT/current 结果保持 `bare_emt_candidate` 或 `diagnostic_current`。

## 与 PyQCD 技能交接

| 技能 | 交接责任 |
|---|---|
| `pyqcd-gauge` | 生成或读取规范链接、Wilson flow/smearing、Clover/path 原语；本技能记录其路径、方向和归一化。 |
| `pyqcd-tmd-algorithm` | 接收带几何、流时间和 representation provenance 的胶子场/矩阵元；本技能不替代 TMD operator、soft/rapidity 或 matching。 |
| `pyqcd-statistics` | 对逐组态矩阵元执行共享索引重采样、协方差和 SVD 门；bare 观测量也需保留样本轴。 |
| `pyqcd-analysis` | 只绘制已标明状态的 bare/gauge-fixed 结果、归约 ownership 和未闭合 blocker，不把 candidate 重命名。 |

## 来源与证据上限

来源技能冻结于
`90bcbb8739c80b41a3965f41dab50bdfc9841b16`。详细轴序、公式和 blocker 台账见
[`references/physics-contract.md`](references/physics-contract.md)。

方法来源：

- arXiv:hep-lat/0203008：improved/clover field-strength method class；
- arXiv:1310.4263：gauge-dependent gluon spin 与 Chern-Simons context。

这些来源不自动验证本库 legacy EMT、Chern-Simons、Wilson-line power divergence、
operator mixing 或 continuum conversion。

当前证据上限是选定 bare contraction 的 array algebra；真实多 rank halo/performance、
gauge-fixing quality、EMT/current equation-level provenance 与 renormalized
observable 尚未闭合。`renorm` 出现在技能名中不构成已重整化证据。
