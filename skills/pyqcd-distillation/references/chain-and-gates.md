# 蒸馏链：定义、张量、算法与安全门

## 1. Physical definition

先写清：

- 目标 observable 是 2pt、3pt、4pt 还是矩阵元，connected 还是 disconnected；
- 粒子、`J^PC`、flavor、source/sink interpolator、current insertion；
- 初末动量、动量转移、源汇时间和 insertion time；
- 是否需要 backward/parity partner、projector 和 trace；
- 输出保留哪些自由指标，哪些必须闭合。

若 topology 或 free-index closure 未定义，不能从“代码能跑出一个数组”反推物理对象。

## 2. Convention

| 对象 | 必须固定 |
|---|---|
| gamma | DeGrand-Rossi、编号到矩阵的映射、Euclidean 符号 |
| 时间边界 | 夸克 APBC；复合态的有效边界；backward 的负宇称伙伴 |
| 源汇 | operator conjugation、quark ordering、creation/annihilation count |
| Fourier | `exp(+ipx)` 或 `exp(-ipx)`、源汇符号、动量单位 |
| basis | `(Nev,Lz,Ly,Lx,Nc)`、展平顺序 `z*Ly*Lx+y*Lx+x` |
| phase | `p_basis_sink/source` 与 `q_sink` 分开；共轭腿会反转 phase |
| normalization | color trace、`1/(2E)`、root factors、complex projection |
| gauge links | forward/backward direction、boundary、raw/smeared 来源 |

distillation basis phase 可写为

\[
E_p(z,y,x)=
\exp[-i2\pi(p_z z/L_z+p_y y/L_y+p_x x/L_x)]E(z,y,x).
\]

它不是带 gauge links 的 covariant momentum smearing。同一 basis 的 VdV 两条相位抵消，
VVV 三条非共轭腿会出现 `u_p^3`，再与 `q_sink` 合并。

## 3. Tensor formula

先为每个张量写表：

```text
physical meaning
axis order and extent
time and direction label
spin and eigenvector indices
flavor
conjugate/transpose state
momentum and displacement label
```

典型形状：

```text
E:     (Nev,Lz,Ly,Lx,Nc)
peram: (t_source,t_sink,d_sink,d_source,ev_sink,ev_source)
seq:   (t_sink,4,4,ev_sink,ev_source) for one fixed source
VdV:   V^dagger O V with two eigenvector spaces
VVV:   baryon vertex with three eigenvector spaces
```

sequential perambulator 的来源约定为

\[
\tau_{seq}(t,t_s)=\gamma_5\tau(t_s,t)^\dagger\gamma_5.
\]

实际 reader/producer 可能交换 `t_source/t_sink` 或把 batch axes 放前。应以 producer
代码和最小矩阵 oracle 为准，不能按变量名推断。

## 4. Algorithm

1. 生成或读取 LapH eigenvectors，并决定是否压缩/truncate。
2. 用 valence Dirac operator 求 inverses，形成 perambulator。
3. 用 eigenvector vertices 和 link/displacement 构造 `V^dagger O V`。
4. 对 sink/source/current operator 枚举 Wick contractions。
5. 为每个 diagram 生成 `einsum`/optimized contraction plan。
6. 收缩并作 momentum、parity/spin projection 和 trace。
7. 保存 raw correlated objects，再交给统计和谱学层。

## 5. LapH/eigenvector gates

- 独立计算 `G=E^dagger E` 和 `max(abs(G-I))`；同时报告 diagonal normalization 和
  off-diagonal overlap。
- 不依赖只比较少量 overlap 的内部 `check()` 作为充分条件。
- 压缩检查 group length、divisibility、zero norm、linear dependence、output rank。
- 若声称 `Z_N` stochastic vectors，验证 unit-modulus roots 和 normalized distribution；
  固定的高位 phase 不能自动视为通用 roots-of-unity。
- 动量 list 先 deduplicate；zero momentum、单轴和 conjugate pair 做 phase oracle。

## 6. Gauge/smearing gates

明确区分三类链：

```text
valence inverse action: historical 4D stout links
operator/flow vacuum: thin links
distillation basis: multi-step stout links
```

对每个 smearing step 验证：

1. identity 和 `rho=0` 为不动点；
2. 输出 finite；
3. `U U^dagger=I`；
4. `det(U)=1`；
5. 仅更新声明方向；
6. 随机 SU(3) 场与逐点 staple oracle 一致；
7. gauge covariance；
8. backend/CPU-GPU 数值对照。

来源 lqcddb 的 Stout 文档/实现存在 rank、transpose、zero-generator、roll-axis 风险；
PyQCD 的移植通过这些基本不变量测试也不自动证明历史生产等价。

## 7. Perambulator/vertex gates

- 先确认 stored perambulator convention，再写 `gamma5` Hermiticity。
- `.conj()` 只有在输出 index order 同时完成 transpose 时才是 adjoint。
- 顺序传播子用显式 spin/eigenvector matrix multiplication 对照。
- flavor 即使数值相等也必须保持标签。
- vertex 必须确认是 `V^dagger O V`、其 conjugate，还是 displaced/link version。
- 三点 ratio 显式对齐 insertion 和 sink axes；不能依赖 broadcasting。
- conserved/current insertion 要记录 link direction、midpoint phase 和 normalization。

## 8. Wick and contraction gates

每个 diagram 保存：

```text
numerical coefficient and Fermi sign
perambulator flavor and source/sink times
gamma/charge-conjugation factors and transposes
vertex identity, momentum, displacement, time
free output indices and final shape
connected components in the quark-line graph
```

来源 audit 的关键限制：

- `wick()` 的 `P/V/G` 默认空字符串，生产调用必须显式给 prefix。
- `conjugate_operator` 返回的 coefficient 必须并入总系数。
- unknown operator token 必须报错，不能静默消失。
- `identify_equivalent_diagrams()` 只比较 perambulator index 结构，不足以保证 gamma、
  vertex、time 和 topology 等价。
- `ignore_dis` 是 time-label heuristic，不是 graph connectedness proof。
- 参数化 gamma 必须推导 component-wise transpose sign，不能用泛型 `Gamma^T=Gamma`。
- grouped/ungrouped result 要在 small deterministic tensors 上逐项对照后才能启用。

## 9. GEVP

解

\[
C(t)v_n(t,t_0)=\lambda_n(t,t_0)C(t_0)v_n(t,t_0),
\qquad
\lambda_n=e^{-E_n(t-t_0)}.
\]

要求：

- `C(t0)` 经记录 conditioning 后 Hermitian positive definite；
- 保留 complex Hermitian off-diagonal；
- 与 `scipy.linalg.eigh(C(t),C(t0))` 或独立小矩阵对照；
- 检查 `C(t0)`-orthogonality 和 normalization；
- level crossing/near degeneracy 时用 eigenvector overlap 跨时间追踪 state；
- 报告 `t0`、window、matrix rank、conditioning 和 state assignment stability。

来源 lqcddb 的 GEVP 曾把 complex matrix 投影为 real；这是 source implementation
limitation，不是 GEVP 物理要求。PyQCD 当前实现保留 complex Hermitian data。

## 10. MPI

先区分两种并行：

```text
lattice domain decomposition: MPI ranks own lattice subdomains
meta-task parallelism: (step,conf) tasks distributed over ranks
```

两者不能混用语义。domain decomposition 至少检查：

- `getDefaultGrid` 的 direction order 和 lattice divisibility；
- rank/coord mapping，包括 `XYZT_FASTEST`/`TZYX_FASTEST`；
- local time list 和 root/non-root return；
- size-one 与 multi-rank gather shape 一致；
- `TGather` 要求每 rank leading size 相等，否则使用 variable-count 设计；
- collective failure 在所有 rank 同步退出；
- gather/reduce 在 gauge context 外。

元任务 pipeline 则由 `pyqcd-infra` / `pyqcd-pipeline` 定义，保持 `(step,conf)` 完整、
GPU 绑定、rank0 analysis/plots/report 和失败退出码。

## 11. I/O

canonical 规则：

- 优先 HDF5；旧 `.npy/.npz` 只作兼容读取。
- 保存 shape、axis names、dtype/complex、endian、backend/device、seed、command、
  package version 和 input hash。
- read/write round-trip 必须逐位或按明确容差一致。
- basename-only path、cwd auto fallback 和 site-specific fallback 不是可移植发布语义。
- 二进制 VdV/VVV reader 检查 endian、Nev auto-detect、truncation 和 complex pair
  约定。
- ASCII writer/reader 检查 `.gz`、header、complex interleave 和目录创建行为。

来源 lqcddb 的 `write_data_ascii` basename 行为、legacy `numpy.compat` 依赖和
`savetxt` 兼容性属于 known limitation；PyQCD 自有 writer/reader 虽已移植和测试，
但外部旧文件仍需 provenance 核对。

## 12. 最小验收组合

对任何新拓扑至少包括：

1. 一个手算或 independent `einsum` 小张量 oracle；
2. shape、axis、conjugation 和 free-index closure 检查；
3. Fermi sign/flavor balance 检查；
4. 一个真实组态上的残差/有限性/backend 对照；
5. MPI size-one vs size-N 或 serial reference；
6. HDF5 round-trip 和 metadata readback；
7. 明确列出仍未被 oracle 覆盖的 4pt、disconnected 或 GEVP 物理系统。
