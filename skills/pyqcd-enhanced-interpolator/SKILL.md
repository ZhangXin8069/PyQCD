---
name: pyqcd-enhanced-interpolator
description: |
  Use when a PyQCD task needs to construct or review kinematically enhanced
  boosted-hadron interpolator kernels, Euclidean gamma-plus/minus algebra,
  meson or nucleon source/sink kernels, local direct-minus-exchange Wick
  contractions, or checkerboard momentum projection.
metadata:
  provenance:
    source_commit: 90bcbb8739c80b41a3965f41dab50bdfc9841b16
---

# pyqcd-enhanced-interpolator - 增强内插算符

## 目的与边界

本技能用于推导、实现审查和验证以下数组级对象：

- boosted-hadron 的 Euclidean `gamma_±` 与 quark-component kernel；
- enhanced meson 或 nucleon 的 source/sink spin kernels；
- 传播子末四轴为 `(sink_spin, source_spin, sink_color, source_color)` 的
  local Wick contraction；
- checkerboard local density 的 spatial momentum projection 与 MPI 归约契约。

本技能不执行 Dirac inversion、配置 I/O、smearing、拟合或连续极限。它也不因算符名称
中的 "enhanced" 自动获得 overlap 或 signal-to-noise 改善证据。

在修改 gamma algebra、source conjugation、pion mapping 或 baryon contraction 前，必须
阅读 [`references/physics-contract.md`](references/physics-contract.md)。

## 物理契约

### 1. gamma 基底和增强分量

先通过 `pyqcd-conventions` 固定一个数值 DeGrand-Rossi gamma 基底、boost 方向、
momentum 符号、时间边界和 spin/color 轴序。令 `parallel` 与 hadron boost 方向一致：

```text
gamma_+ = (gamma_t + i gamma_parallel) / sqrt(2)
gamma_- = (gamma_t - i gamma_parallel) / sqrt(2)
Q_+     = gamma_- gamma_+ / sqrt(2)
```

必须保留 `Q_+` 的论文归一化。按该约定通常有 `gamma_+^2 = gamma_-^2 = 0`，且
`Q_+^2 = sqrt(2) Q_+`。`Q_+` 是 scaled component map，不是幂等 projector；不得静默
改成 `gamma_- gamma_+ / 2`，也不得仅凭 `projector` 名称假设 `Q_+^2 = Q_+`。

### 2. 两种不同的 source conjugation

普通 bilinear 的 Euclidean adjoint 与 nucleon source diquark 的复共轭操作不同：

```text
bilinear:        M_bar     = gamma_t M^dagger gamma_t
source diquark:  D_source  = gamma_t D_sink^* gamma_t
```

第二式没有 transpose。把 `D_sink^*` 改成 `D_sink^dagger` 会交换 source spin indices，
必须作为关键 mutation 单独做数值区分。

### 3. Canonical pion mapping

generic meson helper 接收的是 `bar(q) Gamma q` 中的 Dirac-bar matrix。论文的 projected
pion 对象从 `u_+^dagger gamma_5 d_+` 出发，因此必须映射为：

```text
Gamma_Dirac_bar = gamma_t gamma_5
K               = (gamma_t Q_L^dagger gamma_t) Gamma_Dirac_bar Q_R
```

在文档约定的 Clifford algebra 下，canonical wrapper 应给出
`K(gamma_t gamma_5) = sqrt(2) gamma_+ gamma_5`；把裸 `gamma_5` 传给 generic helper
会得到 zero kernel。generic 参数不是论文对任意 bilinear 的物理白名单。

### 4. Diquark 与 Wick 缩并

nucleon sink diquark 和 source diquark 分别为：

```text
D_sink        = C gamma_5 Gamma
Gamma         in {I, gamma_t, gamma_+}
D_source_bar  = gamma_t D_sink^* gamma_t
```

局部 `uud` contraction 必须保留两个颜色 epsilon，并对两条 identical `u` line 实现
direct-minus-exchange：

```text
C_N = C_direct - C_exchange
```

exchange 交换的是两条 `u` line 的 source spin/color ownership。只检查输出 shape 不能
验证该项；修改 einsum 时必须用显式 spin/color loop oracle。

### 5. Checkerboard 与 MPI

local contraction 和 momentum phase 都必须是完整布局 `(e,t,z,y,xh)`，且 `e = 2`。
phase 必须在调用前使用 global coordinates 构造。归约只允许发生在拥有同一组 global
timeslices 的 `spatial_comm` ranks 上，并使用 mpi4py-style buffer `Allreduce`。

CuPy buffer 的 `cuda_aware_mpi=True` 只能在目标 MPI stack 已被直接验证后设置；本技能
不提供隐含 host fallback，也不构造 phase 或 spatial subcommunicator。

## 工作流

1. 记录 gamma basis ID、boost direction、momentum sign、projector/diquark choice、
   normalization、propagator source/sink metadata、backend 和 precision。
2. 用数值 oracle 验证 Clifford algebra、`gamma_±` 归一化、nilpotency 和
   `Q_+^2 = sqrt(2) Q_+`；不要仅核对 shape。
3. 从 `pyqcd-conventions` 的固定 gamma 对象构造 kernels。Pion 使用 canonical wrapper；
   其他 bilinear 必须先给出独立算符推导。
4. 将传播子交给 local contraction，以显式 spin/color loop 核对 direct-minus-exchange。
5. 独立构造 global-coordinate checkerboard phase，再做 momentum projection 和允许的
   spatial reduction。
6. 按下方验收门形成交接包；未通过的门保持 fail closed，不用默认值补齐。

## 验收门

- [ ] `gamma_t`、`gamma_parallel`、`gamma_5`、`C` 来自同一数值基底和 backend，输入是
  finite、numeric 的精确 `(4,4)` matrices。
- [ ] `gamma_+`、`gamma_-` 的符号与 `1/sqrt(2)` normalization 正确，boost direction
  与 momentum phase 一致。
- [ ] `Q_+ = gamma_- gamma_+ / sqrt(2)` 且 `Q_+^2 = sqrt(2) Q_+`；测试明确证明它不是
  idempotent projector。
- [ ] 非 Hermitian、非对称测试矩阵能够区分 `gamma_t M^dagger gamma_t` 与
  `gamma_t M^* gamma_t`；source diquark 路径不含 transpose。
- [ ] canonical pion 使用 `Gamma = gamma_t gamma_5`，数值 oracle 得到
  `sqrt(2) gamma_+ gamma_5`；裸 `gamma_5` 的 zero result 已被专门拒绝或说明。
- [ ] 传播子末四轴为 `(4,4,3,3)`；显式 loop oracle 验证 direct-minus-exchange 的
  spin、color 和 Fermi 相对符号。
- [ ] projection 输入为 `(2,t,z,y,xh)`，phase 使用 global coordinates，communicator
  只含同 global timeslice ranks。
- [ ] 未把 CPU algebra、shape、fake-CuPy 或 stub collective 证据写成 interacting、
  GPU 或 MPI 已验证。

## 技能交接

| 对象 | 交接内容 |
|---|---|
| `pyqcd-conventions` | gamma basis、轴序、单位、时间边界、复数投影、结果状态与 metadata 最低集合；任何不一致先停在本层 |
| `pyqcd-physics-correlator` | 提供 observable、`J^{PC}`、flavor、momentum 约定、源汇时间、Wick topology、propagator inventory；本技能只实现其 spin-kernel 与 local contraction 部分 |
| `pyqcd-propagator` | 提供实际传播子及其 source/sink、solver residual、smearing、backend、precision 和轴序；若 basis 或轴序不匹配，不以 reshape 或默认 gamma 修补 |

向下一层交接时同时给出：算符公式、gamma basis、boost/momentum 方向、kernel choices、
归一化、传播子元数据、contraction einsum 或显式 loop oracle、phase 定义、MPI ownership、
验证命令与尚未验证的边界。

## 来源与证据上限

来源是只读目录
`skills/LQCD_Skills/skills/pyquda_enhanced_interpolator/`，提交
`90bcbb8739c80b41a3965f41dab50bdfc9841b16`。物理来源包括
arXiv:2606.02447v2 的 nucleon construction 与 arXiv:2501.00729v2 的 projected-pion
identity。来源脚本未复制到本技能。

本技能当前只建立 PyQCD 自有的实现与审查契约，最高证据是 formula/algebra contract。
没有在本任务中运行 PyQCD 实现，也不证明 production gamma basis、CUDA-aware MPI、
multi-rank pipeline、interacting overlap、effective-mass plateau、SNR 或 fit stability。
只有与 conventional operator 共用 configurations、propagators、source count、precision、
momentum、fit rule 和 cost 的相关比较通过后，才能报告增强效果；否则输出只能标为
`enhanced_operator_candidate` 或 `paper_normalized_kernel`。
