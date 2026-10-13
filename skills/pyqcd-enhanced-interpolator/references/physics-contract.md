# Enhanced interpolator physics contract

本参考文件保存需要精确复核的公式、轴序、oracle 和 metadata。入口规则见
[`../SKILL.md`](../SKILL.md)。

## 1. Input domain

所有 public Dirac-matrix 输入必须满足：

- 同一 NumPy/CuPy backend；
- numerical 且 finite；
- exact shape `(4,4)`；
- 与调用者声明的 Euclidean Clifford basis 一致。

传播子的末四轴固定为：

```text
(sink_spin, source_spin, sink_color, source_color) = (4,4,3,3)
```

前导轴可以是一批 lattice/batch axes。任何错误 shape、非数值对象、`NaN` 或 `Inf`
都应在 gamma algebra 或 Wick contraction 前 fail closed。

## 2. Gamma basis and normalization

选择一个四维 Hermitian Euclidean basis：

```text
gamma_mu^dagger = gamma_mu
{gamma_mu, gamma_nu} = 2 delta_mu_nu I
```

调用者必须显式给出基底、`gamma_t`、boost 方向 `parallel`、`gamma_5` 和 charge
conjugation `C`。`parallel` 必须与 hadron momentum/phase 方向相同。

```text
gamma_plus  = (gamma_t + i gamma_parallel) / sqrt(2)
gamma_minus = (gamma_t - i gamma_parallel) / sqrt(2)
gamma_plus^2 = gamma_minus^2 = 0
```

论文归一化的 quark-component map 是：

```text
Q_plus = gamma_minus gamma_plus / sqrt(2)
Q_plus^2 = sqrt(2) Q_plus
```

因此 `Q_plus` 不是 idempotent projector。minimum oracle 应同时验证：

1. 两个 light-cone gamma 的矩阵值和 order；
2. nilpotency；
3. `Q_plus` 的 `1/sqrt(2)` normalization；
4. `Q_plus` 的 non-idempotence，而不是只验证 shape。

## 3. Two source operations

普通 bilinear 使用 Euclidean adjoint：

```text
M_bar = gamma_t M^dagger gamma_t
```

nucleon source diquark 使用 complex conjugation，无 transpose：

```text
D_source_bar = gamma_t D_sink^* gamma_t
```

对一般的非 Hermitian、非对称复数矩阵，这两者不同。critical-mutant test 必须使用
这样的矩阵；实对称矩阵或 diagonal matrix 不能区分错误实现。

source diquark 的两个 source-spin indices 属于固定的 contraction ownership。不能把
`conj()` 替换成 `conj().T`，也不能用其余 spin algebra 的通过结果外推到该路径。

## 4. Meson kernel and canonical pion

generic Dirac-bar kernel 为：

```text
K_enh = (gamma_t Q_L^dagger gamma_t) Gamma Q_R
```

其中 `Gamma` 是 `bar(q) Gamma q` 中的 matrix，不是 `q^dagger Gamma q` 中的 matrix。
如果省略 `Q_L`，实现可以使用 `Q_L = Q_R`；该选择必须记录。

标准 projected-pion identity 从：

```text
u_plus^dagger gamma_5 d_plus
```

出发。因为 `bar(u) = u^dagger gamma_t`，Dirac-bar input 必须是：

```text
Gamma = gamma_t gamma_5
```

在约定 basis 下，canonical wrapper 的精确目标为：

```text
K(gamma_t gamma_5) = sqrt(2) gamma_plus gamma_5
K(gamma_5)         = 0
```

generic helper 接受任意合法 `(4,4)` matrix 只是 algebra extension，不表示任意
bilinear 已被论文验证。新增 meson channel 必须提供独立 operator identity、flavor
assignment 和 normalization。

对 connected local meson density，来源实现的 backward antiquark line 和整体符号为：

```text
S_antiquark_back = gamma_5 S_antiquark^* gamma_5
C_local = -Tr_{spin,color}[
    Gamma_sink S_quark
    Gamma_source_bar S_antiquark_back
]
```

整体负号属于该算子/矩阵约定；不能从本 helper 单独推广到所有 meson convention。

## 5. Nucleon kernels and direct-minus-exchange

实现对应的 operator class 为：

```text
N_alpha^Gamma(x) = epsilon_abc
    [u_a^T(x) C gamma_5 Gamma d_b(x)] u_{c,alpha}(x)
```

diquark choices 为：

```text
Gamma in {I, gamma_t, gamma_plus}
D_sink = C gamma_5 Gamma
D_source_bar = gamma_t D_sink^* gamma_t
```

外部 spin projector choices 为：

```text
T in {(I + gamma_t)/2, gamma_t, gamma_plus}
```

名称 `parity`、`time`、`plus` 只标识已实现 choice；`plus` 不应因名称被当作数学上的
idempotent projector。

局部 `uud` contraction 是：

```text
C_N = C_direct - C_exchange
```

两项共享两个颜色 epsilon、spin projector 与 diquark factors，但 exchange 交换两条
identical `u` line 的 source spin/color ownership。

修改 contraction 时必须使用小数组 explicit spin/color nested-loop oracle，并逐项比较：

- direct term；
- exchange term；
- `direct - exchange` 的相对符号；
- color epsilon 的 two-epsilon contraction；
- propagator 最后四轴 `(sink_spin,source_spin,sink_color,source_color)`。

只比较最终 shape、只比较 `direct + exchange`，或用一个对称 toy propagator 都可能漏掉
关键 mutation。

## 6. Checkerboard momentum projection and MPI

local contraction 与 phase 必须统一为：

```text
(e,t,z,y,xh), e = 2
```

projection 定义为：

```text
C(t,p) = sum_{e,z,y,xh} C_local(e,t,z,y,xh) phi_p(e,t,z,y,xh)
```

phase `phi_p` 不由本层构造。若约定为
`exp[-i p . (x - x0)]`，必须使用 global coordinates，并明确 checkerboard parity 到
global coordinate 的映射。

多 rank 归约 rules：

1. `spatial_comm` 只能连接相同 global timeslices 的 ranks；
2. collective 必须是 mpi4py-style buffer `Allreduce(send, recv)`；
3. CuPy buffer 只有在目标 MPI stack 已直接证明 CUDA-aware 后才允许
   `cuda_aware_mpi=True`；
4. `cuda_aware_mpi` 只接受 Python/NumPy Boolean；
5. 不提供 hidden host fallback，也不构造 subcommunicator 或 phase。

正确性的本地 oracle 可以验证 layout 与 reduction algebra，但不能升级成真实 MPI 或
CUDA-aware 证据。

## 7. Required provenance

每个增强算符产物至少记录：

```text
gamma_basis_id
gamma_t / gamma_parallel / gamma_5 / C definitions
boost_direction
momentum_sign_and_phase
projector_kind
diquark_kind
operator_normalization
propagator_source_sink_axes
source_coordinates
backend_and_precision
configuration_and_source_metadata
```

若后续声称 enhancement，还必须冻结 conventional/enhanced 对照的 configurations、
propagators、smearing、source count、precision、momentum、fit window、covariance rule
和 cost。

## 8. Evidence ceiling

来源快照包含 CPU/stub 级 gamma、Wick、layout 和 collective contract 证据，但这些证据
未在本任务中于 PyQCD 重跑，也不证明：

- target propagator 使用同一 gamma basis；
- CuPy、CUDA-aware MPI 或 multi-rank pipeline 可运行；
- interacting ensemble 上 overlap、plateau 或 SNR 改善；
- configuration I/O、solver、renormalization 或 fit pipeline 已闭合。

在这些证据出现前，只允许报告 operator algebra、shape、rejection gate 和 candidate
status，不得报告 physical enhancement。

## 9. Sources

- Read-only source snapshot:
  `skills/LQCD_Skills/skills/pyquda_enhanced_interpolator/`,
  commit `90bcbb8739c80b41a3965f41dab50bdfc9841b16`.
- arXiv:2606.02447v2, Sections 2.1-2.2 and Eqs. (2.6), (2.9)-(2.13):
  nucleon kernels and source/sink contractions.
- arXiv:2501.00729v2, page 2:
  projected-field pion identity and Euclidean `gamma_±` convention.
