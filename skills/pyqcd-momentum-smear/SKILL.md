---
name: pyqcd-momentum-smear
description: Use when a PyQCD task needs to design, implement, or audit link-phased Wuppertal momentum smearing for quark sources, sink propagators, or sequential sources, including source/sink mode ownership, Fourier-phase signs, one-smear enforcement, or global-coordinate MPI behavior.
metadata:
  short-description: Link-phased Wuppertal momentum smearing
---

# pyqcd-momentum-smear — 联动量涂抹

## 目的与边界

本技能定义 PyQCD 中 link-phased Wuppertal momentum smearing 的物理与数据契约：
从 source 端点开始，在每个 Wuppertal hop 中相化空间规范链接，必要时处理 sink 端点，
并在 sequential source 上严格执行一次且仅一次涂抹。输出是 quark source、propagator
或 sequential-line artifact，不是 hadron correlator、distillation basis 或拟合结果。

PyQCD 的 `pyqcd.vertex.apply_momentum_smearing` 只给已有蒸馏 basis 乘 U(1) 相位，
不能替代本技能定义的 gauge-link momentum smearing。需要真实求逆与 MPI 执行时转交
`pyqcd-propagator`；收缩、Wick 结构、$\gamma_5$ 与 overall sign 转交
`pyqcd-physics-correlator`；后端、HDF5、MPI 规划和产物发布转交 `pyqcd-infra`。

详细公式、符号链、轴序和元数据见
[`references/momentum-contract.md`](references/momentum-contract.md)。实现或审查
nonforward/sequential workflow 前必须读取该文件。

## 核心物理与数据契约

### 三种动量不可混用

固定并分别记录：

$$
k_{\rm smear},\qquad P_i,P_f,\qquad q=P_f-P_i.
$$

$k_{\rm smear}$ 只控制 quark wave packet 的中心；$P_i,P_f$ 是最终 Fourier 投影的
hadron momentum；$q$ 是 insertion transfer。常用 $k\simeq P/2$ 或 $P/3$
只能作为经验起点，必须通过 overlap 实测选择，不能写成 API 恒等式。

### 每一步都使用 phased links

对空间方向 $j=x,y,z$，令

$$
U_j(x)\longrightarrow e^{+2\pi i k_j/L_j^{\rm global}}U_j(x),
\qquad U_t(x)\longrightarrow U_t(x).
$$

然后执行 Wuppertal/Gaussian iteration。若 $\Phi^{(k)}$ 是单步算子，结果是
$(\Phi^{(k)})^Nq$。禁止先把普通 field 涂抹完、再整体乘 $e^{ikx}$，因为那与
带相位 hopping operator 不等价。相化必须拷贝 gauge，不能原地改写 raw gauge。

PyQUDA 参数化使用

$$
\alpha=\left(\frac{4N}{\rho^2}-6\right)^{-1}>0,
\qquad \rho>0,\quad N\in\mathbb Z_{>0},\quad \rho^2<\frac{2N}{3}.
$$

这里的 $\rho,N$ 合法域不等于对给定 ensemble 的最优参数。

### 轴序、坐标与全局 MPI 边界

- `k_mode=(kx,ky,kz)` 使用 PyQUDA 的 x/y/z 顺序；PyQCD basis momentum
  `(pz,py,px)` 使用 z/y/x 顺序。两者必须显式映射，不能只写“momentum”。
- PyQCD 全局规范场布局为 `(Nt,Nz,Ny,Nx,4,3,3)`，Lorentz 标签
  `0=x,1=y,2=z,3=t`。PyQUDA 局部 fermion 常为
  `(parity,Lt,Lz,Ly,Lx//2,spin,color)`。
- source 坐标若按 `(x,y,z,t)` 给出，必须与数组存储 `(t,z,y,x)` 分开记录。
- rank-local Fourier phase 使用 global coordinate
  `(x_global,y_global,z_global)=(x_0+x,y_0+y,z_0+z)` 和 global extents。
- 多 rank 时必须核对 `grid_coord`、global/local extents、precision、边界条件与 gauge
  revision；仅比较相同局部 shape 不足以证明分解相同。

### Fourier 与 sequential 相位

物理 $+P_f$ sink 投影使用

$$
\phi_{\rm sink}(P_f)=e^{-iP_f\cdot(x-x_0)}.
$$

若上游产生的是 $e^{+iP_f\cdot(x-x_0)}$，它只能作为 conjugate algebraic candidate。
只有下游 contraction 明确对 sequential line 做 dagger/conjugation 时才可采用；
`sequential12` 一类接口若只选取 sink time slice，不执行该 dagger。insertion 上的
$e^{\pm iq\cdot x}$ 由 contraction 层独立实现。任何符号链不闭合的场景必须停止，
不能靠“结果看起来正确”选择相位。

### 单次涂抹与 one-smear

S-to-P 只允许 source mode；S-to-S 分别记录 source 与 sink mode。sequential source
若已经完成 active endpoint 涂抹，后续 inversion 必须声明 `already_smeared=True`
且不再传 sink mode；若尚未涂抹，则声明 `already_smeared=False`、给出一个明确
mode，并在 inversion 前恰好应用一次。缺失、重复或模糊计数均 fail closed。

### Artifact ownership

momentum-smeared quark source/propagator/sequential line 与 blending 的
`(N,t,z,y,x,color)` Laplacian/stochastic basis 是不同 artifact。不得把完成涂抹的
propagator 直接作为 blending `eigvecs`。两者只能通过显式 basis-source callback、
Dirac solve 或 observable assembly 会合，并保留独立 artifact ID。

## 最小工作流

1. 冻结 ensemble、gauge checksum/revision、global/local shape、grid、boundary、
   precision、action/mass、source coordinate、$k_i/k_f$、$P_i/P_f/q$ 与 artifact ID。
2. 在任何 solve 前验证 mode、坐标、global offset、布局、solver 参数、涂抹参数和
   S-to-P/S-to-S/one-smear 契约；无效输入不能先消耗一次求逆。
3. 复制 raw gauge，构造只含空间相位的新 gauge；保留 raw gauge 供 Wilson line 或
   covDev 等用途。
4. 对每条 quark line 独立记录 phase ownership，再完成 source kernel、可选 sink
   endpoint 和 inversion。
5. 对 sequential source 走显式 one-smear 路径，并单独审计 downstream dagger 后才
   选择 phase candidate。
6. 保存每次 solve 的 setup/solve/total time、iterations、reported residual 和独立
   计算的 true residual；MPI 归约在 `useGauge` 上下文外执行。
7. 按 `pyqcd-physics-correlator` 的 einsum 与 $\gamma_5$ Hermiticity 契约收缩，
   把轴序、source/sink orientation、相位和整体符号一并交接。

## 验收门

- link-phase oracle：每个 Wuppertal step 都使用正确正负号，time links 保持为零相位。
- 坐标 oracle：global extent/offset 与 rank 分解改变时，相位仍按全局坐标一致。
- 单次涂抹门：已涂抹路径、未涂抹路径和双涂抹输入分别得到明确的接受或拒绝。
- Fourier/sequential 门：$P_f$ sink 与 conditional conjugate candidate 分开断言，
  并覆盖 insertion $q$ 符号。
- 求解器门：保存 true residual；每次 inversion 的标签和输出可一一追溯。
- 收缩门：$\gamma_5$ Hermiticity、source/sink orientation 与 overall sign 由独立
  reference 或 sign-flip control 验证。
- 产物门：shape、轴序、dtype/backend、gauge lineage、mode、phase、command 和版本
  可 round-trip；非 root rank 不写共享输出。
- 无运行时证据时只能给出静态、函数级或合成测试结论。`no-runtime-claim`：不得声称
  已在 GPU、MPI、QUDA 或真实 interacting ensemble 上完成验证。

## 与 PyQCD 技能的交接

| 交接对象 | 必须交付 |
|---|---|
| `pyqcd-propagator` | gauge/read layout、raw/smeared 对象、每条 line 的 mode、one-smear 状态、solver 参数与 residual |
| `pyqcd-physics-correlator` | operator、$P_i/P_f/q$、Fourier 符号、sequential dagger owner、spin/source/sink 轴序 |
| `pyqcd-infra` | backend/precision、global/local/grid、HDF5 shape、rank0 写入和 round-trip 证据 |
| `pyqcd-pipeline` | artifact ID、configuration ID、输入 checksum、任务/产物 ownership |
| `pyqcd-conventions` | 单位、$\gamma$ 基、边界条件、动量顺序和证据状态 |

## 来源与证据边界

改写来源：
`skills/LQCD_Skills/skills/pyquda_momentum_smear/`，来源快照提交
`90bcbb8739c80b41a3965f41dab50bdfc9841b16`。本文只改写物理、数据与验证契约，
不复制外部 scripts，也不把来源实现当作 PyQCD 生产 API。

主要物理依据为 arXiv:1602.05525 的 Eq. (24) 与 Eqs. (26)-(28)；
arXiv:2009.10691 仅作为 momentum-smeared distillation 的补充背景。
来源工作树中的局部/synthetic checks 不能外推为真实 GPU、MPI、QUDA 或完整
interacting 2pt/3pt sign closure。
