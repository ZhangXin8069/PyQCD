---
name: pyqcd-distillation
description: Use when a PyQCD task defines, audits, or implements LapH distillation through Wick contractions, perambulators, vertices, GEVP, MPI partitioning, or lattice tensor I/O.
---

# pyqcd-distillation - LapH 蒸馏安全链

## 目的与边界

所有蒸馏任务按同一因果链处理：

`physical definition -> convention -> tensor formula -> numerical algorithm -> implementation`.

本技能负责把物理定义落成可审计的 LapH/Wick/perambulator/GEVP/MPI/I/O 契约。它不负责
替代 `pyqcd-propagator` 的 PyQUDA 求逆、`pyqcd-physics-spectrum` 的谱模型、
`pyqcd-statistics` 的通用重采样，或 `pyqcd-analysis` 的最终图表。

状态词只按下列含义使用：

- `implemented`：接口/模块存在。
- `tested`：有单元、oracle、回归或真实数据审计证据，限定到检查范围。
- `closed`：物理定义、约定、张量、算法、输入输出和验收门对声明任务完整闭合。
- `physical-validated`：真实物理结论经过独立检查并带有受控系统误差。

程序运行成功、形状正确或参考脚本可导入都不等于 `closed` 或
`physical-validated`。

## 按需参考

- 读 [references/chain-and-gates.md](references/chain-and-gates.md) 获取物理定义、
  约定、张量公式、算法、Wick、perambulator、GEVP、MPI、I/O 和验收门。
- 读 [references/implementation-status.md](references/implementation-status.md) 获取
  lqcddb 来源限制、PyQCD 当前覆盖、测试范围和未闭合项。

## 最小工作流

1. **写物理定义**：指定目标 correlator/矩阵元、粒子量子数、源汇、流插入、flavor、
   动量、connected/disconnected topology 和最终可观测量。定义不完整时不得先写
   einsum。
2. **固定约定**：记录 DR gamma 基、Euclidean 符号、APBC、源汇顺序、Fourier 符号、
   颜色/自旋归一化、投影、basis momentum `p_basis` 和独立顶点动量 `q_sink`。
3. **审计 LapH 数据**：检查 `E` 的 shape、空间展平顺序、Gram、压缩矩阵、随机噪声、
   gauge links 和 momentum phases。
4. **构造 perambulator**：先核对实际 reader/producer 的源汇、spin、eigenvector 轴，
   再作顺序传播子或三点收缩。不得靠注释猜 conjugation/transpose。
5. **构造 vertex 与 Wick 项**：显式写出 `V^dagger O V`、displaced/link 和 conjugate
   结构；每个 Wick term 保存 Fermi sign、flavor、时间标签、gamma 转置和 topology。
6. **生成并验证 contraction plan**：先在小确定性张量上对手算或直接
   `einsum/opt_einsum` 对照，再冻结生产 plan。动态等价合并默认关闭，直到 tensor
   identity 已证明。
7. **收缩与闭合指标**：确认每个重复指标恰好收缩两次，输出自旋/动量/link 指标由
   显式 projector 或 trace 闭合，不依赖广播或 `ravel()[:4]` 隐藏残轴。
8. **谱学交接**：若使用 GEVP，先验证 `C(t0)` Hermitian positive definite、复数
   非对角元保留、state tracking 和 `t0` 稳定性，再交给谱学技能解释。
9. **并行与落盘**：先 dry-run MPI plan，再运行；gather/reduce 在 gauge context 外，
   rank 0 发布 canonical 文件；保存形状、轴、dtype、backend、seed、命令和 hash。
10. **按声明范围验收**：分别报告 `implemented/tested/closed/physical-validated`，
    不把 2pt 或 3pt oracle 外推到 4pt 或真实物理结果。

## 验收门

1. **LapH**：独立计算 `max(abs(E^dagger E-I))`、残差和 rank；压缩后的 zero norm、
   线性相关、随机分布和 phase convention 必须检查。
2. **Gauge/smearing**：明确 raw、valence stout、operator-flow thin-link 和
   distillation stout 是不同对象。smearing 至少检查 identity、`rho=0` 不动点、
   finite、unitarity、determinant、covariance 和只更新预期方向。
3. **Perambulator**：源/汇时间、Dirac indices、eigenvector indices、conjugate/
   transpose 和 flavor 全部显式记录；顺序传播子用
   `gamma5 peram^dagger gamma5` 的小矩阵 oracle 验证。
4. **Vertex**：确认是 `V^dagger O V`、其 conjugate，还是 link/displaced 版本；
   momentum phase 分字段记录 `p_basis_sink/source` 与 `q_sink`。
5. **Wick**：显式提供 operator prefix，保存 Fermi sign、共轭算符系数、flavor balance、
   连通分量和 free indices；不得静默忽略未知 token。
6. **Contraction**：默认 `use_equivalence=False`；等价合并只有在 gamma tensors、
   time labels、vertices 和 connected topology 全部证明相同后启用。
7. **GEVP**：`C(t0)` 经记录条件化后 HPD；保留 complex Hermitian off-diagonal；
   用 `scipy.linalg.eigh(C(t),C(t0))` 或独立小矩阵核对；近简并用 overlap 追踪 state。
8. **MPI**：验证 rank grid、local time list、root 语义、size-one vs multi-rank shape、
   equal leading size 和 collective failure 同步退出。
9. **I/O**：canonical HDF5 保存 shape/axes/dtype/endian/units/hash/command；读写
   round-trip；禁止把自动 cwd fallback 当成可移植恢复策略。

## 当前实现边界

| 层 | PyQCD 当前状态 | 重要边界 |
|---|---|---|
| eigenvector/perambulator binary readers | implemented, tested | 固定历史格式，必须核对 actual producer |
| VdV/VVV、sequential perambulator | implemented, tested | 轴序和源汇方向需要显式 metadata |
| autowick/dynamic contraction | implemented, 部分 tested | PJN 3pt 有四项显式 oracle；PJNNJNP 4pt 无独立 12 图 oracle |
| GEVP complex Hermitian | implemented, tested | 来源 lqcddb 曾丢复部；PyQCD port 已修复该合同 |
| Stout port | implemented, tested 基本不变量 | 未证明历史生产等价的全部物理范围 |
| MPI meta-task pipeline | implemented, tested | lattice-domain `_mpi_transport` 仍是独立预留层 |
| legacy distillation regression | closed 于基线复现范围 | 不等于所有物理量 physical-validated |
| general physical validation | not established | 必须按具体 observable 和拓扑单独升级 |

详细限制见
[references/implementation-status.md](references/implementation-status.md)。

## 交接

- `pyqcd-analysis`：接收已闭合的自旋/投影 correlator、矩阵元或 ratio；不要在分析
  层修补拓扑、Fermi sign、轴或 basis。
- `pyqcd-statistics`：负责 jackknife/bootstrap、covariance、rank、GEVP convergence
  诊断和 fit status；本技能定义相关物理对象的 tensor identity。
- `pyqcd-physics-correlator`：负责物理定义、算符、Wick term、传播子清单和 einsum
  语义；本技能补 LapH 数据与算法实现门。
- `pyqcd-propagator`：负责 PyQUDA 求逆、源构造、sequential source 和 covDev；
  perambulator 身份与收缩由本技能核验。
- `pyqcd-infra`：负责 backend、HDF5/ASCII、VdV/VVV readers、MPI 调度、GPU 绑定和
  内存规划；本技能规定数据身份及 gather/write 顺序。
- `pyqcd-conventions`：统一 gamma、轴、单位、边界、basis phase 和状态元数据。

## 来源清单

- `skills/donghx/lqcddb/SKILL.md`
- `skills/donghx/lqcddb/references/code-audit.md`
- `skills/donghx/lqcddb/references/wick-and-diagrams.md`
- `skills/donghx/lqcddb/references/correlator-implementation.md`
- `skills/donghx/lqcddb/references/eigenvectors-and-gauge.md`
- `skills/donghx/lqcddb/references/statistics-and-spectroscopy.md`
- `skills/donghx/lqcddb/references/mpi-io-performance.md`
- `skills/sush/lqcddb/` 中对应镜像 references
- `skills/donghx/Markdown/flow_equation.md`
- `skills/donghx/Markdown/pol_2pt.md`
