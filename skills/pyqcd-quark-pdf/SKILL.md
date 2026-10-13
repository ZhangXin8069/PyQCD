---
name: pyqcd-quark-pdf
description: Use when a PyQCD task concerns connected-quark gradient-flow quasi-PDFs, four-dimensional gauge/fermion flow, ringed normalization, SFTE conversion, channel matching, or u-d flavor combinations.
---

# pyqcd-quark-pdf - 连通夸克梯度流准 PDF

## 目的与边界

本技能定义连通夸克梯度流准 PDF 的长期参考链：

`operator scheme -> 4D flow -> distillation C3/C2 -> bare fit -> ringed normalization -> SFTE -> channel-specific matching -> physical-PDF claim`

本技能负责方案、约定、数据身份和阶段升级门。它不实现通用传播子求解、通用
Wick 收缩、统计基元或图表工具，也不把有限流时间的连通矩阵元直接称为物理 PDF。
胶子 TMD-PDF、断连项和完整 flavor-singlet PDF 属于相邻技能或其他工作流。

状态词只按下列含义使用：

- `implemented`：接口或脚本存在，不表示数值正确。
- `tested`：有单元、契约、合成数据或受控回归证据，范围必须同时说明。
- `closed`：声明范围内的物理定义、约定、算法、输入输出和验收门已完整闭合。
- `physical-validated`：真实系综的物理结论有独立验证、受控系统误差和明确适用范围。

`tested`、`closed` 和 `physical-validated` 不能由文件存在或程序退出码自动升级。

## 按需参考

- 读 [references/chain-and-status.md](references/chain-and-status.md) 获取 4D 流方程、
  ringed 归一化、SFTE、channel matching 公式和逐阶段状态矩阵。
- 读 [references/data-and-validation.md](references/data-and-validation.md) 获取轴序、
  参考数据路径、生产门、验收门和状态措辞。

## 最小工作流

1. **固定物理对象**：记录 `Gamma`、极化通道、连通/断连范围、flavor、`Pz`、`z`、
   核子源汇、投影和最终要交付的物理量。默认先做连通 `u-d`，不要先宣称单味
   `u` 或 `d` 是完整 PDF。
2. **固定约定**：记录 DR gamma 基、Euclidean 符号、时间反周期边界、源汇顺序、
   Fourier 符号、颜色/自旋归一化、thin-link 与 smearing 类型。
3. **执行四维同步流**：从同一 thin-link gauge 轨迹同步演化 gauge field 与
   fermion field；使用耦合 RK3 的 `U0,U1,U2`，不能改成终点规范场上的 3D smearing
   或一次性 `exp(t Delta)`。
4. **构造流化端点**：计算 `K(t)V` 并保留完整 4D 信息；时间窗口裁剪只控制存储，
   不能用裁剪把 4D flow 偷换成 3D flow。多流时间 checkpoint 必须同时续接规范场
   和费米子场。
5. **形成 C3/C2**：在同组态、同方向、同 bootstrap replica 内先形成 ratio，再作
   VDV 偶奇组合和方向平均；检查 `tau`、`t_sep`、`delta_z`、动量和 mask 轴。
6. **拟合裸矩阵元**：把 `ratio_v4` 和共享索引交给 `pyqcd-bare-matrix` 的拟合契约。
   连接层不得修补上游 mask、fallback 或坏组态。
7. **施加 ringed normalization**：独立测量每个 ensemble/flow time 的 `Z_chi`。
   一个双局域算符乘一个 `Z_chi`，不是 `sqrt(Z_chi)` 或 `Z_chi^2`；不得用简并
   `u-d` 真空动能作分母。
8. **执行 SFTE**：使用 Brambilla-Wang 2024 修正的一圈系数、相同 `Z_chi`、line-mass
   因子和显式尺度窗口。不要与 hybrid/ratio 的同一线性发散减除重复叠加。
9. **执行 channel matching**：unpolarized、helicity、transversity 分别使用对应
   LaMET kernel；单态通道还要处理断连和 quark-gluon mixing，transversity 相对符号
   在外部定义核对前保持 `pending_external_physics_confirmation`。
10. **只按证据升级状态**：有限流裸量、ringed 量、MSbar bilocal 和 matched PDF
    必须是四个不同状态。

## 验收门

1. `B_mu(t)`、两个流化端点和 Wilson line 必须来自同一 thin-link 历史；拒绝
   HYP 后流化数据进入 thin-link matching。
2. 保存 `tau=t/a^2`、`epsilon`、步数、window、边界、源汇、动量、方向、投影、
   输入路径和软件版本。
3. 用 zero-flow、自由场热核、epsilon 减半、direct/restart、规范协变和
   `V^dag V` 检查流化实现。
4. `Z_chi` 必须有独立测量、原始噪声/组态身份和固定质量处方；模板 JSON 不能升级为
   production。
5. ringed 归一化与裸拟合必须使用同一组态顺序和同一 bootstrap index；每个 replica
   先平均动能再取倒数，再与该 replica 的裸矩阵元相乘。
6. SFTE 前保存并检查 `a/r_flow`、`r_flow/|z|`、`sqrt(t)Pz`、`t Lambda_QCD^2`
   窗口 mask；阈值全局固定，不能逐点挑最小误差。
7. 多 ensemble 比较必须固定物理流时间或流半径；相同 `t/a^2` 不是相同物理流时间。
8. 交付必须列出统计、拟合窗、流时间、格距、匹配尺度/阶数、有限体积、有限 `z`
   和 distillation 系统误差；缺一项就降低状态。

## 当前实现边界

| 层 | 参考工作流 | PyQCD 当前包 |
|---|---|---|
| 4D 同步 gauge/fermion flow + flowed vertex | implemented, tested | 未实现；现有流引擎是 Wilson 规范流 |
| finite-flow connected C3/C2 bare diagnostic | implemented, tested, 在该诊断范围内 closed | 只有通用蒸馏/拟合基础，不能冒充此链路 |
| ringed `Z_chi` | implemented, tested；正式物理验收仍为 false | 未实现 |
| SFTE | implemented, tested 接口；未 closed | 未实现 |
| channel matching / final PDF | diagnostic implemented；未 closed、未 physical-validated | 只有相邻的通用/胶子匹配代码，不可复用为夸克通道结论 |

详细状态矩阵和未实现项见
[references/chain-and-status.md](references/chain-and-status.md)。

## 交接

- `pyqcd-analysis`：接收 ratio、matrix element、窗口和图像输入；不要在这里重新定义
  4D flow、ringed 归一化或 matching。
- `pyqcd-statistics`：提供共享组态/bootstrap index、协方差、AIC/fallback 和误差传播；
  本技能只规定配对顺序与物理对象。
- `pyqcd-physics-correlator`：定义双局域算符、Wilson line、源汇投影和
  distillation 收缩指标。
- `pyqcd-propagator`：产生独立动能真空迹所需的求逆、checkpoint 和残差证据；API
  执行与设备布局由其负责。
- `pyqcd-infra`：负责 HDF5/二进制 I/O、后端精度、MPI/GPU 绑定和内存规划；全体积
  stencil 的多 rank 假设必须先验证。
- `pyqcd-conventions`：统一 gamma、轴、单位、边界、复数投影和证据状态。
- `pyqcd-tmd-chain` / `pyqcd-tmd-algorithm`：只有任务转向胶子 TMD、soft/rapidity 或
  完整 TMD 几何时使用。

## 来源清单

- `skills/donghx/gradient-flow-quark-pdf/SKILL.md`
- `skills/donghx/gradient-flow-quark-pdf/references/theory-and-matching.md`
- `skills/donghx/gradient-flow-quark-pdf/references/code-and-data.md`
- `skills/donghx/gradient-flow-quark-pdf/references/production-and-validation.md`
- `skills/donghx/quark-pdf-bare-matrix/references/physics-conventions.md`
- `skills/donghx/quark-pdf-bare-matrix/references/fit-selection.md`
- `skills/donghx/fit_bare/SKILL.md`
- `skills/donghx/Markdown/flow_equation.md`
- `skills/donghx/Markdown/gradient_flow.md` 第 12-15 节
