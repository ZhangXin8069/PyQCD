---
name: pyqcd-statistics
description: |
  Use when a PyQCD result needs jackknife or bootstrap resampling, correlated
  covariance fits, SVD conditioning, fit-window selection, ratio or
  Feynman–Hellmann statistics, effective-mass diagnostics, or GEVP validation;
  use pyqcd-pipeline for orchestration and pyqcd-analysis for product-specific
  outputs.
metadata:
  openclaw:
    emoji: 📊
---

# pyqcd-statistics — 重采样、协方差与拟合纪律

## 目的与边界

本技能只处理“样本如何变成带误差的物理量”：估计量、重采样、协方差、拟合诊断和
系统误差。它不决定算符或谱式；算符用 `pyqcd-physics-correlator`，谱模板用
`pyqcd-physics-spectrum`，具体产物链用 `pyqcd-analysis`。

## 统计契约

调用代码前先写清：`Ncfg`、样本索引、重采样方法与次数、中心估计量、误差因子、
协方差定义、随机种子、拟合窗口、参数化和 SVD cut。所有相关量（不同 `z/b/Pz/t_sep`
和方向）使用同一组态索引，不能分别重排后再拼协方差。

| 方法 | 契约 | 必查边界 |
|---|---|---|
| delete-one jackknife | `x̄_k=(Σx−x_k)/(N−1)`；误差按声明的 prefactor 计算 | `N>1`，样本顺序可追溯 |
| bootstrap | 保存每次重采样索引或 seed、输入样本轴、样本数和 resample size；大批量用可复现分块 | 不能把非标准默认值写成物理要求 |
| SEM | 空样本轴报错；单样本保持输出 shape 并返回 `NaN` | 不得用零误差伪装已有统计涨落 |
| 复数协方差 | 明确 Hermitian 协方差（残差共轭）或实/虚部堆叠协方差 | 不得直接把复数乘积误称通用协方差 |
| 相关拟合 | `FitParams.svdcut="auto"` 默认解析为正 `1e-12`；保留全协方差并执行可辨识门 | `None` 表示严格不调节，不能静默改成伪逆或自动正则 |

## 协方差秩与可辨识门

`FitParams.svdcut` 的默认值是字符串 `"auto"`，进入 `gvar/lsqfit` 前解析为
`1e-12`。调用者显式传 `None` 时严格关闭 SVD 调节；若可辨识门已通过但 covariance
仍含零模，必须在调用 `lsqfit` 前报错，不得暗中恢复默认 cut。模型 Jacobian、有限输入、
prior 状态和可执行 contract 命令的完整规则见
[`references/identifiability.md`](references/identifiability.md)。

`covariance_sample_rank` 先在未调节的相关矩阵上计算原始样本信息秩；
`covariance_effective_rank` 则返回当前 `gvar.regulate` / `lsqfit` 的数据自由度。
`svdcut=None` 时二者相同；正 `svdcut` 抬升小相关本征值而不删模，负 `svdcut` 才删除
小模。因此调节后的 `effective_rank` 不能替代原始 `sample_rank`。
covariance 的舍入修复只可把数值负小模投影到零；已知精确核/零模必须保持为零，不能
为了让矩阵正定而抬升并伪造样本信息。

通用 `FitParams`/`lsqfit` 适配器的无-prior 拟合必须同时满足 `Ndata > Nparam`、
`effective_rank > Nparam` 和 `sample_rank >= Nparam`，其中 `Nparam` 是自由参数数。
这个正自由度门是适配器的报告契约，不是普遍的可辨识定理；专用
`fit_dispersion` 的满秩三点拟合语义见 `pyqcd-physics-spectrum`。`cond=inf` 或协方差
奇异本身不自动判失败，但不能用 bootstrap 扩增、协方差对角化或重复抽样制造原始信息。

秩不足时上层 ratio、energy、disconnected 和 TMD ratio 必须跳过 `lsqfit`，保存 NaN 与
`statistically_unidentifiable` 状态；TMD 拟合和 `c0` plateau 状态分别落盘。模型
Jacobian、有限性、prior 字段、状态解释和验证命令统一见上述 reference。

有效秩只回答“数据是否可能约束这些自由参数”。通过秩门后的低 Q、大 `chi2/dof`、先验
主导和窗口不稳定仍是彼此独立的拟合质量诊断。

## 非线性多态拟合的状态分层

对两态及以上的非线性模型，交付采用两层状态：第一层是
`data_identifiability`，第二层分别记录 `fit_quality_status` 与
`physical_result_status`。在有限性和形状守卫通过后，`data_identifiability` 只由已声明
的 sample/effective/model Jacobian 秩门与 prior 门决定：无 prior 且所需秩门通过才可写
`True`/`identifiable`；使用 prior 必须写 `None`/`prior_constrained`；秩门失败写
`False`/`statistically_unidentifiable`。条件数、`chi2`、参数误差、拟合窗口或重采样
漂移不能改变这一字段。

`fit_quality_status` 必须独立反映 GOF、残差/pull、posterior-prior 收缩和有限率等诊断；
`physical_result_status` 必须独立反映谱式、边界/约束、窗口和重采样稳定性对目标物理
主张的支持。满秩或优化器收敛不得自动升级任一状态；证据不足时保留
`not_assessed`/`unverified`。条件数必须注明对应矩阵、形状、参数/特征顺序、白化/
列缩放和 SVD 状态，不能把孤立的 `cond` 当成结论。每个参数、窗口和重采样方案还要
记录 finite/usable 计数与比例及稳定性摘要，细则见
[`references/identifiability.md`](references/identifiability.md)。

AICc、Fisher 信息、profile likelihood 和重采样稳定性在本技能中都是独立诊断，不是带有
通用数值阈值的硬门；若项目另有验收规则，必须连同规则和结果显式记录，不能反向覆盖
`data_identifiability`，也不能由 rank 通过推导“物理结果可靠”。

## 推荐顺序

1. 对齐组态、方向和时间轴，检查缺失值、无效端点与复数投影；无效点应 mask，不能
   用零填充后当作物理数据。
2. 按契约生成 jackknife/bootstrap 样本，立即保存中间统计量与索引元数据。
   公开入口为 `from pyqcd.analysis import sem, resample`。`resample` 的输入样本轴可由
   `axis` 指定，输出重采样轴统一在 axis 0；详细的
   `seed/rng/chunk_size`、dtype 与后端语义见
   [`identifiability.md`](references/identifiability.md)。
3. 由样本估计均值和协方差，分别计算未调节 `sample_rank` 与遵循 `gvar` SVD 语义的
   `effective_rank` 后执行可辨识门；模型近简并、有限性或 prior 情形转读
   [`identifiability.md`](references/identifiability.md)。
4. 先扫描拟合窗口，再看 `χ²/dof`、`Q`、参数 pull、posterior/prior 收缩、能级顺序
   和相邻窗口稳定性；“能拟合”不等于“数据约束了参数”。
5. 对多个窗口、方向、动量和重采样方式做交叉检查，分开报告统计误差与系统漂移。

## PyQCD 对象的专门检查

- 有效质量只能使用两端数据都存在、有限且满足边界条件的时间点；介子用 log/cosh，
  重子按 forward/backward 反宇称结构处理，具体谱式由 `pyqcd-physics-spectrum` 给出。
- 三点 ratio 先显式对齐 `tau` 与 `t_sink` 轴，再处理根号因子和归一化；不同轴位置
  不能依赖广播“碰巧正确”。复杂 ratio 取实部必须有对称性依据。
- FH 常数窗、`c0` plateau 和逐 `z` 拟合共享组态索引；小样本奇异协方差应报告，
  不能静默换成无关拟合。专用 `fit_constant_window` 使用未调节的普通全协方差逆，
  因而要求 `sample_rank == effective_rank == n_tsep`；低秩时返回 NaN 与
  `statistically_unidentifiable`，自适应窗不得再用缺失的 `chi2` 滑窗。该严格门不应
  外推为所有单参数拟合的通用定理，具体状态与绘图规则见
  [`identifiability.md`](references/identifiability.md)。
- 质子能谱的一态/两态 AICc 只做模型比较。一态胜出时 `c0/E0/chi2` 可按各自状态保留，
  `c1/dE` 必须为 NaN 且激发态标为 `practically_unidentifiable`；不得用 AICc 偏好升级
  拟合质量或物理结论，落盘与色带门见上述 reference。
- GEVP 要求 `C(t0)` 经记录的条件化后 Hermitian 正定；保留复数非对角元，并用
  `scipy.linalg.eigh(C(t), C(t0))` 或独立小矩阵核对。近简并时用跨时间本征向量重叠追踪态。

## 输出与交接

统计结果至少随 `Ncfg`、方法、seed、covariance/SVD、样本/有效/所需秩、窗口和拟合诊断
落盘；状态字段与 contract 命令见
[`references/identifiability.md`](references/identifiability.md)。交给 `pyqcd-analysis`
时同时给出可画图的中心值/误差、未通过窗口和不可辨识状态；交给
`pyqcd-tmd-algorithm` 时还需给出 `z/b/Pz/tau/±z` 的共享索引证明。

如果 posterior 仍接近 prior、窗口间漂移大或结果依赖单个坏样本，状态降级为“测试通过”
或“未验证”，不要把它写成真实物理结论。
