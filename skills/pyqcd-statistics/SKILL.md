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

## 配对比较、空集合与检验功效

三个容易把「没有证据」写成「证据表明没有」的陷阱，适用于所有共享组态的分析：

1. **配对比较**：两个观测量共用同一批组态时，其涨落高度相关。必须对**差值**
   做 delete-one jackknife，`σ_Δ² = (n-1)/n Σ_c (Δ_c - Δ̄)²`；分别求误差再线性
   相加会重复计入共模噪声，把真实差异淹没。
2. **检验功效**：`σ` 小只说明差值落在误差内，不说明差值为零。报告必须同时给出
   中位 `|Δ|/|参考值|`。若该比值 ≳0.5 而中位 `σ` ≲1，正确判语是「无判别力 /
   不可判定」，不是「一致」。
3. **空集合**：物理掩码（如流窗、有效插入窗）可能选不到任何格点。判定逻辑必须先
   显式处理空集合并返回「不可判定」；落入默认分支会把「没有数据」误报成
   「存在显著差异」。只在掩码内计算功效，不要用全体格点的统计代替。

另：`n` 个组态的 delete-one JK 给出 `n-1` 维协方差，秩上限为 `n-1`；多个
`(z,b,tsep)` 格点共享同一批组态，其 `σ≥2` 计数不是独立的假设检验，只能作为
整体趋势描述。

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

经验数字（2026-10-29，9 组态 delete-one JK）：两态联合拟合有 4 个 C2 参数
（$m_0,m_1,A_0,A_1$）加 3 个 C3 振幅，共 7 个自由参数，而协方差秩上限只有 8。
此时 `chi2/dof` 在 0.2–1.2 之间，只能说明「没有看到张力」；振幅误差由
少数组态的重采样涨落决定，`|B00|/sigma` 全部小于 2。**不要把低 `chi2/dof`
或优化器收敛写成模型验证**，也不要把「误差小」当成「值可靠」：
本轮的相对误差 `err/|B00|` 中位数为 2.14，10/12 个格点大于 1。

同一轮把 `tsep` 扩到 24 后又得到三条更硬的教训：

- **相关参数的显著性会互相交换。** 单参数 `B00 = 1.04e-2±2.3e-3`（4.5σ）
  与 `B01 = −2.20e-2` 同时出现。固定插入 τ 且除以 `exp(−m0·tsep)` 后，
  物理上可提取的组合量是 `P(τ) = B00 + B01·e^{−dm·τ}`（`dm = m1−m0`），
  其中 `B01` 是设计矩阵的 `left0*right1` 列，不是 `B10`。
  逐 delete-one 复制品算组合值得 `corr(B00,B01) = −0.76`、
  `P(1) = +3.2e-4 ± 2.9e-3`（0.11σ）。
  报单参数显著性等于报一个可以被另一个参数抵消的组合；**误差必须从同一批
  复制品的组合量求**，把两个参数误差平方相加会重复计入共模涨落。
- **低 `chi2/dof` 完全掩盖退化。** C2 单独两态拟合在 `t=5…16` 给出
  `A1/A0 = −1.2e5`（两个指数项近共线、质量不可单独识别），`chi2/dof` 却只有
  0.63。识别退化的判据是振幅比量级、参数之间是否近简并，而不是拟合优度。
- **窗口不是越多越好。** 尾部窗口 `[16,24]` 上 `B00` 的误差涨到 `1e-1`，
  参数被噪声推动；加入它只会污染「窗口稳定性」判断。只保留数据仍有
  信噪比的窗口，并单独标出被排除的窗口。

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

裸矩阵元的 physical `Tmin`、Sumratio-difference AIC、complete bootstrap coverage 与
quality/fallback 双 envelope 的专门契约转交 `pyqcd-bare-matrix`。
