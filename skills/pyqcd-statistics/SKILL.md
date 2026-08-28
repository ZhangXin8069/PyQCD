---
name: pyqcd-statistics
description: |
  Use when a PyQCD result needs jackknife or bootstrap resampling, correlated
  covariance fits, SVD conditioning, fit-window selection, ratio or
  Feynman–Hellmann statistics, effective-mass diagnostics, or GEVP validation;
  use pyqcd-analysis for product-specific plotting and orchestration.
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
| bootstrap | 保存每次重采样索引或 seed、样本数和 resample size | 不能把非标准默认值写成物理要求 |
| 复数协方差 | 明确 Hermitian 协方差（残差共轭）或实/虚部堆叠协方差 | 不得直接把复数乘积误称通用协方差 |
| 相关拟合 | 保留全协方差，记录条件数与 `svdcut`；奇异时显式回退 | 回退方式、自由度和误差必须入报告 |

## 推荐顺序

1. 对齐组态、方向和时间轴，检查缺失值、无效端点与复数投影；无效点应 mask，不能
   用零填充后当作物理数据。
2. 按契约生成 jackknife/bootstrap 样本，立即保存中间统计量与索引元数据。
3. 由样本估计均值和协方差；`Ncfg` 不足以支持数据维度时启用有记录的 SVD cut 或
   对角回退。
4. 先扫描拟合窗口，再看 `χ²/dof`、`Q`、参数 pull、posterior/prior 收缩、能级顺序
   和相邻窗口稳定性；“能拟合”不等于“数据约束了参数”。
5. 对多个窗口、方向、动量和重采样方式做交叉检查，分开报告统计误差与系统漂移。

## PyQCD 对象的专门检查

- 有效质量只能使用两端数据都存在、有限且满足边界条件的时间点；介子用 log/cosh，
  重子按 forward/backward 反宇称结构处理，具体谱式由 `pyqcd-physics-spectrum` 给出。
- 三点 ratio 先显式对齐 `tau` 与 `t_sink` 轴，再处理根号因子和归一化；不同轴位置
  不能依赖广播“碰巧正确”。复杂 ratio 取实部必须有对称性依据。
- FH 常数窗、`c0` plateau 和逐 `z` 拟合共享组态索引；小样本奇异协方差应报告，
  不能静默换成无关拟合。
- GEVP 要求 `C(t0)` 经记录的条件化后 Hermitian 正定；保留复数非对角元，并用
  `scipy.linalg.eigh(C(t), C(t0))` 或独立小矩阵核对。近简并时用跨时间本征向量重叠追踪态。

## 输出与交接

统计结果必须随 `Ncfg`、方法、seed、covariance 约定、条件数、SVD cut、窗口、自由度、
`χ²/dof` 和 Q 值落盘。交给 `pyqcd-analysis` 时同时给出可画图的中心值/误差和未通过
的窗口；交给 `pyqcd-tmd-algorithm` 时还需给出 `z/b/Pz/tau/±z` 的共享索引证明。

如果 posterior 仍接近 prior、窗口间漂移大或结果依赖单个坏样本，状态降级为“测试通过”
或“未验证”，不要把它写成真实物理结论。
