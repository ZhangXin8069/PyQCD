# 3000 配对 bootstrap 的拟合结果与解释

日期：2026-10-18T09:03:54.609435+08:00。L32x64，1700 cfg；不是 L32x96。本版独立于旧的 1000 样本版本。

3000 组配置级 bootstrap 每组有放回抽1700次；seed=20261015，前1000组索引与旧版相同。
所有flow、距离、时间和表示共享索引。配置均值、原始JK协方差和中心值不因增加bootstrap而改变。
bootstrap误差采用ddof=0，不除sqrt(3000)。新旧中心、前1000组能量和参数样本均执行回归核验。

## 固定示例窗口

Wilson R/a=8、T/a=10..16、tau={0.8,1.0,1.4,1.8}，参考tau_r=0.8。
拟合 $a[V(\tau)-V(\tau_r)]/2=d[(8\tau)^{-1/2}-(8\tau_r)^{-1/2}]+c(\tau-\tau_r)$。

|表示|模型|d|1000样本误差|3000样本误差|chi2/dof|Q|有效bootstrap|
|---|---|---|---|---|---|---|---|
|fundamental|relative|0.4422048|0.001871117|0.001838091|143.0761|7.292223e-63|3000/3000|
|fundamental|relative_c|0.3480703|0.006155768|0.006114733|2.82188|0.09298742|3000/3000|
|adjoint|relative|1.059689|0.06603147|0.06557733|0.5500907|0.5768975|3000/3000|
|adjoint|relative_c|1.441427|0.5736708|0.5711568|0.6535705|0.418839|3000/3000|

## 完整误差覆盖

|方法|表示|候选数|有限中心|完整3000误差|
|---|---|---|---|---|
|Polyakov_pair_Cornell|fundamental|94|0|0|
|Polyakov_pair_Cornell|adjoint|94|0|0|
|Polyakov_pair_relative|fundamental|104|5|0|
|Polyakov_pair_relative|adjoint|104|4|0|
|Polyakov_single|fundamental|30|0|0|
|Polyakov_single|adjoint|30|0|0|
|Wilson_Cornell|fundamental|186|66|66|
|Wilson_Cornell|adjoint|186|2|0|
|Wilson_relative|fundamental|204|93|93|
|Wilson_relative|adjoint|204|42|35|

Polyakov独立signed诊断的覆盖（不是实对数自由能链）：

|方法|表示|候选数|有限中心|完整3000误差|
|---|---|---|---|---|
|Polyakov_single_signed|fundamental|15|11|0|
|Polyakov_single_signed|adjoint|15|14|0|
|Polyakov_pair_signed|fundamental|52|32|0|
|Polyakov_pair_signed|adjoint|52|24|0|

## 不可省略的解释

- 更多bootstrap仅改善误差估计的抽样稳定性，不增加1700 cfg的独立信息；不会让误差按sqrt(1000/3000)物理缩小。
- 3000/3000可用才有完整误差。某些1000样本中完整的窗口可能在新增2000组遇到失败；不能丢掉新增失败样本。
- no finite central/JK/positive dof 的失败仍保留，没有用prior、abs、clip或补样本强行提供数字。
- 不按chi2/Q/AIC切选。基本表示领先模型在固定示例的chi2很差，小统计误差不能说明模型合理；含c项只有1个自由度。
- 基本/伴随分别分析，d已含表示因子；不得把基本d乘Casimir自动作为伴随d。
- 单Polyakov只有一条长线；pair为两条，且是颜色平均有限温度自由能，不是零温singlet势。
- a*m_s=d_s/sqrt(8*tau)+...，反项delta_m=-m_s，减除exp(+m_s*L)。差分只能约束flow变化，不能定绝对有限质量常数。
- 当前仍是单格距、有限时间有效系数，模型/窗口/流半径/离散/跑动和自相关分箱系统误差未控制，没有自动应用于PDF。

## 图和数据

- `figures/01_Wilson_relative_flow_fits.pdf` / `01_Wilson_Cornell_flow_fits.pdf`
- `figures/02_d_window_stability.pdf` / `03_Polyakov_signal_and_d_diagnostics.pdf`
- `figures/05_Polyakov_d_flow_fits.pdf` / `06_Polyakov_signed_independent_diagnostics.pdf`
- `figures/07_bootstrap1000_vs_3000.pdf`
- `analysis/bootstrap1000_vs_3000.csv` / `d_results.csv` / `report.json`
- `analysis/bootstrap_indices.npz` / `loop_resamples.npz` / `fit_samples.npz`
- `analysis/validation.json` / `independent_validation.json`
