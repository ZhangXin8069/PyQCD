# 尺度窗口完整性核验

CPU 审计 3856234，2026-10-17T15:57:21.535489+08:00。

全部 1548 个候选均有索引图页；有限中心值 333（含隔离的 signed 诊断）；完整 bootstrap 146。
逐候选尺度条件、1000 replica 轴、误差政策、来源 hash、PDF 页码核验全部通过。

## 几个固定对照（不按 Q 或误差挑选）

### `d_Wilson_relative_fundamental_baseline_R8_T10_16_low08_d_only`

- tau=`[0.8, 1.0, 1.4, 1.8]`；状态=`identifiable`。
- 参数=`[0.4450345672308367]`；完整误差=`[0.0018094678707617545]`。
- bootstrap=`1000/1000`；chi2/dof=`163.00287342731986`；Q=`1.6171545839577115e-71`。

### `d_Wilson_relative_fundamental_baseline_R8_T10_16_low08_d_c_tau_difference`

- tau=`[0.8, 1.0, 1.4, 1.8]`；状态=`identifiable`。
- 参数=`[0.3463009066971638, -0.007900939695937028]`；完整误差=`[0.006042873594376889, 0.00046069919196554677]`。
- bootstrap=`1000/1000`；chi2/dof=`3.215066135605528`；Q=`0.07296324275802367`。

### `d_Wilson_relative_fundamental_baseline_R10_T10_16_low08_d_only`

- tau=`[0.8, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0]`；状态=`identifiable`。
- 参数=`[0.5980197956316141]`；完整误差=`[0.00244388369043091]`。
- bootstrap=`1000/1000`；chi2/dof=`467.5758096033916`；Q=`0.0`。

### `d_Wilson_Cornell_fundamental_baseline_R8_T10_16_low08_d_b`

- tau=`[0.8, 1.0, 1.4, 1.8]`；状态=`identifiable`。
- 参数=`[0.5845816245331359, -0.09169880323806145]`；完整误差=`[None, None]`。
- bootstrap=`990/1000`；chi2/dof=`0.0068876437223812405`；Q=`0.9931360217313994`。

## 结论边界

- 主尺度条件下可得到一些窗口依赖的有效 d；不能只看 Q 高或统计误差小来宣布物理通过。
- 更强分辨条件使可用 tau 与几何上限的交集变窄，甚至不足以辨识拟合；这正是单格距的限制。
- Polyakov 对数与 signed 原始拟合分开；后者可以输出诊断中心／profile，但不填补前者。
- 对照值与完整状态请看 `analysis/d_distance_comparison.csv` 和 `analysis/all_fit_results.csv`。
