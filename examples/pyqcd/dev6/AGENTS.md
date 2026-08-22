# AGENTS.md — examples/pyqcd/dev6

**dev6** —— 基于已有收缩结果（tag-test8 管线产物）补充 test0/test6 同类型图表。

## 输入（只读，不重算）

`${HOME}/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/conf<cid>/`，405 组态
（4050..24300 步 50），每 conf：`corr_pp_P0/P2_<cid>.npy`（质子 2pt，
P=(0,0,0)/(0,0,2)，(Nt=72,) float64）+ `VdV_mom/VVV_mom_<cid>.npy`
（顶点积，本任务不消费——目标图表类型均不需要）。

## 运行

```bash
python main.py --debug                 # 前 5 组态 → 0_debug/（冒烟）
python main.py                         # 全量 405 组态 → v<ts>/（CPU 分钟量级）
python verify_dev6.py <run_dir>        # 产物齐全性 + 形状 + 物理断言
```

## 输出（v<ts>/）

- test6 型 7 图：corr2_raw / eff_mass / sem_comparison / eff_mass_GeV /
  eff_mass_fit_dirs / meff_corr / meff_hist（通道 P0/P2 替代 x/y/z/ave）
- test0 型 2 图：correlators_all_channels / meff_all_channels（docker 风格 2×2；
  pion 面板注明数据缺失）
- 数据：corr2_{P0,P2}.npy + 1_fit_data.npz + 2_fit_report.txt +
  analysis_summary.json

## 关键约定与结论

- 统计/图表/A 型 meff 链全部复用 pyqcd.analysis；B 型拟合照抄 logs/test6
  （C=c0·e^{−E0t}(1+c1·e^{−dEt})，逐样本 lsqfit svdcut=1e-6，窗 [6,12]，
  unit=0.197/a GeV）。
- `ratio_3pt_all_channels.png` **跳过**：需连通 3pt/perambulators 数据，输入
  数据集中不存在（绝不虚构）；已在 summary.json 的 plots_expected_missing 注明。
- A/B 两独立方法 E0 互差 <150 MeV（verify 断言，实测 35/57 MeV，方法系统差量级）；
  P0 ≈1.1 GeV 质子质量量级（stab1 已验证结论）、P2 色散一致（p≈0.98 GeV）。
- pyqcd 最小修改：`pyqcd/analysis/_correlators.run_meff_jackknife` 对输入中
  缺失的通道跳过而非 KeyError（行为对完整数据不变）。
