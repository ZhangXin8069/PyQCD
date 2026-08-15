# AGENTS.md — pyqcd/analysis

Jackknife/Bootstrap/meff/ratio_3pt、disconnected/3pt 编排、色散拟合（`_dispersion.py`）、
c0 裸矩阵元提取（`_ratio_fit.py`，zengch 逻辑移植）、
三方向差异分析与作图（`_ana_3dir.py`，独立实现，功能对齐 refer/huangcl/05_ana_3dir_diff_sem）。

## _ana_3dir.py（输入数据路径 → 分析 → 作图）

- 顶层入口：`analyze_3dir(data_root, out_root, AnaParams, jackknife=False)`。
- 数据约定：`<data_root>/<conf>/Pz<Pz>/{x,y,z,ave}_dir/ratio.npy`（(Nsample,Ntsep,Ntins,Nz)）
  与 `corr2_{dir}.npy`（(Nsample,Ntsep)）。
- 分析：有效质量 `log(C(t)/C(t+1))`（np.roll）、归一化协方差（相关系数）、mean±sem；
  统计复用 `_disconnected.py` 的 `sem`/`cov_mat`。
- 作图：matplotlib 函数内延迟导入 + `use('Agg')`；输出
  `<out_root>/<conf>/{ratio,corr2,eff_mass}/hist_*.png` + `ana_3dir_summary.json`。
- 测试：`bash logs/test0/run-local.sh`（17 项断言：存在性/数值自洽/meff 恢复/相关矩阵性质）。
