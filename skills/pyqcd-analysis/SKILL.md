---
name: pyqcd-analysis
description: |
  Use when extracting masses, energies, matrix elements, c0, ratios, effective
  masses, dispersion relations, or plots from existing PyQCD ensemble data;
  use pyqcd-statistics for the resampling and covariance contract, and do not
  use this skill to generate propagators.
metadata:
  openclaw:
    emoji: 📈
---

# pyqcd-analysis — 数据分析与作图

## 目的与边界

输入盘上已有的组态级关联器/OPE 数据，输出质量、能量、矩阵元、`c0`、诊断表和图表。
本技能不生成规范场、传播子或物理算符；统计估计量、协方差和 SVD 纪律交给
`pyqcd-statistics`，谱式交给 `pyqcd-physics-spectrum`。

## 模块地图

| 需求 | 入口 | 回归入口 |
|---|---|---|
| 图表工具 | `pyqcd.analysis._plots` | 各分析套件共用 |
| fit/χ²/dof/ASCII 表 | `pyqcd.analysis._fitter` | 各分析套件共用 |
| 色散检验/拟合 | `_dispersion.dispersion_check`、`fit_dispersion` | `python -B -m pyqcd.testing._dispersion_identifiability_contract` |
| sem/resample/covariance | `pyqcd.analysis._disconnected` | `pyqcd-statistics` 规定契约 |
| 3pt/2pt 真空扣除 + 逐 z 拟合 | `_ratio2pt.run_ratio2pt` | `bash logs/test0_ratio/run-local.sh` |
| 单 fit/对比/nofit 图 | `_ana_ratio.ana_ratio_plot_all` | `bash logs/test0_anaratio/run-local.sh`（25 项） |
| 三方向裸矩阵元 | `_bare_matrix.run_bare_matrix` | `bash logs/test0_bare/run-local.sh` |
| 有效能量 E0 | `_proton_energy.run_energy` | `bash logs/test0_energy/run-local.sh`（8 项）/ test6 |
| FH 变换与常数窗 | `_fh.run_fh` | `bash logs/test0_fh/run-local.sh`（38 项） |
| 三方向差异与直方图 | `_ana_3dir.analyze_3dir` | `bash logs/test0/run-local.sh` |
| disconnected TMD/c0 | `_tmd_ratio.run_disconnected_tmd_ratio`、`plateau_c0`、`plot_tmd_*` | `python examples/pyqcd/test9_gluon_tmd_nucleon.py --only-plot --conf-ids 6250` |

常用三方向数据结构：
`<root>/<conf>/Pz<Pz>/{x,y,z,ave}_dir/ratio.npy` 与 `corr2_{dir}.npy`。

## 分析数据流

1. **选择入口**：先按上表定位已有功能；只有没有对应入口时才写新分析脚本。
2. **整理时间轴**：按组态自身 `t_src` 平移；介子可逐组态折叠，重子 `P^+` 不折叠
   （backward 是反宇称伙伴）。
3. **交给统计层**：按 `pyqcd-statistics` 对齐索引、生成重采样、估计协方差并记录
   `Ncfg/seed/SVD/window`；不要在图表脚本里悄悄改估计量。
4. **拟合与诊断**：读取 `pyqcd-physics-spectrum` 的模板，报告窗口、χ²/dof、Q、
   参数 pull、prior 主导和相邻窗口稳定性。
5. **结果与图**：中间量、JSON、图表进入 `v<YYYYMMDDHHMM>/`；统一使用 `_plots`，
   真实数据和合成数据分开标注。

## 物理专属要点

### 矩阵元

| 方法 | 目标与适用范围 |
|---|---|
| ratio plateau | $R=C_3/C_2$（零动量弹性时根号因子为 1）；快速首看或单一 `t_sep` |
| summation | $S(t_{sep})=\sum_\tau R$ 的斜率；多 `t_sep` 时减轻激发态污染 |
| C₂+C₃ 联合二态 | 共享 $E_n,Z_n$，以 $B_{nm}=Z_n M_{nm}Z_m$ 拟合；高精度场景 |

### 色散入口与真实数据先例

`fit_dispersion` 的模型、$E$ 空间似然、可辨识性、`dof` 与约束 covariance 以
[`pyqcd-physics-spectrum`](../pyqcd-physics-spectrum/SKILL.md) 的“色散拟合契约”为唯一
物理来源；动量和单位约定以 `pyqcd-conventions` 为准。本层只整理测得的能量及其误差/
协方差，调用 `_dispersion`，并原样报告诊断状态：两动量只使用 `dispersion_check`；
`dof=0` 时不输出 GOF；`covariance_valid=False` 时不画对称高斯误差带。

- P2 2pt 可带相位负号；能量拟合取 `abs(corr2)`，ratio 的负/负应相消。
- 若窗口内两通道同为负，先定全局 `sgn*C` 再拟合；不能把拟合盆地跳变当物理信号。
- 已验证参考：P0 质子 `meff≈1.12 GeV`，P2 约 `1.5–1.56 GeV`；不覆盖的新数据必须
  重新验证，不能套用基准。

## 验收与交接

合成数据应恢复解析 `meff/E0/c0`；真实数据应提供输入形状、组态数、窗口、统计证据、
输出清单和未覆盖项。TMD 结果还要交给 `pyqcd-tmd-chain` / `pyqcd-tmd-algorithm`，
报告成文交给 `pyqcd-docs`，批量运行交给 `pyqcd-pipeline`。
