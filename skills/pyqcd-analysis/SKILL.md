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

## 按需参考

- 逐组态时间平移、APBC 边界符号、介子 fold/重子不 fold 和有效质量定义：
  [`references/preprocessing.md`](references/preprocessing.md)。
- `ratio_3pt` 已实现入口、summation 与 C2+C3 联合拟合的真实状态：
  [`references/three-point-methods.md`](references/three-point-methods.md)。

## 模块地图

| 需求 | 入口 | 回归入口 |
|---|---|---|
| 图表工具 | `pyqcd.analysis._plots` | 各分析套件共用 |
| fit/χ²/dof/ASCII 表 | `pyqcd.analysis._fitter` | 各分析套件共用 |
| 色散检验/拟合 | `_dispersion.dispersion_check`、`fit_dispersion` | `python -B -m pyqcd.testing._dispersion_identifiability_contract` |
| sem/resample/covariance | `pyqcd.analysis._disconnected` | `pyqcd-statistics` 规定契约 |
| 3pt/2pt 真空扣除 + 逐 z 拟合 | `_ratio2pt.run_ratio2pt` | `bash data/from-logs/logs/test0_ratio/run-local.sh` |
| 单 fit/对比/nofit 图 | `_ana_ratio.ana_ratio_plot_all` | `bash data/from-logs/logs/test0_anaratio/run-local.sh`（25 项） |
| 三方向裸矩阵元 | `_bare_matrix.run_bare_matrix` | `bash data/from-logs/logs/test0_bare/run-local.sh` |
| 有效能量 E0 | `_proton_energy.run_energy` | `bash data/from-logs/logs/test0_energy/run-local.sh`（8 项）/ test6 |
| FH 变换与常数窗 | `_fh.run_fh` | `bash data/from-logs/logs/test0_fh/run-local.sh`（38 项） |
| 三方向差异与直方图 | `_ana_3dir.analyze_3dir` | `bash data/from-logs/logs/test0/run-local.sh` |
| disconnected TMD/c0 | `_tmd_ratio.run_disconnected_tmd_ratio`、`plateau_c0`、`plot_tmd_*` | `python -m pyqcd.testing.tmd.pdf_workflow --only-plot --conf-ids 6250` |
| 有限横距 ratio-v2 | `_gradient_flow_ratio.calculate_gradient_flow_tmd_ratios` | `python -B -m pyqcd.testing._gradient_flow_gluon_contract` |
| 跨组态有限横距 ratio-v2 | `_gradient_flow_ratio.calculate_gradient_flow_tmd_ratios_from_configurations` | 同上 |
| 双极化 source-resolved C2 | `_tmd9.compute_2pt_polarizations_multi`、`load_multi_2pt_polarizations` | `python -B -m pyqcd.testing._tmd_physical_contract` |
| 二维有限横距 ratio-v3 | `_gradient_flow_ratio.calculate_gradient_flow_tmd_ratios_2d` | `python -B -m pyqcd.testing._gradient_flow_gluon_contract` |
| 二维跨组态 ratio-v3 | `_gradient_flow_ratio.calculate_gradient_flow_tmd_ratios_2d_from_configurations` | 同上 |
| 有限横距输入审计 | `pyqcd.testing.tmd.audit_finite_b_inputs` | `python -B -m pyqcd.testing._tmd_physical_contract` |
| 连通 3pt/2pt ratio | `_correlators.run_3pt_ratio`、`_analyse.ratio_3pt` | `pyqcd.pipeline._steps.step_analysis` |
| 02_ratio 生产链（OPE/2pt） | `_ratio2pt.run_ratio2pt` | `bash data/from-logs/logs/test0_ratio/run-local.sh`（22 项） |

常用三方向数据结构：
`<root>/<conf>/Pz<Pz>/{x,y,z,ave}_dir/ratio.npy` 与 `corr2_{dir}.npy`。

## 分析数据流

1. **选择入口**：先按上表定位已有功能；只有没有对应入口时才写新分析脚本。
2. **整理时间轴**：按组态自身 `t_src` 逐组态平移，并显式处理跨 APBC 边界的符号；
   介子可逐组态折叠，重子 `P^+` 不折叠（backward 是反宇称伙伴）。详细契约见
   [`references/preprocessing.md`](references/preprocessing.md)。
3. **交给统计层**：按 `pyqcd-statistics` 对齐索引、生成重采样、估计协方差并记录
   `Ncfg/seed/SVD/window`；不要在图表脚本里悄悄改估计量。
4. **拟合与诊断**：读取 `pyqcd-physics-spectrum` 的模板，报告窗口、χ²/dof、Q、
   参数 pull、prior 主导和相邻窗口稳定性。
5. **结果与图**：中间量、JSON、图表进入 `v<YYYYMMDDHHMM>/`；统一使用 `_plots`，
   真实数据和合成数据分开标注。

## 物理专属要点

### 矩阵元

| 方法 | 当前状态与入口 |
|---|---|
| ratio plateau | **已实现**：`_analyse.ratio_3pt` 是数值基元，`_correlators.run_3pt_ratio` 与 `pipeline._steps.step_analysis` 是现有连通 3pt/2pt 入口；零动量弹性时根号因子为 1 |
| 02_ratio 生产链 | **已实现**：`_ratio2pt.run_ratio2pt` 完成 OPE/2pt 比值、逐 `z` 拟合与 `c0/chi2` 图 |
| summation | **仅模板，未落地**：没有 PyQCD 公共实现或生产入口；不得把公式写成现有 API |
| C₂+C₃ 联合二态 | **仅谱式模板，未落地**：共享 $E_n,Z_n$ 的定义见 `pyqcd-physics-spectrum`，PyQCD 尚无该联合拟合生产入口 |

实现状态、公式和入口的逐项说明见
[`references/three-point-methods.md`](references/three-point-methods.md)。

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

连通裸矩阵元的 physical `Tmin`、Ratio/Sumratio、AIC 与逐 replica fallback 契约转交
`pyqcd-bare-matrix`；连通夸克梯度流准 PDF 的 4D flow、ringed normalization 与
SFTE 转交 `pyqcd-quark-pdf`。
