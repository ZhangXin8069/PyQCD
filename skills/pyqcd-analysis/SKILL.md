---
name: pyqcd-analysis
description: |
  PyQCD 数据分析与作图技能：从组态级关联器/OPE 数据提取物理结果并控统计与系统误差。
  覆盖：源时刻平移、关联器折叠、jackknife/bootstrap 重采样、gvar/lsqfit 协方差拟合、
  有效质量、多态关联拟合（强子谱）、矩阵元提取（比值法/求和法/C₂+C₃ 联合拟合）、
  色散关系与光速、拟合诊断（χ²/dof、Q 值、AIC、SVD cut）、物理单位换算，以及
  PyQCD 既有分析功能链（02_ratio → 03_ana_ratio → 03_bare_matrix → 04_energy
  → 06_FH → 05_ana_3dir）的入口与测试。触发于："分析关联器"、"拟合数据"、
  "提质量"、"有效质量"、"矩阵元"、"色散关系"、"光速"、"jackknife"、"重采样"、
  "lsqfit"、"ratio 拟合"、"画 ratio 图"，或收缩完成需要出结果时。
metadata:
  openclaw:
    emoji: 📈
---

# pyqcd-analysis — 数据分析与作图

## 目的与边界

输入**盘上已有**的组态级关联器/OPE 数据（h5/npy；单组态无物理意义，只有系综平均
带误差才有），产出质量/振幅/矩阵元/c0 等结果与图表。不产传播子计算代码。
优先复用 `pyqcd.analysis` 既有模块（下表），新分析代码按"扁平自包含脚本 +
物理操作内联可见"风格写（拟合模型函数除外——lsqfit 要求 callable）。

## 模块地图（已验证入口）

| 需求 | 入口 | 参考套件 |
|---|---|---|
| 图表工具（errbar/scatter/hist + 10 色） | `pyqcd.analysis._plots` | 各套件共用 |
| lsqfit 封装 fit / χ²/dof / ASCII 报告表 | `pyqcd.analysis._fitter` | 各套件共用 |
| 统计基元 sem/resample/cov_mat | `pyqcd.analysis._disconnected` | 各套件共用 |
| 3pt/2pt 真空扣除比值 + 逐 z 拟合 | `pyqcd.analysis._ratio2pt.run_ratio2pt(data_root, out_root, sampa, ...)` | `bash logs/test0_ratio/run-local.sh`（18 断言） |
| 纯画图（单 fit/对比/nofit） | `pyqcd.analysis._ana_ratio.ana_ratio_plot_all` | `bash logs/test0_anaratio/run-local.sh`（25 断言） |
| 三方向裸矩阵元 | `pyqcd.analysis._bare_matrix.run_bare_matrix(...)` | `bash logs/test0_bare/run-local.sh`（18 断言） |
| 有效能量 E0(GeV) | `pyqcd.analysis._proton_energy.run_energy(...)` | `bash logs/test0_energy/run-local.sh`（8 断言）/ test6 |
| FH 变换 + 常数窗拟合 | `pyqcd.analysis._fh.run_fh(...)`；`_ratio_fit.fit_constant_window` / `fh_adaptive_windows` | `bash logs/test0_fh/run-local.sh`（38 断言） |
| 三方向差异分析+直方图 | `pyqcd.analysis._ana_3dir.analyze_3dir(data_root, out_root, AnaParams)` | `bash logs/test0/run-local.sh` |
| disconnected TMD 比值/c0 | `pyqcd.analysis._tmd_ratio.run_disconnected_tmd_ratio / plateau_c0 / plot_tmd_*` | examples/pyqcd/test9 |

数据结构约定：`<root>/<conf>/Pz<Pz>/{x,y,z,ave}_dir/ratio.npy` + `corr2_{dir}.npy`。

## 方法论要点（拟合纪律）

### 预处理顺序（不可颠倒）

1. **先平移后折叠**：各组态按自身 t_src 平移到 t_src=0
   （`np.roll(C, -t_src, axis=-1)`；未吸收边界相位时 roll 过界段乘 −1，
   用单组态 C(T−1) 核实是否需修正）；再逐组态折叠
   （介子 $C_\text{fold}(t)=\frac{C(t)+C(T-t)}2$）。**重子 P⁺ 投影不得折叠**
   （backward 是另一粒子）。
2. **gvar 数据集**：`gv.dataset.avg_data(C_all)`（保留全协方差）；或 jackknife
   显式构造（cov = (n−1)·样本协方差 ddof=0）。两条工作流：
   A) gvar 原生贯穿（误差与关联自动传播，首选）；
   B) jackknife 逐样本重拟合（派生量无法 gvar 表达时；每次拟合用全样本协方差、
   仅均值变）。
3. **小样本协方差奇异**：Ncfg≲Ndata 必加 SVD cut。PyQCD 实战约定
   **svdcut=1e-6**（10 组态基线）；诊断用 `gv.dataset.svd_diagnosis`。
   相关 vs 无关拟合互查：中心值一致而误差不同 → 相关拟合可靠。

### 拟合模板与先验

- 能量差参数化 $E_n=\sum_{k\le n}\Delta E_k$，先验键写 `'log(dE0)'` 等强制 ΔE>0
  （防态序翻转）；振幅先验宽 gvar(0,10)；由有效质量平台估 m_est 定 dE0 先验。
- **t_min 扫描选窗**判据（依序）：① Q>0.05；② m(t_min) 相邻稳定
  （|Δm|<σ_m）；③ 取最小合格 t_min（最大化数据使用）；④ AIC 交叉核对。
- **拟合诊断表**（每次拟合全查）：χ²/dof∈[0.5,2]；Q>0.05；
  posterior 比 prior 窄（prior 主导 = 数据无约束力，dev7 小样本先例）；
  E₁>E₀；AIC 低者优。ASCII 报告表用 `_fitter`。

### 矩阵元三法（C₃(τ,t_sep)，目标 M₀₀）

| 方法 | 要点 | 适用 |
|---|---|---|
| 比值法 | $R=C_3/C_2^\text{snk}\times\sqrt{\cdot}$（零动量弹性根号=1 → R=C₃/C₂），τ 平台即 M；plateau 加权平均 `lsqfit.wavg` | 快速首看/单 t_sep |
| 求和法 | $S(t_\text{sep})=\sum_\tau R$ 对 t_sep 线性拟合取斜率；激发态压低多一阶 | 多 t_sep、中等统计 |
| C₂+C₃ 联合二态拟合 | 共享 E_n/Z_n，B_nm=Z_n·M_nm·Z_m 显式参量化 | 高精度、激发态重要 |

Lorentz/Dirac 结构不影响拟合流程——运动学分解与算符重整化 Z_J 在提取裸 M 之后做。

### 色散关系与单位

格点动量 $\hat p_i=2\sin(\pi n_i/L)$；拟合 $E^2=m^2+c^2\hat p^2$
（c≈1 检验离散效应；多动量能量须同数据集 gvar 保跨动量关联）。
单位换算 $m[\mathrm{MeV}]=m_\text{lat}\times197.3269804/a[\mathrm{fm}]$
（PyQCD 惯用 unit=0.197/a GeV·fm）；a 的不确定度经 gvar 全传播。

## PyQCD 实战约定（真实数据教训）

- **P2 相位负号**（logs/stab1）：P2 通道 2pt 带 phase 负号 → ratio 负/负相消
  自洽；能量提取取 |corr2|。corr_pp (Nt,) 先转平移不变切片矩阵 (Nt,Nt)
  （C[sink,src]=C((sink−src) mod Nt)，dev7 staging 同式）。
- **全局符号 sgn·C**（dev6）：窗口内两通道 C<0（相位残留 π）时先定全局符号
  再拟合，消除拟合盆地歧义；B 型 4 参数形状模型逐样本 lsqfit svdcut=1e-6。
- **窗口参考**：dev6/dev7 用窗 [6,12]；02_ratio 三代表窗 (6,11,2)/(7,11,3)/(9,11,4)
  @dt_max=20。
- **缺失守卫**：通道数据缺失自动跳过不崩溃（`run_meff_jackknife` 守卫先例）；
  c0(z≤4)≈0±0.03 属 disconnected 通道统计不足的正常预期，勿当 bug 修。
- **版本目录**：产物一律 `v<YYYYMMDDHHMM>/`（test12 约定）+ summary/verify JSON。
- 物理断言基准（勿改动已验证结论）：P0 meff 平台 ≈1.12 GeV（质子）、
  P2 ≈1.5–1.56 GeV（色散）、E0 与 meff 平台一致。

## 工作流程

1. 判定需求归属：功能链已有模块（查上表）→ 直接调用入口 + 跑对应 run-local.sh 回归；
   新分析 → 按"预处理→重采样→有效质量→t_min 扫描→协方差拟合(SVD cut)→诊断"
   顺序写扁平脚本，参数硬编码顶部。
2. 数据落盘：中间量/图/JSON 进 `v<ts>/`；图用 `_plots` 工具统一风格。
3. 验证：合成数据须能精确恢复解析真值（meff/E0/c0）；真实数据对照
   AGENTS.md 已验证物理断言；跑相关测试套件断言门（exit 0/1）。
4. 结果入中文 LaTeX 报告时遵循 pyqcd-docs 约定。

## 错误处理

| 场景 | 处理 |
|---|---|
| 协方差奇异/病态 | SVD cut（svdcut=1e-6 起）或对角协方差回退 |
| 无 t_min 过 Q>0.05 | 加 SVD cut；仍败则查折叠/符号/窗 |
| 拟合盆地跳变（符号翻转） | 全局 sgn·C 约定后再拟合 |
| prior 主导（pull 大 shrink≈1） | 如实报告统计不足，不加窗硬拟合 |
| 输入形状不符 | 先查切片矩阵约定 (sink,src) 与 dir 动量置换 |

## 与其他技能配合

- 拟合模板物理来源 → `pyqcd-physics-spectrum`；TMD 物理链产出（c0/hR/Z_R）分析
  → `pyqcd-tmd-chain`；管线化批量运行 → `pyqcd-pipeline`；IO/后端 → `pyqcd-infra`；
- 报告成文 → `pyqcd-docs`。
