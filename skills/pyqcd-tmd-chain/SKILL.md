---
name: pyqcd-tmd-chain
description: |
  PyQCD 核心物理链技能：梯度流重整化方案下核子胶子 TMD-PDF 全链计算——
  ①Wilson flow 梯度流（RK3、τ=3a²）→ ②Clover/对偶场强胶子 TMD 算符 O 组合
  → ③不相连比值 c0 → ④自重整化 Z_R 与混合方案（λ 外推）→ ⑤准 TMD-PDF
  cos/sin 双通道 + NLO 匹配（Z_ij/CS 核/SFTX）→ ⑥连续极限外推。
  触发于："梯度流"、"胶子 TMD"、"TMD-PDF"、"Z_R 自重整化"、"匹配核"、
  "准 PDF"、"CS 核"、"连续极限外推"、"跑 test9"。
metadata:
  openclaw:
    emoji: 🌊
---

# pyqcd-tmd-chain — 梯度流胶子 TMD-PDF 物理链

## 目的与边界

仓库核心目标：**计算使用梯度流重整化方案的核子中的胶子 TMD-PDF**。
本技能串联 `pyqcd/renorm` 六步物理链，给出每步的入口 API、输入输出与
物理校验不变量。全链示例：`python examples/pyqcd/tmd_gradient_flow_demo.py`；
实战：`examples/pyqcd/test9_gluon_tmd_nucleon.py`；自洽断言：
`test9_verify.py`（A 流方程/E 递减/unitarity、B 2pt 谱线、C TMD OPE、D 分析、E PDF）。

## 六步链与 API 地图

### ① 梯度流 — `pyqcd/renorm/_gradient_flow.py`

Wilson flow（Lüscher 2010），**RK3 积分**，τ=3a² 方案
（Monahan–Orginos 2017 / NieMiera 2025）。流时间场强平滑紫外涨落，
为 OPE 算符提供重整化方案。
校验：流能量密度 E(τ) 随 τ **单调递减**；unitarity 保持。

### ② 胶子 TMD 算符 — `pyqcd/operator/_gluon_ope.py` + `renorm/_tmd.py`

Clover 场强 $F_{\mu\nu}$、对偶 $\tilde F$、staple Wilson 线，组合

$$O = M^{tx;tx}+M^{ty;ty}-2M^{xy;xy}.$$

扩展变体：`gluon_ope_operator_z0(mu2,nu2,direction)` 支持 −z Wilson 线；
固定规范 FF 算符 `gluon_ff_operator_z0`（无 Wilson 线，±z、交叉 μ₂ν₂ 对）；
Lorentz 指派表 `get_ope_lorentz_pairs`（unpol/helicity/gauge_fix×2）。
螺旋度 ΔG 双场强算符 F·W†·F̃·W：`operator/_helicity.py`（±z 支）。
宇称投影与反周期边界：`contraction/_baroperator.parity_and_boundary`
（P±=½(γ₀±γ₄)；pp: t_sink<t_src；pm: 反号）。
张量布局 gauge `(Nt,Nz,Ny,Nx,4,3,3)`；γ 用 DeGrand-Rossi 基。

### ③ 不相连比值 c0 — `pyqcd/analysis/_tmd_ratio.py`

disconnected 通道 OPE/2pt 真空扣除比值 → plateau 均值：
`plateau_c0(ratio, dt_max=20, dt_start=7, dt_end=10, cut=6)`（抗奇异协方差）；
顶层 `run_disconnected_tmd_ratio(corr_2pt_all, ope_all, conf_ids, ...)`
直接产出 c0_plateau。
校验：c0(z≤4)≈0±0.03 属小统计正常预期（dev7 实测），非 bug。

### ④ 自重整化 Z_R 与混合 — `renorm/_zr.py` + `_hybrid.py`

- Z_R 参数化全局拟合（arXiv:2510.17758 Eq.3-8）：`fit_ZR(par_ini, datasets, mu_)`；
  hB 数据 z₀ 归一化+插值 loader：`build_hB_dataset(c0_zx, z_fm)`、
  boot 协方差 `boot_covariance(samples, n_rep=200, seed=0)`；
  参数误差逐样本环：`fit_ZR_samples` / `summarize_ZR_samples`
  （单坏样本 NaN 不中断）。
- 混合方案：短距比值 + 长距 Z_R，λ 外推
  `fit_hR_lambda(par_ini, lambda_range, lamb, hR_data, cov_kind='boot')`
  （boot 全协方差选项）；傅里叶 → 准 PDF。

### ⑤ 准 TMD-PDF 与 NLO 匹配 — `renorm/_tmdextract.py` + `_matching.py`

- cos 型准 TMD：`quasi_tmd_pdf(hR_z, z_grid, b_perp, pz_gev, ...)`；
- sin 型 collinear 准 PDF：`quasi_pdf_gluon(h_z, z_grid, pz_gev)`
  （g̃=(2Pz/x)∫h·sin(xPz z)，x→0 保护；与 cos 型互补）；
- CS 核两动量工程封装：`cs_kernel_two_momentum(c01, c02, pz1_gev, pz2_gev,
  z_ref=1, ...)`（z_ref+clamp）；
- TMD 混合方案单圈匹配 `tmd_matching_hybrid(...)`：Z_ij 矩阵结构
  （δ + α_sC_A/2π 核），复用 `_matching._matching_kernels`（A_s=α_s/4π，
  zengch 约定），快度演化 + 软函数；胶子单圈匹配核 g₀..g₃；
  collinear 比值 `C/C_gluon_ratio` 三分区+5/6·Si 项（对照 matching_cc.py 已修复版）。

### ⑥ 连续极限外推 — `renorm/_extrapolate.py`

a/Pz/mπ/L 联合外推：`fit_hR_PDF_extrap_boot(rows, x_grid, max_x=1.0, ...)`
——协方差加权（Cholesky 白化+lstsq，非正定回退单位阵）+ 逐样本误差带
（固定 lx/hx/bx/cx，仅 xg0/fx/dx/kx 自由；批量化优于逐样本 Minuit）。

### 成图 — `pyqcd/analysis/_tmd_ratio.py`

`plot_tmd_c0` / `plot_tmd_ratio` / `plot_tmd_pdf(x_grid, xg_quasi, xg_matched,
b_grid_fm, cs_kernel, ...)` 四件套（TMD-PDF 链标准图组）。

## 工作流程

1. 冒烟先行：`test9_gluon_tmd_nucleon.py --smoke`（1 组态 1 动量）验证链路通；
   分析复用已算数据用 `--only-plot --conf-ids ...`。
2. 逐步跑链并记录中间量（h5 落盘，见 pyqcd-infra）；每步过物理不变量再进下一步。
3. 断言门：`python examples/pyqcd/test9_verify.py [run_dir]` A–E 全 PASS 为通过标准。
4. 出图 + 中文报告遵循 pyqcd-docs；批量并行见 pyqcd-infra（GPU 绑定/显存公式）。

## 物理校验清单（声明完成前逐项核对）

| 不变量 | 判据 |
|---|---|
| 流能量 | E(τ) 单调递减；unitarity 保持 |
| 2pt 谱线 | P0 平台 ≈1.12 GeV 量级一致 |
| c0 | disconnected 小统计 ≈0±ε 正常；勿硬拟合 |
| 匹配 | α_s 幂次结构 δ+O(α_sC_A/2π)；快度依赖经 CS 核进入 |
| 外推 | 协方差白化后残差无结构；误差带覆盖逐样本分布 |

## 错误处理

| 场景 | 处理 |
|---|---|
| Z_R 单样本拟合失败 | fit_ZR_samples 环内跳过（NaN 不中断），汇总时报告坏样本数 |
| boot 协方差非正定 | Cholesky 失败回退单位阵白化（_extrapolate 内建） |
| sin 变换 x→0 除零 | quasi_pdf_gluon 内建保护，勿在外层重复处理 |
| GPU 显存不足 | 按 pyqcd-infra 公式 N·a=n·b 降批次或回 CPU torch |

## 与其他技能配合

- 上游规范场/流基础 → `pyqcd-gauge`；算符收缩推理 → `pyqcd-physics-correlator`；
- 统计拟合纪律 → `pyqcd-analysis`；管线批量 → `pyqcd-pipeline`；IO/后端 → `pyqcd-infra`；
- 结果报告 → `pyqcd-docs`。
