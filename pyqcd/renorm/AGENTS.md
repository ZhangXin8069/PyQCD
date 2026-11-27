# AGENTS.md — pyqcd/renorm

★ 重整化核心链：自重整化 Z_R（`_zr.py`）、混合方案（`_hybrid.py`）、NLO 匹配（`_matching.py`）、连续极限外推（`_extrapolate.py`）、梯度流（`_gradient_flow.py`）、TMD 提取（`_tmd.py`/`_tmdextract.py`）。refer/zengch 逻辑移植 + 理论文档新写。
第三轮：修复 `_matching.C/C_gluon_ratio` 误植（matching_cc 三分区+5/6·Si 项）；
collinear 准 PDF sin 变换 `_tmdextract.quasi_pdf_gluon`；ZR 逐样本重拟合环
`_zr.fit_ZR_samples/summarize_ZR_samples`；协方差加权外推
`_extrapolate.fit_hR_PDF_extrap_boot`（Cholesky 白化+lstsq+逐样本带）。

dev8 对照轮（2026-08-24，对照 matching_new/fit_zr_new/fit_ratio）：修复耦合归一漏乘
4π 三处——`_matching.hR_PDF:81`、`_zr.Z_MS:31`、`_tmdextract.tmd_matching_hybrid:216`
（约定 `A_s ≡ α_s/(4π)`，用真耦合处须 ×4π，同 zengch 端 `alpha_s=A_s(mu)*4π`）；
`quasi_tmd_pdf` 逐列循环批量化为单次 GEMM（7.8–68x，逐位一致）。
求和规则断言改对称宽域 ±1.9（正半轴截断丢失 ξ<0 支路，+26% 不随 dx 收敛）。

## 两圈转换 / 线质量核 / 解析尾（吸收 refer/donghx 20261127）

| 模块 | 内容 | 顶层入口 |
|---|---|---|
| `_nnlo.py` | 局域归一化 `D_Γ=ζ_Γ/C_ψ` 严格展开（`V`/`T`）、胶子 `U/H/P` 端点严格逆、短距 `C₀` NLO/NNLO | `local_ratio_factor`、`gluon_endpoint_factor`、`short_distance_c0` |
| `_line_mass.py` | 树级流化传播子的线自能闭式 `F(x)`、长距展开余项 `R_K`、H3 对数差模型与有限段差分 | `line_mass`、`h3_log_difference`、`finite_segment_kernel` |
| `_tail.py` | LA/NLA 长距尾 `(C+B/λ)e^{−ηλ}` 的解析 Fourier（含 `E₁`）与截断误差诊断 | `analytic_tail_fourier`、`abrupt_cutoff_error` |

关键不变量（均由 `pyqcd/testing/_nnlo_line_mass_tail_contract.py` 断言）：
- `D_V^{NLO}=1+a_s(4L_f+10/3)`、`D_V^{NNLO}=…+a_s²(26L_f²+85.35918L_f+30.92223)`，
  `L_f²` 系数 `26=(f₁₁²+β₀f₁₁)/2` 是独立 RG 检查；
- 两圈 ringed 常数在 `J₂−N₂` 中必须相消（传任意常数结果不变）；
- `C₀^{NNLO}` 前因子是 `α_s²/(2π²)`，不是 `(α_s/2π)²`（差 2 倍，易错点）；
- `F(0)=0`、`F(∞)=1`、`F(x)=1−3/(2√πx)+O(x⁻³)`；小 `x` 必须走 Taylor 分支，
  直接求值在 `x≲3e-3` 损失超过 1e-11 有效位；
- 解析尾与直接求积在 1e-16 内一致，且尾闭式把 `hR_x` 的 `λ_max` 截断误差变成
  可测量的 `abrupt_cutoff_error`。

来源：`refer/donghx/_Gradient_Flow_GluonPDF/{NNLO_note,METHOD,THEORY}.zh-CN.md`、
`refer/donghx/_codex/Markdown/{linear_devergence,baryon_da,gradient_flow_quark_pdf}.md`；
文献 arXiv:2610.40225v1 Eq.(32)(34)(40)、arXiv:2311.16799v2 Eq.(3.9)(3.11)、
arXiv:2006.12370v3 Eq.(22)–(25)。代码为独立实现，不 import refer。
