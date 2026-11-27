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

## 生产分支接线（stab57）

四个入口新增可选参数，**默认路径逐位不变**（`stab56` 产物可直接复现）：

| 入口 | 新参数 | 行为 |
|---|---|---|
| `_matching.hR_PDF` | `alpha_s`、`return_metadata` | 可显式固定耦合（与 `tmd_matching_hybrid` 同语义），metadata 声明两圈核缺口 |
| `_tmdextract.tmd_matching_hybrid` | `matching_order` | 只接受 1；传 2 抛 `NotImplementedError` 而非静默退回一圈 |
| `_hybrid.hR_x` | `tail_model`、`tail_params`、`return_metadata` | `'none'`(默认)/`'power_exp'`/`'la_nla'`，把 `λ_max` 之后的尾部积到 ∞ |
| `_gradient_flow_quasi.match_one` | `endpoint_order` | `1`(默认) 重求和 / `2` 严格二圈级数逆 |

**两圈的三个真实入口**（不可互换、不可混用于同一张表）：
1. 端点：`match_one(endpoint_order=2)`；
2. 局域归一化：`_nnlo.local_ratio_factor(order=2)`；
3. 短距：`_nnlo.short_distance_c0(order=2)`。

**两圈 LaMET 匹配核 `g^(2)(ξ)` 不在本库**——已吸收的参考材料给的是有限转换系数
（`D_Γ`、`C₀`）与端点（`U/H/P`），不是动量空间匹配核。`hR_PDF` 与
`tmd_matching_hybrid` 都在 metadata 里写明 `two_loop_kernel="unavailable"` 及原因，
绝不以下述近似冒充：不得把 `(5/6)L` 型共线对数当成完整两圈核（曾试写过，已撤回）。

**尾部补全的量级**（`hR_x`，`z_max=1.0` fm、`Pz=4`、`λ_max≈1.27`、`λ₀=0.5`）：
`power_exp` 补上 `~3e-11`（相对主体 `2.7e-11`，可忽略）；`la_nla` 取
`(C,B,η)=(1,0.2,0.5)` 时补上 `~3.5e-3`（相对 `2.8e-3`，不可忽略）。两者差两个
量级，说明**尾部模型选择本身就是系统误差**，metadata 的 `tail_relative_to_cut`
就是给报告的该分量。

**`power_exp_tail_fourier` 的精度契约**（GL 节点，`n=88`）：`λ_L/λ₀ ≥ 1` 时
`~1e-6`，`0.33` 时 `~7e-7`，`0.2` 时 `~1e-4`，`0.1` 时 `~9e-4`；`check=True`
用 184 节点自洽复核，超过 `check_tolerance=1e-3` 抛 `ArithmeticError`。
`numpy.polynomial.laguerre.laggauss` 在 `n ≥ 192` 自身出现 0/0，故节点上限 96。

## 默认路径冻结（历史产物可复现，stab58）

`pyqcd.renorm` 四个入口的**默认路径是冻结的**，由
`pyqcd/testing/_renorm_default_golden_contract.py` 用 `np.array_equal`
（NaN 位置须一致）锁定 `stab56` 的全精度黄金值：

| 入口 | 冻结值来源 |
|---|---|
| `hR_x`(cos / hermitian / `tail_model='none'`) | 3 点全精度数组 |
| `hR_PDF`(默认 `alpha_s`) | 首尾 4+3 点 + 数组和 |
| `match_one`(`endpoint_order=1`) | 前 6 个实部 + `nansum` |
| `tmd_matching_hybrid`(`matching_order=1`) | 3 点全精度数组 |

规则：
1. 新能力只能是**可选参数**，默认值与默认分支不得改动；metadata 只许加键，
   既有键的值不得变。
2. `analytic_tail_fourier` 在**实** `C,B` 时走 `Re(A₊)/Im(A₊)` 的历史代数路径
   （`(A₊±A₋)/2` 会差 1 ULP，已实测破坏逐位复现）；仅复幅度才走双通道。
3. 本契约失败时只有两条路：撤回改动，或显式升版并重算全部下游产物。
   不允许"看起来差不多"放行。
