# AGENTS.md — logs/dev5_2

**dev5_2：tag:test9 系列全量分析（all）—— 梯度流重整化核子胶子 TMD-PDF 因果推导与图表深度映照**

本目录为 `tag:test9` 系列（`test9 → test9_1 → test9_2`）的第三版详细分析（`all` 全量），在 `dev5`（25 页，字体偏小）与 `dev5_1`（33 页，物理>55%）基础上，进一步**明确物理因果链**并**强化物理-图表相互映照**。

## 产物
- `analy_test9_20260822.tex`（901 行，59186 字节）与 `analy_test9_20260822.pdf`（36 页，1.3MB，`xelatex` 两遍，`Overfull=0` `Float too large=0` `Missing=0`）
- 编译：`cd /root/PyQCD/logs/dev5_2 && xelatex -interaction=nonstopmode -halt-on-error -file-line-error analy_test9_20260822.tex`（两遍）
- 报告标题：`tag test9 系列全量分析（all）：梯度流重整化核子胶子 TMD-PDF 工作流与全链物理公式因果推导及图表深度映照`，日期 2026-08-22，`logs/dev5_2 专题`

## 核心改进（vs dev5_1）
- **字体规范化**：正文 10.54pt（`ctexart`）、表 `\small`、代码 `\footnotesize`，禁用 `\tiny`/`\scriptsize`（`dev5` 曾用）；`tabcolsep 4pt` + `emergencystretch 3em` + `nolinkurl` 断行保证 `Overfull=0` 且可读
- **因果清晰化**：新增 `§5 全链因果总览`（`keybox` 因→果主链：UV 发散→梯度流→Wilson 线→O 组合→激发态→Z_R→NLO），各 `A5.1..A5.7` 子节均按“前因→推导→后果→代码 `file:line`→图表验证”五步展开
- **物理占比**：`>62%`（20 页公式与校验，原 `dev5_1` 55%），新增 7 个深度块（`Eq.dualprop Eq.C2expand Eq.ZRrg` 等）各含新方程、量纲/对称/极限校验
- **图表-物理三段式映照**：`A9` 与第 7 章每图（`F1 c0(z,b)` `F3 R` `F4 准 TMD` `F5 NLO` `F6 vs b` `F7 K(b)` 等 13 类 122 图）均配 `keybox` 三段：**物理意义**（测什么可观量）/**对应物理结果**（数值与趋势）/**物理角度解析**（QCD 含义、理论预期、系统误差），与 `Eq.Odef Eq.quasiTMD Eq.Zij Eq.TMDmatch` 直接呼应

## 目录结构（本目录）
- `analy_test9_20260822.tex`：主报告源码（`references/latex-template.md` 模板，三视角 `§2-4` + 15 步 `§5` + 关键片段 `§6` + 图表详览 `§7` + 日志汇编 `§8` + 参考清单 `§9` + 结论 `§10`）
- `analy_test9_20260822.pdf`：编译产物（36 页，含 6 张代表性插图 `width=0.9\textwidth`）
- `analy_test9_20260822.aux/.log/.out/.toc`：编译中间产物（不入库，仅验证用）
- 本 `AGENTS.md`：本目录约定（由 `init` 技能生成）

## 工作流（test9 系列 9 步，对应 `pyqcd/pipeline/_tmd9.py:5-17`）
1. 蒸馏 2pt（多动量 `MOMENTA_Z/ALL`）→ 2. 读 `.lime` 规范场（`read_gauge_lime` 幺正性扫描）→ 3. Wilson flow `τ=3a² ε=0.05 60 步 RK3`（`wilson_flow`）→ 4. TMD OPE 逐 `t` 空间求和 `(nz=13,nb=5,Nt=72)`（`tmd_matrix_elements_time`）→ 5. 不相连因子化 `C3=C2·OPE` → 6. 真空扣除+比值 `R` → 7. 逐 `(z,b)` 拟合 `c0`（`run_disconnected_tmd_ratio` `lsqfit` `svdcut=1e-6`）→ 8. 自重整化 `hR` → 9. 准 TMD 傅里叶 `quasi_tmd_pdf` → NLO 匹配 `tmd_matching_hybrid` → CS 核 `cs_kernel_from_ratio` → 图表+`tmd_summary.json`

## 物理公式全链（因果，`§5.5` 详推）
- 梯度流 `∂_t B_μ=D_ν G_{νμ}` `Z=P_{ah}[ΩV†]` `RK3` `Eq.flow_cont--RK3` → `E(t)=¼G²` `t²⟨E⟩=0.3` 定 `t0`
- Clover `F=-i/8 Σ(P-P†)` `Eq.clover` `O(a²)` + 对偶 `½εF` `Eq.dual` → `TrF\tilde F` 拓扑
- Wilson 线 `W=ΠU_z` `Eq.Wline` → staple `W_⊏` `Eq.stapleW`（`b→0` 退化直线的极限校验）
- `M^{μλ;νρ}=ΣTr[F W F W†]` `Eq.Mfund` → `O=M^{tx;tx}+M^{ty;ty}-2M^{xy;xy}` `Eq.Odef`（`2^{++}` 可乘化，`b→0` 回准 PDF）
- `M^{ti;it}+M^{ji;ij}=2p0²M_pp` `Eq.Mpp` → `-M_pp=½∫e^{-ixν}xg` `Eq.MppPDF` → `x\tilde g=1/𝒩∫dz e^{-ixzPz}h_R` `Eq.quasiTMD`
- 2pt `C2=Σ|Z|²e^{-Et}` `Eq.C2spec` → `R=c0+c1e^{-dEdτ}+c1e^{-dE(dt-dτ)}` `Eq.ratioModel`（两态严格导出，`Nsample=200` 满秩）
- `Z_R` `Eq.ZR` + `Z_MS` `Eq.ZMS` → 混合 `Eq.hybrid` + `λ` 外推 `Eq.lambdaExtrap` → `Z_{ij}=δ+α_sC_A/2π M` `Eq.Zij` `g0..g3` `Eq.g123` → TMD 匹配 `Eq.TMDmatch` + 快度 `e^{½lnK}` + SFTX `Eq.SFTX`

## 图表-物理映照（`§7` 清单 13 类，三段式）
- `F1 c0(z,b)`：裸矩阵元指数衰减 `ξ~0.6fm`/`ξ_b~0.3fm` 反映禁闭与横向退相干
- `F3 R`：平台 `0.35±0.08` 平坦验证基态，`c1~ -2c0` 符号负为 `N(1440)` 相消
- `F4 准 TMD`：峰 `x~0.3` `⟨x⟩~0.48` 符合 CT18，`b` 压低 30% 对应 `k⊥~0.4GeV`
- `F5 NLO`：小 `x` 抬升 15% 源于 `g2` 共线对数，`b` 越大抬升越小因 `K(b)` 负
- `F6 vs b`：`b=0.5fm` 1/e 衰减对应胶子半径 0.4fm < 电荷半径
- `F7 K(b)`：负值 `-0.2→-0.8` 符合一圈 `-C_A/π ln(bμ)`，误差 ±0.6 需 200 组态压至 ±0.1
- 其余 `F8 全方向` `F9 eff_mass` `F10 fit_dirs` 等同理（立方等价 <0.1%、P000 1.09GeV 等）

## 验证
- `xelatex` 两遍 `Overfull=0 Float=0 Missing=0`（`grep -c` 实测）
- `pdfinfo` 36 页 1.3MB，`pdftotext` 抽检标题/三视角/因果总览/图表三段式齐全
- 物理断言 `test9_verify.py` A/B/C/D/E 全过（`E递减` `unitarity` `谱线色散` `OPE衰减` `⟨x⟩~0.5`）
- 代码-物理映射表 11 行（`pyqcd/*:line`），参考源 43 条（38 仓库+5 参考）

## 与根 AGENTS.md 的关系
本目录为 `logs/` 按 tag 归档的子目录（`logs/AGENTS.md:1` 约定）；根 `AGENTS.md` 的 `logs/` 表项与 `test9` 系列说明已同步更新至 `dev5_2`（36 页 `all` 全量）。

## 后续
- 统计限制（10 组态大 `z,b` 噪声）需扩至 200 组态+多 `t_src` 平均
- 连续/手征/体积外推（`renorm/_extrapolate.py`）待接入；多 `a` 与 `mπ→0.135GeV` 外推
- `HYP` 对照与 `sftx` `t` 依赖残留 `O(tΛ²)~2%` 待系统误差分析
