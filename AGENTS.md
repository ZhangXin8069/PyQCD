# AGENTS.md — PyQCD

**PyQCD By ZhangXin**：格点 QCD 研究仓库（lattice-pdf 迁移），核心目标为
**计算使用梯度流重整化方案的核子中的胶子 TMD-PDF**。

## 概要命令

```bash
source ./env.sh                      # 环境（若存在）
python examples/pyqcd/conftest.py    # 全量测试（17 项：γ基/Z_R/梯度流/TMD算符/匹配/混合/提取链/标度/HYP/τ极限/比值拟合/HYP-流一致/后端一致/端到端meff/求和规则/核心链/TMD-NLO匹配）
python examples/pyqcd/verify_consistency.py   # 一致性验证（vs docker-v20260805 输出，A–E 全 0 差异）
python examples/pyqcd/tmd_gradient_flow_demo.py   # 梯度流 TMD 全链示例
bash logs/test0/run-local.sh        # ana_3dir 三方向差异分析+作图测试（test12 风格，17 项断言）
bash logs/test0_ratio/run-local.sh  # 02_ratio 3pt/2pt 比值+拟合+图测试（18 项断言）
bash logs/test0_anaratio/run-local.sh  # 03_ana_ratio 纯画图测试（25 项断言）
bash logs/test0_bare/run-local.sh   # 03_bare_matrix 三方向裸矩阵元测试（18 项断言）
bash logs/test0_energy/run-local.sh # 04_proton_energy 有效能量测试（8 项断言）
bash logs/test0_fh/run-local.sh     # 06_FH_bare_matele FH 变换测试（38 项断言）
bash logs/stab1/run-local.sh        # 全功能真实数据实战（docker 基线 10 组态，45 项断言 + 106 图 + 报告）
bash logs/test6/run-local.sh        # pyqcd 独立复现 04_proton_energy（879 组态三方向，逐位一致 + 7 图 + 12 断言）
bash examples/test0/run-local.sh    # 蒸馏管线一致性测试（调用 pyqcd 复现 docker-v20260805 全量输出）
python examples/test0/main.py verify --run-dir examples/test0/v<ts>   # 一致性验证（A–E 项）
cd docs && xelatex <文档>.tex        # 编译中文 LaTeX 文档（xelatex，两遍）
```

## 目录结构

| 目录 | 内容 |
|---|---|
| `pyqcd/` | 主包（lattice/tools/vertex/contraction/operator/analysis/renorm/pipeline/testing） |
| `examples/` | 成功实例（docker-v20260805 基线）+ pyqcd 规范示例/测试 + `test0/` 蒸馏管线一致性套件 |
| `docs/` | 51 篇中文 LaTeX 笔记（xelatex 编译，文件名统一中文） |
| `refer/` | 参考代码/文献（zengch/donghx/huangcl/sush/zhangxin/papers/books），只读 |
| `logs/` | 按 tag 归档产物（stab0/ 等）+ test0/ 与 test0_*/ 数据分析功能测试套件（test12 风格）+ stab1/ 全功能真实数据实战套件 + test6/ pyqcd 独立复现套件（.ref_run/ 存 refer 实跑真值，verify_04_repro.py 数值比对） |
| `cpp/` | C++ 后端占位 |

## 蒸馏管线一致性测试（examples/test0）

调用 pyqcd 包复现成功实例 `examples/docker-v20260805/output/output_20260802_120104`
的全量结果（10 组态 9 步：vertex→2pt→ope→3pt→4pt→analysis→plots→report）：
中间数据 + 图表 + LaTeX 报告完整保存于版本目录 `examples/test0/v<YYYYMMDDHHMM>/`
（test12 约定），逐项数值一致。`main.py` 只含测试/编排代码（计算委托
`pyqcd.pipeline.run_pipeline`，实现于 `pyqcd/pipeline/_steps.py`，照抄 docker 逻辑自包含）。
冒烟：`python examples/test0/main.py run --conf-ids 6250`（Nconf<2 时 disconnected
拟合自动跳过，统计无意义）；全量：`bash examples/test0/run-local.sh`（~3-5h）。
一致性容差：中间数据 rel<1e-6、分析结果 rel<1e-8；verify 按组态数自适应
（Nconf=10 时 B/C/D 统计量严格比对）。已验证：conf6250 中间数据逐位一致
（rel=0.000e+00），全量 237/237 PASS。

## 数据分析与作图（pyqcd/analysis/_ana_3dir.py）

输入数据路径 → 分析并作图（独立实现，功能对齐 refer/huangcl/05_ana_3dir_diff_sem）：
读取三方向（x/y/z/ave）ratio.npy 与 corr2.npy → 有效质量、归一化协方差（相关系数）、
直方图（mean±sem）；顶层入口 `analyze_3dir(data_root, out_root, AnaParams)`，
数据结构 `<root>/<conf>/Pz<Pz>/{x,y,z,ave}_dir/ratio.npy` + `corr2_{dir}.npy`。

## 数据分析与作图功能链（pyqcd/analysis，独立实现，功能对齐 refer/huangcl 02/03/04/06 步）

| 模块 | 功能（参考脚本） | 顶层入口 | 测试套件 |
|---|---|---|---|
| `_plots.py` | 图表工具全集：plot_errbar/scatter/hist + single/multi 封装 + 10 色 | — | 各套件共用 |
| `_fitter.py` | calc_chi2(_dof)/fit（lsqfit 封装）/FitParams/ASCII 报告表 | — | 各套件共用 |
| `_ratio2pt.py` | 02_ratio：2pt+OPE → 真空扣除 ratio → 逐 z 拟合 → ratio/c0/chi2 图 | `run_ratio2pt` | logs/test0_ratio（18 项） |
| `_ana_ratio.py` | 03_ana_ratio：纯画图（单 fit 图+对比图+nofit 图） | `ana_ratio_plot_all` | logs/test0_anaratio（25 项） |
| `_bare_matrix.py` | 03_bare_matrix：三方向 ratio+平均+拟合+图 | `run_bare_matrix` | logs/test0_bare（18 项） |
| `_proton_energy.py` | 04：corr2 + E0 拟合 + eff_mass 图（GeV） | `run_energy` | logs/test0_energy（8 项） |
| `_fh.py` | 06：6 方向 ratio 平均 → FH 变换 → 常数拟合 → FH/参数/对比图 | `run_fh` | logs/test0_fh（38 项） |

统计基元 sem/resample/cov_mat 复用 `_disconnected.py`；各套件合成数据
（物理可解析：meff/E0/c0 精确恢复）经 makedata 生成，verify 断言
产物存在性 + 解析形状 + 参数恢复。

## 全功能真实数据实战（logs/stab1）

docker-v20260805 基线（10 组态真实数据）驱动全部分析功能链实战：
02_ratio → 03_ana_ratio → 04_proton_energy（P2/P0）→ 06_FH → 05_ana_3dir
+ 中文 LaTeX 报告。数据适配：corr_pp (Nt,) → 平移不变切片矩阵 (Nt,Nt)；
P2 2pt 带 phase 负号（ratio 负/负相消自洽，能量提取取 |corr2|）。
物理断言：P0 meff 平台 ≈ 1.12 GeV（质子质量，已验证结论）、P2 ≈ 1.56 GeV
（色散）、E0 与 meff 平台一致；拟合用 svdcut=1e-6（10 组态协方差奇异）。
分析报告：logs/stab1/docs/stab1_analysis.pdf（代码+物理+日志+交叉四视角）。

## 核心物理链（pyqcd/renorm）

1. **梯度流**（`_gradient_flow.py`）：Wilson flow（Luescher 2010），RK3 积分，
   τ=3a² 方案（Monahan–Orginos 2017 / NieMiera 2025）。
2. **胶子 TMD 算符**（`operator/_gluon_ope.py` + `renorm/_tmd.py`）：
   Clover F_μν、对偶 F̃、staple Wilson 线、组合 O = M^{tx;tx}+M^{ty;ty}−2M^{xy;xy}。
3. **自重整化**（`_zr.py`）：Z_R 参数化与全局拟合（arXiv:2510.17758 Eq.3-8）。
4. **混合方案**（`_hybrid.py`）：短距比值 + 长距 Z_R，λ 外推，傅里叶→准 PDF。
5. **NLO 匹配**（`_matching.py` + `_tmdextract.py`）：胶子单圈匹配核 g_0..g_3；
   TMD 混合方案匹配 `tmd_matching_hybrid` 用 Z_ij 矩阵结构（δ + α_sC_A/2π 核），
   复用 `_matching_kernels`（A_s = α_s/4π，zengch 约定），快度演化 + 软函数。
6. **连续极限**（`_extrapolate.py`）：a/Pz/mπ/L 联合外推。

## 关键约定

- 张量布局：gauge `(Nt,Nz,Ny,Nx,4,3,3)`；γ 矩阵 DeGrand-Rossi 基。
- 后端：numpy/cupy 切换（`pyqcd.tools.set_backend`）。
- 编译：docs 与 logs 的 tex 一律 xelatex（中文）；`\quad` 后跟中文需空格。
- 测试：无 pytest 框架依赖，examples/pyqcd/conftest.py 直接运行。
- refer/ 只读参考：pyqcd 逻辑照抄但不 import。
- git tag 约定：stab<N>/dev<N>/bug<N>/test<N>（当前 stab1）。

## 反模式

- 不 import refer/、不 import examples/（照抄逻辑，自包含）。
- 不修改 refer/ 与成功实例基线的"已验证物理结论"（pn 2pt=0、meff≈1.12 GeV 等）。
