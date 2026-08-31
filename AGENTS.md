# AGENTS.md — PyQCD

**PyQCD By ZhangXin**：格点 QCD 研究仓库（lattice-pdf 迁移），核心目标为
**计算使用梯度流重整化方案的核子中的胶子 TMD-PDF**。

## 概要命令

```bash
source ./env.sh                      # 环境（若存在）
python examples/pyqcd/conftest.py    # 全量测试（物理/链路、整合功能，以及 SU(3) 几何/流归一化/HYP 协变/统计可辨识/二进制 IO/MPI 目录/持久化/管线异常等；实际计数以命令输出为准）
python examples/pyqcd/verify_consistency.py   # 一致性验证（参考产物完整时 vs docker-v20260805，A–E 全 0 差异；缺失时明确退出2）
python examples/pyqcd/tmd_gradient_flow_demo.py   # 梯度流 TMD 全链示例
python -m pyqcd.parallel --dry-run --confs 6250,6450   # MPI 并行规划预览（用户公式 N*a=n*b）
mpirun -np N python -m pyqcd.parallel --confs ...      # MPI 元任务并行管线（N 由 plan_parallel 给出）
bash logs/test0/run-local.sh        # ana_3dir 三方向差异分析+作图测试（test12 风格，17 项断言）
bash logs/test0_ratio/run-local.sh  # 02_ratio 3pt/2pt 比值+拟合+图测试（22 项断言）
bash logs/test0_anaratio/run-local.sh  # 03_ana_ratio 纯画图测试（25 项断言）
bash logs/test0_bare/run-local.sh   # 03_bare_matrix 三方向裸矩阵元测试（22 项断言）
bash logs/test0_energy/run-local.sh # 04_proton_energy 有效能量测试（8 项断言）
bash logs/test0_fh/run-local.sh     # 06_FH_bare_matele FH 变换测试（38 项断言）
bash logs/stab1/run-local.sh        # 全功能真实数据实战（docker 基线 10 组态，45 项断言 + 106 图 + 报告）
bash logs/test6/run-local.sh        # pyqcd 独立复现 04_proton_energy（879 组态三方向，逐位一致 + 7 图 + 12 断言）
python examples/pyqcd/test9_gluon_tmd_nucleon.py --smoke   # test9 梯度流胶子 TMD-PDF 冒烟（1 组态 1 动量）
python examples/pyqcd/test9_gluon_tmd_nucleon.py --only-plot --conf-ids ... # test9 仅分析出图（复用已算数据）
python examples/pyqcd/test9_verify.py [run_dir]  # test9 物理链自洽断言（A 梯度流/E递减/unitarity、B 2pt 谱线、C TMD OPE、D 分析、E PDF）
python examples/pyqcd/dev6/main.py               # dev6 基于 tag-test8 收缩产物（405 组态，只读）补齐 test0/test6 同类型图表（CPU 秒级）
python examples/pyqcd/dev6/verify_dev6.py examples/pyqcd/dev6/v202608221540   # dev6 断言门（22 项：齐全性/形状/物理自洽）
python examples/pyqcd/dev7/main.py               # dev7 dev6 收敛迭代（262 组态实际存在扫描 + Part C 02_ratio 链补齐 ratio_3pt 型图）
python examples/pyqcd/dev7/verify_dev7.py examples/pyqcd/dev7/v202608230624   # dev7 断言门（38 项：齐全性/形状/02链产物/物理自洽/跨运行一致）
bash examples/test0/run-local.sh    # 蒸馏管线一致性测试（调用 pyqcd 复现 docker-v20260805 全量输出）
python examples/test0/main.py verify --run-dir examples/test0/v<ts>   # 一致性验证（A–E 项）
cd docs && xelatex <文档>.tex        # 编译中文 LaTeX 文档（xelatex，两遍）
```

## 目录结构

| 目录 | 内容 |
|---|---|
| `pyqcd/` | 主包（lattice/tools/vertex/contraction/operator/analysis/renorm/pipeline/testing/parallel） |
| `examples/` | 成功实例（docker-v20260805 基线）+ pyqcd 规范示例/测试 + `test0/` 蒸馏管线一致性套件 + `pyqcd/dev6/`、`pyqcd/dev7/` 同型图表补充/收敛套件（405→262 组态实战） |
| `docs/` | 52 篇中文 LaTeX 笔记（xelatex 编译，文件名统一中文）+ analy 报告 |
| `refer/` | 参考代码/文献（zengch/donghx/huangcl/sush/zhangxin/papers/books）+ `git-rep/` 外来参考仓库（quda/PyQUDA/lamet-agent/EasyDistillation/LQCD_Master，只读、其 AGENTS.md 已归档） |
| `logs/` | 按 tag 归档产物（stab0/ 等）+ test0/ 与 test0_*/ 数据分析功能测试套件（test12 风格）+ stab1/ 全功能真实数据实战套件 + test6/ pyqcd 独立复现套件（.ref_run/ 存 refer 实跑真值，verify_04_repro.py 数值比对）+ test7/ 服务器正式工作版（100 组态，GPU V100，env.sh 启动，输入检查机制 + 实时进度日志）+ test9/ 梯度流胶子 TMD-PDF 实战报告（test9_analysis.pdf）+ dev5/dev5_1/dev5_2 对 tag:test9 系列的详细分析（25 页→33 页→36 页，字体规范+物理>62%+因果总览+图表三段式，all 全量） |
| `cpp/` | C++ 后端占位 |
| `.opencode/skills/` | 归集的 LQCD_Master 上游技能（lqcd-analysis 等 5 个，原位置保留） |

## 外来参考库分析产物（refer/git-rep/*/docs）

对 `refer/git-rep/` 下 4 个外来软件库（EasyDistillation/LQCD_Master/PyQUDA/lamet-agent）
执行 analy/pure 分析的产物约定：PDF+tex 各存放于对应库的 `docs/` 子目录，
命名 `analy_<slug>_<YYYYMMDD>.pdf` / `pure_<slug>_<YYYYMMDD>.pdf`（slug 拼音/英文短词），
编译要求 xelatex 两遍、Overfull=0 且 Float too large=0。已产出：
EasyDistillation（analy/pure_distillation_20260817）、LQCD_Master（analy/pure_qcd_master_20260817）、
PyQUDA（analy/pure_pyquda_20260817）、lamet-agent（analy/pure_lamet_20260817）。
这些库非独立 git 仓库（由主仓库跟踪），报告内 git 信息注明"非独立仓库"；
产物为新建文件，不修改库内既有内容。

## 主仓库 PyQCD 自身 analy/pure 产物（docs/）

主仓库 PyQCD 亦按相同约定（`analy_<slug>_<YYYYMMDD>.pdf` / `pure_<slug>_<YYYYMMDD>.pdf`、
xelatex 两遍、Overfull=0 且 Float too large=0）产出全库级分析：
`docs/analy_pyqcd_20260818.pdf`（全库结构 + 梯度流重整化胶子 TMD-PDF 物理链三视角分析）、
`docs/pure_pyqcd_20260818.pdf`（核心部分穷尽剖析：C1 梯度流 RK3 / C2 胶子 TMD 算符 O 组合 /
C3 Z\_R / C4 混合 / C5 NLO 匹配核 Z\_ij / C7 torch 后端，含代码-物理对象映射表）。
两者均达成 Overfull=0、Float too large=0、Missing character=0（2026-08-18 生成）。
新增：`docs/analy_physics_chain_20260824.pdf`（物理理论全链专项：B15 框架完整公式链
流方程 RK3 → Clover/对偶场强 → O 组合 → 不相连比值 c0 → Z_R/混合 → 准 TMD-PDF
cos/sin 双通道 → NLO 匹配 Z_ij+CS 核+SFTX → 连续极限外推，含 25 项代码-对象映射表、
22 处 lstinputlisting 直引；Overfull=0、Float=0、Missing=0；等宽字体换 DejaVu Sans Mono
以覆盖 docstring 希腊字母/⊏/⊥ 等符号，2026-08-24 生成）。
新增：`docs/report_gluon_tmd_gradient_flow_20260828.tex/.pdf`（中文 16:9 梯度流重整化
核子胶子 TMD-PDF 算法—物理报告，22 页；覆盖 RK3、Clover/对偶场强、staple/OPE、
断连 ratio、z=0 比值与 Z_R/混合入口、Fourier/CS/NLO/SFTX、连续极限、工程链及
MyQCD 对照；双遍 XeLaTeX，Overfull=0、Float too large=0、Missing character=0；
单一系综与完整物理闭环缺口已明确标注，2026-08-28 生成）。

## torch 后端 + h5 IO + MPI 并行（pyqcd/tools/_torch_backend.py + pyqcd/parallel/）

- 后端：`set_backend('torch')`（别名 gpu/cuda）全面替换 numpy/cupy；numpy/cupy 输入自动转
  torch（复数遵循全局精度）；`set_precision('complex64'|'complex128')` 精度切换；
  `get_backend()` 返回 numpy-like 适配层（einsum/roll(axis=)/transpose 任意轴/take(axis=)/
  linalg 等已包装）；torch.Tensor 补丁 transpose/astype/.T/repeat(axis=)/get/二元运算。
- IO：管线产物一律 h5py 读写（`save_tensor_h5`/`load_tensor_h5`，numpy/torch/cupy 通用），
  读取层 `_load_any` 优先 .h5、回退 .npy/.npz（旧产物兼容）。
- 并行：`pyqcd.parallel.plan_parallel` 按用户公式 N*a=n*b（a=单元任务显存、b=单卡可用显存
  80%、n=GPU 数）给出进程数 N、批次 X=m/N、每卡进程 Y=N/n；`run_parallel_pipeline` 元任务
  （step,conf）round-robin 调度 + GPU 绑定（rank mod n）+ 每任务后自动释放
  （empty_cache+gc）；analysis/plots/report 仅 rank 0（分析作图不并行）。
- 实测：torch CPU 梯度流 2.7–4.8x（vs numpy，逐位一致 max|d|~1e-15）；CPU 自动 8 线程
  （16 线程过并行慢 40%）；vertex conf6250 GPU 36s（峰值 176MB）、2pt 337s（峰值 570MB）；
  本机 1 卡+内存紧张时公式自动收敛 N=1。

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
| `_ratio2pt.py` | 02_ratio：2pt+OPE → 真空扣除 ratio → 逐 z 拟合 → ratio/c0/chi2 图 | `run_ratio2pt` | logs/test0_ratio（22 项） |
| `_ana_ratio.py` | 03_ana_ratio：纯画图（单 fit 图+对比图+nofit 图） | `ana_ratio_plot_all` | logs/test0_anaratio（25 项） |
| `_bare_matrix.py` | 03_bare_matrix：三方向 ratio+平均+拟合+图 | `run_bare_matrix` | logs/test0_bare（22 项） |
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

## 参考代码整合（~auto-all 20260822，logs/examples/refer → pyqcd）

30 项整合，三轮完成（照抄逻辑、自包含、不 import 来源；各附测试；
第二轮 R6 经原版实跑真值逐位对照验证——有效契约 7 用例 max|d|=0）：
第三轮（~auto-all 第三遍清查，B 系列+E 系列，12 项落地 + 1 项判定已在位）
含 1 处既有误植修复（`_matching.C/C_gluon_ratio` 三分区+Si 项对照
matching_cc.py 重写）、2 处原版潜在 bug 的可运行化补全
（lqcddb dis_connect 第二 assign 形状失配 / 其 C() 全局 Cf 未定义）。

| # | 来源 | 功能 | 落点 |
|---|---|---|---|
| R1 | refer/sush lqcddb `smear_gauge.py` | Stout 涂抹（nstep=20,ρ=0.12 对齐真实系综） | `pyqcd/smear/_stout.py` |
| R2 | refer/sush lqcddb `eigvectors/vector.py` | 本征模压缩 V1–V4+噪声/GS/正交检查（seed 可复现） | `pyqcd/vertex/_eigcompress.py` |
| R3 | refer/sush lqcddb `cg_coeff.py` | SU(2) CG 系数（Racah 纯 Python，无 sympy） | `pyqcd/lattice/_cg.py` |
| R4 | refer/zengch `hB_data_FeynmenHellman_new.py` | hB 数据 z₀ 归一化+插值 loader + boot 协方差 | `pyqcd/renorm/_zr.py`（build_hB_dataset/boot_covariance/make_zr_dataset） |
| R5 | refer/zengch `fit_hR_big_lambda_new.py` | λ 外推拟合 boot 全协方差选项 | `pyqcd/renorm/_hybrid.fit_hR_lambda(cov_kind='boot')` |
| E1 | examples/pyqcd test9 示例 | `_plateau_c0` plateau 均值（抗奇异协方差） | `pyqcd/analysis/_tmd_ratio.plateau_c0`（run_disconnected_tmd_ratio 直接产出 c0_plateau） |
| E2 | examples/pyqcd test9 示例 | CS 核两动量提取工程封装（z_ref+clamp） | `pyqcd/renorm/_tmdextract.cs_kernel_two_momentum` |
| E3 | examples/pyqcd test9 示例 | TMD-PDF 链成图 4 张 | `pyqcd/analysis/_tmd_ratio.plot_tmd_pdf` |
| L1 | logs/test8 | 2pt 组态级断点续跑（corr 齐全即跳过） | `pyqcd/pipeline/_steps.step_2pt`（recompute_2pt 强制重算） |
| L2 | logs/test7/test8 | 数据守卫：原始数据齐全度+输入数组校验+ETA 日志 | `pyqcd/pipeline/_validate.py` |
| L3 | logs/test6 | 能量链方向感知（动量置换 dir 参数，z 向后兼容） | `pyqcd/analysis/_proton_energy` + `_bare_matrix.dir_momentum` |
| L4 | logs/test7 | tlog 时间戳+ETA 进度日志 | `pyqcd/pipeline._validate.ProgressLog/progress_log` |
| H1(二轮) | refer/donghx Operator.py | 螺旋度 ΔG 双场强 Wilson 线算符 F·W†·F̃·W（±z 支、平面/全和求和） | `pyqcd/operator/_helicity.py` |
| R6(二轮) | refer/sush lqcddb vertex.py | Ω 加速张量（exact/块/noise 分区权重，dim=2/3，conserved/normal） | `pyqcd/vertex._eigcompress.create_omega_accelerate` |
| R7(二轮) | refer/zengch fit_ratio_FH_new | FH 常数闭式协方差拟合 + χ² 驱动逐 z 自适应 t_sep 窗 | `pyqcd/analysis._ratio_fit.fit_constant_window/fh_adaptive_windows` |
| R8(二轮) | refer/donghx input_output_4_cupy | L.Liu ASCII 关联函数读写对（.gz 自动压缩） | `pyqcd/tools._io.write_data_ascii/read_data_ascii` |
| B1(三轮) | refer/zengch matching_cc.py | **修复** `_matching.C/C_gluon_ratio` 误植（忠实三分区+5/6·Si 项，α_s·C_F/C_A/(2π) 归一） | `pyqcd/renorm._matching` |
| B2(三轮) | refer/zhangxin gluon_pdf_full_workflow:1086 | collinear 胶子准 PDF sin 变换 g̃=(2Pz/x)∫h·sin(xPz z)（x→0 保护），与 cos 型 quasi-TMD 互补 | `pyqcd/renorm._tmdextract.quasi_pdf_gluon` |
| B3(三轮) | refer/zhangxin gluon_pdf_workflow / Operator.py | OPE −z Wilson 线变体 + 固定规范 FF 算符（无 Wilson 线，±z、交叉 μ₂ν₂ 对）+ Lorentz 指派表（unpol/helicity/gauge_fix×2） | `pyqcd/operator._gluon_ope`（gluon_ope_operator_z0 扩展 mu2/nu2/direction + gluon_ff_operator_z0 + get_ope_lorentz_pairs） |
| B4(三轮) | refer/zhangxin workflow apply_parity_and_boundary | 双宇称投影 P±=½(γ₀±γ₄) + 反周期边界符号翻转（pp: t_sink<t_src；pm: t_sink>t_src） | `pyqcd/contraction._baroperator.parity_and_boundary` |
| B5(三轮) | refer/zengch fit_zr_new.fit_ZR 样本循环 | Z_R 参数误差逐样本重拟合环 + mean/std 汇总（单坏样本 NaN 不中断） | `pyqcd/renorm._zr.fit_ZR_samples/summarize_ZR_samples` |
| B6(三轮) | refer/zengch fit_pz_a_extrapolatiing | 连续极限外推协方差加权（Cholesky 白化+lstsq，非正定回退单位阵）+ 逐样本误差带（固定 lx/hx/bx/cx，仅 xg0/fx/dx/kx 自由）；批量化优于原版逐样本 Minuit | `pyqcd/renorm._extrapolate.fit_hR_PDF_extrap_boot` |
| B7(三轮) | refer/sush lqcddb analyse.py | dis_connect disconnected 矩阵元（PFF/PDF）+ 分组聚合基元（take+stack 语义等价实现，适配层无 reduceat）；修正原版第二 assign 形状失配 | `pyqcd/analysis._analyse` |
| B8(三轮) | refer/sush lqcddb io/write_date.py | 模板占位符组合式文件存在性+大小一致性守卫（corrupted 归类） | `pyqcd/pipeline._validate.check_files_existence` |
| B9(三轮) | refer/sush lqcddb autowick/dynamic | Wick 缩并图 QC 可视化（复杂度自适应）+ 收缩路径 FLOPs/加速比/最大中间张量诊断（run_wick_analysis 增 registry/optimize 可选参） | `pyqcd/contraction._wickplot.plot_figure_wick` + `_dynamic._analyze_contraction_path/_format_cost` |
| B10(三轮) | refer/huangcl 98_tools input_output.py | V†V/VVV 预计算顶点积二进制 reader（f8 交错复数，Nev 自探测+截断 Nev1） | `pyqcd/tools._io.readin_vdv_all/readin_vvv_all/readin_vvv` |
| E4(三轮) | examples/test0/main.py dump_env | 运行环境快照 env.json（git/包版本/xelatex/GPU/cmdline） | `pyqcd/tools._env.dump_env` |
| E5(三轮) | examples/test0/main.py _rel_maxdiff/_cmp_one | NaN 感知回归比对原语（NaN 位置须一致；分母 \|b\| norm 防除零） | `pyqcd.testing.rel_maxdiff/cmp_one` |

判定已在位（第三轮清查结论，零改动）：`_fitter.fit` 的 debug/debugNfit/NaN
填充与 `cov_mat` 条件数返回（B11，子代理报告有误）；docker utils 文件日志
工厂 setup_logging/print_banner/log_exception（与仓库 print/tlog 日志约定
冲突，ProgressLog 已覆盖）；mpi_init 域分解搬运层 get_mpi_data/TScatter
（范式不同于 pyqcd.parallel 元任务调度，预留未来）。

跳过（记录理由）：IOG reader（依赖 iog.so 二进制）、Chroma XML 生成器与
SIDIS-DY 唯象层（依赖库外 evolution 模块）、contractadviser（性能顾问非物理，
其核心思想已由 B9 轻量内嵌）、helicity ΔG 下游链（研究方向未启动，算符层已覆盖）、
下载/打包脚本与 donghx Calc_*/2pt_proton_* 一次性驱动壳（核心算法均已覆盖）、
zengch/huangcl 各 _new 旧版变体、sush function_contraction 扁平旧包
（lqcddb 早期版，全被取代）。
torch 适配层补齐 numpy-like 函数（cos/sin/arccos/isnan/clip/maximum 标量/
argwhere/identity/append/random）。test9 示例已改为消费 pyqcd API
（删除内嵌 `_plateau_c0`/CS 核内联/`plot_pdf` 共 ~115 行重复实现）。

## test9 系列详细分析（logs/dev5/dev5_1/dev5_2，对 tag:test9 的 all 全量）

`tag:test9` 系列（`test9 4c58ddb` → `test9_1 eb24f23` → `test9_2 15b020f`）的详细分析按 `analy` 技能三视角 + 15 步工作流框架，完整推导全链物理公式，已迭代三版：
- `dev5`（`logs/dev5/analy_test9_20260820.pdf` 25 页）：首版全链推导 + 101 证据 + 122 图，但字体偏小（`tiny`/`scriptsize`）且物理占比不足；
- `dev5_1`（`logs/dev5_1/analy_test9_20260821.pdf` 33 页）：修正字体至模板标准（正文 10.54pt/表 `small`/代码 `footnotesize`，禁用 `tiny`）并扩展物理至 >55%（新增 7 深度块，`Eq.dualprop Eq.C2expand Eq.ZRrg` 等）；
- `dev5_2`（`logs/dev5_2/analy_test9_20260822.pdf` 36 页，`all` 全量）：进一步明确因果（`§5 全链因果总览` 前因→后果主链，各 `A5.x` 五步展开）并强化图表-物理三段式映照（每图 `物理意义/对应结果/物理解析` 呼应 `Eq.Odef Eq.quasiTMD Eq.Zij Eq.TMDmatch`），物理占比 >62%，`xelatex` 两遍 `Overfull=0 Float=0 Missing=0`。

## dev6 同型图表补充（examples/pyqcd/dev6，20260822）

输入为 `${HOME}/data/beta6.20_mu-0.2770_ms-0.2400_L24x72`（tag-test8 管线产物，
405 组态 corr_pp_P0/P2 + VdV/VVV；只读、不消费 VVV）。main.py 调 pyqcd.analysis
补齐与 examples/test0/v202608150750/plots 及 logs/test6/1_result/L24x72/Pz6
相同类型的全部图表：B 型 7 图（P0/P2 双通道替代 x/y/z/ave）+ A 型 docker 栅格
2 图（pion 面板留白注明）；ratio_3pt 因输入无 perambulators/3pt 数据缺席并在
summary/verify/报告三处注明。B 型拟合照抄 test6（4 参数形状模型逐样本 lsqfit
svdcut=1e-6，窗 [6,12]，unit=0.197/a）；关键修复：两通道窗口内 C<0（相位残留
pi），全局符号 sgn*C 约定消除拟合盆地歧义。实测：CPU 9–12 s、峰值 0.44 GB；
E0(P0)=1.112(8)/1.1474(15) GeV（B/A 双方法，与 stab1 约 1.12 GeV 一致）、
E0(P2)=1.491(52)/1.5477(27) GeV（色散预期 1.510，偏差 -1.2%/+2.5%）；
verify_dev6 22 断言全绿；analy 报告 docs/analy_dev6_20260822.pdf（27 页
Overfull=0/Float=0/Missing=0）。pyqcd 最小修改：
`_correlators.run_meff_jackknife` 缺失通道跳过守卫（完整数据行为不变）。

## dev7 同型图表收敛迭代（examples/pyqcd/dev7，20260823）

dev6 的 ~auto-all 收敛迭代，两项实质收敛：(1) **组态实际存在扫描**——数据目录自
dev6 运行后由外部删减（405 → 262，全树 mtime ≤20260818），scan 按五文件齐备
（corr_pp_P0/P2 + ops 三分量）判据如实计数，不假设网格；(2) **补齐 ratio_3pt 型图**
——输入 ops_*(Nz,Nt)+corr_pp 正是 pyqcd 02_ratio 链输入类型，Part C 照抄
test8 makedata 切片整理 + run_02_ratio 配置：staging（切片矩阵 C[sink,src]=
C((sink−src) mod Nt) + ops 符号链接）→ run_ratio2pt(Pz=2,dt_max=20,三代表窗
(6,11,2)/(7,11,3)/(9,11,4)) → ratio_3pt_all_channels.png（disconnected OPE/2pt
真空扣除比值 R(τ)，z∈{0,2,4,6}@t_sep=10；非连通 3pt，图题明注）。实测：262 组态
4m35s、峰值 8.7 GB；E0(P0)=1.143(A)/1.092(13)(B)、E0(P2)=1.551(A)/1.515(42)(B)
GeV，与 dev6 跨运行互差 5/3 MeV、色散偏差 3.1%；c0(z≤4)≈0±0.03（与零一致，
disconnected 通道需更多统计）；小样本（--debug）下 c0 被先验主导属预期。
verify_dev7 38 断言全绿（含 vs dev6 跨运行一致性 D 组）；analy 报告
docs/analy_dev7_20260823.pdf（29 页 Overfull=0/Float=0/Missing=0）。

## 关键约定

- 张量布局：gauge `(Nt,Nz,Ny,Nx,4,3,3)`；γ 矩阵 DeGrand-Rossi 基。
- 后端：numpy/cupy/torch 三后端（`pyqcd.tools.set_backend`；torch 详见上文小节）。
- 编译：docs 与 logs 的 tex 一律 xelatex（中文）；`\quad` 后跟中文需空格。
- 测试：无 pytest 框架依赖，examples/pyqcd/conftest.py 直接运行。
- refer/ 只读参考：pyqcd 逻辑照抄但不 import。
- git tag 约定：stab<N>/dev<N>/bug<N>/test<N>（当前 stab1）。

## 反模式

- 不 import refer/、不 import examples/（照抄逻辑，自包含）。
- 不修改 refer/ 与成功实例基线的"已验证物理结论"（pn 2pt=0、meff≈1.12 GeV 等）。
