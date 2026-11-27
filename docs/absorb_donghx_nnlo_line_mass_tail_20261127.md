# 吸收 refer 新增内容：两圈转换 / 线质量核 / 解析尾（2026-11-27）

## 1. 范围与基线

- 仓库：`/public/home/zhangxin/PyQCD`，分支 `main`，基线标签 `stab53`
  （`git describe --tags --abbrev=0 HEAD`）。
- 待吸收的新增外来内容：`stab53..HEAD` 区间的两次提交
  （`50ddfebd`、`e1d96bfc`），全部落在 `refer/` 下，共 **400 个 candidate**
  （`absorb_inventory.py --base stab53 --scope commits`）：
  `refer/donghx/_Gradient_Flow_GluonPDF` 279、`refer/huangcl/_0xxx` 45、
  `refer/donghx/Note` 32、`refer/sush/_distillation` 23、
  `refer/donghx/_codex` 12、`refer/sush/_codex` 9。
- 清单统计：binary 144、python 137、markdown 62、shell 55；风险信号
  `executable-bit` 32、`dynamic-execution` 25、`network-access` 4、
  `scan-truncated` 1。全部 `origin=likely-foreign`。
- 本轮**不执行任何外来代码**，只做静态阅读与公式复算；`refer/` 工作树未改动。

## 2. 本库既有覆盖盘点（物理层，排除 `pyqcd/einsum`）

| 能力 | 现状 | 证据 |
|---|---|---|
| AIC/AICc 权重 | 部分：`_proton_energy` 有权重但不参与合成；`_bare_sumratio` 有逐 replica 加权 | `_proton_energy.py:148,165,312`；`_bare_sumratio.py:186,281` |
| GEVP | 部分：`(N,N,Nt)` 多算符，但无 replica 轴、无本征矢诊断 | `_analyse.py:633` |
| FH 链 | 部分：多方向平均与多 nex 计算齐全，拟合只取首个 nex，无 ratio_type 变体 | `_fh.py:504,523,533` |
| 色散关系 | 窄：只有 `(m,k2,k3)`，无模型比较 | `_dispersion.py:21` |
| 匹配核阶数 | 只有一圈（NLO），无两圈系数、无 cusp 两圈项 | `_matching.py:48,122,180`；`_tmdextract.py:256,494` |
| 三方向 ratio | 有 `x/y/z` 置换与等权平均，无中点投影、无奇分量剔除 | `_bare_matrix.py:30,132` |
| 后端/精度对照 | 只有测试层脚本，无生产 API | `testing/_double_precision_contract.py` 等 |

结论：**两圈（NNLO）有限转换系数、流化 Wilson 线的一圈闭式核、长距尾的解析
Fourier 三项在库内完全缺失**（`grep -iE "nnlo|two.?loop|cusp|exp1"` 在
`pyqcd/` 命中 0），而它们在新增 refer 中已有完整公式；其余项目（replica-gated
AIC、中点投影、GEVP 权重、色散模型比较）库内已有等价或更强实现，记为「已覆盖」。

## 3. 吸收矩阵（关键项）

| candidate 组 | 正向价值 | 处置 | 目标 | 验证 |
|---|---|---|---|---|
| `NNLO_note.zh-CN.md` + `THEORY.zh-CN.md`（`D_V`、`C₀^{NNLO}`） | 两圈有限转换的严格展开与两圈常数相消证明 | **adapt** | `pyqcd/renorm/_nnlo.py` | 27 项契约 |
| `probe_nnlo_cpu.py` | 两圈系数的 40 位精确代数审计（ringed 常数不变性、RG 自洽） | **adapt**（只取代数，丢弃 `SLURM_JOB_ID` 硬门槛与 mpmath 依赖） | `_nnlo.py` 的 `_clausen_pi_over_3` / `_polylog_quarter` 双路径校验 | 契约 `test_ringed_two_loop_constant_cancels_in_the_ratio`、`test_rg_consistency_fixes_the_double_logarithm` |
| `linear_devergence.md` §17.2–17.6（`F(x)`、`R_K`、H3） | 流化开放 Wilson 线的闭式一圈核与 H3 参数计数 | **adapt** | `pyqcd/renorm/_line_mass.py` | 10 项契约 |
| `baryon_da.md` §4.3（LA/NLA 尾 + `E₁` 闭式） | 把 `hR_x` 的 `λ_max` 截断误差变为可测量 | **adapt** | `pyqcd/renorm/_tail.py` | 5 项契约 |
| `fit_all_operators_aic.py`（replica gate + AIC） | 逐 replica 重裁候选池与等权 cut 合并 | **已覆盖** | `analysis/_bare_sumratio.py:207` 已有同构实现 | 现有 22 项契约 |
| `helicity_operator_comparison.py` 的 `midpoint_projection`（`327-347`） | 中点插入点平均并保留配对 JK | **已覆盖**（本库取 `_plateau_points` 的整段平均） | `analysis/_tmd_ratio.py:89` | 现有契约 |
| `helicity_operator_comparison.py` 的逐 JK 真空减除（`205-238`） | 每个 delete-one 样本内重算分母 | **已覆盖** | `analysis/_disconnected.py:117`（`ensemble_covariance_ratio` 逐 replica 重算） | 现有契约 |
| `NP_T1_weight_fit.py` / `NP_T1_lsqeffgevp.py` | p 值 × 逆方差权重 | **reject**：`dE**2` 参数化与物理 `ΔE` 混淆，权重与 `save_chi` 乘除不一致，系综与路径硬编码 | — | 静态审查 |
| `1_get_3pt.py` / `2_ratio.py` / `2_fit_z0.py` | 设计意图 | **reject**：三者当前运行即报错（`resample` 缺参、`ImportError`、`AttributeError`），只作设计参考 | — | 静态审查 |
| `huangcl/compare_precision.py`、`compare_contract.py` | 精度/性能对照 | **reference-only**：`compare_contract.py` 直接 import `pyqcd.einsum.test` 内部 API，且 einsum 已冻结 | — | — |
| `sush/_distillation/*`（`e32p29.py`、`benchmark.py`、`wick_test.py`） | 收缩驱动与 benchmark | **reference-only**：依赖私有 `lqcddb`、`libthreept.so` 与外部数据根，不可移植 | — | — |
| `sush/gpu-debug-report.md` | pyqcd GPU 路线实测数字 | **reference-only**：结论性证据，不入代码 | — | — |
| 其余 PDF/PNG（144 个二进制） | 图表与笔记 | **reference-only**：只读归档，不进代码路径 | — | — |

## 4. 改动内容

新增三个物理模块（均为独立实现，不 import `refer/`）：

- `pyqcd/renorm/_nnlo.py`
  - `flow_log(tau,a_fm,mu_gev)` → `L_f`
  - `local_ratio_coefficients(channel)` / `local_ratio_factor(...)` → `D_Γ`
  - `gluon_endpoint_coefficients(nf)` / `gluon_endpoint_factor(...)` → `U/H/P`
  - `short_distance_c0(z_fm,mu_gev,alpha_s,order)` → `C₀` NLO/NNLO
- `pyqcd/renorm/_line_mass.py`
  - `flow_radius`、`one_loop_amplitude`、`line_mass_kernel`（`F(x)`，含小 `x` Taylor 分支）、
    `line_mass_kernel_remainder`（`R_K`）、`line_mass`、`effective_slope`、
    `finite_segment_kernel`、`h3_log_difference`
- `pyqcd/renorm/_tail.py`
  - `tail_shape`、`tail_amplitude_fit`、`analytic_tail_fourier`、`tail_fourier_continuum`、
    `abrupt_cutoff_error`

导出接入 `pyqcd/renorm/__init__.py`（`__all__` 同步）；文档同步
`pyqcd/renorm/AGENTS.md`；测试新增 `pyqcd/testing/_nnlo_line_mass_tail_contract.py`
并在 `pyqcd/testing/__init__.py` 与 `conftest.py` 注册。

## 5. 推理依据（关键物理判断）

1. **`a_s ≡ α_s/(4π)`**：与 `_const.A_s` 一致。一条自洽检查是
   `D_V^{NLO}=1+a_s(4L_f+10/3)`，其中 `4L_f+10/3` 可由
   `j₁-n₁ = C_F(3L_f+5/2)` 且 `C_F=4/3` 得到；本实现逐点复现到 1e-16。
2. **两圈 ringed 常数必须相消**：`J₂-N₂` 中共同常数抵消，但它不等于把物理
   `Z_χ` 设为 1。契约用 `ringed_constant ∈ {-50, 0, 100}` 断言 `a₂` 完全不变。
3. **`L_f²` 系数 26 是独立 RG 检查**：`(f₁₁²+β₀f₁₁)/2`，`f₁₁=3C_F=4`、`β₀=9`。
   本实现得 26.000000000，反证系数抄写无误。
4. **`C₀^{NNLO}` 前因子是 `α_s²/(2π²)`**：等于 `(α_s/2π)²` 的 2 倍，差一个因子 2
   是最容易抄错的地方，契约用比值 `quad_only / (α_s/2π)² = 2×42.943457131817` 锁定。
5. **`F(x)` 必须分段**：定义式在 `x≲3e-3` 时 `erf` 与分子抵消，相对误差 >1e-11；
   用 `x/√π(3/4 − x²/12 + x⁴/80 − x⁶/560 + x⁸/4320)` 替代后，`x=1e-12` 到
   `x=1e8` 全区间相对误差 ≤4e-15（对 200 位 mpmath 参考值实测）。
6. **`R_K` 的符号**：`K = 1/r − 3/(2√π z) + r²/(2√π z³) + R_K`，故 `R_K` 含
   `−erfc/r + e^{−z²/r²}/(√π z) − r²e^{−z²/r²}/(2√π z³)`；契约用 `x∈{5,8,12}` 的
   恒等重排验证到 1e-14。
7. **H3 参数计数 `N_f+1`**：`N_f−1` 组独立对数差各一个截距 + 共同 `d`、`k₃`。
   契约直接断言 `2+(N_f−1) == N_f+1` 与输出行数 `Nz(N_f−1)`。
8. **有限段差分不能用中点**：`H₃,ij[z_L,z_R] = (r_i²−r_j²)/Δu · (z_L⁻²−z_R⁻²)/(2(z_R−z_L))`；
   契约断言它与中点 `z⁻³` 在 `[0.6,1.0]` fm 上差异不可忽略。
9. **解析尾与直接求积一致到 1e-16**：`q_tail = (1/π)Re[C e^{−sλ_L}/s + B E₁(sλ_L)]`，
   `s=η−ix`；用 `scipy.integrate.quad` 从 `λ_L` 积到 ∞ 对照，`max|Δ|=1.1e-16`。
10. **不吸收 `huangcl` 的三个新脚本**：静态检查发现 `1_get_3pt.py` 调
    `resample(corr2_avg)` 缺 `Nsample` 必填参数、`2_ratio.py` 的三个函数在
    `ratio_tools.py` 中不存在、`2_fit_z0.py` 引用 dataclass 不存在的
    `output_dir` 字段——运行即报错，只能当设计意图参考。

## 6. 验证结果

| 检查 | 命令 | 结果 |
|---|---|---|
| 新契约单跑 | `python -B -m pyqcd.testing._nnlo_line_mass_tail_contract` | `Ran 27 tests ... OK` |
| 全量回归（改动后） | `python -B -m pyqcd.testing` | `158 passed, 2 skipped, 2 failed` |
| 全量回归（改动前基线） | 同上 | `156 passed, 2 skipped, 3 failed` |
| 新增用例数 | 158 − 156 = 2 = 本次注册的 1 个包装（内含 27 条断言）+ 基线里偶发失败的 `test_einsum_tune_isolation_contracts` 本轮转绿 | 一致 |
| 文档 API 契约 | `test_docs_api_contracts` | PASS |
| 发布元数据契约 | `test_release_metadata_contracts` | PASS |
| `form` 审计 | `bash skills/form/scripts/form-audit.sh` | `PASS form-audit(pyqcd): registered exceptions=18 unexpected=0` |
| `git diff --check` | — | 无输出（0 冲突标记） |
| 大文件检查 | `form-large-files-check.sh` | `通过 已检查=0 限制=10000000 字节` |

失败项说明（**与本次改动无关，为登录节点无 GPU 驱动所致**）：

- `test_gauge_observables_contracts`：`cupy.cuda.runtime.CUDARuntimeError:
  cudaErrorInsufficientDriver`（契约要求 CUDA 必须可用，不得 skip）。
- `test_gpu_backend_consistency`：同一根因。

基线里另有 `test_einsum_tune_isolation_contracts` 失败（`强杀必须在组级上限附近生效`，
实测 3.36 s vs 上限 1.5 s×2），本次未触碰 `pyqcd/einsum` 与本项相关文件，改动后
该失败**已不再出现**（属负载依赖的时序敏感项，登录节点并发高时偶发）。

GPU 侧验证：已提交 Slurm 作业到 `gpu-debug`（唯一允许的 GPU 测试分区）：
`sbatch data/tmp/gpu-contract.sbatch`（作业 `4057218`，`--gres=gpu:1 --mem=32G
--time=00:30:00`，只跑 `test_gpu_backend_consistency` /
`test_gauge_observables_contracts` / `test_torch_backend_consistency`）。
首次提交 `4057216` 因 `--mem=64G` 触发节点调度约束被撤回（未占用节点资源）。

## 7. 服务器使用规范

- 全部静态检查、单元测试、`form` 审计在登录节点执行，CPU 线程限制
  `OMP_NUM_THREADS=3`、`MKL_NUM_THREADS=3`，单次全量回归约 4 分 45 秒。
- GPU 验证只提交 `gpu-debug` 分区（30 分钟上限），未使用 `na100`/`h20`/`nv100` 等
  生产 GPU 分区；提交前先 `scancel` 了因内存约束无法调度的前一个作业，避免占位。
- 未在登录节点编译 CUDA/KUPL 归档，未触发 `make-nv.sh`。
- `pyqcd/einsum` 全部内容未改动（冻结状态），本次新增代码不 import 它。

## 8. 后续步骤

1. GPU 作业 `4057218` 完成后取日志，确认 2 项 GPU 契约转绿并记录到本文档。
2. 若要把两圈系数真正接入生产链（`hR_PDF` / `tmd_matching_hybrid`），需要新增
   `order=2` 分支与相应用户参数——本轮只提供系数代数，不改动既有 NLO 路径。
3. 尾部解析式接入 `hR_x` 需要在 `_hybrid.py` 增加"尾部分段 + 解析积分"路径，
   并同时更新 `pyqcd/testing/_renorm_numerical_guards_contract.py` 的复数通道断言。
4. 提交与打标签按用户指令执行（`push` + `tag`），标签沿 `stab<N>` 递增。
