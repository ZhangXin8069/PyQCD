# 梯度流核子胶子 TMD-PDF 全链生产计划

**目标:** 对 `beta6.20_mu-0.2770_ms-0.2400_L24x72` 的 12 个组态
`6250,6450,6650,6850,7050,7250,7450,7650,7850,8050,8250,8450`
完成可恢复的蒸馏顶点、2pt、3pt、Wilson flow、胶子 OPE、ratio、准 PDF 与
诊断性 TMD-PDF 计算，保存中间和最终产物，并形成中文 LaTeX 报告。

**架构:** 生产驱动只编排现有 PyQCD 公共入口，不复制物理算法。三个互斥 worker
分别绑定 V100 与两块 P100，每个 worker 处理不相交组态；所有缓存沿用 canonical
HDF5 契约，分析层只消费通过 shape/dtype/finite/身份校验的产物。分析状态固定为
`diagnostic_only/physical_acceptance=false`，因为本任务只有一个格距，且没有同几何
soft/rapidity、胶子-夸克混合和连续极限证据。

**技术栈:** Python 3.10、PyQCD、PyTorch/CUDA、CuPy、h5py、NumPy/SciPy、
lsqfit/gvar/iminuit、Matplotlib、XeLaTeX。

**规格:** 用户 2026-11-01 与 2026-11-02 指令；`AGENTS.md`；`skills/pyqcd-tmd-chain`；
`skills/pyqcd-tmd-algorithm` 的 geometry/renormalization/validation/gpu-production
references；`skills/pyqcd-pipeline` 的 runbook/guards references。

**全局约束:** CPU、RAM 与 GPU 显存占用不得超过 80%；磁盘不设置人为上限
（2026-11-02 用户明确覆盖此前的 80%/95% 讨论）。不修改 `/public`；提交只包含
显式路径；report 必须双遍或三遍 XeLaTeX，并满足 Overfull、Float too large、
Missing character 均为零；总任务实际 wall time 不少于六小时；不得伪造物理闭环。

## 恢复现场审计（2026-11-01 18:14）

原 TUI 运行 `20261101T012354.TgopEG` 的进程与 session id 均已消失，现按
checkpoint 手动恢复。断点状态如下：

| 组态 | distillation | flow | OPE | 恢复动作 |
|---|---|---|---|---|
| 6250 | complete | complete | complete | strict validation 已通过，不重算 |
| 6450 | complete | complete | partial | 从 OPE 继续，严格缓存命中已有调用 |
| 6650 | complete | complete | partial | 从 OPE 继续，优先完成 tau=2/3 |
| 6850 | partial | pending | pending | VVV/VdV 已落盘；阶段重入后先命中顶点缓存，再完成 2pt/3pt 与 flow |
| 7050–8450 | pending | pending | pending | 按原三 worker 分片执行 |

容量预算以已完整的 6250 为单组态基准：生产目录最终约 42.897 GiB；当前目录约
14.06 GiB，尚需新增约 28.835 GiB。磁盘不作为本次停止条件，但仍需在文件系统容量
耗尽前完成原子写入与完整性校验。

恢复顺序固定为：资源门通过 → 三个原 worker 从 checkpoint 续跑 → strict loader 与
finite-b audit → Task 3 分析 → Task 4 报告和最终日志。任何阶段不得删除或覆盖已验证
缓存，也不得把 `diagnostic_only` 结果升级为物理接受。

## 资源门状态（2026-11-02 04:22）

CPU、RAM 与 GPU 显存仍按 80% 门校验。三个 worker 曾因磁盘占用约 88% 被
`SIGSTOP` 暂停；用户随后明确“硬盘不设置上限”，已用 `SIGCONT` 恢复，checkpoint、
GPU 内存和进程状态保持不变，未删除任何数据。

## 并行审计与代码加固（2026-11-02 04:36）

四个并行子代理均已完成只读审计，报告保存在
`data/runs/tmd-full-20261101/analysis/agents/`：

- `production_audit.md`：258 个 HDF5 可读，已存在产物 finite/contract 检查通过；
  审计快照生产为 `203/456` 逻辑产物，当前仍在续跑。
- `physics_chain_audit.md`：确认当前数据集最多支持 `diagnostic_only`；soft/rapidity、
  胶子--夸克混合、完整 matching 和连续极限必须跳过。
- `analysis_report_plan.md`：给出完整图表/JSON/报告清单，并识别 Sumratio 轴、
  effective-mass 时间轴、manifest provenance 与报告 fail-open 风险。
- `verification_audit.md`：CPU-only 契约审计，最终工作树聚合为 124/124 通过。

已在主工作树完成并验证以下加固：

1. `analyze_full_production.py` 的 Sumratio 直接消费
   `(sample,z,b,tsep,insertion)`，新增 `expand_sumratio_inputs()` 并加入微型回归测试。
2. effective-mass 使用物理中心时间 `t=1..Nt-2`，平台窗、NPZ 和图的横轴统一；
   新增解析指数恢复测试。
3. 正式分析默认强制 12 个目标组态；部分运行必须显式 `--allow-partial`。
4. `build_full_report.py` 对缺失/空图像、缺失 JSON、非完整组态和非 `diagnostic_only`
   状态 fail-closed；报告中的组态数和 worker 数从分析 JSON 派生。

回归结果：`_tmd_full_analysis_contract` 12/12、`_tmd_full_report_contract` 3/3、
`_tmd_full_production_contract` 11/11 通过。

## 自动收尾器（2026-11-02）

`data/runs/tmd-full-20261101/run_finalize.sh` 已通过 `bash -n`、dry-run 和独立
`setsid` 启动验证。它等待 12 个组态目录、12 份 flow history 和三个 active worker
checkpoint 全部 complete，并强制总任务至少运行到 2026-11-02 10:09:49 +0800。
随后按顺序执行全量 `python -m pyqcd.testing`、full analysis、strict
`verify_full_production`、中文 TeX/Markdown 报告生成、三遍 XeLaTeX、三零检查、
final `log` 归档和 `git diff --check`。收尾器只写运行目录、`docs/` 与 `logs/`，
不删除生产数据、不自动提交或推送。

### 指令再次确认（2026-11-02 05:23）

用户再次确认原始范围：使用 `6250–8450` 全组态，完成 VVV/VdV、2pt、3pt、梯度流、
裸/重整化矩阵元、ratio、准 PDF 和诊断 TMD-PDF 的生产与分析，并输出 effective-mass、
平台拟合及物理量图表到 `docs/`、日志到 `logs/`。CPU、RAM、GPU 显存继续按 80% 门；
磁盘按用户后续明确指令不设人为上限；非格点 matching 自动跳过。不得因任务长或局部
结果可用而缩小范围；最终物理状态仍为 `diagnostic_only`、`physical_acceptance=false`。

### Task 1: 可恢复生产驱动与资源门

**Blocked by:** None (can start immediately)

**Delivers:** 一个 worker 可按组态分片执行 VdV/VVV、双极化 2pt、连接 3pt、flow
history、三流时 flowed gauge、三维方向 full-channel/legacy OPE 与二维 staircase OPE；
每阶段可断点恢复，运行前后记录 CPU/GPU/RAM/disk 使用。

**文件:**
- 新建: `pyqcd/testing/tmd/run_full_production.py`
- 新建: `pyqcd/testing/_tmd_full_production_contract.py`
- 修改: `pyqcd/testing/__init__.py`

**接口:**
- 消费: `compute_vertices_multi`、`compute_2pt_polarizations_multi`、
  `compute_3pt_for_config`、`flow_gauge_for_config`、`compute_tmd_ope_time`、
  `compute_tmd_ope_time_2d`、`free_gpu_memory`。
- 产出: `run-worker --run-dir <path> --conf-ids ... --backend ... --device ...`
  及唯一 worker checkpoint、资源 JSONL、strict HDF5 caches。

- [ ] **Step 1: 写失败测试**
      在 `_tmd_full_production_contract.py` 用 mock 阶段函数验证组态有序执行、
      已完成宿主缓存跳过、失败 checkpoint、GPU/RAM/disk 80% 硬门与非法参数拒绝。
- [ ] **Step 2: 运行确认失败**
      运行: `PYTHONPATH=data/deps:. python -B -m pyqcd.testing._tmd_full_production_contract`
      预期: 因新模块不存在而 FAIL。
- [ ] **Step 3: 最小实现**
      实现 CLI、阶段状态机、原子 JSON checkpoint、资源守卫、flow history 与缓存复用；
      不在驱动内重写任何物理公式。
- [ ] **Step 4: 运行确认通过**
      运行同一 contract；预期全部 PASS。
- [ ] **Step 5: 交付检查**
      `git diff --check`；只跑单组态 mock/CPU smoke，不启动完整生产。

### Task 2: 12 组态生产

**Blocked by:** Task 1

**Delivers:** 12 个组态的三方向 VVV/VdV、`nopol/pol35` source-resolved 2pt、
`t_sep=12` 连接 3pt、`tau=1,2,3` flowed gauge/flow action history、1D/2D 胶子
OPE 与 legacy OPE 全部落盘。

**文件:**
- 产出: `data/runs/tmd-full-20261101/data/conf<ID>/*.h5`
- 产出: `data/runs/tmd-full-20261101/flow_history/*.npz`
- 产出: `data/runs/tmd-full-20261101/workers/*.json`
- 产出: `data/runs/tmd-full-20261101/env/*.json`

**接口:**
- 消费: Task 1 CLI；`/public` 的 gauge/eigen/perambulator。
- 产出: 通过 strict loaders 可重算的完整中间缓存。

- [ ] **Step 1: 写失败测试**
      在 Task 1 contract 中断言 `tau=1,2,3`、`z=0..12`、1D `b=0..6`、
      2D `b_x=0..6,b_y=0..2` 与三方向 metadata 进入调用。
- [ ] **Step 2: 运行确认失败**
      运行 contract；MUST FAIL before implementation。
- [ ] **Step 3: 最小实现**
      启动 3 个 worker：V100 处理 6 组态，两块 P100 各处理 3 组态；CPU 每进程 4 线程。
- [ ] **Step 4: 运行确认通过**
      以 strict loader、逐文件 shape/dtype/finite 和 worker checkpoint 审计全部 12 组态；
      记录实际耗时、GPU 峰值与磁盘增量。
- [ ] **Step 5: 交付检查**
      `/public` 未改动；所有资源采样低于 80%；失败组态只做有界断点重试。

### Task 3: 统计、矩阵元与 PDF/TMD-PDF 分析

**Blocked by:** Task 2

**Delivers:** effective mass、谱学平台、方向/动量比较、流作用量、1D/2D ratio-v2/v3、
c0/Sumratio、流时与流窗比较、准 PDF、诊断 TMD-PDF、CS kernel 与完整状态 JSON。

**文件:**
- 新建: `pyqcd/testing/tmd/analyze_full_production.py`
- 新建: `pyqcd/testing/_tmd_full_analysis_contract.py`
- 修改: `pyqcd/testing/tmd/pdf_workflow.py`
- 修改: `pyqcd/testing/__init__.py`
- 产出: `data/runs/tmd-full-20261101/analysis/**`
- 产出: `data/runs/tmd-full-20261101/plots/**`

**接口:**
- 消费: `load_multi_2pt_polarizations`、`load_full_channel_tmd_ope_all`、
  `load_full_channel_tmd_ope_2d_all`、`load_tmd_ope_all`、
  `calculate_gradient_flow_tmd_ratios*`、`run_disconnected_tmd_ratio`、
  `load_verified_plateau_inputs`、`run_tmd_pdf_chain`。
- 产出: 带轴名、单位、Nconf、JK/bootstrap、SVD、窗口、状态和来源 hash 的 NPZ/JSON/PNG。

- [ ] **Step 1: 写失败测试**
      用合成 2 组态确认三方向 momentum 映射、2D staircase identity、复 helicity 相位、
      c0 平台状态传播、PDF 缺失 soft/rapidity 时不得升级物理状态。
- [ ] **Step 2: 运行确认失败**
      运行: `PYTHONPATH=data/deps:. python -B -m pyqcd.testing._tmd_full_analysis_contract`
      预期: FAIL。
- [ ] **Step 3: 最小实现**
      复用现有统计入口，补充 tau-qualified 产品命名和 aggregate summary；绘图全部由
      已验证数组生成。
- [ ] **Step 4: 运行确认通过**
      运行新 contract、`_tmd_physical_contract`、`_gradient_flow_gluon_contract`、
      `_fit_identifiability_contract` 与分析安全 contracts。
- [ ] **Step 5: 交付检查**
      检查所有图中无 NaN 冒充零误差；不可识别/平台失败明确显示。

### Task 4: 中文报告与日志归档

**Blocked by:** Task 3

**Delivers:** 中文 `report_tmd_full_production_20261101.tex/.pdf`、结果摘要 Markdown、
数据 README、final log，以及可复现命令、图表、源码映射、状态和误差账本。

**文件:**
- 新建: `docs/report_tmd_full_production_20261101.tex`
- 新建: `docs/report_tmd_full_production_20261101.pdf`
- 新建: `docs/summary_tmd_full_production_20261101.md`
- 新建: `data/runs/tmd-full-20261101/README.md`
- 修改: `docs/AGENTS.md`
- 新建: `logs/v<timestamp>.txt`
- 修改: `logs/AGENTS.md`

**接口:**
- 消费: Task 3 的 JSON/NPZ/PNG、worker/资源日志和测试输出。
- 产出: 双遍/三遍 XeLaTeX 报告与最终归档。

- [ ] **Step 1: 写失败测试**
      报告验收脚本检查源文件、PDF、`Pages`、三项零诊断、图表路径与数据状态。
- [ ] **Step 2: 运行确认失败**
      运行报告验收，PDF 尚不存在时预期 FAIL。
- [ ] **Step 3: 最小实现**
      按 15 步结构写报告；物理解释与代码映射逐图绑定；明确 nonphysical matching 层
      被跳过，最终状态为 diagnostic-only。
- [ ] **Step 4: 运行确认通过**
      XeLaTeX 至少两遍；grep 三项零诊断；无可裁切/重叠；运行 `git diff --check`。
- [ ] **Step 5: 交付检查**
      只登记本次文件；不自动暂存、提交、推送或打标签。
