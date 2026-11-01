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

**规格:** 用户 2026-11-01 指令；`AGENTS.md`；`skills/pyqcd-tmd-chain`；
`skills/pyqcd-tmd-algorithm` 的 geometry/renormalization/validation/gpu-production
references；`skills/pyqcd-pipeline` 的 runbook/guards references。

**全局约束:** 每个资源占用不得超过 80%；不修改 `/public`；提交只包含显式路径；
report 必须双遍或三遍 XeLaTeX，并满足 Overfull、Float too large、Missing character
均为零；总任务实际 wall time 不少于六小时；不得伪造物理闭环。

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

