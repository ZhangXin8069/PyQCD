# dev20 以来外来内容吸收与胶子 TMD-PDF 物理链纠错计划

**目标：** 在不修改 `refer/` 原始参考树、`refer/git-rep` 子模块和不可逆数据的前提下，逐候选吸收 `dev20` 快照引入的外来内容，修复 PyQCD 可证明的实现/状态错误，并把“诊断链”与“物理闭合链”严格分开。

**范围：** `dev20..HEAD` 与 `dev20` 快照重点目录 `refer/donghx/_Gradient_Flow_GluonPDF`；`refer/**` 拼音系列为主对象；`refer/git-rep` 只建立来源/固定 OID/边界，不 debug 其内部实现。`/public` 只读取用户明确引入的真实数据。

**基线：** `dev20=519ef9c`，当前 `HEAD=3e1edd4`，冻结参考重点目录由 `fc3a7c9` 跟踪，181 个文件、约 4.46 MB；`python -m pyqcd.testing` 基线为 `110 passed / 0 skipped / 0 failed`。

**不可改变边界：** 不执行外来安装/构建脚本中的任意命令，不删除或改写 `refer/**`，不把 smoke、单组态、单 Pz、单流时或 `soft_factor=1` 结果升级为物理 PDF，不提交/推送/打标签。

## 验收标准

1. 参考目录和 `/public` 数据有完整来源、哈希、格式、缺失项和处置记录；所有吸收候选有唯一 `keep/adapt/reference-only/reject` 处置。
2. 对 Python/C++ 参考的 Clover、dual、Wilson line/staple、流积分和坐标轴做独立小张量 oracle；发现的偏差有文件行号、复现命令、根因和回归断言。
3. 对 ratio、重采样、真空扣除、协方差/SVD、plateau/fit 做共享索引和极限校验；无统计意义的输入不得产生有限物理结果。
4. 对 Fourier、GF→MSbar、CS/soft、NLO 匹配做符号、量纲、x 支撑、alpha_s→0 和几何窗口校验；错误的 x 依赖必须修复并加回归。
5. `pdf_workflow` 和 `verify_pdf_workflow` 对当前严格 HDF5 缓存、部分组态、单 Pz 和失败状态给出真实状态，不生成伪“matched PDF”通过结论。
6. 相关定向测试、全量 `python -m pyqcd.testing`、`python -m py_compile`、`git diff --check` 均有本次会话证据；无法运行的真实大体积重算明确标注未验证。

## 工作分解

### Task 0: 来源与数据账本

**Blocked by:** None

**文件：** `data/runs/absorb/**/inventory.json`、`data/runs/agent-reports/*.md`、最终 `docs/` 审计报告。

**消费：** `git diff dev20..HEAD`、`absorb_inventory`、参考 copy 脚本、`/public` 的 stat/hash/小样本读取。

**产出：** 候选矩阵、固定来源/许可/OID、Python/C++ 重点入口、数据可用性、未验证项。

### Task 1: 算符与流几何回归

**Blocked by:** Task 0

**文件：** `pyqcd/renorm/_gradient_flow.py`、`pyqcd/renorm/_tmd.py`、`pyqcd/operator/_gradient_flow_gluon_ope.py`、对应测试。

**消费：** `refer/donghx/_Gradient_Flow_GluonPDF/flowed_gluon_ope.cpp`、`ope_components.py` 与技能 geometry reference。

**产出：** 小张量独立 oracle、方向/端点/迹/dual/求和语义断言；只修有根因的偏差。

### Task 2: 统计与 ratio 失败关闭

**Blocked by:** Task 0

**文件：** `pyqcd/analysis/_tmd_ratio.py`、`_disconnected.py`、`_fitter.py`、`pyqcd/pipeline/_tmd9.py`、对应测试。

**消费：** `refer/.../calc_ratio_matched.py`、`fit_bare_latest.py`、现有真实缓存。

**产出：** C2/C3/OPE 共享重采样、真空扣除与 ensemble estimator、plateau/fit 状态和 z=0 归一化的回归证据。

### Task 3: GF→MSbar、Fourier 与 TMD 匹配纠错

**Blocked by:** Task 1

**文件：** `pyqcd/renorm/_gradient_flow_quasi.py`、`_tmdextract.py`、`_matching.py`、`_hybrid.py`、`pyqcd/renorm/__init__.py`、对应测试。

**消费：** `refer/.../gluon_flow_to_quasi.py`、`lambda_fourier_quasi_matching_v2.py`、`match_unpolarized_quasi_from_extrap_v2.py`、`docs/胶子TMD梯度流连续极限.tex`。

**产出：** 显式 z=0 policy、局域 spin-2 公共导出、x 依赖快度标度、signed-x、α_s→0 单位核、z_max/端点失败关闭；所有输出只到诊断状态。

### Task 3b: bare-matrix Sumratio-difference AIC

**Blocked by:** Task 2

**文件：** `pyqcd/analysis/_bare_sumratio.py`、`_tmd_ratio.py`、对应 contract。

**产出：** 物理 `Tmin`、forward difference、同 cut omission-AIC、逐 replica quality/fallback、`M_quality/M_filled` 分栏；PDF 链只在全点 quality 且无 fallback 时采用 Sumratio 输入。

### Task 4: 真实工作流与验证脚本

**Blocked by:** Tasks 1–3

**文件：** `pyqcd/testing/tmd/pdf_workflow.py`、`pyqcd/testing/tmd/verify_pdf_workflow.py`、相关测试。

**消费：** 当前 conf6250 严格缓存、`/public` 数据、参考来源账本。

**产出：** 传参化 tau/eps、样本传播、几何窗口、chain status、严格缓存文件名兼容；部分运行显示 PARTIAL/诊断而非虚报全通过。

### Task 5: 回归与归档

**Blocked by:** Tasks 1–4

**命令：** `python -m pyqcd.testing`；相关 `python -B -m pyqcd.testing.<contract>`；`python -m py_compile`；`git diff --check`；必要时运行小体积真实数据 smoke。

**产出：** 定向/全量结果、未验证真实大体积项、最终审计报告与 `log final` 归档。

## 收敛与停止

- 通过标准：Task 0–4 的所有可执行门通过，且 Task 5 的全量回归通过；单组态或缺少 soft/rapidity/多 Pz 时只允许 `diagnostic`/`partial` 状态，不允许“物理 PDF 完成”。
- 若某物理修复依赖缺失的多尺度数据，记录 `unverified`，不做数值猜测；继续完成可证明的代码/状态修复。
- 若连续三次修复由同一根因假设失败，停止逐症状修改，回到 Task 1/3 的 oracle 和调用栈重新定位。
