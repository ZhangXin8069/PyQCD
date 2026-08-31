---
name: pyqcd-pipeline
description: |
  Use when running or validating the PyQCD distillation pipeline, reproducing the
  docker baseline, resuming configuration-level jobs, checking input/output guards
  or runtime cleanup, collecting ETA or environment snapshots, or coordinating
  examples/test0 and test9; use pyqcd-infra for backend/MPI details,
  pyqcd-analysis for product outputs, and pyqcd-statistics for statistical
  contracts.
metadata:
  openclaw:
    emoji: 🏭
---

# pyqcd-pipeline — 管线编排与可复现运行

## 目的与边界

本技能负责编排已经实现的计算步骤、输入检查、断点续跑、持久化和结果验证，不在入口
中重写物理算法、统计估计或后端适配。管线实现位于 `pyqcd/pipeline/_steps.py`，示例/回归
编排位于 `examples/test0/main.py`；分析/绘图产品转交 `pyqcd-analysis`，统计契约转交
`pyqcd-statistics`，报告转交 `pyqcd-docs`。

按需读取：

| 任务 | Reference |
|---|---|
| 运行、重跑、失败清理、OPE 命中/重算/load-only、发布竞态、基线一致性和归档 | [`references/runbook.md`](references/runbook.md) |
| 输入/输出守卫、OPE contract 字段/状态/检查顺序、进度和环境快照 | [`references/guards-and-metadata.md`](references/guards-and-metadata.md) |

## 九步管线

```text
env → vertex → 2pt → ope → 3pt → 4pt → analysis → plots → report
```

| 阶段 | 主要产物 | 责任边界 |
|---|---|---|
| env | `env.json` | 记录 git、依赖、XeLaTeX、GPU、命令行 |
| vertex/2pt/ope/3pt/4pt | 组态级中间数据 | 计算委托 `pyqcd.pipeline`，逐步检查 shape |
| analysis/plots | JSON、拟合、图表 | 产品输出由 `pyqcd-analysis` 负责；统计契约由 `pyqcd-statistics` 定义 |
| report | `.tex/.pdf` | 由 `pyqcd-docs` 完成双遍编译和版式验收 |

## 常用入口

```bash
python examples/test0/main.py run --conf-ids 6250
bash examples/test0/run-local.sh
python examples/test0/main.py verify --run-dir examples/test0/v<ts>
python examples/pyqcd/verify_consistency.py
python examples/pyqcd/test9_gluon_tmd_nucleon.py --smoke
python examples/pyqcd/test9_verify.py <run_dir>
python -B -m pyqcd.testing._ope_channel_contract
python -B -m pyqcd.testing._field_strength_cache_contract
python -B -m pyqcd.testing._pipeline_runtime_contract
python -B -m pyqcd.testing._pipeline_persistence_contract
```

冒烟只证明链路、形状或受控断言；`Nconf<2` 时 disconnected 统计没有物理意义。全量
一致性必须与声明的 docker-v20260805 参考产物比较，不能把文件存在当作数值一致。

## 推荐流程

1. 运行前：解析组态集合和输入模板，执行数据守卫，保存 `env.json`；记录后端、精度、
   进程/设备映射和目标输出目录。未指定 `run_dir` 时统一使用配置中的 `OUTPUT_DIR`，
   显式 `run_dir` 原样保留。
2. 运行中：按 `(step, conf)` 记录开始/结束、耗时和 ETA；组态级产物齐全时按断点语义
   跳过，`recompute_2pt=True` 才强制重算；完整性必须由可读内容门确认。
3. 运行后：逐步核对产物、shape、元数据和退出码；使用 verify 对照基线，区分中间数据
   误差与分析结果误差。
4. 只有输入守卫、计算步骤和验证都通过，才进入报告；失败项保留 raw/intermediate
   及摘要，禁止用空文件或默认值掩盖失败。

持久化不变量是：最终数组可读、含 `data`、shape/dtype 符合且非空；原子发布、断点完成门
以及 `size=1` MPI 的 `recompute_2pt` 透传规则见
[`references/runbook.md`](references/runbook.md)。`pyqcd.parallel --dry-run` 只证明规划与
collective preflight，不证明输入存在或计算成功；实现细节转 `pyqcd-infra`。

## OPE 复用硬门

修改或消费 OPE 缓存前必须按上表同时读取运行行为与字段守卫；入口只保留四条不可绕过的门：

1. strict resume 的单位是三个 component 加一个 combined 的完整 artifact set；全套通过同一
   请求的严格 contract 才能命中。
2. 任一 artifact/contract/payload/组合关系失败都必须完整重算；legacy、`stale` 或来源不可
   正向验证的产物永不构成 resume hit。
3. `load_ope` 是 historical load-only：legacy 返回 `missing` 且不伪造 spec；合法 metadata
   对应的来源过期时仍返回 combined 和已有 spec，标记 `stale`、记录警告且不触发重算。
4. 要求当前来源的下游必须显式拒绝 `stale`；source stat、逐文件 replace 和发布后复查都不
   提供完整 ABA 防护或四文件线性化保证。

## 持久化与异常清理

原子 HDF5 发布、步骤完成门、GPU 后同步和清理异常优先级以 `runbook` 与
`guards-and-metadata` 为唯一细节来源。硬门是：best-effort 清理不得覆盖主异常；计算成功
后的同步失败仍须报告；Torch CPU 不触碰 CUDA runtime。

## 验收标准

| 层级 | 证据 |
|---|---|
| 冒烟 | 指定组态完成、产物 shape 正确、统计限制已标注 |
| 回归 | 基线比较的容差、NaN 位置、组态数和 PASS 数可复现 |
| 真实运行 | 输入守卫、逐组态日志、环境快照、摘要和错误清单齐全 |
| TMD 链 | 另过 `pyqcd-tmd-chain`/`pyqcd-tmd-algorithm` 的物理门，不以管线成功替代物理验证 |

## 常见故障

| 现象 | 处理 |
|---|---|
| 输入缺失/损坏 | 由守卫列出具体文件和大小，修复输入后只重跑受影响组态 |
| 中间量超差 | 先比对 backend/precision、版本和输入，再定位首个超差阶段 |
| 单组态失败 | 保留日志并断点重跑；连续同点失败时停止扩散，查共性根因 |
| 统计步骤报奇异 | 转 `pyqcd-statistics` 检查 `Ncfg`、协方差和 SVD，不能静默换估计量 |
| 并行 OOM | 转 `pyqcd-infra` 按显存公式重新 dry-run，不删任务集合 |

## 交接

向分析层交付输入清单、组态索引、产物路径、shape、退出码、日志和验证摘要；产品输出 →
`pyqcd-analysis`，统计/拟合契约 → `pyqcd-statistics`。向报告层交付已核实的图表/JSON/
源码证据。后端与 MPI → `pyqcd-infra`，物理链 → `pyqcd-tmd-chain`，文档 → `pyqcd-docs`。
