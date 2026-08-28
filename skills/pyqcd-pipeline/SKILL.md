---
name: pyqcd-pipeline
description: |
  Use when running or validating the PyQCD distillation pipeline, reproducing the
  docker baseline, resuming configuration-level jobs, checking input/output guards,
  collecting ETA or environment snapshots, or coordinating examples/test0 and test9;
  use pyqcd-infra for backend/MPI details and pyqcd-analysis for product analysis.
metadata:
  openclaw:
    emoji: 🏭
---

# pyqcd-pipeline — 管线编排与可复现运行

## 目的与边界

本技能负责编排已经实现的计算步骤、输入检查、断点续跑和结果验证，不在入口中重写
物理算法、统计估计或后端适配。管线实现位于 `pyqcd/pipeline/_steps.py`，示例/回归
编排位于 `examples/test0/main.py`；分析、绘图和报告分别转交
`pyqcd-analysis`、`pyqcd-docs`。

按需读取：

| 任务 | Reference |
|---|---|
| 运行、重跑、基线一致性和结果归档 | [`references/runbook.md`](references/runbook.md) |
| 输入/输出守卫、进度和环境快照 | [`references/guards-and-metadata.md`](references/guards-and-metadata.md) |

## 九步管线

```text
env → vertex → 2pt → ope → 3pt → 4pt → analysis → plots → report
```

| 阶段 | 主要产物 | 责任边界 |
|---|---|---|
| env | `env.json` | 记录 git、依赖、XeLaTeX、GPU、命令行 |
| vertex/2pt/ope/3pt/4pt | 组态级中间数据 | 计算委托 `pyqcd.pipeline`，逐步检查 shape |
| analysis/plots | JSON、拟合、图表 | 由 `pyqcd-analysis` 的统计契约解释 |
| report | `.tex/.pdf` | 由 `pyqcd-docs` 完成双遍编译和版式验收 |

## 常用入口

```bash
python examples/test0/main.py run --conf-ids 6250
bash examples/test0/run-local.sh
python examples/test0/main.py verify --run-dir examples/test0/v<ts>
python examples/pyqcd/verify_consistency.py
python examples/pyqcd/test9_gluon_tmd_nucleon.py --smoke
python examples/pyqcd/test9_verify.py <run_dir>
```

冒烟只证明链路、形状或受控断言；`Nconf<2` 时 disconnected 统计没有物理意义。全量
一致性必须与声明的 docker-v20260805 参考产物比较，不能把文件存在当作数值一致。

## 推荐流程

1. 运行前：解析组态集合和输入模板，执行数据守卫，保存 `env.json`；记录后端、精度、
   进程/设备映射和目标输出目录。
2. 运行中：按 `(step, conf)` 记录开始/结束、耗时和 ETA；组态级产物齐全时按断点语义
   跳过，`recompute_2pt=True` 才强制重算。
3. 运行后：逐步核对产物、shape、元数据和退出码；使用 verify 对照基线，区分中间数据
   误差与分析结果误差。
4. 只有输入守卫、计算步骤和验证都通过，才进入报告；失败项保留 raw/intermediate
   及摘要，禁止用空文件或默认值掩盖失败。

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

向分析层交付输入清单、组态索引、产物路径、shape、退出码、日志和验证摘要；向报告层
交付已核实的图表/JSON/源码证据。后端与 MPI → `pyqcd-infra`，物理链 →
`pyqcd-tmd-chain`，文档 → `pyqcd-docs`。
