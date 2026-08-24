---
name: pyqcd-pipeline
description: |
  PyQCD 蒸馏管线技能：run_pipeline 九步全流程（env→vertex→2pt→ope→3pt→4pt→
  analysis→plots→report）的运行、一致性验证（vs docker-v20260805 基线）、
  组态级断点续跑、数据守卫（齐全度/大小一致性/输入校验）、ETA 进度日志与
  env 快照。触发于："跑管线"、"蒸馏管线"、"一致性测试"、"复现 docker 基线"、
  "断点续跑"、"数据检查"、"跑 examples/test0"。
metadata:
  openclaw:
    emoji: 🏭
---

# pyqcd-pipeline — 蒸馏管线运行与验证

## 目的与边界

调用 `pyqcd.pipeline.run_pipeline` 复现成功实例基线
`examples/docker-v20260805/output/output_20260802_120104` 的全量结果
（10 组态 9 步）。编排代码放 examples/test0/main.py（只含测试/编排，
计算全部委托 `pyqcd/pipeline/_steps.py`——照抄 docker 逻辑、自包含）。

## 管线九步与命令

```text
env → vertex → 2pt → ope → 3pt → 4pt → analysis → plots → report
```

```bash
python examples/test0/main.py run --conf-ids 6250        # 冒烟（Nconf<2 时 disconnected 拟合自动跳过，统计无意义）
bash examples/test0/run-local.sh                         # 全量（~3-5h）
python examples/test0/main.py verify --run-dir examples/test0/v<ts>   # 一致性验证 A–E
```

## 关键约定

- **版本目录**：中间数据+图表+LaTeX 报告完整保存于 `examples/test0/v<YYYYMMDDHHMM>/`
  （test12 约定）；产物 IO 一律 h5py（save_tensor_h5/load_tensor_h5，见 pyqcd-infra）。
- **一致性容差**：中间数据 rel<1e-6；分析结果 rel<1e-8。verify 按组态数自适应
  （Nconf=10 时 B/C/D 统计量严格比对）。已验证：conf6250 中间数据逐位一致
  （rel=0.000e+00），全量 237/237 PASS。
- **断点续跑**（L1）：2pt 组态级——corr 齐全即跳过该组态；
  强制重算用 recompute_2pt=True。
- **数据守卫**（L2/B8）：`pyqcd.pipeline._validate` ——原始数据齐全度检查、
  输入数组校验、模板占位符组合式文件存在性+大小一致性守卫
  （check_files_existence，corrupted 归类）。
- **进度日志**（L4）：`ProgressLog/progress_log` —— tlog 时间戳 + ETA。
- **环境快照**（E4）：`pyqcd.tools._env.dump_env` → env.json
  （git/包版本/xelatex/GPU/cmdline），每 run 必存。

## 工作流程

1. 运行前：dump_env 存档 → 数据守卫检查输入齐全度（不齐早停并报缺清单）。
2. 运行中：progress_log 记步时与 ETA；中断后按断点续跑语义重启（已完成组态自动跳过）。
3. 运行后：verify 对照基线（A–E）；汇总 summary/timing/verify JSON 入 v<ts>/。
4. 失败处理见下表；回归通过后再进入 analysis/plots/report 步。

## 错误处理

| 场景 | 处理 |
|---|---|
| 输入文件缺失/损坏 | check_files_existence 报清单后早停，不半途崩 |
| 中间量与基线超差 | 先查后端/精度是否一致（set_backend/set_precision，见 pyqcd-infra）再查改动 |
| Nconf<2 统计步骤 | disconnected 拟合自动跳过属预期，冒烟模式勿比统计量 |
| 单组态失败 | 断点续跑重跑该组态；连续 3 组态同点失败 → 查共性根因（debug 纪律） |

## 与其他技能配合

- 并行加速（元任务调度/GPU 绑定）→ `pyqcd-infra`（plan_parallel/run_parallel_pipeline）；
- 分析步细节 → `pyqcd-analysis`；物理链正确性 → `pyqcd-tmd-chain`；
- 报告产物 → `pyqcd-docs`。
