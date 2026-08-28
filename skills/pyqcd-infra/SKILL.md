---
name: pyqcd-infra
description: |
  Use when a PyQCD task crosses backend selection, tensor I/O, ASCII or VdV/VVV
  readers, MPI meta-task scheduling, GPU binding, or memory planning; for one
  focused mode, read the corresponding reference linked below.
metadata:
  openclaw:
    emoji: 🧱
---

# pyqcd-infra — 基础设施路由

## 目的与边界

本入口只负责把基础设施问题路由到一个聚焦的契约：计算后端、数据 I/O、或 MPI 元任务。
它不含物理定义、传播子算法和统计拟合，也不把 PyQUDA 的 lattice domain decomposition
与 PyQCD 的元任务调度混为一谈。

## 按需参考

| 任务信号 | 必读 reference | 交付对象 |
|---|---|---|
| `set_backend`、torch/cupy、精度、CPU/GPU 一致性 | [`references/backend.md`](references/backend.md) | backend/precision 与误差证据 |
| h5、npy/npz、ASCII、V†V/VVV、env.json | [`references/io.md`](references/io.md) | 可 round-trip 的文件与元数据 |
| MPI、显存公式、GPU 绑定、并行管线、OOM | [`references/parallel.md`](references/parallel.md) | dry-run、任务退出码和 rank0 产物 |

只涉及一个模式时只读对应文件；跨模式任务按“backend → I/O → parallel”依赖顺序读取。

## 跨模式不变量

1. 结果形状、轴名、dtype、backend、设备、seed 和版本必须随产物保存。
2. 业务代码使用 numpy 语义，后端差异集中在适配层；I/O 转换不得静默丢失复数、精度或轴。
3. 并行前先做 dry-run，运行后核对每个 `(step, conf)` 任务和 rank0 产物；缺失项只能按
   数据守卫重跑，不能用空文件填充。

## 基础设施交接

提交给 `pyqcd-pipeline`、`pyqcd-propagator` 或 `pyqcd-analysis` 时，明确给出：
输入/输出路径、shape/轴序、dtype/backend、进程与设备映射、退出码、round-trip 或数值
对照误差，以及未覆盖的后端/规模。发现接口行为不确定时先查当前源码签名，再引用对应
reference 的 API，不凭旧示例扩展能力。

## 路由

- 传播子求解与 `useGauge` 上下文 → `pyqcd-propagator`；纯规范量 → `pyqcd-gauge`。
- 分析和拟合 → `pyqcd-analysis` / `pyqcd-statistics`；TMD 物理契约 → `pyqcd-tmd-chain`
  / `pyqcd-tmd-algorithm`；全管线编排 → `pyqcd-pipeline`。
