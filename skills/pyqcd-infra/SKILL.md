---
name: pyqcd-infra
description: |
  Use when a PyQCD task crosses backend selection, tensor I/O, ASCII or VdV/VVV
  readers, MPI meta-task scheduling, rank-divergent failure handling, GPU
  binding, or memory planning; not for PyQUDA lattice-domain decomposition or
  physical-observable definitions.
metadata:
  openclaw:
    emoji: 🧱
---

# pyqcd-infra — 基础设施路由

## 目的与边界

本入口只负责把基础设施问题路由到一个聚焦的契约：计算后端、数据 I/O、或 MPI 元任务。
它不含物理定义、传播子算法和统计拟合，也不把 PyQUDA 的 lattice domain decomposition
与 PyQCD 的元任务调度混为一谈。OPE strict cache 的 artifact 命中、source identity 和
多文件发布由 `pyqcd-pipeline` 负责；本技能只定义 HDF5 等 I/O 的表示、可读性和 round-trip
边界，不把单文件校验或发布后检查扩展解释为锁、完整 ABA 或跨文件线性化保证。

## 按需参考

| 任务信号 | 必读 reference | 交付对象 |
|---|---|---|
| `set_backend`、torch/cupy、精度、`complex64` eigcompress、CPU/GPU 一致性 | [`references/backend.md`](references/backend.md) | backend/precision 与误差证据 |
| h5、npy/npz、ASCII、V†V/VVV、`create_omega_accelerate`、env.json | [`references/io.md`](references/io.md) | 可 round-trip 的文件、元数据与收缩权重契约 |
| MPI、显存公式、GPU 绑定、并行管线、OOM | [`references/parallel.md`](references/parallel.md) | dry-run、任务退出码和 rank0 产物 |

只涉及一个模式时只读对应文件；跨模式任务按“backend → I/O → parallel”依赖顺序读取。

## 跨模式不变量

1. 结果形状、轴名、dtype、backend、设备、seed 和版本必须随产物保存；`complex64` 不得
   被适配层或压缩路径静默升为 `complex128`。
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
