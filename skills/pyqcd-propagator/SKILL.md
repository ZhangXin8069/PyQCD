---
name: pyqcd-propagator
description: |
  Use when a PyQCD task must solve Wilson or Clover quark propagators with PyQUDA,
  choose a source or multigrid setup, construct sequential sources, or implement a
  covariant nonlocal displacement; use pyqcd-physics-correlator for the required
  contractions and pyqcd-infra for backend, I/O, and MPI planning.
metadata:
  openclaw:
    emoji: 🌀
---

# pyqcd-propagator — PyQUDA 传播子生产

## 目的与边界

本技能把 `pyqcd-physics-correlator` 给出的传播子清单变成可执行的 PyQUDA 作业：
读规范场、初始化格点和 Dirac 算子、构造源、求逆、做残差/形状检查并落盘。它只负责
数据生产，不重新推导 Wick 缩并、不做谱拟合或图表；后者分别转交
`pyqcd-physics-correlator`、`pyqcd-physics-spectrum` 和 `pyqcd-analysis`。

API 细节见 [`references/solver.md`](references/solver.md)；顺序源和协变位移见
[`references/sequential-and-covdev.md`](references/sequential-and-covdev.md)。

## 求解前契约

在启动 MPI 前写出并保存：

| 项目 | 必须固定 |
|---|---|
| 组态 | 路径/格式、格点尺寸、规范作用量参数、组态 ID |
| 费米子 | Wilson/Clover、质量或 κ、`csw`、各向异性、求解容差/最大迭代 |
| 源 | 点/壁/体积/已存传播子、位置、动量相位、smear 参数和源时刻 |
| 目标 | flavor、sink/source 轴序、投影、所需 `t_sep`、保存格式 |
| 几何 | raw/smeared 规范场用途、covDev 方向/长度、Wilson 线约定 |

轴序、单位、γ 矩阵和反周期边界先按 `pyqcd-conventions` 固定；显存、后端和进程数
按 `pyqcd-infra` 的 dry-run 结果执行。源清单不完整时先停，不用默认参数补齐。

## 最小工作流

1. **初始化**：通过 `core.getMPIComm/getMPIRank/getMPISize` 确认 rank；先
   `core.init(...)`，再创建 `LatticeInfo`，并设定时间边界 `t_boundary=-1`。
2. **读场**：从实际格式读入规范场并 `toDevice()`；若要 stout/其他涂抹，先复制
   `gauge_raw`，分别保留 raw 与 smeared 对象。
3. **建算子**：按质量、Clover 系数、容差和 multigrid 配置 Wilson/Clover Dirac
   算子；把所有求逆放入同一 `useGauge` 上下文。
4. **构造并求逆**：逐源时刻和源类型调用 point/wall/volume 或传播子源求逆；顺序源
   固定 sink、投影和汇动量后再求逆，所有标签随结果保存。
5. **检查与落盘**：检查 solver residual、shape、dtype、source/sink 轴序和元数据；
   优先使用 `saveH5` 或 `pyqcd-infra` 的 HDF5 约定，旧 `.npy/.npz` 仅作兼容回退。
6. **收缩交接**：只按上游给出的 einsum 收缩；MPI 归约用 `gatherLattice`，且在
   `useGauge`/`gauge.use()` 关闭后执行。非 root rank 不写共享输出。

## 不能破坏的边界

| 风险 | 强制措施 |
|---|---|
| PyQUDA 局部布局与 PyQCD 全局布局混用 | 每次调用前标注 parity、tzyx、spin/color 轴，并在 gather 前核对形状 |
| smear 改写 raw 场 | `gauge_stout = gauge_raw.copy()`；raw 仅作 Wilson 线/covDev |
| 归约 API 猜错 | 不调用不存在的 `core.allreduce`；使用已核实的 `gatherLattice` |
| 上下文死锁或场串用 | 求逆与 covDev 的 `use()` 不嵌套，gather 在上下文外 |
| 结果无从复现 | 保存组态、源、参数、后端、MPI 映射、seed、命令、版本和 residual |

## 验收与交接

最小验收应包括：自由场/合成场小格点的形状与 γ₅ 厄米性检查、一个实际组态的残差
记录、raw/smeared 用途核对，以及重启读写 round-trip。只有这些证据通过，才把数据交给
`pyqcd-analysis` 或 TMD 链；缺失 GPU/PyQUDA 时只能做静态检查和 CPU smoke，不能声称
真实求逆完成。

## 常见问题

| 现象 | 诊断方向 |
|---|---|
| OOM | 先按 `pyqcd-infra` 重新规划 rank/批次，再降低规模；不静默删任务 |
| 不收敛 | 检查组态完整性、Clover/质量、边界、初始猜测和 multigrid 配置 |
| 收缩结果形状异常 | 回查 parity、tzyx 与 spin/color `[sink, source]` 轴序 |
| 非 root 写出竞争 | 只让 rank 0 写，并在写前完成全局归约 |

## 路由

传播子清单与 Wick 结构 → `pyqcd-physics-correlator`；谱模型 →
`pyqcd-physics-spectrum`；统计/拟合 → `pyqcd-statistics`；后端、HDF5、ASCII、VdV/VVV
和 MPI → `pyqcd-infra`；纯规范场 → `pyqcd-gauge`；批量运行 → `pyqcd-pipeline`。
