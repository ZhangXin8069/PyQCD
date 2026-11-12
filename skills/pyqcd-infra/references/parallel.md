# Parallel reference — 本机资源探测

## 适用范围

任务涉及本机可用资源（CPU 线程 / 内存 / GPU 与显存 / 磁盘空闲）、批次规模估算、
显存预算或 OOM 判据时读取。**PyQCD 不再提供 MPI 任务级并行**（`parallel.einsum`、
`plan_parallel`、`run_parallel_pipeline`、KUPL/CUDA 的 MPI 分片与运行时预加载均已
整体删除），因此这里没有 rank、collective、dry-run 或产物聚合语义。

## 入口

```python
from pyqcd.parallel import detect_resources

snapshot = detect_resources()
# {'cpu_threads': 8, 'mem_total_mb': 15900, 'mem_avail_mb': 14000,
#  'n_gpu': 1, 'gpu_vram_mb': 8187, 'gpu_usable_mb': 6549,
#  'disk_free_mb': 3672045}
```

单项探测也可单独调用：`cpu_threads()`、`total_memory_mb()`、
`available_memory_mb()`、`gpu_info()`、`free_disk_mb(path)`。

`pyqcd.parallel` 只做只读探测：不调度任务、不起子进程、不导入任何分布式运行时。
探测结果是一次快照，不是配额或锁；作业运行期间其它进程会改变这些数字。

## 判据

- `gpu_info()` 返回 `(n_gpu, vram_per_gpu_mb, usable_vram_mb)`，其中
  `usable_vram_mb = 0.8 × 单卡显存`（沿用“单卡可用 80%”约定）。
- 三态语义必须保留：`None` 是 unknown，`0` 是成功探测到的 known-zero，正数是
  known-positive。已知无 GPU 返回 `(0, 0, 0)`；**所有探测器都失败**返回
  `(None, None, None)`——不能把探测失败当成 0 卡。
- `cpu_threads()` 优先 `sched_getaffinity`（尊重 taskset 绑定），回退 `os.cpu_count()`。
- 内存优先读 `/proc/meminfo` 的 `MemAvailable`；`MemTotal` 是容量而不是可用内存，
  不能在 `MemAvailable` unknown 时替代它。取不到时返回 `None`，不得伪造 0。
- 批次规模与显存预算由调用方按自己的任务形状决定；`detect_resources` 只给上限参考。

## 常见失败

- 只读容器里 `/proc/meminfo` 不可读 → 字段为 `None`，不要据此拒绝运行。
- `nvidia-smi` 超时或 `torch` 未装 → `gpu_info()` 走失败分支返回 `None` 三元组。
- `torch` 首次导入要数秒到数十秒（本机实测约 22 s 才拿到完整快照）；只需要 CPU
  线程数时用 `cpu_threads()`，不要为了一个数字调用 `detect_resources()`。
- 磁盘探测针对给定路径（默认当前目录），跨挂载点时必须显式传入目标路径。

## 已删除的 MPI 入口

旧的 `python -m pyqcd.parallel --dry-run`、`plan_parallel`、`run_parallel_pipeline`、
`get_mpi_context`、`pyqcd.testing._mpi_*_contract` 与
`pyqcd.testing.parallel_einsum` 已随 MPI 支持一并删除；不要再引用这些命令、契约
入口或产物（`rank0` 输出、任务退出码聚合、collective rendezvous）作为验收证据。
需要并行时按物理/算法层自行拆分进程（例如多个 `python -m pyqcd.testing...` 进程
各跑不同组态），由调用方负责结果聚合。
