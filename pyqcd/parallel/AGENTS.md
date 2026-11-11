# AGENTS.md — pyqcd/parallel

资源探测与 MPI 元任务调度。`plan_parallel` 按 `N*a=n*b` 规划进程数和每卡进程，
`run_parallel_pipeline` 负责 `(step,conf)` round-robin、GPU 绑定及任务后释放。

`einsum.py` 提供**任务级**并行入口 `einsums(tasks, ...)`：把一批相互独立的
`contract(...)` 按工时估计（`contract_path` 的 opt_cost，逐元素方程以元素数兜底）
LPT 装箱，分给多个 MPI 进程；`plan_einsums` 只规划不执行。

另有 `init(config, ...)` + `run(state)` 的分步形式：`init` 只接受列式字典
（`task_id`/`equation`/`operands`/`results`），只做校验与规划；`run` 是启动器，
无返回值，结果**原地写入**调用方传入的 `results` 缓冲区（不分配新数组/显存）。
用于把规划与执行分别计时。

## 入口与约束

```bash
python -m pyqcd.parallel --dry-run --confs 6250,6450          # 管线规划
python -m pyqcd.parallel.einsum --dry-run                      # 批量 einsum 规划
python -m pyqcd.parallel.einsum --worker <run_dir>              # worker（内部使用）
python -m pyqcd.testing.parallel_einsum verify --equation S1 --procs 2
python -m pyqcd.testing.parallel_einsum bench --equation S1 --procs 4
mpirun -np 4 python -m pyqcd.testing.parallel_einsum bench-collective --equation S1
```

- 分析、作图和报告只在 rank 0 执行。
- 不得在无 MPI/CUDA 的普通配置初始化中启动真实并行任务。
- 失败路径必须有确定性退出与错误传播，不能用空结果掩盖缺卡或通信失败。
- `einsums` 只做分发与汇总，**不得**改变 `pyqcd.einsum` 的行为或 dtype。
- **collective 默认关闭**（`collective=False`）：它让每个 rank 各持一份整批输入，
  真实生产对内存/显存敏感。默认在 mpirun 内只由 rank 0 执行、其余 rank 返回
  `None`；`collective=True` 才分摊。开启时必须打印 `COLLECTIVE_MEMORY_WARNING`。
- 包 `__init__` 走 `__getattr__` 惰性解析：`pyqcd.parallel` 的导入不得拉起
  `pyqcd.pipeline`（h5py/cupy，~0.6 s）或 `pyqcd.einsum`。
- 调度只读 `cpu_threads()` 与 `available_memory_mb()`；`detect_resources()`
  会 `import torch` 拿显存（本机 1.0 s），不得放进批量 einsum 的热路径。
