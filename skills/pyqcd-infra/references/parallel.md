# Parallel reference — MPI meta-task scheduling

## 适用范围

任务涉及 MPI 元任务、GPU 绑定、显存规划、并行管线或残缺产物补跑时读取。本机制
不是 lattice domain decomposition；传播子内部的 PyQUDA grid 仍由 `pyqcd-propagator`
处理。

## 规划

按用户给定公式

\[
N\,a=n\,b,
\]

其中 `a` 是单元任务显存，`b` 是单卡可用显存（通常取 80%），`n` 是 GPU 数，求进程数
`N`；批次 `X=m/N`，每卡进程 `Y=N/n`。实际调用前以 `plan_parallel(...)` 的返回值
为准，不手算后覆盖。

```bash
python -m pyqcd.parallel --dry-run --confs 6250,6450
mpirun -np N python -m pyqcd.parallel --confs 6250,6450
```

## 调度与验收

- `run_parallel_pipeline` 按 `(step, conf)` round-robin，rank `mod n` 绑定 GPU。
- 每个任务完成后释放 torch cache 并触发垃圾回收；analysis/plots/report 仅 rank 0。
- 正式运行前保存 dry-run、设备/显存和任务清单；运行后核对每个任务退出码、rank0
  产物和模板/大小守卫。
- OOM 时按公式降低批次/进程数，或回退 CPU；不要静默改变任务集合。
