# AGENTS.md — pyqcd/parallel

资源探测与 MPI 元任务调度。`plan_parallel` 按 `N*a=n*b` 规划进程数和每卡进程，
`run_parallel_pipeline` 负责 `(step,conf)` round-robin、GPU 绑定及任务后释放。

## 入口与约束

```bash
python -m pyqcd.parallel --dry-run --confs 6250,6450
```

- 分析、作图和报告只在 rank 0 执行。
- 不得在无 MPI/CUDA 的普通配置初始化中启动真实并行任务。
- 失败路径必须有确定性退出与错误传播，不能用空结果掩盖缺卡或通信失败。
