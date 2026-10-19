# N170_t4 测试交接记录

## 目的

这是 F48P30 的独立内存/显存测试方案。它沿用 `mpi_readVVV.py`，但把实际截取和参与收缩的本征向量数从 200 降到 170。

当前首轮只测试组态 `3460`。该组态在 `N200_t4` 中曾因 GPU 显存不足失败，适合验证这次降 Nev1 是否有效。

## 当前参数

```text
Nev=200       原始 perambulator/VVV 文件的 eigen 数
Nev1=170      程序实际保留并参与收缩的 eigen 数
NevVVV=200    VVV 文件读取参数
NevVdV=200    VdV 文件读取参数
MPI ranks=2   每个任务使用 2 个 rank / 2 张 GPU
tsource_interval=4
tsep=0..35
```

数据仍来自 F48P30 的 Nev200 数据目录；程序读取后只截取前 170 个 eigen，不会修改原始 VVV、VdV 或传播子文件。

## 内存估算

代码会为每个 rank 预分配：

```text
VVV_rank: (48, 33, Nev1, Nev1, Nev1), complex128
```

两个 rank 的 VVV 主数组估算：

```text
Nev1=200: 约 405.5 GB（377.7 GiB）
Nev1=170: 约 249.0 GB（231.9 GiB）
```

单个 VVV 时间片约从 3.93 GiB 降至 2.42 GiB。VdV、perambulator 以及收缩临时数组也会随 Nev1 降低，但 `NevVVV=200` 的单次文件读取仍会保留少量临时开销。

GPU 显存只能先估算：N200 的 3460 日志在申请约 8.19 GB 临时数组时 OOM；Nev1=170 时该类三次规模中间数组预计约 5 GB。必须以本次实际运行结果为准。

## 已确认的输入

组态 3460 的以下目录均存在：

```text
/public/group/lqcd/perambulators/.../light/3460
/public/group/lqcd/perambulators/.../charm/.../3460
/public/group/lqcd/eigensystem/.../3460
/public/group/lqcd/VdaggerV/.../Nev200/3460
/nexdata/project/lqcd/changx/VVV/.../Nev200/3460
```

VVV 已检查为 33 个文件，每个文件大小为 `12288000000` bytes。

## 本次提交状态

首轮测试已提交：

```text
Job ID: 3602603
任务名: 3460_VS_N170_T4
资源: 2 MPI ranks / 2 GPUs
当前状态: PENDING
```

提交记录保存在 `status/submitted_3460.txt`。目前结果文件尚未生成。

## 操作命令

进入本目录后生成输入并检查：

```bash
cd /public/home/changx/mission/scatter/DstarN/corr/test/vs/N170_t4
python ../../../run/make_inputs.py --overwrite
python ../../../run/check_configs.py
cat todo.txt
```

确认 `todo.txt` 只有 `3460` 后提交一个任务：

```bash
bash /public/home/changx/mission/scatter/DstarN/corr/test/vs/submit_vs.sh N170_t4 1
```

任务名应为：

```text
3460_VS_N170_T4
```

查询：

```bash
squeue -u changx
sacct -j <JobID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList
```

## 成功判据

日志中出现：

```text
组态 3460 成功结束
```

结果目录出现三个文件：

```text
results/corr_N_conf3460.npz
results/corr_Dbar_conf3460.npz
results/corr_DbarN_conf3460.npz
```

并且日志不再出现 `cupy.cuda.memory.OutOfMemoryError`、`signal 9` 或 `Unable to allocate`。

## 失败处理

先保留日志和 `status/failed_3460.txt`，不要直接覆盖或删除。若仍然 GPU OOM，下一档可测试 `Nev1=160`；若成功，再把 `expected_conf.txt` 扩展为更多可用组态，重新运行 `make_inputs.py --overwrite` 和 `check_configs.py`。
