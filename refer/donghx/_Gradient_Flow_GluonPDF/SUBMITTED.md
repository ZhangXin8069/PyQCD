# L32x96 B 方案提交记录（2026-10-21）

本目录代码已经冻结于 `request.json`；不要修改可能仍被排队任务读取的源码。
正式提交记录与完整 sbatch 命令见 `submission.json`，精确流时间见 `flow_schedule.csv`。

| 阶段 | 作业号 | 本次检查时状态 |
|---|---|---|
| CPU 代数及完整输入头检查 | 3863493 | COMPLETED，退出码 0:0 |
| GPU 首末流时间测试（conf10000） | 3863494 | RUNNING，na800-sug / gpu043 |
| 20 conf × 7 flow 数组 | 3863495_[0-139] | PENDING，afterok:3863494 |
| 全部输出 CPU 审计 | 3863496 | PENDING，afterok:3863495 |

状态为提交后的初始快照，不代表 GPU 回归或批量生产已完成。GPU 测试失败会阻断后续数组。
所有 GPU 作业指定 `--partition=na800-sug --reservation=donghx_223`，数组未设置 `%` 并发上限。

CPU 门已核实 20 个组态、2980 个输入文件头及文件指纹；权重、时间/APBC 映射、
六 Wick 项收缩、C2 降维、三通道投影的小张量代数检查通过。
这些检查不构成真实 GPU 数值回归、带权 C3 时间窗系统误差或物理 matching 的验证。

采用 L32x64 当前正式七点 tau64=0.2..0.8，按 a64=0.0897 fm、a96=0.0775 fm
匹配物理流半径；tau96=tau64*(a64/a96)^2。额外 tau64=0.1 诊断点未自动纳入。
固定 epsilon96=0.006698097814776275，nstep=40,60,...,160。
保存 Pz=0,+/-3,+/-4,+/-5,+/-6；tsep/a=7..12，所有 dt=0..tsep，
四个 current anchors 为 0,24,48,95；三通道 connected u、d 及差分 u-d。

保留 basis 和 flowed gauge；vertex/VVV 不保存磁盘缓存。
basis 预计约 2.68 TiB（不是已使用量，也不是用户配额已获确认）。
结果位置：`C3_C2_schemeB/<flow_tag>/<conf>/tJxxx/tsepxx/dtxx/correlators.npz`。
读取时使用 `common.read_case` 并核验 receipt，不以文件存在代替完整性检查。

查询状态：

```bash
squeue -j 3863493,3863494,3863495,3863496
sacct -j 3863493,3863494,3863495,3863496 --format=JobID,JobName,Partition,State,ExitCode,Elapsed
```

同物理流半径不等于同物理动量或同物理源汇距离；本次没有执行归一化、拟合或 matching。
