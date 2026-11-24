# CPU 验证迁移到 na800-sug

2026-11-17 按用户要求，使用 scontrol 直接调整尚未启动的 CPU 验证任务的
Partition=na800-sug、ReservationName=donghx_223，保留原 job ID 和全部依赖。
已核对 scontrol 中两个任务均未申请 GPU。

- 4021217：输入及代数检查，8 CPU、8G，已于 14:48:19 在 gpu043 启动。
- 4021230：最终全量验收，4 CPU、16G，等待生产数组 4021227 全部成功。

完整依赖链仍为：4021217 → 4021219 → 4021222 → 4021227_[0-29] → 4021230。
原提交脚本、request.json、submission.json 保留最初记录；此次调度器资源覆盖以
scheduling_override_20261117.json 以及 Slurm 当前状态为准。未重复提交生产数组。

后续核对：4021217 于 14:49:05 正常结束（COMPLETED 0:0，用时 46 秒），
30 conf / 210 conf–flow 的输入检查均通过；4021219 已在 gpu043 自动开始运行。
