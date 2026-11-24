# gpu025 的 L24x72 quark gradient flow 已提交

新数组 4021476_[22-23]，na100-ins，reservation=donghx_250，固定 gpu025。
每项 4 GPU、4 MPI、8 CPU、240G、72h，两个 worker 可同时使用 8 GPU。

接管 worker 22、23 的剩余 42 个 conf–flow，共 10,584 case；已通过的任务由原程序跳过，
原有部分输出和 basis 按原 guards 校验复用。原任务 4020162_22、4020162_23 原为 PENDING，
交接时先 hold，在最终验收改绑成功后取消，已用 sacct 确认取消。没有中断运行中的 worker。
新数组在原两项结束后启动，防止相同 worker 同时写出。

最终验收 4020163 已改为 afterany:4020162,afterok:4021476，原数组其余 worker 保持运行。
原数组有两项预期 CANCELLED，因此 afterany 用于等待它结束；最终验收仍读取并检查全部
560 个 conf–flow 的每一个 case，没有放宽任何数值完成条件。

Slurm 在提交时仍将 gpu025 标为 DOWN，原因为 2026-11-17 11:08 意外重启。
任务已排队；实际启动需要节点恢复。本操作未修改管理员节点状态。

新 launcher 和原 request/manifest 的哈希均核对通过，数值内核、物理设置和结果目录不变。
新调度日志在本目录 logs/；科学输出仍位于原生产目录 C3_C2_schemeB/ 和 basis_cache/。
详情见 submission.json 及各个 *_verified.txt。
