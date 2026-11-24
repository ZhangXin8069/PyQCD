# 实际调度方式：24个worker覆盖350个conf-flow任务

2026-11-04，350元素直接数组被 `AssocMaxSubmitJobLimit` 拒绝，没有注册生产作业。
实际查询账号的 `MaxSubmitJobs=200`、`MaxJobs=200`；当时已有约166个任务。
因此采用独立 `scheduler_adapter/`，不改动已排队检查任务所绑定的冻结数值代码。

- 所有50个conf和7个flow均保留，没有缩减物理/数值任务。
- 24个worker数组元素，无 `%` 并发上限。每个worker负责 `range(worker,350,24)`。
- 每次allocation只完成一个完整conf-flow，成功后用 `scontrol requeue` 重新排队同一个数组元素；
  下一次从剩余任务继续。这不会额外创建job ID，也不把约15个flow捆绑进一次72小时上限。
- 科学计算失败时直接失败，不自动requeue。断点basis及已验证case保留。
- `scheduler_adapter/request.json` 冻结适配层哈希，并绑定原 `request.json` 哈希。
- `scheduler_adapter/worker_manifest.csv` 精确记录350个conf-flow与24个worker的无重无漏映射。
- 实际job ID及资源以根目录 `submission.json` 为准；原 README 中的350元素直接数组只是最初被拒绝的方案。
- 最后CPU全审计依赖整个24-worker数组最终成功结束，不能把中途requeue视为全部完成。

输入检查和数值计算不变：B权重、a=0.105fm、650/150、7个匹配物理rf、P0±4±5±6、T7..12、全部dt、z0..15。
GPU仍为na100-ins/donghx_250、排除gpu025/gpu028，每次单节点4GPU、240G。
精确basis预计占用已由CPU门确认约3.57TiB（另需flowed gauge和C3/C2结果存储）。

首次提交适配层（已经提交则拒绝重复）：

```bash
/public/home/donghx/AnaConda/bin/python -B -u scheduler_adapter/submit_packed.py
```
