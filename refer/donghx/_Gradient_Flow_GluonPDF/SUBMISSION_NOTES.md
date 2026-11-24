# CPU 提交补充说明

预检3954819已退出失败，原因是L96存储单份公共Gamma模板；原源码/lock保存在00_audit，修正只做明确模板广播后的全分量比较。
修正后预检3954822在CPU上完成，三套真实fixture、169个合成全链产品、虚部保留、连续/flow/m0和PDF测试全部通过。

生产bootstrap：3954826 (L64,8flow)、3954827 (L96,7flow)、3954828 (L24,7flow)。
ratio/跨flow C2审计：3954829。L64裸拟合：3954830 (72个channel/flavor/flow逻辑任务)。

后续大数组被调度器以AssocMaxSubmitJobLimit拒绝，没有产生那些作业。未改QOS、未取消原GPU任务。
改用 `submit_packed_tail_v2.py` 完成剩余提交：每ensemble一个CPU作业，8个进程处理全部63个裸拟合；全阶段分析一个8CPU作业、绘图一个4CPU作业、最终审计一个CPU作业。
数值核心、数据、索引和冻结source_lock不改；新增dispatch有独立packed_tail_lock_v2。

全部jobid、依赖和命令以submission.json为准。后续检查可用：

```bash
/public/home/donghx/AnaConda/bin/python -B submit_packed_tail_v2.py
```

该入口只补尚未记录的提交，不重交已有作业。不要用重复sbatch绕过账户限制，也不要修改正在运行/排队所读源码。
