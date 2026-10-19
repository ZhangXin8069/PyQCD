# 2026-10-09：L32x64 gluon 全组态、14 个 L32x96 flow checkpoint

本次先阅读 `/public/home/donghx/.codex/Markdown/gradient_flow.md`，并与
`slurm/v5/submit_all_tau_v5.sh`、`slurm/l32x64_v1/submit_all_tau.sh` 交叉核对。
用户要求的 flow 清单是 gluon 的14点，不是 quark 的8点。
`epsilon=.01`；a=.0897fm；zdir2；zcount25。

上一轮 20conf campaign 只作为可追溯内核/exporter依赖，不改其源码：
`3825771` 内核验证、`3825768` 旧结果导出、`3825769` 两新增quark-checkpoint gate、
`3825770` 全部38数组项，均已由本次 `sacct` 确认 COMPLETED / exit 0。

## 本次提交

- `3825824`：`audit_cpu.slurm`。已确认在 `a192c1t` 的 `cpu074` 运行。
  CPU 上重新枚举全部 thin-link LIME 组态；复核14个真实数据 epsilon-halving gate；
  完整审核所有历史 raw；拆分导出 TU/SU/TH/SH 和明确的target/diagnostic。
- `3825825`：`dispatch_cpu.slurm`，依赖 `afterok:3825824`，已确认 PENDING / Dependency。
  审核成功后直接运行固定的 `allconf.py submit`，自动登记所有缺口的CPU数组和final审核。
  有缺口时数组按conf分组，至多1000组态/批，无 `%N` 并发上限；48核、64G、2天。

运行代码和binary的绑定为 `campaign.json`；准备的实际组态清单为 `inventory.json`；
精确请求为 `requests.json`；审核计数为 `validation/initial_audit.json`。
旧 `manifest_L32x64.tsv` 有1700条，但本次正式数量以CPU重新枚举/验证结果为准。

本次CPU准备已确认：1700个有效LIME输入，conf1000..17990，共14tau、23,800请求。
内核 self-test 通过；14个历史epsilon gate均重新核验通过，最大整体OPE相对L2差
`2.180709988342686e-5`，最大plaquette差 `7.511529667048045e-6`。
  完整raw审核及分量导出已经完成，结果见下述审核与恢复记录。

调度回执 `submission.json` 在 dispatch 运行时生成，逐批原子保存实际production job ID；
当前不要把登记了自动补交误写成所有production已经开始，也不要把历史目录存在
误写成23,800份结果已全部验证。

## 审核结果与账户提交上限恢复（2026-10-09）

3825824已COMPLETED/exit0，耗时38分27秒。23,800请求中23,038份raw
通过审核并拆分导出，762份缺失或无效，分属600conf组。
缺口按tau：.2=239、.5=82、.6=84、1.4=154、3.8=203；其他9点各1700齐全。
证据为validation/initial_audit.json及initial_audit_rows.jsonl。

3825825实际FAILED/exit1，原因AssocMaxSubmitJobLimit：账户MaxSubmitJobs=200，
600项数组被拒绝，旧dispatch没有登记生产数组。
恢复入口submit_bundled_recovery.py和compute_bundle_cpu.slurm：
每项依次处理10conf组，共60项；资源仍48CPU/64G/2天，无脚本并发%N上限。
已提交production 3826064、afterok最终审计3826065，回执submission.json。
本次检查production仍有运行/等待；最终完成必须由final_audit.json确认。
以上取代本文件早先的“审核进行中”和“dispatch等待”的历史快照。

## 后续核验命令

```bash
squeue -j 3826064,3826065
sacct -j 3825824,3825825,3826064,3826065 -X --format=JobID,JobName,State,ExitCode,Elapsed,NodeList -P
tail -15 logs/audit_3825824.out
python -B -c 'import json; print(json.dumps(json.load(open("submission.json")), indent=2))'
```

若审核成功、没有缺口，将只提交final审核，不重复flow。
如果有缺口，final审核依赖所有生产数组成功，`validation/final_audit.json` 必须
missing_count=0 且作业 COMPLETED/exit0，才可宣布全部完成。
任何审计错误保留于JSONL和日志；不会修改历史raw，也不会调整算符符号以迎合结果。

首次提交前 Python AST 与三个Slurm shell语法检查均通过。

## GLIBC失败项补交（2026-10-09 16:40 CST）

此前3826064共60项：9项在a192c1t成功，51项在cpu6248R/cpu004失败；
失败均为binary启动时缺少GLIBC_2.33/2.34，不是算符数值异常。
此前补齐110份，当前缺口652份，分属510conf；
按tau为0.2=209、0.5=72、0.6=74、1.4=114、3.8=183。
提交前核对冻结代码/请求清单hash、原missing_groups身份、完成回执、
metadata hash、requested identity和8个artifact大小；选中的51个bundle恰好
覆盖所有剩余缺口，不包含此前已完成的110份。数值验证仍在CPU计算节点执行。

新生产数组为**3827362**，仅使用已验证能运行binary的a192c1t；
每项10conf、48CPU、64G、2天，没有`%N`并发上限。
未改动physics代码、binary、冻结manifest、此前submission.json或已完成产品。
任务入口仍为原compute_bundle_cpu.slurm，以提交参数覆盖partition；完整回执为
recovery_glibc_v2.json。

```bash
sbatch --parsable --partition=a192c1t --job-name=gf_g64_glibc_retry --array=3-24,26,28,30-40,42-57 --export=ALL,GROUPS_PER_TASK=10,GROUP_COUNT=600 /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/gluon_l32x64_allconf_20261009/compute_bundle_cpu.slurm
scontrol update JobId=3826065 Dependency=afterok:3827362 Partition=a192c1t
squeue -j 3827362,3826065
```

3826065已成功改为afterok:3827362及a192c1t，当前PENDING/Dependency，
不再是DependencyNeverSatisfied。无须新建或取消其他审计任务。
启动检查：3827362_3/_4/_5在cpu073运行，stderr均为空；
conf8110、conf8910、conf9710的tau1.4首份产品已有COMPONENTS_VALIDATED，
证明本次程序正常进入flow、OPE与分量验证。此为启动快照，不是全部完成声明。
最终仍需3826065成功及validation/final_audit.json中missing_count=0。
