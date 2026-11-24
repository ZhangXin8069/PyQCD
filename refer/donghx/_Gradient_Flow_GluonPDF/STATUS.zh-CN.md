# na100-ins 作业状态检查

检查时间：2026-11-17T10:18:37.668721+08:00（Asia/Shanghai）。本次读取实时队列、作业记账、日志和已有验收 JSON；没有在登录节点加载科学数组，也没有修改或重交生产任务。

## 结论

任务未全部正常完成。当前 `squeue -u donghx` 无排队或运行任务。最近提交的 L24x72 quark gradient-flow 第 121–200 个 conf 批次中断；昨天的显式 dp 性能测试已完成。另有 11 月 14 日的三个 ensemble gluon 相同物理 flow 半径补算批次在 pilot 阶段失败，尚未进入生产。

## 最新 L24x72 quark 80 conf 批次

目录：`quark_l24x72_B_conf121to200_matched_radius_20261116`。

- 目标：conf 34000–49800，步长 200，共 80 conf × 7 flow = 560 个 conf–flow 任务。
- 生产数组：`4014512_[0-23]`，24 个 worker 最终均为 `FAILED`。
- 2026-11-17 09:43:29–09:43:51 集中退出：6 个运行中的 worker 退出码为 `0:7`（信号 7 / SIGBUS），其余 18 个启动时以 `1:0` 退出。
- 已有 52/560 个有效 PASS 完成标记，即 9.2857%；这些标记记录了 13,104 个完成 case。
- 完成标记中的状态、任务编号、case_count=252 和 manifest SHA256 均一致，未发现无效完成标记。本次没有重新逐文件读取数值数组；这不等于重新通过了整批数值验收。
- 全部 7 个 flow 完成的 conf：34000、34200、34400、34600、34800、35000、35200。
- conf 35400 完成 3/7 个 flow；其他未完成任务也可能已有可复用的部分 case，不能仅由总完成标记判断它们完全没有计算。
- 尚有 508 个 conf–flow 任务没有完整 PASS 标记，见 `remaining_conf_flow_tasks.csv`。
- 总验收作业 `4014513` 因生产依赖失败取消，没有生成 `summary/complete.json`。

这次集中退出的根因尚未确认。最新 quark 日志没有明确 Python 数值异常；多节点同时中断提示可能存在公共运行环境或文件系统问题，但不能仅凭信号 7 就确认根因。另一个分区上的 CPU 验收 `3961260` 也在 09:44:01 失败，可作为后续排查公共故障的线索。

`gpu025/026/027/029` 的节点状态显示 `maint`，但这些节点已有长期带 `MAINT` 标志的预约：`donghx_250` 自 2026-10-26 生效至 2027-10-26。因此，`maint` 本身不能证明今天发生了新增维护。

## 先前 L24x72 quark 120 conf

| 批次 | PASS conf–flow 任务 | 验收结论 |
|---|---:|---|
| 第 1–50 conf | 350/350 | 完整最终验收通过，作业 3929748 |
| 第 51–100 conf | 350/350 | 已保存输出验收通过，4014488_[0-9] 和 4014500 完成；部分原始 perambulator 已消失，未完成原始输入溯源重验 |
| 第 101–120 conf | 140/140 | 完整最终验收通过，4001961 于 2026-11-16 14:02:59 完成 |

第 51–100 conf 的原总验收 `3931036` 由于原始 perambulator 文件缺失而失败；2026-11-16 的单独输出验收确认了 88,200 个已保存 case 的完整性与数值结构，跨 flow 的 C2 相对 L2 为 0。该结果不替代原始输入的端到端溯源验收。

## 显式 opt_einsum dp 性能测试

目录：`quark_c3c2_explicit_dp_benchmark_20261116`。

`4016165`、`4016174`、`4016199`、`4016204`、`4016208` 均 `COMPLETED 0:0`，warm 与 fresh 测试的结果 JSON 均为 `PASS`。两个 GPU 测试分别于 2026-11-16 18:25:47 和 18:33:46 结束。生产代码未切换为显式 dp。

## 近期 gluon 和 Wilson line 批次

| 批次目录 | 已有验收记录与实时记账 |
|---|---|
| `gluon_flow_L24x72_gpu025_tau06_12_20261114` | `PASS_ALL_COMPLETE`，976/976 conf，6,832/6,832 flow 点；GPU 生产 4000769_[0-7]、总验收 4000770 均完成 |
| `gluon_flow_gpu025_dense_tau_20261113_v2` | `PASS_ALL_COMPLETE`，L32x64/L32x96 合计 2,587/2,587 conf，12,935/12,935 flow 点；生产与总验收完成 |
| `hyp4d10_landau_wilson_100conf_20261114` | 最终验收 PASS，C32P29/E32P29/F32P30 各 100 conf，共 300/300；生产 4001299_[0-23]、验收 4001300 完成 |
| `gluon_flow_three_ensemble_matched_rf_gpu025_20261114` | 未完成；三个 pilot 失败，生产 4002258/4002259 因依赖失败取消，总验收 4002260 失败 |

相同物理 flow 半径的 gluon 补算 pilot 作业为 `4002251`、`4002253`、`4002255`。三个 stderr 都在 `code/common.py:77` 的 `assert md['execution'] == read(ROOT/'request.json')['execution']` 触发 `AssertionError`，发生于 2026-11-14 13:46–13:47。这与 11 月 17 日 quark 的集中 SIGBUS 是两项不同问题。需要先查明并修正 execution 元数据契约，再重新通过 pilot；当前不能将该新增 flow 网格视为已补齐。

## 恢复时的依据

L24x72 quark 应保留当前生产目录、manifest 和已有输出，按缺少完整 PASS 标记的 508 个任务恢复；启动前先验证节点环境和存储访问，恢复时仍须逐 case 校验已有文件。gluon 相同物理 flow 半径补算应先解决 pilot 的元数据断言，重新验证后再进入生产。

原始查询保存在 `squeue.txt`、`sacct_recent_na100_ins.txt`、`sacct_selected.txt`、`sinfo.txt`、`reservations.txt`；结构化状态为 `status_snapshot.json`。检查脚本为 `collect_status.py`。
