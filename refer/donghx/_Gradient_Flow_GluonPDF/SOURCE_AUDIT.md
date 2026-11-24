# L24×72 性能审查：源码依据与优化边界

日期：2026-11-05。审查对象为原冻结生产目录
`../quark_l24x72_B_50conf_matched_radius_20261104_v2/`。

本文件记录源码可证实的重复工作，不提供未经计时验证的加速承诺。
CPU job 3929749 将另行生成 `PERFORMANCE_REPORT.md`、`performance_summary.json`
和 `case_timings.csv`，以旧数组 3926911 的日志为基础统计。

## 恢复情况

原 50 个 conf × 7 个匹配物理流半径，共 350 个 conf–flow。
补交时有 61 个完整 task 标记，289 个 task 尚无完整标记，
这不等于只有 61×252 个已发布 case：部分未完成 task 也已经产生有效 case。
恢复前用 CPU 核验所有 61 个已标完整 task；GPU 对未完成 task 逐 case 验证后仅补缺项。

## 1. 外部 VVV 的重复构造

`gpu_worker.py:c2_oracle` 对每个 case、每个动量分别构造 source/sink VVV。
随后 `contract_case` 对同一 source/sink 和同样七个动量再生成一遍。
同一个 current 的不同 `(tsep,dt)` 还会重复使用某些 source/sink 时间。

完整的一个 current 包含

\[
N_{\rm case}=\sum_{T=7}^{12}(T+1)=63.
\]

不考虑断点命中时，仅这两个阶段对应

\[
2\;\text{stages}\times 63\;\text{cases}\times7\;\text{momenta}
\times2\;\text{endpoints}=1764
\]

次 `baryon_vvv` 调用/current；四个 current 合计 7056 次/conf–flow。
这是调用数，不是耗时或可实现的加速倍数。

可优化为有界的**进程内**缓存，身份必须包括 conf、eigen 文件身份、时间、NE、
动量/相位约定和 source/sink 共轭角色。不能恢复持久化 VVV/vertex 缓存，
不能假设 GPU_MEMORY 的阶段快照就是显存峰值。

## 2. 收缩路径缓存生命周期过短

`reference/spin2_contractions.py:Contractor` 已缓存 equation 与 shape 对应的
`opt_einsum.contract_expression`，但 `c2_oracle` 和 `contract_case`
各自每个 case 都重新创建 `Contractor('optimal')`。

候选改进是让同一 rank 在多个 case 之间复用 Contractor。需先检查缓存项只持有路径表达式、
不会因形状增长或数组常量绑定造成内存累积，并比较路径规划时间与 GPU 收缩时间。

## 3. 原始右时间循环内的重复外部投影

`contract_case` 在 `sigma_b × momentum` 循环中调用 `c3_reduced`。
其中 `external_perambulators` 及若干外部旋量投影依赖同一个 case 的外部输入，
并不都依赖 `sigma_b`。可分离不变量，但必须保留 current 的完整 spin 指标、
六个 Wick 项、符号、B 权重一次性施加及原始时间有序求和。

## 4. manifest 哈希重复 I/O

`common.py:manifest_hash` 已实现单进程缓存与 stat-change 拒绝机制。
但 `load_vertices`、NPZ/receipt 发布等路径仍直接对约 22 MB manifest 再次 SHA。
可以统一使用缓存身份；不可移除或弱化校验，且应保留生产树不可变要求。
该重复 I/O 的源码证据不意味着它是 GPU 总耗时的主因。

## 5. basis 批次和多 flow 共享

原 `launch.py` 固定 `FLOW_MODE_BATCH=16`。每批再次执行 gauge/fermion flow，
可考虑增大批次，或多个目标共用 checkpoint。但这会涉及显存、QUDA restart、
不同时间窗的输入支持域，必须单独回归。

七个独立目标共需的步数为

\[
40+60+80+100+120+140+160=700,
\]

最大目标为 160 步。\(700/160=4.375\) 只是步数比，
不能直接解释为整个计算或 basis 的 4.375 倍实际提速。

## 实施顺序

1. 此次先恢复原冻结生产内核，保留已有验证依据。
2. 隔离 successor，先加入 C2 oracle、VVV、J、H、外部投影、Wick 收缩、I/O 的分阶段计时与显存峰值。
3. 先验证进程级 Contractor 与有界内存 VVV 复用，再考虑不变量提取和 flow checkpoint。
4. 固定 conf=10000，选最小/最大 flow、tJ=0/71、T=7/12，覆盖接触点、平台中点、APBC。
5. 逐块对照 C2、H、六个 Wick 项、u/d、三个通道及所有 ±P/±z；使用原 manifest 误差门。
6. 回归通过并有实测 wall time/峰值显存后，再另开冻结版本用于生产；不原地修改当前提交源码。

目前没有把上述数值优化候选混入 GPU 数组 3929747，也未提交优化基准 GPU 作业。
