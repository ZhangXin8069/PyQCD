# Pz=0 B方案补算提交回执

2026-10-20 21:09 CST 已提交。会话标题：L32x64 B方案20conf/7flow零动量三极化补算。

| 阶段 | Slurm job | 前置条件 |
|---|---|---|
| CPU小张量/投影/输入身份门 | 3861970 | 无 |
| GPU smoke：10000，tau=.2/.8 | 3861971 | afterok:3861970 |
| 140个conf-flow数组 | 3861972_[0-139] | afterok:3861971 |
| CPU全量35280-case审计 | 3861973 | afterok:3861972 |

已用scontrol核对GPU两级均：partition=na800-sug，reservation=donghx_223，
1node/4tasks/4GPU/240G/72h，无数组%并发上限。源码与输入请求在request.json冻结，
之后不得直接修改该版本的计算代码；失败修复应使用独立successor。

原始请求：同B权重、同20conf（10000..10760，步长40）、同7flow（.2..8）、
Pz=0 unpolarized/helicity/transversity。三极化共用开放current-spin收缩并保存各自复数投影；
u/d/u-d配对保留；z方向、±位移、tsep7..12、dt全覆盖、四current和窗口均保持不变。
每个conf-flow252处，总35280处。不是额外为三个通道提交三倍相同Wick任务。

独立结果根：`C3_C2_schemeB_Pz0/`。统一读取入口：`common.read_case`。
任务号到conf-flow的准确对应由CPU门生成`manifest.json`，顺序conf-major，
flow依次.2,.3,.4,.5,.6,.7,.8。完整元数据/命令/图示边界见README.md。

21:12 CST前核查：CPU门3861970在cpu014以COMPLETED/0:0完成，耗时2分9秒，
status/tests.json和status/cpu_gate.json均PASS，全部140任务输入身份通过。
manifest SHA256=e6ea7784df23adef5c3593a1ab80a221ed1aee422925da25ce243082c0acfe71。
GPU smoke3861971已在gpu043运行；140数组与CPU末端审计仍等待依赖。
此处只是提交后的历史快照，不是GPU回归完成证明。

状态必须查询Slurm和status/回执，不把本提交记录当作已计算完成：

```bash
sacct -j 3861970,3861971,3861972,3861973 \
  --format=JobID,JobName,State,ExitCode,Elapsed,NodeList -P
squeue -j 3861971,3861972 -o '%.20i %.22j %.12T %.12M %.26R'
```

CPU小张量/单动量/三投影检查已经PASS（status/tests.json）；完整输入门另看
status/cpu_gate.json。GPU真实输入C2/H/六Wick检查需smoke_task0.json和smoke_task6.json；
全量完成只能以summary/complete.json的PASS及所有raw payload核验为准。
不代表已完成Zchi/Ward、GF matching、零流时间外推或m0MSbar确定。

VVV/vertex仅显存使用、不保存缓存；复用的basis/gauge保留且不改写；
原100组bootstrap索引仍与B方案非零动量样本一致，后续需显式CPU分析才能产生ratio/裸拟合。
