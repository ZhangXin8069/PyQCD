# 2026-11-07 提交后状态

本轮已经完成：独立目录、配置、输入审计代码、配对局域锚点恢复、伴随表示候选线质量减除、非零距离 GF 端点转换、线性 small-flow-time AIC/窗口敏感性/外部系数敏感性、逐阶段绘图代码及 CPU 提交链。
静态 Python AST 和 bash 语法检查已通过。尚不能宣称数值单测或结果图生成完成。

最后一次 Slurm 查询中：

| Job ID | 阶段 | 状态 |
| --- | --- | --- |
| 3939927 | 数值单测和完整输入审计 | PENDING，Priority |
| 3939928 | tau=0.5 pilot | PENDING，Dependency |
| 3939929 | 全流处理、线性外推及敏感性 | PENDING，Dependency |
| 3939930 | 全部逐阶段图与交付验收 | PENDING，Dependency |

CPU 分区已满，原本可见的单核空位随后也被调度占用。没有在登录节点运行科学数组、拟合或绘图。
为利用零碎空位，已将本次四个单线程作业的资源申请缩减到 1 CPU、6 GiB、30 分钟，见 resource_override_20261107.json；冻结的科学源码和配置未改动。

因此，此时**尚无本轮数值外推结果或新图可供检查**。
成功后会自动写入 calculation_complete.json、RESULTS.md 和 complete.json；这些文件若不存在，不能把提交成功当成计算完成。
后续先查 sacct 和 logs，若失败应按日志修复并保留版本记录，不能覆盖冻结源或用零补缺失点。

```bash
sacct -j 3939927,3939928,3939929,3939930 --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
```

物理约定和限制以 README.md 与 config.json 为准：
伴随 d_A=0.8699083237267272 只是外部有限窗候选，不是最终定稿；没有用 d_F=0.374137。
self-ratio 分母为局域 TU，不直接套用非局域端点转换。主外推使用同一 bootstrap 的局域 -C3/C2 锚点恢复分子幅度，端点转换后的 z=0 保留 NaN。
参考缩放图使用固定 tau=0.5 的 bare 局域幅度，仅是共同显示尺度，不是 MSbar 局域归一化。

会话：L32x64 TU self-ratio 的外部伴随线减除、GF 端点转换与线性小流时间试算。
