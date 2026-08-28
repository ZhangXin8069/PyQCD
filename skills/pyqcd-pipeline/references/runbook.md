# 管线运行 reference

## 基线与输出目录

默认一致性基线是
`examples/docker-v20260805/output/output_20260802_120104`。每次运行使用独立的
`examples/test0/v<YYYYMMDDHHMM>/` 目录，不能覆盖基线或另一轮运行。记录实际组态列表，
不要把预期网格当作存在性证明。

## 冒烟到全量

```bash
python examples/test0/main.py run --conf-ids 6250
bash examples/test0/run-local.sh
python examples/test0/main.py verify --run-dir examples/test0/v<ts>
```

冒烟适合检查导入、单组态路径和文件格式；它不能验证 ensemble 统计。全量回归至少保存
运行命令、backend/precision、组态 ID、开始结束时间和失败清单。

## 断点续跑

`step_2pt` 对每个组态检查必需的 corr 产物：齐全则跳过，缺失或损坏则只重算该组态。
需要比较新算法时显式传 `recompute_2pt=True`；不要通过删除未知文件强制重跑。重启前
重新执行输入守卫并核对已有产物的 shape/dtype，防止部分写入被误判为完成。

## 一致性验证

验证报告至少分开列出：

- 中间张量的相对最大差（通常目标 `<1e-6`）；
- 分析结果的相对差（通常目标 `<1e-8`）；
- NaN 位置是否一致、组态数和各阶段 PASS/FAIL 数；
- 缺失基线时的明确退出状态，而不是把空比较当作通过。

若首个阶段已经超差，停止继续比较下游，先核对输入、精度、后端和版本。只在基线和
当前运行使用相同物理/数值约定时解释逐位差异。

## TMD 示例

`test9_gluon_tmd_nucleon.py --smoke` 用于最小链路检查；`--only-plot` 只消费已算产物。
运行结果需再交给 `test9_verify.py` 和 `pyqcd-tmd-algorithm` 的门控，不能以示例程序
退出 0 宣称完整 TMD-PDF。
