# qcd-einsum vs opt_einsum 3x 硬门禁基准

## 结论

- 状态：**NON_ACCEPTANCE**
- acceptance evaluated：True
- provisional：True
- 说明：目标数据集与本机 perambulator 均不存在；使用固定种子合成 complex128 操作数，形状与真实蒸馏一致，正确性以 opt_einsum/numpy 逐元素对照（rel<=1e-12）为准，性能结论标记 provisional
- 阻断：验收缺少 profiles: ['T1', 'T2']
- 阻断：验收缺少 equations: ['E1', 'E2', 'E4', 'P1', 'P2', 'S1', 'S2', 'S3']
- 阻断：验收缺少 backends: ['cpu']
- 阻断：没有运行 T1/T2 生产规模 profile，不能作为性能验收
- 阻断：oracle is not the frozen opt_einsum submodule
- 阻断：benchmark uses a proxy dataset rather than the requested target

## 口径

- dtype：`complex128`
- threads：`4`（numpy 导入前强制）
- optimize：`dp`
- isolation：`process`
- acceptance profiles：`['T1', 'T2']`；diagnostic profiles：`['T0', 'T0n']`
- 计时：预热后 PROCESS_ISOLATED，每实现不少于 10 次，取中位数
- planner：`production`（双方各自规划一次，只计时稳态执行）
- 可比性门槛：双方中位耗时均 ≥ `0.1` s，否则记 `too_short` 且不计入 3x 判定
- 正确性：`max |actual-expected| / max(|expected|, tiny)` ≤ `1e-12`
- GPU：每次计时前后调用 default stream synchronize

## 环境

- oracle：installed（非冻结站点包）；version=`3.4.0`；frozen=False
- oracle path：`/home/zhangxin/miniforge3/lib/python3.12/site-packages/opt_einsum`
- pyqcd.einsum：`0.1.0`；native=`0.1.0`
- pyqcd path：`/home/zhangxin/PyQCD/pyqcd/einsum/__init__.py`
- Python：`3.12.4`；NumPy：`2.5.4`
- CPU：`8` logical CPUs
- worker pids：`[226560, 226641]`
- gpu device：`{'type': 'gpu', 'name': 'NVIDIA GeForce RTX 4060 Laptop GPU', 'cupy': '14.2.0', 'free_mb': 7096.0, 'total_mb': 8187.5}`

## 结果

| profile | eq | backend | isolation | worker pids | optimize | planner | opt median/s | qcd median/s | speedup | max rel err | planner path match | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L24 | E5 | gpu | process | {'opt_einsum': 226560, 'qcd_einsum': 226641} | dp | production | 0.442571157 | 0.461798015 | 0.958365 | 0.0 | True | ok / FAIL(<3x) |

## 显式 path

- L24/E5/gpu: `[(1, 2), (0, 1)]`；digest=`eda94defbc2a2d4f1b09987a24001e2fca4cbcc3542df6ef1103b11c3aa991d0`
