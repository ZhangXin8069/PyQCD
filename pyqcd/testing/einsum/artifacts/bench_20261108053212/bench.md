# qcd-einsum vs opt_einsum 3x 硬门禁基准

## 结论

- 状态：**NON_ACCEPTANCE**
- acceptance evaluated：True
- provisional：True
- 说明：目标数据集与本机 perambulator 均不存在；使用固定种子合成 complex128 操作数，形状与真实蒸馏一致，正确性以 opt_einsum/numpy 逐元素对照（rel<=1e-12）为准，性能结论标记 provisional
- 阻断：验收缺少 profiles: ['T1', 'T2']
- 阻断：验收缺少 equations: ['E1', 'E2', 'E4', 'S1', 'S2']
- 阻断：验收缺少 backends: ['gpu']
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
- path：oracle planner 生成一次，双方用同一显式 path
- 正确性：`max |actual-expected| / max(|expected|, tiny)` ≤ `1e-12`
- GPU：每次计时前后调用 default stream synchronize

## 环境

- oracle：installed（非冻结站点包）；version=`3.4.0`；frozen=False
- oracle path：`/home/zhangxin/miniforge3/lib/python3.12/site-packages/opt_einsum`
- pyqcd.einsum：`0.1.0`；native=`0.1.0`
- pyqcd path：`/home/zhangxin/PyQCD/pyqcd/einsum/__init__.py`
- Python：`3.12.4`；NumPy：`2.5.4`
- CPU：`8` logical CPUs
- worker pids：`[142260, 142283, 142298, 142350]`
- cpu device：`{'type': 'cpu', 'name': 'x86_64', 'threads': 4}`

## 结果

| profile | eq | backend | isolation | worker pids | optimize | opt median/s | qcd median/s | speedup | max rel err | planner path match | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | E5 | cpu | process | {'opt_einsum': 142260, 'qcd_einsum': 142283} | dp | 0.000124132 | 4.6294e-05 | 2.681384 | 0.0 | True | ok / DIAGNOSTIC(<3x) |
| T0 | S3 | cpu | process | {'opt_einsum': 142298, 'qcd_einsum': 142350} | dp | 0.000997975 | 0.000144682 | 6.897714 | 6.303602812525156e-16 | True | ok / PASS |

## 显式 path

- T0/E5/cpu: `[(1, 2), (0, 1)]`；digest=`eda94defbc2a2d4f1b09987a24001e2fca4cbcc3542df6ef1103b11c3aa991d0`
- T0/S3/cpu: `[(1, 5), (2, 4), (1, 3), (0, 2), (0, 1)]`；digest=`6562510463e4d12fd14a6f31607378b5ecc2a68235715b99b48827beae784e0d`
