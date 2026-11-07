# qcd-einsum vs opt_einsum 3x 硬门禁基准

## 结论

- 状态：**FAIL**
- acceptance evaluated：True
- provisional：True
- 说明：目标数据集与本机 perambulator 均不存在；使用固定种子合成 complex128 操作数，形状与真实蒸馏一致，正确性以 opt_einsum/numpy 逐元素对照（rel<=1e-12）为准，性能结论标记 provisional
- 阻断：验收缺少 profiles: ['T2']
- 阻断：验收缺少 backends: ['gpu']
- 阻断：T1/E1/cpu speedup=2.759424 < 3.0
- 阻断：T1/E4/cpu speedup=1.818159 < 3.0
- 阻断：T1/E5/cpu speedup=0.999661 < 3.0
- 阻断：oracle is not the frozen opt_einsum submodule
- 阻断：benchmark uses a proxy dataset rather than the requested target

## 口径

- dtype：`complex128`
- threads：`4`（numpy 导入前强制）
- optimize：`greedy`
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
- worker pids：`[110389, 112813, 113668, 125480, 125662, 125714, 125744, 125789]`
- cpu device：`{'type': 'cpu', 'name': 'x86_64', 'threads': 4}`

## 结果

| profile | eq | backend | isolation | worker pids | optimize | opt median/s | qcd median/s | speedup | max rel err | planner path match | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | E1 | cpu | process | {'opt_einsum': 110389, 'qcd_einsum': 112813} | greedy | 4.266694929 | 1.546226513 | 2.759424 | 1.2349152584035974e-14 | True | ok / FAIL(<3x) |
| T1 | E2 | cpu | process | {'opt_einsum': 113668, 'qcd_einsum': 125480} | greedy | 21.84825881 | 0.564247697 | 38.721042 | 9.131871518811761e-15 | True | ok / PASS |
| T1 | E4 | cpu | process | {'opt_einsum': 125662, 'qcd_einsum': 125714} | greedy | 0.007397338 | 0.004068588 | 1.818159 | 1.6877409180496513e-13 | True | ok / FAIL(<3x) |
| T1 | E5 | cpu | process | {'opt_einsum': 125744, 'qcd_einsum': 125789} | greedy | 0.000112043 | 0.000112081 | 0.999661 | 0.0 | True | ok / FAIL(<3x) |

## 显式 path

- T1/E1/cpu: `[(0, 3), (1, 2), (0, 3), (0, 4), (1, 3), (0, 2), (0, 1)]`；digest=`660b3312bbd5e51a26dba4bbcdac864976a4bac7aef144a8b37e2c90171f55c3`
- T1/E2/cpu: `[(0, 3), (1, 2), (0, 3), (0, 4), (1, 3), (0, 2), (0, 1)]`；digest=`660b3312bbd5e51a26dba4bbcdac864976a4bac7aef144a8b37e2c90171f55c3`
- T1/E4/cpu: `[(0, 3), (0, 1), (0, 1)]`；digest=`7e6135984bb289a2f76d9b5f5d676bb7fb22621966fcf8f2fdb34b5dd6e416bd`
- T1/E5/cpu: `[(0, 1), (0, 1)]`；digest=`68664fc48547ed679c77b17a0cfb1177b202c210927f6402230eedd6fa9bc74a`
