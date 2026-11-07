# qcd-einsum vs opt_einsum 3x 硬门禁基准

## 结论

- 状态：**FAIL**
- acceptance evaluated：True
- provisional：True
- 说明：目标数据集与本机 perambulator 均不存在；使用固定种子合成 complex128 操作数，形状与真实蒸馏一致，正确性以 opt_einsum/numpy 逐元素对照（rel<=1e-12）为准，性能结论标记 provisional
- 阻断：验收缺少 profiles: ['T2']
- 阻断：验收缺少 backends: ['cpu']
- 阻断：T1/E1/gpu speedup=0.800329 < 3.0
- 阻断：T1/E2/gpu speedup=0.814429 < 3.0
- 阻断：T1/E4/gpu speedup=0.083299 < 3.0
- 阻断：T1/E5/gpu speedup=1.490789 < 3.0
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
- worker pids：`[136999, 137041, 137130, 137163, 137231, 137279, 137312, 137374, 137396, 137513, 137662, 137756, 137895, 137942, 137964, 138022]`
- gpu device：`{'type': 'gpu', 'name': 'NVIDIA GeForce RTX 4060 Laptop GPU', 'cupy': '14.2.0', 'free_mb': 7096.0, 'total_mb': 8187.5}`

## 结果

| profile | eq | backend | isolation | worker pids | optimize | opt median/s | qcd median/s | speedup | max rel err | planner path match | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | E1 | gpu | process | {'opt_einsum': 136999, 'qcd_einsum': 137041} | greedy | 0.003125174 | 0.005118849 | 0.610523 | 3.040583021443104e-14 | True | ok / DIAGNOSTIC(<3x) |
| T0 | E2 | gpu | process | {'opt_einsum': 137130, 'qcd_einsum': 137163} | greedy | 0.003262437 | 0.005174296 | 0.630508 | 7.585285207930786e-16 | True | ok / DIAGNOSTIC(<3x) |
| T0 | E4 | gpu | process | {'opt_einsum': 137231, 'qcd_einsum': 137279} | greedy | 0.000444164 | 0.001905279 | 0.233123 | 1.1140450335419138e-14 | True | ok / DIAGNOSTIC(<3x) |
| T0 | E5 | gpu | process | {'opt_einsum': 137312, 'qcd_einsum': 137374} | greedy | 0.000357257 | 0.000268212 | 1.331995 | 0.0 | True | ok / DIAGNOSTIC(<3x) |
| T1 | E1 | gpu | process | {'opt_einsum': 137396, 'qcd_einsum': 137513} | greedy | 0.383688588 | 0.479413431 | 0.800329 | 1.027637888463412e-14 | True | ok / FAIL(<3x) |
| T1 | E2 | gpu | process | {'opt_einsum': 137662, 'qcd_einsum': 137756} | greedy | 0.377496431 | 0.463510799 | 0.814429 | 3.115634731965751e-14 | True | ok / FAIL(<3x) |
| T1 | E4 | gpu | process | {'opt_einsum': 137895, 'qcd_einsum': 137942} | greedy | 0.001232747 | 0.014799088 | 0.083299 | 2.0108543876810975e-13 | True | ok / FAIL(<3x) |
| T1 | E5 | gpu | process | {'opt_einsum': 137964, 'qcd_einsum': 138022} | greedy | 0.000421549 | 0.000282769 | 1.490789 | 0.0 | True | ok / FAIL(<3x) |

## 显式 path

- T0/E1/gpu: `[(0, 3), (1, 2), (0, 3), (0, 4), (1, 3), (0, 2), (0, 1)]`；digest=`660b3312bbd5e51a26dba4bbcdac864976a4bac7aef144a8b37e2c90171f55c3`
- T0/E2/gpu: `[(0, 3), (1, 2), (0, 3), (0, 4), (1, 3), (0, 2), (0, 1)]`；digest=`660b3312bbd5e51a26dba4bbcdac864976a4bac7aef144a8b37e2c90171f55c3`
- T0/E4/gpu: `[(0, 3), (0, 1), (0, 1)]`；digest=`7e6135984bb289a2f76d9b5f5d676bb7fb22621966fcf8f2fdb34b5dd6e416bd`
- T0/E5/gpu: `[(0, 1), (0, 1)]`；digest=`68664fc48547ed679c77b17a0cfb1177b202c210927f6402230eedd6fa9bc74a`
- T1/E1/gpu: `[(0, 3), (1, 2), (0, 3), (0, 4), (1, 3), (0, 2), (0, 1)]`；digest=`660b3312bbd5e51a26dba4bbcdac864976a4bac7aef144a8b37e2c90171f55c3`
- T1/E2/gpu: `[(0, 3), (1, 2), (0, 3), (0, 4), (1, 3), (0, 2), (0, 1)]`；digest=`660b3312bbd5e51a26dba4bbcdac864976a4bac7aef144a8b37e2c90171f55c3`
- T1/E4/gpu: `[(0, 3), (0, 1), (0, 1)]`；digest=`7e6135984bb289a2f76d9b5f5d676bb7fb22621966fcf8f2fdb34b5dd6e416bd`
- T1/E5/gpu: `[(0, 1), (0, 1)]`；digest=`68664fc48547ed679c77b17a0cfb1177b202c210927f6402230eedd6fa9bc74a`
