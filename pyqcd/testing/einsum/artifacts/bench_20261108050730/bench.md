# qcd-einsum vs opt_einsum 3x 硬门禁基准

## 结论

- 状态：**NON_ACCEPTANCE**
- acceptance evaluated：True
- provisional：True
- 说明：目标数据集与本机 perambulator 均不存在；使用固定种子合成 complex128 操作数，形状与真实蒸馏一致，正确性以 opt_einsum/numpy 逐元素对照（rel<=1e-12）为准，性能结论标记 provisional
- 阻断：验收缺少 profiles: ['T1', 'T2']
- 阻断：验收缺少 backends: ['gpu']
- 阻断：没有运行 T1/T2 生产规模 profile，不能作为性能验收
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
- worker pids：`[108299, 108489, 108669, 108857, 109041, 109159, 109297, 109468]`
- cpu device：`{'type': 'cpu', 'name': 'x86_64', 'threads': 4}`

## 结果

| profile | eq | backend | isolation | worker pids | optimize | opt median/s | qcd median/s | speedup | max rel err | planner path match | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | E1 | cpu | process | {'opt_einsum': 108299, 'qcd_einsum': 108489} | greedy | 0.029578178 | 0.057577588 | 0.51371 | 2.6954898816465338e-14 | True | ok / DIAGNOSTIC(<3x) |
| T0 | E2 | cpu | process | {'opt_einsum': 108669, 'qcd_einsum': 108857} | greedy | 0.072728607 | 0.056375258 | 1.29008 | 2.605993005214077e-15 | True | ok / DIAGNOSTIC(<3x) |
| T0 | E4 | cpu | process | {'opt_einsum': 109041, 'qcd_einsum': 109159} | greedy | 0.000707767 | 0.000534779 | 1.323476 | 3.285996502281411e-14 | True | ok / DIAGNOSTIC(<3x) |
| T0 | E5 | cpu | process | {'opt_einsum': 109297, 'qcd_einsum': 109468} | greedy | 0.000191185 | 0.000200626 | 0.952942 | 0.0 | True | ok / DIAGNOSTIC(<3x) |

## 显式 path

- T0/E1/cpu: `[(0, 3), (1, 2), (0, 3), (0, 4), (1, 3), (0, 2), (0, 1)]`；digest=`660b3312bbd5e51a26dba4bbcdac864976a4bac7aef144a8b37e2c90171f55c3`
- T0/E2/cpu: `[(0, 3), (1, 2), (0, 3), (0, 4), (1, 3), (0, 2), (0, 1)]`；digest=`660b3312bbd5e51a26dba4bbcdac864976a4bac7aef144a8b37e2c90171f55c3`
- T0/E4/cpu: `[(0, 3), (0, 1), (0, 1)]`；digest=`7e6135984bb289a2f76d9b5f5d676bb7fb22621966fcf8f2fdb34b5dd6e416bd`
- T0/E5/cpu: `[(0, 1), (0, 1)]`；digest=`68664fc48547ed679c77b17a0cfb1177b202c210927f6402230eedd6fa9bc74a`
