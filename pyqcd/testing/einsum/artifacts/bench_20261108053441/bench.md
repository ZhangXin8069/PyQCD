# qcd-einsum vs opt_einsum 3x 硬门禁基准

## 结论

- 状态：**NON_ACCEPTANCE**
- acceptance evaluated：False
- provisional：True
- 说明：目标数据集与本机 perambulator 均不存在；使用固定种子合成 complex128 操作数，形状与真实蒸馏一致，正确性以 opt_einsum/numpy 逐元素对照（rel<=1e-12）为准，性能结论标记 provisional
- 阻断：smoke 只覆盖 T0n/CPU，明确不构成验收
- 阻断：oracle is not the frozen opt_einsum submodule
- 阻断：benchmark uses a proxy dataset rather than the requested target

## 口径

- dtype：`complex128`
- threads：`2`（numpy 导入前强制）
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
- worker pids：`[144051, 144133, 144145, 144191, 144242, 144264, 144293, 144319, 144331, 144380, 144409, 144443, 144469, 144872]`
- cpu device：`{'type': 'cpu', 'name': 'x86_64', 'threads': 2}`

## 结果

| profile | eq | backend | isolation | worker pids | optimize | opt median/s | qcd median/s | speedup | max rel err | planner path match | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T0n | E1 | cpu | process | {'opt_einsum': 144051, 'qcd_einsum': 144133} | dp | 0.000208219 | 0.000262432 | 0.793421 | 1.6525023644854348e-15 | True | ok / DIAGNOSTIC(<3x) |
| T0n | E2 | cpu | process | {'opt_einsum': 144145, 'qcd_einsum': 144191} | dp | 0.00026276 | 0.000283942 | 0.9254 | 3.7454673653456204e-16 | True | ok / DIAGNOSTIC(<3x) |
| T0n | E4 | cpu | process | {'opt_einsum': 144242, 'qcd_einsum': 144264} | dp | 0.000245433 | 0.000144913 | 1.693658 | 8.276569340663362e-15 | True | ok / DIAGNOSTIC(<3x) |
| T0n | E5 | cpu | process | {'opt_einsum': 144293, 'qcd_einsum': 144319} | dp | 6.3724e-05 | 0.000107419 | 0.593228 | 0.0 | True | ok / DIAGNOSTIC(<3x) |
| T0n | S1 | cpu | process | {'opt_einsum': 144331, 'qcd_einsum': 144380} | dp | 0.000191856 | 0.000302514 | 0.634205 | 0.0 | True | ok / DIAGNOSTIC(<3x) |
| T0n | S2 | cpu | process | {'opt_einsum': 144409, 'qcd_einsum': 144443} | dp | 0.00017151 | 0.000470768 | 0.36432 | 5.720080164945963e-16 | True | ok / DIAGNOSTIC(<3x) |
| T0n | S3 | cpu | process | {'opt_einsum': 144469, 'qcd_einsum': 144872} | dp | 0.000156802 | 0.000167303 | 0.937234 | 1.2274336556766194e-15 | True | ok / DIAGNOSTIC(<3x) |

## 显式 path

- T0n/E1/cpu: `[(2, 7), (5, 6), (1, 5), (3, 4), (2, 3), (0, 2), (0, 1)]`；digest=`16ac286c07d64a634796bba34f94ce2e34964e4f4c5d0ebe5188e070b163ce3a`
- T0n/E2/cpu: `[(2, 7), (5, 6), (1, 5), (3, 4), (2, 3), (0, 2), (0, 1)]`；digest=`16ac286c07d64a634796bba34f94ce2e34964e4f4c5d0ebe5188e070b163ce3a`
- T0n/E4/cpu: `[(1, 3), (1, 2), (0, 1)]`；digest=`e77630bb320a23f2f9230ed7b81c411b8bc5c4c3580d66a4c020017c18a31d7c`
- T0n/E5/cpu: `[(1, 2), (0, 1)]`；digest=`eda94defbc2a2d4f1b09987a24001e2fca4cbcc3542df6ef1103b11c3aa991d0`
- T0n/S1/cpu: `[(2, 7), (5, 6), (1, 5), (3, 4), (2, 3), (0, 2), (0, 1)]`；digest=`16ac286c07d64a634796bba34f94ce2e34964e4f4c5d0ebe5188e070b163ce3a`
- T0n/S2/cpu: `[(2, 8), (4, 7), (2, 6), (0, 5), (2, 4), (0, 1), (0, 2), (0, 1)]`；digest=`89a98a4aedd600bc05f2bc5f599e94e0adc578568968cd0e15e2617e72110176`
- T0n/S3/cpu: `[(1, 5), (2, 4), (1, 3), (0, 2), (0, 1)]`；digest=`6562510463e4d12fd14a6f31607378b5ecc2a68235715b99b48827beae784e0d`
