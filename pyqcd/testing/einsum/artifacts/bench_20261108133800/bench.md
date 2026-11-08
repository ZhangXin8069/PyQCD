# qcd-einsum vs opt_einsum 3x 硬门禁基准

## 结论

- 状态：**NON_ACCEPTANCE**
- acceptance evaluated：False
- provisional：True
- 说明：目标数据集与本机 perambulator 均不存在；使用固定种子合成 complex128 操作数，形状与真实蒸馏一致，正确性以 opt_einsum/numpy 逐元素对照（rel<=1e-12）为准，性能结论标记 provisional
- 阻断：smoke 只覆盖 T0n/CPU，明确不构成验收
- 阻断：9 组双方中位耗时 < 0.1s，记为 too_short 且不作为实战对照: ['T0n/E1/cpu', 'T0n/E2/cpu', 'T0n/E4/cpu', 'T0n/E5/cpu', 'T0n/S1/cpu', 'T0n/S2/cpu', 'T0n/S3/cpu', 'T0n/P1/cpu', 'T0n/P2/cpu']
- 阻断：oracle is not the frozen opt_einsum submodule
- 阻断：benchmark uses a proxy dataset rather than the requested target

## 口径

- dtype：`complex128`
- threads：`4`（numpy 导入前强制）
- optimize：`dp`
- isolation：`process`
- acceptance profiles：`['T1', 'T2']`；diagnostic profiles：`['T0', 'T0n']`
- 计时：预热后 PROCESS_ISOLATED，每实现不少于 10 次，取中位数
- planner：`shared`（双方共用 oracle 显式 path，只隔离执行层）
- 可比性门槛：双方中位耗时均 ≥ `0.1` s，否则记 `too_short` 且不计入 3x 判定
- 正确性：`max |actual-expected| / max(|expected|, tiny)` ≤ `1e-12`
- GPU：每次计时前后调用 default stream synchronize

## 环境

- oracle：installed（非冻结站点包）；version=`3.4.0`；frozen=False
- oracle path：`/home/zhangxin/miniforge3/lib/python3.12/site-packages/opt_einsum`
- pyqcd.einsum：`0.1.9`；native=`0.1.0`
- pyqcd path：`/home/zhangxin/PyQCD/pyqcd/einsum/__init__.py`
- Python：`3.12.4`；NumPy：`2.5.4`
- CPU：`8` logical CPUs
- worker pids：`[420649, 420663, 420674, 420683, 420697, 420705, 420714, 420728, 420736, 420744, 420753, 420767, 420775, 420784, 420798, 420806, 420815, 420829]`
- cpu device：`{'type': 'cpu', 'name': 'x86_64', 'threads': 4}`

## 结果

| profile | eq | backend | isolation | worker pids | optimize | planner | opt median/s | qcd median/s | speedup | max rel err | planner path match | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0n | E1 | cpu | process | {'opt_einsum': 420649, 'qcd_einsum': 420663} | dp | shared | 0.000190799 | 8.2881e-05 | 2.302084 | 7.617064447661188e-16 | True | ok / too_short(no-comparison) |
| T0n | E2 | cpu | process | {'opt_einsum': 420674, 'qcd_einsum': 420683} | dp | shared | 0.000237485 | 8.8868e-05 | 2.672334 | 3.7454673653456204e-16 | True | ok / too_short(no-comparison) |
| T0n | E4 | cpu | process | {'opt_einsum': 420697, 'qcd_einsum': 420705} | dp | shared | 0.000112052 | 4.8409e-05 | 2.314694 | 4.233940518453961e-16 | True | ok / too_short(no-comparison) |
| T0n | E5 | cpu | process | {'opt_einsum': 420714, 'qcd_einsum': 420728} | dp | shared | 6.3381e-05 | 2.6488e-05 | 2.392819 | 0.0 | True | ok / too_short(no-comparison) |
| T0n | S1 | cpu | process | {'opt_einsum': 420736, 'qcd_einsum': 420744} | dp | shared | 0.000188135 | 8.971e-05 | 2.097146 | 0.0 | True | ok / too_short(no-comparison) |
| T0n | S2 | cpu | process | {'opt_einsum': 420753, 'qcd_einsum': 420767} | dp | shared | 0.000166069 | 6.0161e-05 | 2.76041 | 5.649354260980651e-16 | True | ok / too_short(no-comparison) |
| T0n | S3 | cpu | process | {'opt_einsum': 420775, 'qcd_einsum': 420784} | dp | shared | 0.000107666 | 4.8605e-05 | 2.215122 | 1.2274336556766194e-15 | True | ok / too_short(no-comparison) |
| T0n | P1 | cpu | process | {'opt_einsum': 420798, 'qcd_einsum': 420806} | dp | shared | 0.000339336 | 4.5223e-05 | 7.503615 | 1.0221880692108546e-15 | True | ok / too_short(no-comparison) |
| T0n | P2 | cpu | process | {'opt_einsum': 420815, 'qcd_einsum': 420829} | dp | shared | 0.000315311 | 4.7077e-05 | 6.697772 | 1.0156680662087238e-15 | True | ok / too_short(no-comparison) |

## 显式 path

- T0n/E1/cpu: `[(2, 7), (5, 6), (1, 5), (3, 4), (2, 3), (0, 2), (0, 1)]`；digest=`16ac286c07d64a634796bba34f94ce2e34964e4f4c5d0ebe5188e070b163ce3a`
- T0n/E2/cpu: `[(2, 7), (5, 6), (1, 5), (3, 4), (2, 3), (0, 2), (0, 1)]`；digest=`16ac286c07d64a634796bba34f94ce2e34964e4f4c5d0ebe5188e070b163ce3a`
- T0n/E4/cpu: `[(1, 3), (1, 2), (0, 1)]`；digest=`e77630bb320a23f2f9230ed7b81c411b8bc5c4c3580d66a4c020017c18a31d7c`
- T0n/E5/cpu: `[(1, 2), (0, 1)]`；digest=`eda94defbc2a2d4f1b09987a24001e2fca4cbcc3542df6ef1103b11c3aa991d0`
- T0n/S1/cpu: `[(2, 7), (5, 6), (1, 5), (3, 4), (2, 3), (0, 2), (0, 1)]`；digest=`16ac286c07d64a634796bba34f94ce2e34964e4f4c5d0ebe5188e070b163ce3a`
- T0n/S2/cpu: `[(2, 8), (4, 7), (2, 6), (0, 5), (2, 4), (0, 1), (0, 2), (0, 1)]`；digest=`89a98a4aedd600bc05f2bc5f599e94e0adc578568968cd0e15e2617e72110176`
- T0n/S3/cpu: `[(1, 5), (2, 4), (1, 3), (0, 2), (0, 1)]`；digest=`6562510463e4d12fd14a6f31607378b5ecc2a68235715b99b48827beae784e0d`
- T0n/P1/cpu: `[(2, 14), (0, 12), (9, 12), (5, 11), (2, 10), (4, 9), (6, 8), (0, 7), (3, 6), (1, 5), (0, 4), (1, 3), (1, 2), (0, 1)]`；digest=`45185f9f873f983ead6ff9821016dcbbf07e5b56abd6375f9ad9c74d1569f1d6`
- T0n/P2/cpu: `[(2, 14), (0, 12), (1, 6), (2, 11), (0, 10), (6, 9), (3, 8), (6, 7), (4, 6), (2, 5), (0, 4), (1, 3), (1, 2), (0, 1)]`；digest=`8a0bbd2047ae8bc8214429f4114f2e1d7fe29721d77cef4fe39c1d6e3a735b96`
