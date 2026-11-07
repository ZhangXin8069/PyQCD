# qcd-einsum vs opt_einsum 3x 硬门禁基准

## 结论

- 状态：**NON_ACCEPTANCE**
- acceptance evaluated：True
- provisional：True
- 说明：目标数据集与本机 perambulator 均不存在；使用固定种子合成 complex128 操作数，形状与真实蒸馏一致，正确性以 opt_einsum/numpy 逐元素对照（rel<=1e-12）为准，性能结论标记 provisional
- 阻断：验收缺少 profiles: ['T1', 'T2']
- 阻断：验收缺少 equations: ['P1', 'P2']
- 阻断：验收缺少 backends: ['cpu']
- 阻断：L24/E4/gpu status=error
- 阻断：3 组双方中位耗时 < 0.1s，记为 too_short 且不作为实战对照: ['L24/S1/gpu', 'L24/S2/gpu', 'L24/S3/gpu']
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
- worker pids：`[198179, 198229, 198297, 198368, 201480, 201704, 201839, 201886, 201929, 201989, 202067, 202153]`
- gpu device：`{'type': 'gpu', 'name': 'NVIDIA GeForce RTX 4060 Laptop GPU', 'cupy': '14.2.0', 'free_mb': 7096.0, 'total_mb': 8187.5}`

## 结果

| profile | eq | backend | isolation | worker pids | optimize | planner | opt median/s | qcd median/s | speedup | max rel err | planner path match | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L24 | E1 | gpu | process | {'opt_einsum': 198179, 'qcd_einsum': 198229} | dp | production | 0.136967162 | 0.101641915 | 1.347546 | 1.4702081590871288e-15 | True | ok / FAIL(<3x) |
| L24 | E2 | gpu | process | {'opt_einsum': 198297, 'qcd_einsum': 198368} | dp | production | 0.10191396 | 0.100550504 | 1.01356 | 2.025955229705764e-15 | True | ok / FAIL(<3x) |
| L24 | E4 | gpu | process |  | dp | production |  |  |  |  | True | error / FAIL(<3x) |
| L24 | E5 | gpu | process | {'opt_einsum': 201480, 'qcd_einsum': 201704} | dp | production | 0.387195496 | 0.116754948 | 3.316309 | 0.0 | True | ok / PASS |
| L24 | S1 | gpu | process | {'opt_einsum': 201839, 'qcd_einsum': 201886} | dp | production | 0.080034009 | 0.081171733 | 0.985984 | 2.8188017575463705e-17 | True | ok / too_short(no-comparison) |
| L24 | S2 | gpu | process | {'opt_einsum': 201929, 'qcd_einsum': 201989} | dp | production | 0.002667016 | 0.002644346 | 1.008573 | 1.6411204842913612e-15 | True | ok / too_short(no-comparison) |
| L24 | S3 | gpu | process | {'opt_einsum': 202067, 'qcd_einsum': 202153} | dp | production | 0.002602778 | 0.002687207 | 0.968581 | 1.2712715265102077e-15 | True | ok / too_short(no-comparison) |

## 显式 path

- L24/E1/gpu: `[(2, 7), (5, 6), (1, 5), (3, 4), (2, 3), (0, 2), (0, 1)]`；digest=`16ac286c07d64a634796bba34f94ce2e34964e4f4c5d0ebe5188e070b163ce3a`
- L24/E2/gpu: `[(2, 7), (5, 6), (1, 5), (3, 4), (2, 3), (0, 2), (0, 1)]`；digest=`16ac286c07d64a634796bba34f94ce2e34964e4f4c5d0ebe5188e070b163ce3a`
- L24/E4/gpu: `[(1, 3), (1, 2), (0, 1)]`；digest=`e77630bb320a23f2f9230ed7b81c411b8bc5c4c3580d66a4c020017c18a31d7c`
- L24/E5/gpu: `[(1, 2), (0, 1)]`；digest=`eda94defbc2a2d4f1b09987a24001e2fca4cbcc3542df6ef1103b11c3aa991d0`
- L24/S1/gpu: `[(2, 7), (5, 6), (1, 5), (3, 4), (2, 3), (0, 2), (0, 1)]`；digest=`16ac286c07d64a634796bba34f94ce2e34964e4f4c5d0ebe5188e070b163ce3a`
- L24/S2/gpu: `[(2, 8), (4, 7), (2, 6), (0, 5), (2, 4), (0, 1), (0, 2), (0, 1)]`；digest=`89a98a4aedd600bc05f2bc5f599e94e0adc578568968cd0e15e2617e72110176`
- L24/S3/gpu: `[(1, 5), (2, 4), (1, 3), (0, 2), (0, 1)]`；digest=`6562510463e4d12fd14a6f31607378b5ecc2a68235715b99b48827beae784e0d`
