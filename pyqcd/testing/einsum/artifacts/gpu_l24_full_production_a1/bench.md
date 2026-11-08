# qcd-einsum vs opt_einsum 3x 硬门禁基准

## 结论

- 状态：**NON_ACCEPTANCE**
- acceptance evaluated：True
- provisional：True
- 说明：目标数据集与本机 perambulator 均不存在；使用固定种子合成 complex128 操作数，形状与真实蒸馏一致，正确性以 opt_einsum/numpy 逐元素对照（rel<=1e-12）为准，性能结论标记 provisional
- 阻断：验收缺少 profiles: ['T1', 'T2']
- 阻断：验收缺少 equations: ['P1', 'P2']
- 阻断：验收缺少 backends: ['cpu']
- 阻断：5 组双方中位耗时 < 0.1s，记为 too_short 且不作为实战对照: ['L24/E1/gpu', 'L24/E2/gpu', 'L24/S1/gpu', 'L24/S2/gpu', 'L24/S3/gpu']
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
- worker pids：`[226950, 227008, 227060, 227113, 227232, 227553, 231720, 231938, 232131, 232205, 232264, 232331, 232376, 232423]`
- gpu device：`{'type': 'gpu', 'name': 'NVIDIA GeForce RTX 4060 Laptop GPU', 'cupy': '14.2.0', 'free_mb': 7096.0, 'total_mb': 8187.5}`

## 结果

| profile | eq | backend | isolation | worker pids | optimize | planner | opt median/s | qcd median/s | speedup | max rel err | planner path match | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L24 | E1 | gpu | process | {'opt_einsum': 226950, 'qcd_einsum': 227008} | dp | production | 0.087438569 | 0.084992751 | 1.028777 | 1.4702081590871288e-15 | True | ok / too_short(no-comparison) |
| L24 | E2 | gpu | process | {'opt_einsum': 227060, 'qcd_einsum': 227113} | dp | production | 0.082458765 | 0.189741886 | 0.434584 | 2.025955229705764e-15 | True | ok / too_short(no-comparison) |
| L24 | E4 | gpu | process | {'opt_einsum': 227232, 'qcd_einsum': 227553} | dp | production | 1.086181686 | 20.584370922 | 0.052767 | 6.065796671199516e-14 | True | ok / FAIL(<3x) |
| L24 | E5 | gpu | process | {'opt_einsum': 231720, 'qcd_einsum': 231938} | dp | production | 0.390806481 | 0.116214081 | 3.362815 | 0.0 | True | ok / PASS |
| L24 | S1 | gpu | process | {'opt_einsum': 232131, 'qcd_einsum': 232205} | dp | production | 0.081601116 | 0.08166631 | 0.999202 | 2.8188017575463705e-17 | True | ok / too_short(no-comparison) |
| L24 | S2 | gpu | process | {'opt_einsum': 232264, 'qcd_einsum': 232331} | dp | production | 0.002713367 | 0.002656969 | 1.021226 | 1.6411204842913612e-15 | True | ok / too_short(no-comparison) |
| L24 | S3 | gpu | process | {'opt_einsum': 232376, 'qcd_einsum': 232423} | dp | production | 0.002525801 | 0.006575661 | 0.384114 | 1.2712715265102077e-15 | True | ok / too_short(no-comparison) |

## 显式 path

- L24/E1/gpu: `[(2, 7), (5, 6), (1, 5), (3, 4), (2, 3), (0, 2), (0, 1)]`；digest=`16ac286c07d64a634796bba34f94ce2e34964e4f4c5d0ebe5188e070b163ce3a`
- L24/E2/gpu: `[(2, 7), (5, 6), (1, 5), (3, 4), (2, 3), (0, 2), (0, 1)]`；digest=`16ac286c07d64a634796bba34f94ce2e34964e4f4c5d0ebe5188e070b163ce3a`
- L24/E4/gpu: `[(1, 3), (1, 2), (0, 1)]`；digest=`e77630bb320a23f2f9230ed7b81c411b8bc5c4c3580d66a4c020017c18a31d7c`
- L24/E5/gpu: `[(1, 2), (0, 1)]`；digest=`eda94defbc2a2d4f1b09987a24001e2fca4cbcc3542df6ef1103b11c3aa991d0`
- L24/S1/gpu: `[(2, 7), (5, 6), (1, 5), (3, 4), (2, 3), (0, 2), (0, 1)]`；digest=`16ac286c07d64a634796bba34f94ce2e34964e4f4c5d0ebe5188e070b163ce3a`
- L24/S2/gpu: `[(2, 8), (4, 7), (2, 6), (0, 5), (2, 4), (0, 1), (0, 2), (0, 1)]`；digest=`89a98a4aedd600bc05f2bc5f599e94e0adc578568968cd0e15e2617e72110176`
- L24/S3/gpu: `[(1, 5), (2, 4), (1, 3), (0, 2), (0, 1)]`；digest=`6562510463e4d12fd14a6f31607378b5ecc2a68235715b99b48827beae784e0d`
