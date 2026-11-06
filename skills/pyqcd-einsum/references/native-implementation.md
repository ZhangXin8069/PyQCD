# 原生收缩实现与实测记录

更新日期：2026-11-06。本文只记录当前工作树证据；命令应在新会话定向复跑。

## 1. 文件与职责

| 层 | 路径 | 职责 |
|---|---|---|
| Python API/规划 | `pyqcd/einsum/` | opt_einsum 兼容入口、路径、表达式和回退 |
| Python 原生绑定 | `pyqcd/einsum/_native.py`、`_native_plan.py`、`kupl/einsum.py`、`cuda/einsum.pyx` | 计划编码、加载和指针传递 |
| KUPL C++ | `cpp/kupl/einsum/` | 共享计划核、CPU pack/reduce/GEMM 与 C ABI |
| CUDA C++ | `cpp/cuda/einsum/` | CUDA pack/reduce、cuBLAS batched GEMM 与 C ABI |
| 测试 | `pyqcd/testing/einsum/`、`cpp/kupl/einsum/tests/` | API、数值、ABI、蒸馏与性能门 |

## 2. 执行计划 v2

Header 为 `int32[9]`：

```text
magic=0x51434532, version=2, total_bytes, num_inputs, num_steps,
num_symbols, output_ndim, dtype_id, reserved
```

后续依次为：

1. 8-byte 对齐的 `int64[num_symbols]` 尺寸表；
2. 每个初始 operand 的 rank 与 `int64[rank]` shape；
3. 最终输出 labels；
4. 每步 arity、当前活动 operand positions、各输入 labels、输出 labels。

dtype id：`complex128=0`、`complex64=1`、`float64=2`、`float32=3`。混合精度和
单 operand 内重复 label 当前 fail closed。

## 3. K-lowest 布局

对任意二元步骤，将 label 分成：

- `B`：同时出现在两个输入和输出中的 batch；
- `M`：只属于左输入并保留到输出；
- `N`：只属于右输入并保留到输出；
- `K`：两个输入共有但本步消失的缩并轴。

原生规范布局为：

```text
left  = [B, M, K], K stride = 1
right = [B, N, K], K stride = 1
out   = [B, M, N]
result = left x right^T over K
```

当前 `cpp/kupl/einsum/tests/test_runtime.cpp` 已验证 label/permutation：

```text
left [a,b,c] -> [b,a,c]
right[b,d,c] -> [b,d,c]
out  [a,d,b] -> packed [b,a,d], then inverse output permutation
```

left/right 的 K 均为最低维；KUPL 调用 CBLAS 时对 right 使用 transpose-B。CUDA 镜像
必须保持同一语义。

## 4. 当前加速结构

- `ContractExpression` 与 `contract` 都缓存固定计划对应的原生 program；只缓存
  shape/dtype/path 和控制数组，不缓存输入内容或 output。
- KUPL compiled-context ABI：`applyCompileQcd`、`applyCompiledQcd`、
  `applyDestroyQcd`，避免每次调用重复解析执行计划。
- CUDA compiled context 首次分配并复用 pack/GEMM/中间 workspace；
  `PYQCD_EINSUM_CUDA_STREAMS` 控制 stream pool，batch 分片通过 event 在默认流汇合。
- `ab,...bce,cd->...dae` 识别为 gamma sandwich fast path：一次 e-lane 连续核完成
  两次 4x4 小矩阵夹心，不物化中间张量；按 work 大小选择串行或 OpenMP lane。
- `M=N=1` 的最终闭合走 batch-dot fast path，直接在原布局求和，不做三块 buffer
  分配、pack 或 1x1 GEMM。
- pack 使用增量 odometer、固定 16/8/4-byte copy，并按元素数限制 OpenMP 线程；
  避免每元素除法和 8 线程小任务过度并行。
- E1 类多小 GEMM workload 在 4 线程开发窗口中优于 8/16 线程；线程数必须与
  opt_einsum 对照保持一致并写入结果。

## 5. 当前验证快照

已测试：

- 共享 runtime 测试：`g++ -std=c++17 -Wall -Wextra -Werror ...` 后
  `test_runtime: PASS`。
- KUPL：四 dtype、batch、size-1 广播、输入/输出换序、一元 reduce/permute、外积、
  零维和多步活动槽位；代理记录 93 个构造计划及 UBSan 通过。
- Python：`python -m pyqcd.einsum._native_plan` 覆盖四 dtype；KUPL 真实执行与
  NumPy 对照通过。
- 全量：`python -m pyqcd.testing` 记录 `111 passed, 1 skipped, 0 failed`。
- 构建：KUPL/CUDA 动态库与 Cython 扩展均已产出；KUPL 导出面已包含冻结四入口与
  compiled-context 三入口，CUDA 三入口同步中。
- KUPL compiled context 与 sandwich/batch-dot fast path 已与 NumPy/opt_einsum T0
  示例逐元素一致。

未验证或失败：

- CUDA 一般 batched GEMM 当前返回 `CUBLAS_STATUS_INVALID_VALUE`，状态码 3；不能以
  逐元素 CUDA 路径正确性替代。
- CUDA K-last、一般 GEMM、compiled context、多 stream 与 E5 sandwich 已完成；
  CUDA MPI rank 分片仍未实现（`applyContractMpiQcd` 当前为 fail-closed stub）。
- KUPL MPI 第一阶段已通过真实 2-rank：仅分片一个保留到最终输出的 batch/free 轴，
  rank-local 执行后 Allgatherv；被消去轴、广播分片和多轴分片仍 fail closed。
- KUPL 多进程 fallback 已实现并通过 25 项功能契约；32×64×64 小例为 `0.42x`
  串行速度，说明进程/shared-memory 开销占主导，不能写成性能加速。
- MPI、多 CUDA stream、真实 2-rank/多 GPU 尚未完成数值与性能门。
- 固定 `opt_einsum` 子模块当前未初始化；两次 GitHub TLS/连接失败。开发对照使用本机
  `opt_einsum 3.4.0`，不能替代固定 `fccdd5aa` oracle 的最终证据。
- 指定 `beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg` 未在
  `/public/group/lqcd` 发现；现有近似 fixture 为
  `beta6.20_mu-0.2770_ms-0.2400_L24x72`，不得冒充目标系综结果。

## 6. 开发期性能快照

2026-11-06 开发窗口，T0 合成 operands、`complex128`、4 线程、相同 greedy path、
`contract` 入口，opt_einsum 3.4.0 站点包（非冻结 oracle）：

| 方程 | 开发窗口 qcd/opt 范围 | 状态 |
|---|---:|---|
| E1 8 算子 2pt | `3.0x-3.9x` | provisional pass，波动最高 |
| E2 8 算子 3pt | `9.9x-13.6x` | provisional pass |
| E4 peram×gauge×gamma | `4.0x-5.1x` | provisional pass |
| E5 gamma sandwich | `3.6x-4.3x` | provisional pass |

CUDA GPU/GPU T0/E5 开发窗口（opt_einsum 固定快照、ABBA）：

| GPU | opt_einsum | pyqcd | 倍率 |
|---|---:|---:|---:|
| V100 sm_70 | 1.119 ms | 0.275 ms | `4.07x` |
| P100 sm_60 | 0.922 ms | 0.264 ms | `3.49x` |

CUDA 其余 T0 开发窗口：E1 `3.14x`、E2 `4.12x`、E4 `3.10x`。这些仍是
provisional，需进程隔离、冻结 oracle 和目标/获准代理数据复验。

这些数字不是最终验收：必须换成冻结 `opt_einsum fccdd5aa`、目标/获准代理数据、
ABBA 顺序和至少三轮独立复测。CUDA 对 GPU 的同口径 3x 尚未完成。

## 7. 性能门

旧公共路径仅封装 NumPy/CuPy，实测约 `0.8x-1.08x`，不是新原生性能。目标为
CPU/CPU 和 GPU/GPU 均达到 opt_einsum 的 3x，并在至少 3 轮独立复测中保持。

每次测量至少记录：

```text
backend, dtype, device UUID, MPI ranks/grid, OpenMP threads, CUDA streams,
oracle commit/version, path hash, warmup, iterations,
median, IQR, min/max, peak bytes, max relative error, allclose status
```

性能 runner 必须把 `<3x` 计入非零退出码；必须同时保存 JSON 和 Markdown 摘要。
多传输或多 rank 的吞吐加速需与单 rank/单 stream 算法加速分开报告。
