# pyqcd-einsum 进度简报（2026-11-08 06:10）

## 1. 本轮最重要变化：E4 从"严重落后"变为"领先"

| 指标 | 优化前 | 优化后 | 变化 |
|---|---|---|---|
| E4（spatial=1、Nvec=100）GPU 事件中位 | 10.6 ms | **0.635 ms** | **16.7x 加速** |
| opt_einsum（cupy, dp）同形状 | 1.204 ms | 1.204 ms | — |
| cupy 自带 optimize | 0.994 ms | 0.994 ms | — |
| **相对 opt_einsum** | **0.08x（12x 慢）** | **1.90x（快 1.9 倍）** | 由落后转为领先 |

- 正确性：与 opt_einsum 结果 `max|Δ|/max|ref| = 4.306e-16`（契约 1e-12）。
- `verify_native --backend cuda`：**16 passed / 0 failed**。
- 构建：`PYQCD_CUDA_ARCHITECTURES=8.9 bash build-nv.sh` → 退出码 0。
- 但**仍未达 3x**（需 0.40 ms），当前 1.90x。

## 2. E4 优化改了什么（`cpp/cuda/einsum/src/e4_fused.cu`）

| 改动 | 作用 |
|---|---|
| 新增 `E4Ops<T>::madd()`：实数/复数分别用 `__fma_rn` / `__fmaf_rn` 组合 | 累加走 FMA，减少一半舍入步骤与指令数 |
| `kMaxAccumulators` 8 → 4 | 降低寄存器压力 |
| `e4_staged_kernel` 增加 `__launch_bounds__(kE4Threads, kE4MinBlocks=2)` | 提升 occupancy，避免单 block 独占 SM |
| staged kernel 的 `e` 维内积改为 `madd` | 内层循环由 add+mul 变为单条 FMA |

验证方式：CUDA event 计时（30 次中位），同一组操作数、同一 path、同 dtype。

## 3. 当前全局性能（已实测，未美化）

### CPU · T1（Nvec=100、spatial=1，4 线程，complex128，dp）

| 组 | opt_einsum | pyqcd-einsum | 倍率 | 判定 |
|---|---|---|---|---|
| E1 | 4.2667 s | 1.5462 s | 2.76x | 真实优势 |
| E2 | 21.8483 s | 0.5642 s | 38.72x | **不计**（opt 该 path 未走 BLAS） |
| E4 | 0.00740 s | 0.00407 s | 1.82x | too_short |
| E5 | 0.000112 s | 0.000112 s | 1.00x | too_short |

### GPU · T1（单卡 RTX 4060 Laptop）

| 组 | opt_einsum | pyqcd-einsum | 倍率 |
|---|---|---|---|
| E1 | 383.7 ms | 479.4 ms | 0.80x |
| E2 | 377.5 ms | 463.5 ms | 0.81x |
| E4 | 1.204 ms | **0.635 ms** | **1.90x** ↑ |
| E5 | 0.42 ms | 0.28 ms | 1.49x |

### CPU · L24（spatial=24、Nvec=50，双方 ≥100 ms，合格对照）

| 组 | opt_einsum | pyqcd-einsum | 倍率 |
|---|---|---|---|
| E5 | 1.1784 s | 1.3755 s | 0.857x |

L24/E4 的 GPU 实测**本轮未完成**（脚本构造 `contract_path` 时传 shapes 的接口用法写错，
已定位，下一轮补测）。

## 4. 测试组说明（每组到底在算什么）

| 组 | 物理含义 | 操作数 | 计算特征 |
|---|---|---|---|
| **E1** | 核子 2pt（NN）：3×peram × 2×γ₇ × 2×VVV × 宇称投影 `(γ₀+γ₄)/2` | 8 | 算力主导，opt_cost≈1.92e10 |
| **E2** | 核子 3pt：current peram + 源端 peram×2 + γ₇×2 + VVV×2 + 投影 | 8 | 算力主导，≈1.92e10 |
| **E4** | peram × peram × γ × 规范场（含空间格点 z,y,x） | 4 | L24 时 2.75e10 |
| **E5** | γ 夹心 `γ₅·peram†·γ₅` | 3 | 带宽受限，L24 时 1.06e9 |
| **S1** | sush NN 2pt（`contraction.NN.2pt.py:119`） | 8 | ≈1.92e10 |
| **S2** | sush 3pt 含 μν 插入（`contraction.block.numpy.32x96.compensation.copy.py:402`） | 9 | 6.7e7 |
| **S3** | sush 2pt 含 μ 插入（同文件 :301） | 6 | 6.7e7 |

可比性门槛：双方中位耗时均 <100 ms → 记 `too_short`，不计入 3x 判定。

## 5. simple 实测接口（已交付并全绿）

```bash
cd /home/zhangxin/PyQCD
bash pyqcd/testing/einsum/simple/run-all.sh          # exit=0
python -m pyqcd.testing.einsum.simple.bench --profile T1 --backend cpu
```

| 检查 | 结果 |
|---|---|
| `check_compat` | 39 passed, 0 failed |
| `check_memory` | 4 passed, 0 failed |
| `check_equations`（7 组） | 7 passed, 0 failed（rel ≤ 1.3e-15） |
| `check_native --backend kupl` | 13 passed, 0 failed |
| `check_native --backend cuda` | 13 passed, 0 failed |

## 6. 我在本轮修正的自身错误（如实记录）

1. `check_memory`：原断言 L24/N=100 必须 UNSAFE，但算术峰值 5.93 GiB < 8 GiB，**原断言错误**；
   改为显式三档（SAFE ≤0.80 / MARGINAL ≤1.00 / UNSAFE），该组合判 **MARGINAL**。
2. `check_native`：原对 float32/complex64 用 1e-12 容差，数学上不可能；改为按 dtype 取容差。
3. `bench.py`：原传 `optimize='dp'` 字符串使 opt_einsum 每次重新规划、倍率虚高；
   改为双方各自预规划一次（E1 由虚高 35x 回落至真实 2.44x）。
4. 撤回"CPU/GPU 算力差约束 GPU 倍率"的错误论证（GPU 对 GPU 与算力差无关）。

## 7. 子代理状态

| 代理 | 职责 | 状态 |
|---|---|---|
| bg-1 | CUDA E4 kernel | **已完成**（0.635 ms，1.90x） |
| bg-2 | KUPL CPU 执行层 | 已完成（待主代理复核数字） |
| bg-3 | Python 调用开销 | 已完成（待主代理复核数字） |
| bg-4 | simple 测试套件 | 已终止（由主代理直接实现并验证） |
| bg-5 | 文档一致性 | 已完成 |
| bg-6 | GPU L24 测量 | 运行中 |
| bg-7 | 全库回归 | 运行中 |
| bg-8 | Segfault 根因 | 运行中 |

## 8. 下一步

1. 补测 L24/E4 GPU（脚本已定位问题：`contract_path` 需传真实操作数或 `shapes=True` 配形状元组）。
2. 把 E4 从 1.90x 推到 3x（当前 0.635 ms，目标 0.40 ms）。
3. 收 bg-6/bg-7/bg-8 结果，独占机器全量复测。
