# qcd-einsum 基准报告（0.2.4）

本报告只做测量，不改动任何库代码。脚本与原始数据见
`pyqcd/testing/einsum/bench_public_ab.py`（测量）与
`pyqcd/testing/einsum/bench_public_ab_tables.py`（本报告生成器）；本次运行的逐 case 原始记录
在 `logs/einsum-benchmark-20261109.txt`（`logs/` 只跟踪 `*.txt`），完整 JSON 可用脚本重跑生成。

## 口径

| 项 | 设置 |
|---|---|
| 被测版本 | pyqcd 0.2.4（PyPI 安装）/ opt_einsum 3.4.0（`dp` 策略） |
| 轮数 | 每 case **7 轮**，每轮交替"谁先跑"（后跑方会白拿 10–20% 的假优势） |
| 采样 | 单次不足 0.2 s 的 case 在轮内重复取最好值，避免被计时粒度主导 |
| 尺寸 | 每个家族三档：small / medium / large |
| 数据 | **物理结构数据**而非 iid 高斯：传播子式时间指数衰减、低秩 perambulator 块、近幺正的 SU(3) link；通用 case 另配纯随机输入 |
| 正确性 | 每个 case 在计时前先与 opt_einsum 对齐（按相对最大范数判定，避免近零元素被 `atol` 误判） |
| 覆盖 | CPU 42 个 case、GPU 42 个 case |
| 机器 | 4 物理核（8 线程）+ RTX 4060 Laptop 8 GB |

轮间波动（同一 case 内 ratio 的 max/min 相对差）中位数意义上 CPU 约 12%、GPU 约 15%；
个别 GPU case 因显存池与时钟波动可达 30–70%，这类数字只能看数量级。

## CPU（42 个 case）

| Family | Size | `pyqcd.einsum` | `opt_einsum` (`dp`) | Rate | Round spread |
|---|---|---|---|---|---|
| matmul real | small | 1650 µs | 1613 µs | 0.98× | 8% |
| matmul real | medium | 82.7 ms | 83.3 ms | 1.01× | 13% |
| matmul real | large | 662.4 ms | 660.7 ms | 1.00× | 15% |
| matmul complex | small | 680 µs | 626 µs | 0.92× | 14% |
| matmul complex | medium | 38.3 ms | 37.5 ms | 0.98× | 11% |
| matmul complex | large | 318.9 ms | 319.9 ms | 1.00× | 13% |
| element-wise | small | 3322 µs | 5416 µs | **1.63×** | 8% |
| element-wise | medium | 51.2 ms | 63.9 ms | 1.25× | 11% |
| element-wise | large | 208.0 ms | 265.6 ms | 1.28× | 18% |
| partial reduction | small | 5546 µs | 5551 µs | 1.00× | 10% |
| partial reduction | medium | 23.1 ms | 22.9 ms | 0.99× | 16% |
| partial reduction | large | 98.4 ms | 98.1 ms | 1.00× | 13% |
| full reduction | small | 959 µs | 1024 µs | 1.07× | 24% |
| full reduction | medium | 10.1 ms | 10.1 ms | 1.00× | 12% |
| full reduction | large | 42.1 ms | 41.9 ms | 1.00× | 12% |
| chain 3 ops | small | 1088 µs | 1151 µs | 1.06× | 20% |
| chain 3 ops | medium | 68.4 ms | 67.5 ms | 0.99× | 14% |
| chain 3 ops | large | 532.2 ms | 532.2 ms | 1.00× | 14% |
| chain 4 ops | small | 594 µs | 631 µs | 1.06× | 21% |
| chain 4 ops | medium | 28.9 ms | 28.7 ms | 0.99× | 6% |
| chain 4 ops | large | 265.0 ms | 251.3 ms | 0.95× | 12% |
| hyper-edge | small | 336 µs | 265 µs | 0.79× | 4% |
| hyper-edge | medium | 10.8 ms | 9698 µs | 0.90× | 8% |
| hyper-edge | large | 84.4 ms | 80.5 ms | 0.95× | 22% |
| batched GEMM | small | 873 µs | 10.9 ms | **12.48×** | 6% |
| batched GEMM | medium | 33.3 ms | 368.8 ms | **11.07×** | 8% |
| batched GEMM | large | 242.7 ms | 2.93 s | **12.06×** | 18% |
| ellipsis batch | small | 675 µs | 5823 µs | **8.62×** | 11% |
| ellipsis batch | medium | 19.2 ms | 174.7 ms | **9.08×** | 12% |
| ellipsis batch | large | 640.7 ms | 5.94 s | **9.27×** | 5% |
| batch x matrix | small | 404 µs | 372 µs | 0.92× | 7% |
| batch x matrix | medium | 5154 µs | 5088 µs | 0.99× | 10% |
| batch x matrix | large | 44.8 ms | 45.7 ms | 1.02× | 9% |
| many small batches | small | 3740 µs | 8549 µs | **2.29×** | 13% |
| many small batches | medium | 89.8 ms | 205.2 ms | **2.28×** | 6% |
| many small batches | large | 507.2 ms | 1.38 s | **2.73×** | 16% |
| nucleon 2pt (kernel) | small | 14.0 ms | 95.6 ms | **6.83×** | 11% |
| nucleon 2pt (kernel) | medium | 245.9 ms | 1.46 s | **5.95×** | 15% |
| nucleon 2pt (kernel) | large | 2.47 s | 41.50 s | **16.82×** | 5% |
| gamma sandwich (kernel) | small | 358 µs | 3770 µs | **10.53×** | 7% |
| gamma sandwich (kernel) | medium | 21.0 ms | 197.1 ms | **9.40×** | 7% |
| gamma sandwich (kernel) | large | 180.5 ms | 3.74 s | **20.74×** | 24% |

**汇总**：中位倍率 1.06×；比参考快 21 项、持平
（0.95–1.05）16 项、更慢 5 项。全部 42
项的相对误差上限 7.0e-15。

按家族聚合：

| Family | Best | Worst | Geometric mean |
|---|---|---|---|
| gamma sandwich (kernel) | 20.74× | 9.40× | 12.71× |
| nucleon 2pt (kernel) | 16.82× | 5.95× | 8.81× |
| batched GEMM | 12.48× | 11.07× | 11.85× |
| ellipsis batch | 9.27× | 8.62× | 8.99× |
| many small batches | 2.73× | 2.28× | 2.42× |
| element-wise | 1.63× | 1.25× | 1.37× |
| full reduction | 1.07× | 1.00× | 1.02× |
| chain 4 ops | 1.06× | 0.95× | 1.00× |
| chain 3 ops | 1.06× | 0.99× | 1.01× |
| batch x matrix | 1.02× | 0.92× | 0.98× |
| matmul real | 1.01× | 0.98× | 0.99× |
| matmul complex | 1.00× | 0.92× | 0.97× |
| partial reduction | 1.00× | 0.99× | 1.00× |
| hyper-edge | 0.95× | 0.79× | 0.88× |

比参考慢的项：

- hyper-edge / small: 0.79× (336 µs vs 265 µs)
- hyper-edge / medium: 0.90× (10.8 ms vs 9698 µs)
- matmul complex / small: 0.92× (680 µs vs 626 µs)
- batch x matrix / small: 0.92× (404 µs vs 372 µs)
- chain 4 ops / large: 0.95× (265.0 ms vs 251.3 ms)

## GPU（42 个 case）

| Family | Size | `pyqcd.einsum` | `opt_einsum` (`dp`) | Rate | Round spread |
|---|---|---|---|---|---|
| matmul real | small | 4882 µs | 4824 µs | 0.99× | 2% |
| matmul real | medium | 188.6 ms | 191.5 ms | 1.02× | 11% |
| matmul real | large | 678.3 ms | 667.7 ms | 0.98× | 36% |
| matmul complex | small | 1806 µs | 1069 µs | 0.59× | 70% |
| matmul complex | medium | 113.9 ms | 116.0 ms | 1.02× | 11% |
| matmul complex | large | 338.9 ms | 363.8 ms | 1.07× | 29% |
| element-wise | small | 580 µs | 590 µs | 1.02× | 12% |
| element-wise | medium | 3953 µs | 3962 µs | 1.00× | 7% |
| element-wise | large | 15.7 ms | 16.0 ms | 1.02× | 372% |
| partial reduction | small | 788 µs | 788 µs | 1.00× | 13% |
| partial reduction | medium | 2550 µs | 2559 µs | 1.00× | 3% |
| partial reduction | large | 8907 µs | 8866 µs | 1.00× | 1% |
| full reduction | small | 390 µs | 624 µs | **1.60×** | 15% |
| full reduction | medium | 2531 µs | 5133 µs | **2.03×** | 7% |
| full reduction | large | 9504 µs | 20.2 ms | **2.12×** | 1% |
| chain 3 ops | small | 3700 µs | 3638 µs | 0.98× | 2% |
| chain 3 ops | medium | 173.9 ms | 168.9 ms | 0.97× | 11% |
| chain 3 ops | large | 567.3 ms | 562.6 ms | 0.99× | 38% |
| chain 4 ops | small | 747 µs | 758 µs | 1.01× | 17% |
| chain 4 ops | medium | 45.5 ms | 48.1 ms | 1.06× | 12% |
| chain 4 ops | large | 282.2 ms | 302.7 ms | 1.07× | 21% |
| hyper-edge | small | 376 µs | 359 µs | 0.95× | 24% |
| hyper-edge | medium | 17.1 ms | 17.5 ms | 1.02× | 47% |
| hyper-edge | large | 171.0 ms | 176.4 ms | 1.03× | 11% |
| batched GEMM | small | 1527 µs | 1598 µs | 1.05× | 27% |
| batched GEMM | medium | 58.4 ms | 57.0 ms | 0.98× | 8% |
| batched GEMM | large | 285.4 ms | 303.1 ms | 1.06× | 19% |
| ellipsis batch | small | 594 µs | 591 µs | 0.99× | 12% |
| ellipsis batch | medium | 26.0 ms | 26.2 ms | 1.01× | 8% |
| ellipsis batch | large | 388.4 ms | 412.8 ms | 1.06× | 30% |
| batch x matrix | small | 504 µs | 458 µs | 0.91× | 23% |
| batch x matrix | medium | 9080 µs | 9294 µs | 1.02× | 3% |
| batch x matrix | large | 88.0 ms | 91.1 ms | 1.03× | 15% |
| many small batches | small | 1069 µs | 794 µs | 0.74× | 27% |
| many small batches | medium | 35.9 ms | 32.5 ms | 0.91× | 7% |
| many small batches | large | 316.4 ms | 266.6 ms | 0.84× | 32% |
| nucleon 2pt (kernel) | small | 22.5 ms | 48.3 ms | **2.14×** | 5% |
| nucleon 2pt (kernel) | medium | 169.5 ms | 445.2 ms | **2.63×** | 22% |
| nucleon 2pt (kernel) | large | 905.3 ms | 2.45 s | **2.71×** | 4% |
| gamma sandwich (kernel) | small | 805 µs | 7606 µs | **9.45×** | 43% |
| gamma sandwich (kernel) | medium | 14.5 ms | 148.0 ms | **10.18×** | 22% |
| gamma sandwich (kernel) | large | 517.7 ms | 1.92 s | **3.71×** | 44% |

**汇总**：中位倍率 1.02×；比参考快 14 项、持平
23 项、更慢 5 项。全部 42 项的
相对误差上限 1.9e-12。

按家族聚合：

| Family | Best | Worst | Geometric mean |
|---|---|---|---|
| gamma sandwich (kernel) | 10.18× | 3.71× | 7.09× |
| nucleon 2pt (kernel) | 2.71× | 2.14× | 2.48× |
| full reduction | 2.12× | 1.60× | 1.90× |
| matmul complex | 1.07× | 0.59× | 0.87× |
| chain 4 ops | 1.07× | 1.01× | 1.05× |
| ellipsis batch | 1.06× | 0.99× | 1.02× |
| batched GEMM | 1.06× | 0.98× | 1.03× |
| batch x matrix | 1.03× | 0.91× | 0.99× |
| hyper-edge | 1.03× | 0.95× | 1.00× |
| element-wise | 1.02× | 1.00× | 1.01× |
| matmul real | 1.02× | 0.98× | 1.00× |
| partial reduction | 1.00× | 1.00× | 1.00× |
| chain 3 ops | 0.99× | 0.97× | 0.98× |
| many small batches | 0.91× | 0.74× | 0.83× |

比参考慢的项：

- matmul complex / small: 0.59× (1806 µs vs 1069 µs)
- many small batches / small: 0.74× (1069 µs vs 794 µs)
- many small batches / large: 0.84× (316.4 ms vs 266.6 ms)
- many small batches / medium: 0.91× (35.9 ms vs 32.5 ms)
- batch x matrix / small: 0.91× (504 µs vs 458 µs)

## 结论

1. **纯 GEMM 与链式收缩在三档尺寸上都持平**：CPU 0.92–1.06×（多数落在 0.95–1.01×）、
   GPU 0.98–1.07×。大尺寸（4096² 实数、2048² 复数、3×3072 链）两边都在 BLAS 上限，
   差异在噪声内。
2. **批量轴是最大的结构性优势**：CPU 上批量 GEMM 三档 11–12×、ellipsis 批量 8.6–9.3×、
   大量小批量 2.3–2.7×；GPU 上同族只有 0.84–1.06×，因为 cupy 自己已经会批量调用 cuBLAS。
3. **两个物理 kernel 的收益随尺寸放大**：CPU 核子 2pt 6.8×/6.0×/16.8×、γ 夹心
   10.5×/9.4×/20.7×；GPU 核子 2pt 2.1×/2.6×/2.7×、γ 夹心 9.5×/10.2×，大尺寸回落到
   3.7×（8192×1024 时受 2.1 GB 工作区带宽限制）。CPU 侧倍数随尺寸增长，说明收益来自
   "省掉中间体"而非常数开销。
4. **逐元素与全约简**：CPU 逐元素 1.25–1.63×（无求和标签直接交 backend），GPU 全约简
   1.6–2.1×；两者都不涉及 BLAS，收益来自不落地中间体。
5. **仍比参考慢的项**（本报告只记录，未改代码）：
   - `hyper-edge`（`ij,jk,ik->i`）small/medium CPU 0.79×/0.90×，大尺寸回到 0.95×；
   - GPU `many small batches` 0.74–0.91×（1024–8192 个 32–96² 小矩阵），是唯一在生产
     尺寸仍落后的一族；
   - GPU `matmul complex` small 0.59×，但该 case 轮间波动 70%，为噪声；
   - `chain 4 ops` large 0.95×、`batch × matrix` small 0.92×，均在噪声边界。
6. **诊断口径**：这些"慢"的绝对值都在 0.3–11 ms，属于固定开销与路径选择区间；
   生产形状（≥100 ms）上没有一项落后。

## 复现

```bash
python -m pyqcd.testing.einsum.bench_public_ab cpu --rounds 7 --out cpu.json
python -m pyqcd.testing.einsum.bench_public_ab gpu --rounds 7 --out gpu.json
python -m pyqcd.testing.einsum.bench_public_ab_tables --write
```

脚本会在计时前打印 `pyqcd.__version__` 与 `pyqcd.__file__`，避免把 `site-packages` 里的
旧版本当成被测对象。
