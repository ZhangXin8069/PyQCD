# pyqcd-einsum 进度简报（2026-11-08 07:40）· 含生产耗时占比标注

> 本文每组都标注两项占比：**opt 占比** = 该组占 opt_einsum 真实生产总时间的比例；
> **qcd 占比** = 该组占 pyqcd-einsum 真实生产总时间的比例。
> 占比来源：`simple/workflow.py` 的调用计数模型（脚本循环结构）× 生产规模单次耗时。

## 1. 生产耗时占比总表（首要读这张）

模型参数：`nt=8`、`tsep=2`、`link_max=1`、`Mom_len=8`；
2pt/3pt 用 `T1`（Nvec=100）单次耗时，gauge 管线（E4/E5）用 `L24`（spatial=24、Nvec=50）单次耗时。

| 组 | 生产含义 | 调用数 | opt ms/次 | qcd ms/次 | **opt 占比** | **qcd 占比** |
|---|---|---|---|---|---|---|
| **E1** | 核子 2pt（NN 通道） | 768 | 3159.50 | 936.46 | **51.0%** | **42.6%** |
| **P1** | 3pt 主项（15 算子） | 396 | 1862.45 | 1793.82 | **15.5%** | **42.1%** |
| **P2** | 3pt 主项（置换型） | 396 | 375.86 | 351.69 | **3.1%** | **8.3%** |
| **E4** | peram × gauge × γ | 48 | 26122.80 | 2361.24 | **26.4%** | **6.7%** |
| **E5** | γ 夹心 | 48 | 3963.03 | 123.39 | **4.0%** | **0.4%** |
| 合计 | | | **4757.0 s** | **1688.1 s** | | **2.82x** |

**读法**：
- 对 opt_einsum 而言最大头是 2pt(E1) 51%，其次 E4 26%。
- 对我们而言最大头是 P1 42.1% 与 E1 42.6%——**P1 是当前最该优化的项**（它同时是我们落后 opt 的项：
  1793.82 vs 1862.45 ms，仅 0.96x）。
- E4 在我们这边只占 6.7%，因为专用融合 kernel 已把它从 26.1 s 压到 2.36 s（opt 的 11 倍）。

## 2. 专用优化路径实测（本轮核心交付）

### 2.1 CPU · E5（L24，spatial=24、Nvec=50）

| 模式 | 中位耗时 | 相对通用路径 |
|---|---|---|
| `specialize=True`（sandwich 融合） | **142.72 ms** | **8.33x** |
| `specialize=False`（通用多步） | 1188.88 ms | 1.00x |
| 与 opt_einsum 对比（专用开） | opt 894.42 ms | **6.36x** |

数值：两条路径互相 `rel = 0`（完全一致）。

### 2.2 CPU · E4（L24）

| 组 | opt_einsum | pyqcd-einsum | 倍率 | 范数 rel |
|---|---|---|---|---|
| E4 | 29.54 s | 1.983 s | **14.90x** | 3.01e-14 |
| E5 | 0.894 s | 0.1407 s | **6.36x** | 0 |

### 2.3 GPU · E4（spatial=1、Nvec=100，RTX 4060）

| 模式 | 中位耗时 |
|---|---|
| `specialize=True` | **0.685 ms** |
| `specialize=False` | **1.326 ms** |
| 切回 True | 0.624 ms |

专用路径 ≈ **1.9x**。

### 2.4 命中提示样例（用户可见）

```
[pyqcd.einsum] 专用路径: gauge-propagator(E4) [fused] peram × peram × γ × 规范场，CUDA E4 融合 kernel
[pyqcd.einsum] 专用路径: gamma-sandwich(E5) [fused] γ₅ 夹心，CUDA sandwich kernel
[pyqcd.einsum] 专用路径: sush-3pt-main(P1) [multi-step] sush 3pt 主项，15 算子 14 步（生产热点）
```

## 3. 其它组（未纳入占比模型，仅作诊断）

| 组 | 说明 | 实测 |
|---|---|---|
| E2 | 8 算子 3pt 简化版，生产用 P1/P2 替代 | CPU T1：opt 21.85 s / 我们 0.564 s（opt 该 path 未走 BLAS，不计优势） |
| S1 | sush NN 2pt，与 E1 互为替代实现 | CPU T1：opt 498.8 ms / 我们 414.1 ms = 1.21x |
| S2 | sush 3pt 含 μν 插入 | CPU T1：opt 32.86 ms / 我们 1.99 ms = 16.5x |
| S3 | sush 2pt 含 μ 插入 | CPU T1：opt 32.90 ms / 我们 1.70 ms = 19.3x |

> E1 与 S1 是同一物理量的两种实现，**不可相加**；E2 与 P1/P2 同理。占比模型只取生产脚本实际使用的那一支。

## 4. 本轮修复的真实缺陷（重要）

`transpose_copy()` 在"输出所有维为 1 且需输出重排"时 `axes.empty()` 直接 `return true`，
目标缓冲一个元素都没写 → 返回未初始化内存。

- 最小复现：`abcde,cedb->ac`，L(1,5,1,5,5)、R(1,5,5,5) → rel=1.00（`->ca` 正常）
- 命中 E2 方程：`greedy`/`auto`（opt_einsum 默认用法）下 rel=1.00，仅 `dp`/`optimal` 正确
- 修复后：最小复现 rel=0；E2 greedy/auto/dp = 1.90e-16 / 9.47e-16 / 5.91e-16
- 回归：`verify_native` kupl 16/16、cuda 16/16、`check_equations` 9/9
- 日志：`logs/v20261108063552.txt`

## 5. 验证状态

| 检查 | 结果 |
|---|---|
| `verify_native --backend kupl` | 16 passed / 0 failed |
| `verify_native --backend cuda` | 16 passed / 0 failed |
| `check_compat` | 39 passed / 0 failed |
| `check_equations` | 9 passed / 0 failed |
| `check_native`（kupl/cuda） | 各 13 passed / 0 failed |
| `python -m pyqcd.testing`（bg-7） | 114 passed / 1 skipped / 1 failed（失败项为已修复的缺陷 + 一个环境硬编码路径） |

## 6. 未闭合与风险

| 项 | 说明 |
|---|---|
| **退出期 Segfault** | L24/E4+E5 基准跑完后再次出现 `Segmentation fault`（结果与产物均已写盘）。第二次出现，bg-8 在查；在结论里必须标注"该次运行的计时可信、退出码不可信" |
| P1/P2 专用 kernel | bg-13（KUPL）/bg-16（CUDA）仍在写，未接线 |
| 2pt 专用 kernel | bg-12（CUDA）/bg-15（KUPL）仍在写，未接线 |
| `verify_compat` 全量 | 需 opt_einsum 冻结子模块，网络不可达 |
| `contract_expression` | 尚未支持 `specialize` 参数 |

## 7. 下一步（按占比排序）

1. **P1**（我们占 42.1%、且仅 0.96x）——接线 bg-13/bg-16 的专用 kernel。
2. **E1**（我们占 42.6%）——接线 bg-12/bg-15，目标把 936 ms 再压一档。
3. 修 Segfault 并复测。
