# pyqcd-einsum 进度简报（2026-11-08 08:00）

## 1. 退出期 Segfault 排查结果：4 次复现均未重现

| # | 场景 | 命令 | 退出码 | 是否崩溃 |
|---|---|---|---|---|
| 1 | 直连脚本 L24/E4（不跑 bench） | `python -X faulthandler -c ...` | **0** | 否 |
| 2 | bench L24/E4 单独 | `bench_native --profiles L24 --equations E4` | **2**（NON_ACCEPTANCE，正常语义） | 否 |
| 3 | bench L24/E5 单独 | `... --equations E5` | **2** | 否 |
| 4 | bench L24/E5,E4 组合（即当初崩溃的组合） | `... --equations E5,E4` | **2** | 否 |

四次都带 `-X faulthandler`，输出中无 `Fatal`/`Current thread`/`Segmentation` 记录；
结果与产物均正常写出并打印汇总。

**判定（如实）**：
- 崩溃**不可确定性复现**，因此不能声称"已修复"，也不能声称"不存在"。
- 最可能原因仍是**并发替换 `.so`**：崩溃两次都发生在后台子代理重建
  `libpyqcdeinsum_kupl.so` / `libpyqcdeinsum_cuda.so` 的时段内。动态库被 `mv -Tf` 原子替换后，
  已映射进程在退出 `munmap` 阶段可能访问到不一致的映射。
- **最终验收必须在无并发构建时进行，并记录真实退出码**；在此之前，涉及崩溃那两次运行的结论必须标注
  "计时可信、退出码不可信"。

## 2. 专用优化路径实测（本轮核心，含生产占比）

### 2.1 生产耗时占比（每组标注）

模型：`nt=8`、`tsep=2`、`link_max=1`、`Mom_len=8`；2pt/3pt 用 T1（Nvec=100），
gauge 管线（E4/E5）用 L24（spatial=24、Nvec=50）。

| 组 | 生产含义 | 调用数 | opt ms/次 | qcd ms/次 | **opt 占比** | **qcd 占比** |
|---|---|---|---|---|---|---|
| **E1** | 核子 2pt（NN 通道） | 768 | 3159.50 | 936.46 | **51.0%** | **42.6%** |
| **P1** | 3pt 主项（15 算子） | 396 | 1862.45 | 1793.82 | **15.5%** | **42.1%** |
| **P2** | 3pt 主项（置换型） | 396 | 375.86 | 351.69 | **3.1%** | **8.3%** |
| **E4** | peram × gauge × γ | 48 | 26122.80 | 2361.24 | **26.4%** | **6.7%** |
| **E5** | γ 夹心 | 48 | 3963.03 | 123.39 | **4.0%** | **0.4%** |
| 合计 | | | **4757.0 s** | **1688.1 s** | | **2.82x** |

### 2.2 专用路径 A/B（同一进程内切换，直接反映收益）

| 场景 | `specialize=True` | `specialize=False` | 收益 |
|---|---|---|---|
| CPU · E5（L24） | **142.72 ms** | 1188.88 ms | **8.33x** |
| GPU · E4（spatial=1、Nvec=100） | **0.685 ms** | 1.326 ms | **1.9x** |

两条路径数值完全一致（`rel = 0` / `5e-16`）。

### 2.3 与 opt_einsum 的对照（L24 CPU，production 口径，10 次中位）

| 组 | opt_einsum | pyqcd-einsum | 倍率 | 范数 rel |
|---|---|---|---|---|
| E4 | 24.68 s | 1.349 s | **18.29x** | 3.01e-14 |
| E5 | 0.813 s | 0.114 s | **7.13x** | 0 |

（单独跑时：E4 26.56 s / 1.379 s = 19.27x；E5 0.846 s / 0.110 s = 7.67x。
多次独立运行结果一致，非单点偶然。）

## 3. 其它组（未纳入占比模型）

| 组 | 说明 | 实测（CPU T1） |
|---|---|---|
| E2 | 8 算子 3pt 简化版；生产用 P1/P2 | opt 21.85 s / 我们 0.564 s（opt 该 path 未走 BLAS，不计优势） |
| S1 | sush NN 2pt，与 E1 互为替代实现 | opt 498.8 ms / 我们 414.1 ms = 1.21x |
| S2 | sush 3pt 含 μν 插入 | 32.86 ms / 1.99 ms = 16.5x |
| S3 | sush 2pt 含 μ 插入 | 32.90 ms / 1.70 ms = 19.3x |

> E1 与 S1 是同一物理量的两种实现，**不可相加**；E2 与 P1/P2 同理。

## 4. 本轮已修的缺陷（E2）

`transpose_copy()` 在"输出所有维为 1 且需重排"时 `axes.empty()` 直接 `return true`，
目标缓冲未写 → 未初始化内存。命中 E2 的 `greedy`/`auto`（默认用法）。

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
| `python -m pyqcd.testing` | 114 passed / 1 skipped / 1 failed（1 failed 为已修缺陷的历史记录 + 环境硬编码路径） |

## 6. 未闭合

1. **Segfault 未定位**（不可复现，需在无并发构建时做最终确认）。
2. 新专用 kernel（bg-12/13/15/16）未接线。
3. `contract_expression` 未支持 `specialize`。
4. `verify_compat` 全量需 opt_einsum 冻结子模块（网络不可达）。
5. GPU L24 全量测量（bg-6）未返回。

## 7. 下一步（按生产占比排序）

1. **P1**（我们占比 42.1%，且仅 0.96x）——接线 bg-13（KUPL）/ bg-16（CUDA）。
2. **E1**（我们占比 42.6%）——接线 bg-12（CUDA）/ bg-15（KUPL）。
3. 停掉所有后台代理后，独占机器复跑 L24 + T1 全量，记录真实退出码，确认 Segfault 是否仅由并发替换引起。
