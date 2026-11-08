# pyqcd-einsum 进度简报（2026-11-08 07:52）

## 1. 子代理健康检查

| 检查项 | 结果 |
|---|---|
| 进程 | 仅本会话 watcher（`agent-tui-watch.py`）在跑；子代理为会话内异步执行，无独立 OS 进程 |
| 活跃写入 | scratchpad 近 30 分钟有持续写入（`p3/`、`n2pt_common.py`、`libp3pt.so`、`libn2pt_bench.so`、`cupy_cache`）→ **代理确实在工作** |
| kernel 源文件 | bg-17/18/19 的 `.wip` 自 06:51–07:18 起未变（在定位编译/内存错误阶段）；bg-20 的 `nucleon2pt.cpp` 07:24 更新过 |
| 构建状态 | `build-kupl.sh` SUCCESS、`build-nv.sh` SUCCESS（代理未把构建改红） |
| 产出 | bg-6 已交 `docs/pyqcd_einsum_gpu_l24_20261108.md` |

## 2. 【新发现·真实 bug】CUDA E4 融合路径在 L24 上慢 33 倍

### 2.1 现象（主代理独立复现，非引用子代理）

形状：`spatial=24`、`Nvec=50`、`n_t=2`，prop 单算子 0.99 GiB，complex128，单卡 RTX 4060。

| 路径 | 中位耗时（3 次） | 相对 |
|---|---|---|
| `specialize=True`（融合 kernel） | **36.436 s** | **33x 慢** |
| `specialize=False`（通用多步） | **1.104 s** | 1.00x |
| 数值一致性 | `rel = 7.42e-14` | 两条路径结果一致 |

即**不是数值错误，而是性能灾难**：融合 kernel 在小 spatial（1–3）上快 1.9x，
在 spatial=24 上退化到通用路径的 1/33。

按用户口径（劣于 opt_einsum 即 bug）：opt_einsum 在 L24 上约 1.09–1.14 s，
融合路径 36.4 s → **慢 32–33 倍，属严重 bug**。

### 2.2 子代理 bg-6 的补充证据

其报告 `docs/pyqcd_einsum_gpu_l24_20261108.md` 记录：
- E4 在 GPU L24 上 qcd 20.37–20.74 s vs opt 1.09–1.14 s（0.053–0.056x）
- **默认融合路径下 `qcd_einsum` worker `returncode=-11`（SIGSEGV）**，多构建状态复现
- **关闭 fused 后不再崩，且回到 0.980x**

→ 融合路径在 L24 上既有性能灾难，也有崩溃风险。

## 3. 根因初判

`cpp/cuda/einsum/src/e4_fused.cu` 的 `e4_staged_kernel` 对 `(t, b, f-tile)` 分块，
并按 `kStagedSharedBudget = 48 KiB` 做暂存。当 `z·y·x` 从 1 增大到 13824 时：
- 每个 block 需重复读取整块 gauge/prop，缓存命中率崩塌；
- 暂存表尺寸随空间维增长，可能退化到全局内存随机访问；
- 缺少"何时不该融合"的判据 → 无条件命中。

## 4. 处置计划

1. **立即（修真 bug）**：在 `match_e4_fused` 增加**收益判据**——仅当空间规模足够小
   （例如 `z*y*x <= 阈值` 或输出/输入元素数低于阈值）时才启用融合，否则回落到通用多步。
   这样 L24 走 1.10 s 的通用路径（≈opt 的 1.0x），spatial=1 仍走 0.68 ms 的融合路径。
2. **随后**：分析 staged kernel 在 spatial=24 的访存模式，决定是否值得为大规模实现分块版本。
3. **验收**：`kernel_acceptance` 口径 + L24 GPU 复测（含崩溃检查）+ 双后端回归。

## 5. 当前全局状态

| 项 | 状态 |
|---|---|
| 构建（kupl/cuda） | 均 SUCCESS |
| `verify_native` | kupl 16/16、cuda 16/16 |
| `verify_compat` | 980/980 PASS |
| `python -m pyqcd.testing` | 115 passed / 1 skipped / 0 failed |
| 已知未修 bug | **E4 CUDA 融合路径在 L24 慢 33 倍且可崩溃**（本节） |
| 未完成 kernel | 3 个 `.wip`（bg-17/18/19 在处理）+ 1 个待优化（bg-20） |
