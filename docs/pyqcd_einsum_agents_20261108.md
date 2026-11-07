# pyqcd-einsum 子代理进度简报（2026-11-08 08:15）

## 1. 总览

共派发 16 个子代理：**12 完成 / 1 主动终止 / 3 运行中**。
核心原则：每个代理只碰自己声明的文件；新 kernel 一律写**新文件**，接线由主代理统一做。

## 2. 已产出并已验证（可用的收益）

| 代理 | 交付 | 已验证结果 |
|---|---|---|
| **bg-1** | `cpp/cuda/einsum/src/e4_fused.cu`：FMA 累加 + 寄存器/occupancy 调整 | E4 GPU 10.6 ms → **0.635 ms（16.7x）**，rel 4.3e-16 |
| **bg-2** | `cpp/kupl/einsum/src/apply_contract.cpp`：CPU 执行层 pack/GEMM 派发 | CPU E1（N=100）1.546 s → **0.531 s**，相对 opt **6.98x** |
| **bg-9** | E4 kernel 继续优化（尝试记录） | 见其报告；已并入 bg-1 的版本 |
| **bg-10** | `sandwich.cu` + `apply_contract.cpp`：E5 夹心路径 | E5 L24 CPU → **114 ms（opt 813 ms，7.1x）**；A/B 专用 142.7 vs 通用 1188.9 ms |
| **bg-7** | 全库回归 + 差分 | 发现 **E2 未初始化内存缺陷**；证明非本次改动引入；`python -m pyqcd.testing` 114 passed/1 skipped/1 failed |
| **bg-8** | Segfault 根因 | 见 §3（结论确定：并发就地截断 `.so`） |
| **bg-5** | 文档一致性 | `cpp/*/einsum/AGENTS.md` 产物名修正 + `docs/pyqcd_einsum_build_matrix_20261108.md` |
| **bg-14** | 兼容层文档 + A/B 工具 | `docs/pyqcd_einsum_compat_layer_20261108.md`、`simple/specialize_ab.py` |

## 3. bg-8 的 Segfault 结论（本轮最重要的排查）

| 假设 | 实验次数 | 崩溃数 | 判定 |
|---|---|---|---|
| 原子替换 / 删除已映射 `.so` | 100 | **0** | 未复现（原子 `rename` 保留旧 inode，映射有效） |
| **就地截断**已映射 `.so` | 80 | **50/50 崩**（截断），0/30（不截断） | **机制成立** |
| spawn 父子进程同持 `.so` | 40 | 0 | 未复现 |
| OpenBLAS 线程池 + fork/spawn | 120 | 0 | 未复现 |
| 真实 bench 负载（L24/E5） | 28 | 静止 0/8；原子 churn 2/14；**尾部就地改写 6/6** | 见下 |

**判定**：事故是**并发重建竞态**，充分条件是"**已映射 inode 被就地截断**"，
而非原子替换。触发者应为绕过 `make.sh` 的直接链接，或
`setup.py::_stage_native_library()` 的 `shutil.copy2`（`O_TRUNC` 就地覆盖）。

**已据此修复**（主代理执行）：`setup.py::_stage_native_library()` 改为
"临时文件 + `os.replace`" 原子落位：

```python
staged = destination.with_name(destination.name + f".tmp.{os.getpid()}")
shutil.copy2(source, staged)
os.replace(staged, destination)
```

**最终验收要求**：停掉所有后台代理后独占复跑，记录真实退出码。

## 4. 已落盘但**尚未接线**的专用 kernel（4 个文件）

| 侧 | 文件 | 大小 | 代理 | 状态 |
|---|---|---|---|---|
| CUDA | `cpp/cuda/einsum/src/nucleon2pt.cu` + `include/nucleon2pt.h` | 32.7 KB / 8.1 KB | bg-12 | 运行中/未接线 |
| CUDA | `cpp/cuda/einsum/src/pattern_3pt.cu` + `include/pattern_3pt.h` | 39.5 KB / 5.0 KB | bg-16 | 撞轮次上限，**未完成验证** |
| KUPL | `cpp/kupl/einsum/src/nucleon2pt.cpp` + `include/nucleon2pt.h` | 39.6 KB / 6.4 KB | bg-15 | 运行中/未接线 |
| KUPL | `cpp/kupl/einsum/src/pattern_3pt.cpp` + `include/pattern_3pt.h` | 44.7 KB / 4.8 KB | bg-13 | 撞轮次上限，**未完成验证** |

**已核实**：`grep` 全仓，这 4 个文件在自身之外**没有任何引用** → 确认全部未接线，
当前所有性能数字都不含它们的贡献。

## 5. 撞轮次上限的两个代理（bg-13 / bg-16）

两者都在"写测试驱动 / 测 before 基线"阶段用尽轮次，**未完成数值验证与性能对照**。
其源码已落盘但**不可假定正确**：
- 未接线，故不影响现有构建与结果；
- 接线前必须由主代理做：单独编译 → 数值对照（rel ≤ 1e-12）→ 与通用路径 A/B 计时。

## 6. 运行中（3 个）

| 代理 | 任务 | 已知进展 |
|---|---|---|
| bg-6 | GPU L24 生产规模测量（production/shared 两组） | 已产出 `artifacts/gpu_l24_v3_shared_E4_a1/`、`..._E5_a1/` |
| bg-12 | CUDA 2pt 专用 kernel | 源文件已落盘，未接线 |
| bg-15 | KUPL 2pt 专用 kernel | 源文件已落盘，未接线 |

## 7. 主动终止

| 代理 | 原因 |
|---|---|
| bg-4 | simple 测试套件；主代理直接实现并当场验证（`run-all.sh` exit=0，76 PASS），避免与其产物冲突 |

## 8. 代理产出的文档

| 文档 | 来源 |
|---|---|
| `docs/pyqcd_einsum_build_matrix_20261108.md` | bg-5 |
| `docs/pyqcd_einsum_regression_20261108.md` | bg-7 |
| `docs/pyqcd_einsum_segfault_analysis_20261108.md` | bg-8 |
| `docs/pyqcd_einsum_compat_layer_20261108.md` | bg-14 |
| `docs/pyqcd_einsum_progress_20261108_*.md` | 主代理 |

## 9. 下一步（接线顺序，按生产占比）

1. **P1/P2**（我们占比 42.1%/8.3%，P1 仅 0.96x）→ 先验证 bg-13/bg-16 的 kernel，再接线。
2. **E1**（我们占比 42.6%）→ 验证 bg-12/bg-15。
3. 接线后必须在**无并发构建**下复跑 L24 + T1 全量，并记录真实退出码。
