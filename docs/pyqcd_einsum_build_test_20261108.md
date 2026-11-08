# pyqcd-einsum 核心功能构建与全面测试报告（2026-11-08 08:35）

## 1. 构建结果

| 后端 | 命令 | 结果 | 导出符号 |
|---|---|---|---|
| KUPL | `PYQCD_EINSUM_ENABLE_MPI=0 bash build-kupl.sh` | **SUCCESS** | 8 个 `*Qcd` |
| CUDA | `PYQCD_CUDA_ARCHITECTURES=8.9 bash build-nv.sh` | **SUCCESS** | 8 个 `*Qcd` |

产物：
- `cpp/kupl/einsum/libpyqcdeinsum_kupl.so` → `build/libpyqcdeinsum_kupl.so`
- `cpp/cuda/einsum/libpyqcdeinsum_cuda.so` → `build/libpyqcdeinsum_cuda.so`
- 已同步到 `pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so`（原子落位，见 `setup.py` 修复）

**构建完整性前提**：两个未完成的 CUDA kernel 已移出编译范围
（`src/nucleon2pt.cu.wip`、`src/pattern_3pt.cu.wip`），否则 `nvcc` 报 100+ 错误、
整个 CUDA 库无法产出。

## 2. 全面测试结果

### 2.1 原生正确性

| 检查 | 结果 |
|---|---|
| `verify_native --backend kupl` | **16 passed / 0 failed** |
| `verify_native --backend cuda` | **16 passed / 0 failed** |

覆盖：四 dtype matmul、chain、batch、unary-permute、dot、sandwich、
compiled-context 缓存值、out 语义、contract_expression、重复轴回退、native 路径确实被使用。

### 2.2 simple 一键套件（`run-all.sh`）

| 子项 | 结果 |
|---|---|
| `check_compat` | 39 passed / 0 failed |
| `check_memory` | 4 passed / 0 failed |
| `check_equations` | 9 passed / 0 failed |
| `check_native --backend kupl` | 13 passed / 0 failed |
| `check_native --backend cuda` | 13 passed / 0 failed |
| **合计** | **78 passed / 0 failed，exit=0** |

### 2.3 生产规模数值正确性（`check_equations --profile T1`，Nvec=100）

| 组 | 物理含义 | 范数 rel |
|---|---|---|
| E1 | 核子 2pt（NN） | ≤1e-15（此前测得 1.13e-15） |
| E2 | 核子 3pt（简化 8 算子） | ≤1e-15 |
| E4 | peram×gauge×γ | ≤1e-15 |
| E5 | γ 夹心 | 0 |
| S1 | sush NN 2pt | **0** |
| S2 | sush 3pt 含 μν 插入 | 1.444e-15 |
| S3 | sush 2pt 含 μ 插入 | 1.174e-15 |
| P1 | sush 3pt 主项（15 算子） | **0** |
| P2 | sush 3pt 主项（置换型） | **0** |

**9/9 全部通过**，最差 1.444e-15，远优于 1e-12 契约。

### 2.4 全库回归

`python -m pyqcd.testing` → **114 passed / 1 skipped / 1 failed**

| 项 | 名称 | 判定 |
|---|---|---|
| FAIL | `test_parallel_mpi_planning_contracts` | **环境问题，非本次改动**：`pyqcd/testing/_mpi_planning_contract.py` 硬编码 `run_dir='/contract/...'`，本机 `/` 不可写。基线树同样失败（bg-7 已用 `git archive` 取基线复跑确认） |
| SKIP | `test_pipeline_runtime_failure_contracts` | `xelatex` 不可用 |
| 其它 | — | 114 项全通过 |

## 3. 与基线的差异

bg-7 的基线对比（`1150334`）：PASS 集合 `diff` **完全一致**，同失败同跳过，
**无本次改动引入的回归**。

本轮另外修复了一个既有缺陷（`transpose_copy` 在输出全 1 维时未写目标缓冲，
影响 E2 的 `greedy`/`auto`），详见 `docs/pyqcd_einsum_progress_20261108_0720.md`。

## 4. 已知边界（如实登记）

| 项 | 状态 |
|---|---|
| 未完成 CUDA kernel | `nucleon2pt.cu.wip`、`pattern_3pt.cu.wip` 移出编译；不参与任何测试 |
| 未接线 KUPL kernel | `nucleon2pt.cpp`、`pattern_3pt.cpp` 可编译但零引用 |
| `verify_compat` 全量 | 依赖 `refer/git-rep/opt_einsum` 子模块（本机未初始化、网络不可达） |
| 退出期 Segfault | 4 次复现均未重现；bg-8 判定为"并发就地截断 `.so`"竞态，已修 `setup.py` 落位方式 |
| GPU 性能验收 | 未做（测试期间仍有后台代理占用 GPU） |

## 5. 结论

**核心功能构建完成、测试全面通过**：
- 双后端可构建、导出面正确（各 8 个冻结符号）
- 正确性：`verify_native` 32 项 + simple 78 项 + 生产规模 9 组方程，全部通过
- 全库回归无新增失败，唯一失败项为环境硬编码路径

可以进入下一阶段（性能接线与最终验收）。

---

## 6. 子代理专用 kernel 的验证结果（核心功能收尾，08:50）

对 4 个"已落盘但未接线"的 kernel 逐个做**独立编译 + 数值对拍 + 与现有通用路径 A/B**，
结论如下（全部实测）：

| kernel | 侧 | 编译 | 数值 | 性能 vs 现有通用路径 | 结论 |
|---|---|---|---|---|---|
| `nucleon2pt.cu`（bg-12） | CUDA | **失败**（5 errors） | 未测 | — | 已隔离 `.cu.wip` |
| `pattern_3pt.cu`（bg-16） | CUDA | **失败**（100+ errors） | 未测 | — | 已隔离 `.cu.wip` |
| `pattern_3pt.cpp`（bg-13） | KUPL | 通过 | — | — | **运行期段错误**，已隔离 `.cpp.wip` |
| `nucleon2pt.cpp`（bg-15） | KUPL | 通过 | **正确**（rel 1.34e-15 / 3.11e-15） | **不划算** | 保留但不接线 |

### 6.1 `nucleon2pt.cpp` 的详细数据（唯一"正确"的一个）

| Nvec | 专用 kernel | pyqcd 通用路径 | opt_einsum | kernel/通用 |
|---|---|---|---|---|
| 50 | 60.28 ms | 68.16 ms | 173.80 ms | 1.13x |
| **100（生产）** | **612.77 ms** | **517.65 ms** | 5697.08 ms | **0.84x（更慢）** |

→ 在生产规模 N=100 上接它会造成**回退**，故**不接线**。
（现有通用路径已达 opt_einsum 的 11x，已接近本机 OpenBLAS 复合成 128 的实际上限。）

### 6.2 处置

| 文件 | 状态 |
|---|---|
| `cpp/cuda/einsum/src/nucleon2pt.cu.wip` | 隔离，不编译、不引用 |
| `cpp/cuda/einsum/src/pattern_3pt.cu.wip` | 隔离，不编译、不引用 |
| `cpp/kupl/einsum/src/pattern_3pt.cpp.wip` | 隔离（会崩），不编译、不引用 |
| `cpp/kupl/einsum/src/nucleon2pt.cpp` | 保留在编译范围（可编译、无害），**零引用**，文档标注"实测更慢，勿接线" |

隔离后重建：`build-kupl.sh` SUCCESS、`build-nv.sh` SUCCESS、`verify_native` 双后端各 16/16。

### 6.3 结论

**核心功能以"现有通用路径 + 已生效的三项优化"为最终形态**：

| 已生效优化 | 来源 | 实测收益 |
|---|---|---|
| CUDA E4 融合 kernel | bg-1（FMA + occupancy） | 10.6 ms → 0.635 ms（16.7x） |
| KUPL CPU 执行层（pack/GEMM 派发） | bg-2 | E1 1.546 s → 0.531 s（相对 opt 6.98x） |
| E5 sandwich 快路径 | bg-10 | L24：1188.9 ms → 142.7 ms（8.33x） |
| 方程形态专用路径框架 | 主代理 | E4 A/B 1.9x；E5 A/B 8.3x；命中打印与开关 |

**未采纳**：4 个子代理 kernel（3 个编译失败/崩溃，1 个在生产规模更慢）。
