# pyqcd-einsum 进度简报（2026-11-08 07:20）

## 1. 本轮最重要：修复一个会静默返回垃圾数据的真实缺陷

### 1.1 现象与最小复现

```python
pe.contract('abcde,cedb->ac', L(1,5,1,5,5), R(1,5,5,5))   # rel = 1.00，返回 5.273e-310
pe.contract('abcde,cedb->ca', L(1,5,1,5,5), R(1,5,5,5))   # 正确
pe.contract('abcde,cedb->ac', L(2,5,1,5,5), R(1,5,5,5))   # 正确
```

触发条件：**输出所有维都是 1，且需要输出轴重排**（`pack_out=1`、`output_perm=1,0`）。

### 1.2 影响面（重要）

命中我们的基准方程 **E2**：
`Geifj,ckdl,agbh,ce,gi,Mbdf,Mhjl,Gka->GM` 在 `greedy` / `auto`（即 opt_einsum 默认用法）下
返回未初始化内存（rel=1.00），只有 `dp` / `optimal` / `optimize=False` 正确。
由于本任务基准一直用 `dp`，此前未被发现。

### 1.3 根因

`cpp/kupl/einsum/src/apply_contract.cpp::transpose_copy()`：

```cpp
for (const CopyAxis& axis : raw) {
    if (axis.extent == 1) continue;     // 所有轴都被跳过
    ...
}
if (axes.empty()) {
    return true;                        // ← 直接返回，目标缓冲一次都没写
}
```

### 1.4 修复与验证

`axes.empty()` 分支先校验 `source_count == destination_count == 1`，再 `memcpy` 单元素。

| 检查 | 修复前 | 修复后 |
|---|---|---|
| 最小复现 `->ac` | rel=1.00 | rel=0 |
| E2 / greedy | rel=1.00 | 1.90e-16 |
| E2 / auto | rel=1.00 | 9.47e-16 |
| E2 / dp | 5.91e-16 | 5.91e-16 |
| `verify_native` kupl | 16/16 | 16/16 |
| `verify_native` cuda | 16/16 | 16/16 |
| `check_equations` | 7/7 | 9/9（含新增 P1/P2） |
| CUDA 侧 E2 greedy/auto | — | 4.48e-16 / 1.43e-15（无同类问题） |

该缺陷由子代理 bg-7 的全库回归发现，并用基线树对比证明**非本次改动引入**。
日志：`logs/v20261108063552.txt`。

## 2. 方程专用优化路径（已可用）

| 能力 | 状态 |
|---|---|
| 形态注册表（9 条格点 QCD 收缩） | 已实现 `pyqcd/einsum/_specialize.py` |
| 命中提示 | `[pyqcd.einsum] 专用路径: <name> [kind] <note>`（每形态每进程一次） |
| 关闭方式 | `contract(..., specialize=False)` 或 `PYQCD_EINSUM_SPECIALIZE=0` |
| 公开查询 | `pe.specializations()` |
| A/B 实测（E4，spatial=1，Nvec=100） | 专用 **0.685 ms** vs 通用 **1.326 ms**（≈1.9x） |

**注**：KUPL 侧的 `specialize=False` 目前不改变执行路径（缺对应 C++ 开关），待补。

## 3. 当前性能（实跑，未美化）

### CPU · T1（Nvec=100，4 线程，complex128，dp）

| 组 | opt_einsum | pyqcd-einsum | 倍率 |
|---|---|---|---|
| E1 | 4.267 s | 0.531 s | **6.98x** |
| E2 | 21.85 s | 0.564 s | 38.7x（opt 该 path 未走 BLAS，不计为优势） |
| E4 | 7.40 ms | 0.87 ms | 8.25x（too_short） |
| S1 | 498.8 ms | 414.1 ms | 1.21x |
| S2 | 32.86 ms | 1.99 ms | 16.5x |
| S3 | 32.90 ms | 1.70 ms | 19.3x |

### CPU · L24（spatial=24、Nvec=50，双方 ≥100 ms，合格对照）

| 组 | opt_einsum | pyqcd-einsum | 倍率 | 范数 rel |
|---|---|---|---|---|
| E4 | 27.20 s | 1.911 s | **14.2x** | 3.01e-14 |
| E5 | 1.178 s | 1.376 s | 0.857x | 0 |

### GPU · T1（单卡对单卡）

| 组 | 倍率 |
|---|---|
| E4 | 1.90x（开专用路径后更高） |
| E5 | 1.49x |
| E1 | 0.80x |
| E2 | 0.81x |

## 4. 真实工作流（`simple/workflow.py` model 口径，CPU/T1，nt=8、tsep=2）

| 组 | 调用数 | opt ms/次 | qcd ms/次 | opt 占比 | qcd 占比 |
|---|---|---|---|---|---|
| E1（2pt） | 768 | 4205.72 | 887.76 | 75.0% | 39.3% |
| P1（3pt） | 396 | 2239.84 | 2244.67 | 20.6% | 51.2% |
| P2（3pt） | 396 | 478.20 | 416.99 | 4.4% | 9.5% |
| **合计** | | **4306.3 s** | **1735.8 s** | | **2.48x** |

判读：**2pt 是最大耗时项（opt 的 75%）**；**3pt 的 P1/P2 是我们最落后的项（0.49–0.85x）**。

## 5. 子代理（9 个运行中）

| 侧 | 代理 | 任务 |
|---|---|---|
| CUDA | bg-9 / bg-11 / bg-12 / bg-16 | E4 冲 3x / 15 算子通用多步 / 2pt 专用 kernel / 3pt 专用 kernel |
| KUPL | bg-13 / bg-15 | 3pt 专用 kernel / 2pt 专用 kernel |
| 双侧 | bg-10（已完成） | E5 夹心路径 |
| 其他 | bg-6 / bg-8 / bg-14 | GPU L24 测量 / Segfault 根因 / 兼容层文档 + A/B 工具 |

新 kernel 均为**新建文件**（`nucleon2pt.*`、`pattern_3pt.*`），不碰正在修改的
`apply_contract.*`、`pack.cu`、`batched_gemm.cu`、`e4_fused.cu`、`sandwich.cu`。

## 6. 全库回归结论（bg-7 交付，`docs/pyqcd_einsum_regression_20261108.md`）

| 检查 | 结果 |
|---|---|
| `verify_native` kupl / cuda | 16/16、16/16 |
| `python -m pyqcd.testing` | 114 passed / 1 skipped / **1 failed** |
| `simple/run-all.sh` | 76 PASS / 0 FAIL |
| `verify_compat` | **无法运行**：oracle 子模块未初始化（`FileNotFoundError`） |
| 失败项 1 | `test_parallel_mpi_planning_contracts`：测试硬编码 `/contract/run`，本机 `/` 不可写（基线同样失败） |
| 失败项 2 | `wick_3pt` 差分：**即本轮修复的缺陷** |

与基线树 `1150334` 逐项对比：PASS 集合完全一致，**无本次改动引入的回归**。

## 7. 未闭合项

1. KUPL 侧 `specialize=False` 开关未实现。
2. 新 kernel（bg-12/13/15/16）未接线到 `apply_contract.*`。
3. `contract_expression` 未支持 `specialize`。
4. `verify_compat` 依赖 opt_einsum 冻结子模块，网络不可达。
5. GPU L24 全量测量未完成（bg-6 在跑）。
