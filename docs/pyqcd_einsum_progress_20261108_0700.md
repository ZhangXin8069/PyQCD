# pyqcd-einsum 进度简报（2026-11-08 07:00）

## 1. 本轮新增：方程专用优化路径（已落地并可用）

### 1.1 机制

| 文件 | 作用 |
|---|---|
| `pyqcd/einsum/_specialize.py` | 形态注册表（9 条格点 QCD 常见收缩）+ 命中判定 + 提示打印 |
| `pyqcd/einsum/_api.py` | `contract(..., specialize=True)`；关闭时置 `PYQCD_EINSUM_DISABLE_FUSED` |
| `cpp/cuda/einsum/src/apply_contract.cu` | `initialize_context` 读取该变量，跳过 `match_sandwich` / `match_e4_fused` |
| `pyqcd/einsum/__init__.py` | 公开 `pe.specializations()`（列出全部已注册形态） |

开关优先级：`contract(specialize=False)` > `PYQCD_EINSUM_SPECIALIZE=0` > 默认开启。
命中时打印 `[pyqcd.einsum] 专用路径: <name> [<kind>] <note>`（每形态每进程一次；
`PYQCD_EINSUM_SPECIALIZE_VERBOSE=1` 可每次打印）。

### 1.2 已注册形态

| 名称 | 类型 | 说明 |
|---|---|---|
| nucleon-2pt(E1) | fused-plan | 8 算子核子 2pt |
| nucleon-3pt(E2) | fused-plan | 8 算子核子 3pt |
| gauge-propagator(E4) | fused | CUDA E4 融合 kernel |
| gamma-sandwich(E5) | fused | γ₅ 夹心 |
| sush-2pt(S1) / sush-3pt-munu(S2) / sush-2pt-mu(S3) | fused-plan | sush 生产方程 |
| sush-3pt-main(P1/P2) | multi-step | 15 算子 14 步（生产热点） |

### 1.3 实测收益（E4，spatial=1、Nvec=100、RTX 4060，10 次中位）

| 模式 | 中位耗时 |
|---|---|
| `specialize=True` | **0.685 ms** |
| `specialize=False` | **1.326 ms** |
| 切回 `True` | 0.624 ms |

专用路径约 **1.9x**；正确性不受影响（该形状 `rel ≈ 5e-16`）。

## 2. CUDA / KUPL 同步分工（11 个代理）

| 形态 | CUDA | KUPL |
|---|---|---|
| 2pt（E1/S1） | bg-12 | bg-15 |
| 3pt 主项（P1/P2） | bg-16 | bg-13 |
| E4 规范场 | bg-9（已完成，16.7x） | — |
| E5 夹心 | bg-10 | bg-10 |
| 通用多步（15 算子） | bg-11 | — |
| 测量/回归/根因 | bg-6 GPU L24、bg-7 全库回归、bg-8 Segfault | |
| 文档/工具 | bg-14 兼容层文档 + `specialize_ab.py` | |

新 kernel 全部为**新建文件**（`nucleon2pt.*`、`pattern_3pt.*`），不碰正在修改的
`apply_contract.*`、`pack.cu`、`batched_gemm.cu`、`e4_fused.cu`、`sandwich.cu`，接线由主代理统一做。

## 3. 当前性能（全部实跑，未美化）

### CPU · T1（Nvec=100，4 线程，complex128，dp）

| 组 | opt_einsum | pyqcd-einsum | 倍率 |
|---|---|---|---|
| E1 | 4.267 s | 0.531 s | **6.98x** |
| E2 | 21.85 s | 0.564 s | 38.7x（opt 该 path 未走 BLAS，不计为优势） |
| E4 | 7.40 ms | 0.87 ms | 8.25x（too_short） |
| S1 | 498.8 ms | 414.1 ms | 1.21x |
| S2 | 32.86 ms | 1.99 ms | 16.5x |
| S3 | 32.90 ms | 1.70 ms | 19.3x |

### CPU · L24（spatial=24、Nvec=50，双方均 ≥100 ms，合格对照）

| 组 | opt_einsum | pyqcd-einsum | 倍率 | 范数 rel |
|---|---|---|---|---|
| E4 | 27.20 s | 1.911 s | **14.2x** | 3.01e-14 |
| E5 | 1.178 s | 1.376 s | 0.857x | 0 |

### GPU · T1（单卡对单卡）

| 组 | 倍率 |
|---|---|
| E4 | 1.90x（专用路径开启后更高） |
| E5 | 1.49x |
| E1 | 0.80x |
| E2 | 0.81x |

## 4. 真实工作流时间构成（`simple/workflow.py`，model 口径，CPU/T1，nt=8、tsep=2）

| 组 | 调用数 | opt ms/次 | qcd ms/次 | opt 占比 | qcd 占比 |
|---|---|---|---|---|---|
| E1（2pt） | 768 | 4205.72 | 887.76 | 75.0% | 39.3% |
| P1（3pt） | 396 | 2239.84 | 2244.67 | 20.6% | 51.2% |
| P2（3pt） | 396 | 478.20 | 416.99 | 4.4% | 9.5% |
| **合计** | | **4306.3 s** | **1735.8 s** | | **2.48x** |

热点判读（已纠正过一次）：**2pt 是最大耗时项（opt 的 75%）**；
**3pt 的 P1/P2 是我们最落后的项（0.49–0.85x）**，属"优化机会最大"，不是"耗时占比最大"。

## 5. 验证状态

| 检查 | 结果 |
|---|---|
| `verify_native --backend kupl` | 16 passed / 0 failed |
| `verify_native --backend cuda` | 16 passed / 0 failed |
| `check_compat` | 39 passed / 0 failed |
| `check_equations`（7 组，T0n） | 7 passed / 0 failed |
| `check_native`（kupl/cuda） | 各 13 passed / 0 failed |
| `simple/run-all.sh` | exit=0 |

## 6. 未闭合项（如实）

1. **KUPL 侧的 `specialize=False` 暂不生效**：`cpp/kupl/einsum/src/apply_contract.cpp` 正被 bg-10 占用，
   屏蔽开关待其释放后补，并复测 A/B。
2. 新 kernel（bg-12/13/15/16）均为独立文件，**尚未接线**到 `apply_contract.*` 的匹配逻辑。
3. `contract_expression` 尚未支持 `specialize` 参数（当前仅 `contract`）。
4. E4/L24 的 `rel` 早期曾报 5.73e-12，经复现确认是**逐元素指标在抵消元素上放大**；
   已改为范数口径判据，实际 3.01e-14（详见 `docs/pyqcd_einsum_progress_20261108_0635.md` §9 勘误）。

## 7. 下一步

1. 收 bg-12/13/15/16 的 kernel，统一接线并做 CPU/GPU 对称复测。
2. 补 KUPL 屏蔽开关，使 `specialize=False` 在两后端都可对照。
3. 独占机器做最终验收：T1 + L24 × {cpu, gpu} × 7 组，`verify_compat` 全量。
