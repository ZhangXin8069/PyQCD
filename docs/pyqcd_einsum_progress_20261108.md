# pyqcd-einsum 进度简报（2026-11-08 05:55）

## 1. 一句话结论

构建与正确性已闭环（KUPL 16/16、CUDA 16/16）；性能**尚未达成 3x**：
CPU 侧仅 E1 为真实优势（2.76x），GPU 侧当前全面落后（0.08–1.49x）。

## 2. 测试组说明（每条方程在算什么）

| 组号 | 含义 | 操作数 | 输出 | 计算特征 |
|---|---|---|---|---|
| **E1** | 核子 2pt 关联函数（NN 通道）：3 个 perambulator 切片 × 2 个 γ₇ × 2 个 VVV 三点张量 × 宇称投影 `(γ₀+γ₄)/2` | 8 | 动量 `M` | 算力主导，opt_cost ≈ 1.92e10 |
| **E2** | 核子 3pt 关联函数：current peram + 源端 peram×2 + γ₇×2 + VVV×2 + 投影 | 8 | `G×M` | 算力主导，opt_cost ≈ 1.92e10 |
| **E4** | perambulator × perambulator × γ × 规范场（含空间格点 z,y,x） | 4 | `T,G,d,b,f,n` | spatial=1 时仅毫秒级（不构成对照）；L24 时 2.75e10 |
| **E5** | γ 夹心 `γ₅·peram†·γ₅` | 3 | 同 prop | 带宽受限；L24 时 1.06e9 |
| **S1** | sush NN 2pt（`contraction.NN.2pt.py:119`） | 8 | `M×N` | 算力主导，opt_cost ≈ 1.92e10 |
| **S2** | sush 3pt 含 μν 插入（`contraction.block.numpy.32x96.compensation.copy.py:402`） | 9 | `M×G` | 6.7e7（Nvec=100） |
| **S3** | sush 2pt 含 μ 插入（同文件 :301） | 6 | `M×F` | 6.7e7（Nvec=100） |

**可比性门槛**：双方中位耗时均 < 0.1 s 的行记 `too_short`，不计入 3x 判定。

## 3. 当前实测（均为本机实跑）

### 3.1 CPU · T1（Nvec=100、spatial=1、n_t=2，4 线程，complex128，dp）

| 组 | opt_einsum | pyqcd-einsum | 倍率 | max rel err |
|---|---|---|---|---|
| E1 | 4.2667 s | 1.5462 s | **2.76x** | 1.23e-14 |
| E2 | 21.8483 s | 0.5642 s | 38.72x ※ | 9.13e-15 |
| E4 | 0.00740 s | 0.00407 s | 1.82x（too_short） | 1.69e-13 |
| E5 | 0.000112 s | 0.000112 s | 1.00x（too_short） | 0 |

※ E2 的高倍率来自 opt_einsum 在该 path 未走 BLAS（单次 2.18 s），**不计为我们的优势**。

### 3.2 GPU · T1（单卡 RTX 4060 Laptop，单 GPU 对单 GPU）

| 组 | opt_einsum | pyqcd-einsum | 倍率 |
|---|---|---|---|
| E1 | 383.7 ms | 479.4 ms | 0.80x |
| E2 | 377.5 ms | 463.5 ms | 0.81x |
| E4 | 1.23 ms | 14.80 ms | 0.08x（too_short） |
| E5 | 0.42 ms | 0.28 ms | 1.49x（too_short） |

### 3.3 CPU · L24（spatial=24、Nvec=50，双方均 ≥100 ms，**合格对照**）

| 组 | opt_einsum | pyqcd-einsum | 倍率 | max rel err |
|---|---|---|---|---|
| E5 | 1.1784 s | 1.3755 s | 0.857x | 0 |

## 4. 根因（已定位）

1. **无算法红利**：E1 的 greedy / dp / branch-2 / auto 代价完全相同（19,204,160,000，差 <0.002%）。
2. **E1/E2 在 GPU 上已被打平**：opt_einsum 443.6 ms / 我们 386.0 ms /
   `cupy.einsum(optimize=True)` 388.1 ms / 手工分步 373.1 ms —— 四方同区间。
3. **E4 CUDA 融合 kernel 有约 13x 冗余**：GPU 事件 10.6 ms，同机 cupy 仅 0.55–0.84 ms。
4. **小规模固定开销**：tiny matmul 经已缓存 context 0.088 ms，经每次新建 context 0.994 ms（11x）。

> 口径纠正：GPU 对 GPU 的倍率**与 CPU/GPU 算力差无关**（此前论证有误，已撤回）。
> 判断依据只有「双方是否都已接近同一硬件的有效吞吐上限」。

## 5. 子代理分工（文件边界互不重叠）

| 代理 | 职责 | 允许改动 | 状态 |
|---|---|---|---|
| bg-1 | CUDA E4 融合 kernel 优化 | `cpp/cuda/einsum/src/e4_fused.cu` | 运行中 |
| bg-2 | KUPL CPU 执行层（pack/GEMM 派发） | `cpp/kupl/einsum/src/apply_contract.cpp` | 运行中 |
| bg-3 | Python 单次调用固定开销 | `pyqcd/einsum/_api.py`、`pyqcd/einsum/cuda/_program.py` | 运行中 |
| bg-4 | PyQCU simple 风格测试套件 | 新建 `pyqcd/testing/einsum/simple/**` | 运行中 |
| bg-5 | 文档/产物名一致性 | `cpp/*/einsum/AGENTS.md`、新建 docs | 已完成 |

## 6. 基准口径改造（已完成）

- `--data real|synthetic`：真实数据缺失时用固定种子合成操作数，产物 `provisional=true`。
- `--planner production|shared`：production=双方各自规划一次、只计稳态执行；shared=共用显式 path。
- 默认 `--optimize dp`。
- `MIN_MEANINGFUL_SECONDS = 0.1` 可比性门槛。

## 7. 已知问题与阻塞

| 项 | 状态 |
|---|---|
| L24/E5 CPU 实跑退出期 Segfault | 一次出现、二次无法复现；判定为子代理并发替换 `.so` 的映射竞态，最终验收需复测确认 |
| 真实数据（`/public/group/lqcd`、`*.lime`） | 本机不存在 → 用合成数据 + opt_einsum 数值对照 |
| opt_einsum 冻结子模块 | 网络不可达（`GnuTLS recv error`）→ 用站点包 3.4.0，`frozen=false` |
| GPU L24 全量 | 未运行 |

## 8. 下一步

1. 收 bg-1/bg-2/bg-3 的优化结果，独占机器复测（禁止并发编译/计时）。
2. GPU 侧 L24 生产规模全量（E1/E2/E4/E5/S1/S2/S3）。
3. 修/确认 Segfault，补齐 `verify_native` 与 simple 测试套件回归。
4. 全量验收报告 + 日志归档 + 单次提交。
