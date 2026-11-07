# pyqcd-einsum 当前性能优势评估（2026-11-08 实测）

> 本文只记录**本次实跑**得到的数据与结论。所有数字均可由文中命令复现；
> 未验证项在末尾单列，不与结论混写。

## 1. 结论摘要

| 判断 | 结论 |
|---|---|
| CPU 侧是否存在真实优势 | **存在，但有条件**：E1 达 2.76x，E4 达 1.82x；E5 与 opt_einsum 持平 |
| GPU 侧是否存在优势 | **不存在，目前是净劣势**：E1 0.80x、E2 0.81x、E4 0.08x、E5 1.49x |
| 「3x 目标」当前达成情况 | CPU 4 条验收方程中 1 条达标（且该条是 oracle 自身异常）；GPU 0 条达标 |
| 3x 在 GPU 上是否物理可达 | E1/E2 **大概率不可达**（算力已打平）；E4/E5 有 10x 级实现冗余，可达 |

## 3. 测试组说明（每条方程到底在算什么）

| 组号 | 物理/工程含义 | 操作数构成 | 输出 | 计算特征 |
|---|---|---|---|---|
| **E1** | 核子 **2pt 关联函数**（NN 通道）：三个 perambulator 时间切片 × 两个 γ₇ × 两个 VVV 三点张量 × 宇称投影 `(γ₀+γ₄)/2` | 8 个（peram×3、γ×2、VVV×2、投影×1） | 单一动量 `M` | 算力主导，opt_cost ≈ 1.92e10 |
| **E2** | 核子 **3pt 关联函数**：current perambulator + 源端 perambulator ×2 + γ₇ ×2 + VVV ×2 + 投影 | 8 个 | `G×M`（动量对） | 算力主导，opt_cost ≈ 1.92e10 |
| **E4** | perambulator × perambulator × γ × **规范场**，含真实空间格点 z,y,x | 4 个 | `T,G,d,b,f,n` | spatial=1 时仅 ~1e8 复乘（毫秒级，**不足以作实战对照**）；L24 时 6.6e9 级 |
| **E5** | **γ 夹心**：`γ₅ · peramb† · γ₅` | 3 个 | 同 prop 形状 | 纯带宽受限，微秒级；L24 时才到 5e8 级 |
| **S1** | sush **NN 2pt**（真实生产脚本 `contraction.NN.2pt.py`） | 8 个（peram×3、γ×2、VVV×2、投影） | `M×N` | 算力主导，opt_cost ≈ 1.92e10 |
| **S2** | sush **3pt 含 μν 插入**（`contraction.block.numpy.32x96.compensation.copy.py:402`） | 9 个（peram×3、VdV×2、sink、γ、μνγ×2） | `M×G` | 中等规模，opt_cost ≈ 6.7e7（Nvec=100） |
| **S3** | sush **2pt 含 μ 插入**（同文件 :301） | 6 个（peram×2、VdV×2、γ、μγ） | `M×F` | 中等规模，opt_cost ≈ 6.7e7（Nvec=100） |

> **可比性门槛（新增，2026-11-08）**：若某组在**双方**的中位耗时都低于 100 ms，
> 则该组仅记为 `too_short`（诊断用），**不得**作为实战性能对照参与 3x 判定。
> 现有 `spatial=1` 的 E4/E5 即属此列；须使用 `L24`（spatial=24、Nvec=50）profile 才有意义。

## 4. 测试口径（保证可复核）

- 同一台机器、同一进程隔离（`--isolation process`，两边各自全新解释器）
- 同一 dtype（complex128）、同一输入（固定种子合成操作数，形状与真实蒸馏一致）
- 同线程数（`PYQCD_BENCH_THREADS=4`，同时钉住 `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB`）
- 同 planner 策略（`dp`），两边各自规划一次，只计时稳态执行
- 预热 2 轮、取 10 次中位数、计时前后同步
- 正确性：逐元素 `max|Δ| / max|ref| ≤ 1e-12`

命令：

```bash
cd /home/zhangxin/PyQCD
PYQCD_EINSUM_ORACLE=installed PYQCD_EINSUM_NATIVE=kupl PYQCD_BENCH_THREADS=4 \
  python -m pyqcd.testing.einsum.bench_native \
  --data synthetic --profiles T1 --backends cpu --iters 10 --warmup 2 --planner shared
```

> 说明：本次 opt_einsum 冻结子模块（`refer/git-rep/opt_einsum`）无法初始化
> （`GnuTLS recv error (-110)`），oracle 使用站点包 **opt_einsum 3.4.0**，
> 产物中 `frozen=false`。数据亦不在本机（`/public/group/lqcd` 不存在），
> 故使用固定种子合成数据，产物标记 `provisional=true`。

## 3. CPU 实测（T1：Nvec=100、spatial=1、n_t=2，4 线程）

| 方程 | opt_einsum 中位耗时 | pyqcd-einsum 中位耗时 | 倍率 | max rel err |
|---|---|---|---|---|
| E1 | 4.266694929 s | 1.546226513 s | **2.76x** | 1.23e-14 |
| E2 | 21.84825881 s | 0.564247697 s | 38.72x ※ | 9.13e-15 |
| E4 | 0.007397338 s | 0.004068588 s | **1.82x** | 1.69e-13 |
| E5 | 0.000112043 s | 0.000112081 s | 1.00x | 0 |

※ **E2 的 38.7x 不能算作真实优势**：opt_einsum 在该收缩 path 上未走 BLAS，
单次耗时 2.18 s，属 oracle 自身实现问题。必须同时看绝对耗时，不能只报倍率。

## 4. GPU 实测（T1，单卡 RTX 4060 Laptop，单 GPU 对单 GPU，cupy oracle）

| 方程 | opt_einsum 中位耗时 | pyqcd-einsum 中位耗时 | 倍率 |
|---|---|---|---|
| E1 | 383.7 ms | 479.4 ms | 0.80x |
| E2 | 377.5 ms | 463.5 ms | 0.81x |
| E4 | 1.23 ms | 14.80 ms | **0.08x** |
| E5 | 0.42 ms | 0.28 ms | 1.49x |

## 5. 根本原因（已定位，非推测）

### 5.1 E1/E2 在 GPU 上已被算力打平

对 E1（Nvec=100）用四种方式实测，全部落在同一区间：

| 实现 | 中位耗时 |
|---|---|
| opt_einsum（cupy，greedy path） | 443.56 ms |
| pyqcd.einsum（CUDA 原生） | 385.95 ms |
| `cupy.einsum(optimize=True)` | 388.09 ms |
| 手工按同一 path 分步 `cupy` | 373.13 ms |

该方程 `opt_cost = 19,204,160,000`，且 **greedy / dp / branch-2 / auto 四种 planner 给出的
代价完全相同**（差 < 0.002%），即没有算法层可挖的路径红利。任何"执行层优化"都无法在这条
方程上拿到 3x。

### 5.2 硬件吞吐参考（**不构成 GPU vs GPU 倍率的约束**）

| 基准 | 实测 |
|---|---|
| CPU OpenBLAS `zgemm` 2000³（4 线程） | 227.6 GFLOP/s |
| GPU cuBLAS `zgemm` 2000³ | 71.3 GFLOP/s |
| GPU cuBLAS `cgemm`（complex64）2000³ | 934.5 GFLOP/s |

> **口径纠正（重要）**：上表只用于描述本机各后端的绝对吞吐。**GPU 对 GPU 的倍率与
> CPU/GPU 算力差无关**——比较双方跑在同一块 GPU 上，硬件算力自然抵消，倍率只反映两套实现
> 的相对效率。此前把"CPU 比 GPU 快 3.2 倍"当作 GPU 侧 3x 的约束条件是**错误论证**，已撤回。
>
> 正确的判断依据只有一条：**双方是否都已接近该硬件对该 dtype 的有效吞吐上限**。
> 若已接近，则倍率必然收敛到 1x——这正是 §5.1 中四种实现落在 373–444 ms 同一区间所显示的事实。

### 5.3 E4 存在真实可挖的浪费（约 13x）

- 该方程走的是 CUDA 融合 kernel（`PYQCD_EINSUM_TRACE=1` 显示
  `PYQCD_EINSUM_CUDA_E4 ... kernel=1`）。
- 输出 320,000 个 complex128 元素、约 1e8 次复乘；GPU 事件计时 **10.6 ms**。
- 同机 `cupy.einsum(optimize=True)` 仅 **0.55–0.84 ms**，opt_einsum+cupy **1.23 ms**。
- 结论：差距在 kernel 实现（约 13x），不在算法，属可优化项。

### 5.4 小规模下反而落后

T0（Nvec=20、spatial=1）CPU：E1 0.51x、E2 1.29x、E4 1.32x、E5 0.95x。
单次调用固定开销（计划解析、指针封装、缓冲区准备）压过计算量。

## 6. 可达性判断

| 场景 | 3x 判断 | 依据 |
|---|---|---|
| CPU E1/E2（算力主导） | **可达**，需把 E1 从 1.55 s 压到 ≤0.7 s | 当前已 12.4 GFLOP/s 有效吞吐，OpenBLAS 峰值 227.6 GFLOP/s |
| CPU E4 | 可达 | 当前 1.82x，仍有 pack/unpack 与派发开销 |
| CPU E5 | **不可达**（带宽受限，两边持平） | 耗时 0.1 ms 级，已是访存下限 |
| GPU E4 | **可达** | kernel 有 13x 冗余 |
| GPU E5 | 可达区间小（当前 1.49x） | 小张量、带宽/启动开销主导 |
| GPU E1/E2 | **大概率不可达** | 四方实现已打平在 373–444 ms；需 2.6x 超越 cuBLAS 主导路径 |

## 7. 未验证 / 限制

- 真实数据未使用：`/public/group/lqcd` 与
  `/home/zhangxin/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime` 在本机不存在。
- opt_einsum 冻结子模块未初始化，oracle 为站点包 3.4.0（`frozen=false`）。
- `L24`（spatial=24）profile 已完成显存预算设计（Nvec=50 时 prop 单算子 1.06 GB、
  峰值约 3.2 GB），尚未产出实测数据。
- GPU 侧未做 MPI 与多流吞吐测量；按用户要求，此类加速不用于与单卡 opt_einsum 对比。

## 8. 复现产物

| 产物 | 路径 |
|---|---|
| T0 CPU 基准 | `pyqcd/testing/einsum/artifacts/bench_20261108050730/` |
| T1 CPU 基准 | `pyqcd/testing/einsum/artifacts/bench_20261108051452/` |
| GPU 基准 | `pyqcd/testing/einsum/artifacts/bench_20261108052558/` |
| 环境与任务交接 | `logs/resume_20261108T053500_pyqcd-einsum.txt` |
| 阶段日志 | `logs/v202611080524.txt` |
