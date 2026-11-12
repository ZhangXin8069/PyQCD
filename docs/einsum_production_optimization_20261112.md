# PyQCD qcd-einsum 生产特化优化记录（2026-11-12）

> 目标：优化 KUPL CPU 与 CUDA GPU 特化执行器，使真实生产场景效率最佳。
> 性能结论只接受计算节点上的生产缩并；`smoke`、`--scale` 与小维合成数据只用于
> 正确性或路径诊断，不能替代生产性能证据。

## 1. 验收与资源约束

- CPU 与 GPU 测试必须分别通过 `sbatch` 提交计算节点，并尽可能同时运行。
- 当前站点 Lua 拒绝 `--gpus=0`；CPU 作业实际申请 `--gpus=1`，同时设置
  `CUDA_VISIBLE_DEVICES=""`、`NVIDIA_VISIBLE_DEVICES=NoDevFiles`，证据中明确记录
  “预留 1 卡但 Python 不使用 GPU”。GPU 作业正常使用 `--gpus=1`。两者均从
  `~/env.sh` 激活环境。
- 单作业 Slurm 时限 10 分钟。运行超过 5 分钟必须检查 `squeue`、`scontrol` 和日志；
  超过 10 分钟必须 `scancel`，并记录 job id、日志与终止原因。
- 数值正确性优先：特化路径必须与同 dtype 的 `contract(..., specialize=False)` 对照。
- 性能必须固定 dtype、线程、设备、输入和计划；记录预热、轮次、中位数及实际执行标签。

## 2. 生产基线口径

当前生产定义为：

```text
lattice_size = [32, 32, 32, 64]
Nx=32, Lt=64, Nev=100, Ns=4, Mom=8
scene=full
```

代表性生产序列为 `2584`、`866`、`855`、`129`、`1863`、`6585`，覆盖元数 6、10、14、16。

## 3. 已发现的结构性问题

### 3.1 `seq=6585` 的计划代价过高

对 full scene 的纯形状规划得到：

| 计划 | FLOPs | 最大中间体元素数 | 实际 pack 量（complex128） |
|---|---:|---:|---:|
| `auto/greedy` | 168,822,272,000 | 327,680,000 | 约 9.98 GiB |
| `branch-2` | 119,302,963,200 | 163,840,000 | 约 4.57 GiB |

`branch-2` 相对当前 `auto/greedy`：

- FLOPs 降低约 29.3%；
- 最大中间体减半；
- 实际 pack 量降低约 54.2%；
- 首次规划约 0.08 s，可按 `(equation, shapes, dtype, optimizer, backend)` 缓存。

这说明当前只看 FLOPs 的 `_cheapest_dedicated_plan` 不足以保证端到端效率，必须同时考虑
内存流量、最大工作集与实际 pack 字节。

### 3.2 GPU 中间槽和 pack buffer 一次性分配过大

历史 A100 生产 trace（`seq=6585`）显示：

- 15 步 device arena 约 16.2 GB；
- pack buffer 另计约 A=5.24 GB、B=0.328 GB、C=2.62 GB；
- 单次带 trace 调用约 0.284 s；
- 普通生产基准中特化路径约 0.50 s，而同场景通用路径约 0.041 s，倍率约 `12.2x`。

当前执行器预先把所有逻辑中间槽分配完，即使它们不会同时存活。按单次消费的生命周期模拟：

| 计划 | 全部中间槽 | 峰值同时存活 | 简单 first-fit 首次分配量 |
|---|---:|---:|---:|
| `auto/greedy` | 约 9.90 GiB | 约 5.29 GiB | 约 5.75 GiB |
| `branch-2` | 约 4.67 GiB | 约 1.66 GiB | 约 2.18 GiB |

因此，逻辑 slot 与物理 buffer 解耦、按最后一次使用回收再复用，是 CPU/GPU 都应优先
实现的共同结构优化。

### 3.3 CUDA 小外积应避免 cuBLAS

历史 trace 的 `bmnk=1,4,1000000,4` 单步在 cuBLAS 上约 43.5 ms。当前工作树已加入
`small_outer_kernel`，但必须用计算节点生产 trace 验证其收益、数值误差和 pool 行为。

### 3.4 CPU BLAS 入口必须匹配实际生产库

历史 CPU trace 中 `seq=6585` 在约 63 s 后触发 cgroup OOM；多个大 GEMM 单步达到
2-22 s。当前工作树增加了 NumPy bundled OpenBLAS 选择和 ILP64 符号探测，但必须在
计算节点确认：

- 实际加载了哪一个 CBLAS library；
- 使用 LP64 还是 ILP64 符号；
- BLAS/OpenMP 线程数是否与生产基准一致；
- 是否仍发生超时或 cgroup OOM。

## 4. 最终改动集

最终工作树改动包括：

- `cpp/kupl/einsum/src/dedicated_apply.cpp`
- `cpp/cuda/einsum/src/dedicated_apply.cu`
- `pyqcd/einsum/_native.py`
- `pyqcd/einsum/_specialize.py`
- `pyqcd/einsum/_threads.py`
- `pyqcd/einsum/kupl/einsum.py`
- `pyqcd/einsum/_api.py`、`_kernel_kupl.py`、`_kernel_cuda.py`
- `pyqcd/testing/einsum/verify_python_hotpath.py`、`verify_plan_gate.py` 与统一套件接线
- `skills/pyqcd-einsum/**`
- 本文档。

每项默认行为均有计算节点 full-scene 或专项契约证据；未覆盖的 GPU 架构、dtype
和线程配比在文末继续标为未验证，不外推。

## 5. 验收矩阵

| 层 | 检查 |
|---|---|
| 构建 | KUPL 与 CUDA 均可重建，ABI 镜像一致，导出符号完整 |
| 路径 | `path` 必须是实际执行的 `kupl_dedicated` / `cuda_dedicated` |
| 正确性 | 四 dtype、生产序列、compiled context、重复调用、`out=` 与失败回落 |
| 性能 | CPU 对 CPU、GPU 对 GPU；同 dtype、线程、设备、输入、计划；至少 10 次中位数 |
| 内存 | 记录峰值 host RSS、device allocated/reserved、workspace 与 pool 上限 |
| 稳定性 | CPU/GPU 作业均不超过 10 分钟；超时/OOM 必须保留完整证据 |

## 6. 收口顺序

- 历史瓶颈已在计算节点复测：CPU BLAS/ILP64、CUDA large GEMM 与 workspace 均定位。
- CPU batch-parallel、CUDA lifetime arena、small-outer canonical、Python 缓存与
  unsupported 负缓存均已集成并验证。
- 计划选择改为 CPU complex128 的保守 branch-2 条件门；CUDA 保留 double fail-open
  性能门。
- 验证结论同步回 `skills/pyqcd-einsum` 和本文件，未验证范围在文末保留。

## 7. 首轮计算节点实测（2026-11-12 18:41-18:44）

### 7.1 CPU / KUPL

作业 `1647655`：A100 节点上的 CPU-only KUPL，20 CPU、8 OMP / 8 BLAS，
`complex128` 为主，完整生产参数。作业 `2:26` 完成，退出码 `0`，峰值 RSS
`15,395,127,296 bytes`（约 14.34 GiB）。

| seq | dtype | generic | dedicated | 特化/通用 |
|---:|---|---:|---:|---:|
| 2584 | complex128 | 3.429 ms | 2.868 ms | 0.836x |
| 866 | complex128 | 92.759 ms | 89.037 ms | 0.960x |
| 855 | float64 | 5.772 ms | 4.706 ms | 0.815x |
| 129 | float64 | 709.528 ms | 1,586.303 ms | 2.236x |
| 1863 | float64 | 28.062 ms | 25.179 ms | 0.897x |
| 6585 | float64 | 2,915.879 ms | 3,806.100 ms | 1.305x |

结论：

- CPU 当前工作树不再出现历史 trace 的 40 s 级退化，NumPy bundled OpenBLAS /
  ILP64 选择已生效；
- 14 算子和 16 算子长链仍慢，说明瓶颈已经从 BLAS 符号选择转移到计划、pack 和
  workspace 结构；
- 四 dtype 独立正确性均通过，最大相对误差分别为 complex128 `1.29e-15`、
  complex64 `4.95e-7`、float64 `0`、float32 `4.25e-7`。

### 7.2 CUDA

作业 `1647656`：单 A100-40GB，16 CPU。生产 benchmark 阶段退出码 `1`，原因是审计
脚本把 topology 路径与 dedicated 路径的失败计数合并；生产 case 本身均数值正确。

| seq | dtype | generic | dedicated | 特化/通用 |
|---:|---|---:|---:|---:|
| 2584 | complex128 | 3.268 ms | 0.141 ms | 0.043x |
| 866 | complex128 | 2.799 ms | 1.252 ms | 0.447x |
| 855 | float64 | 2.339 ms | 0.524 ms | 0.224x |
| 129 | float64 | 13.733 ms | 20.004 ms | 1.457x |
| 1863 | float64 | 3.890 ms | 0.722 ms | 0.186x |
| 6585 | float64 | 26.952 ms | 43.082 ms | 1.598x |

`seq=6585` 的 dedicated 路径已从历史约 0.50 s 降到约 43 ms，证明
`small_outer_kernel` 消除了原 `m=4,n=1,000,000,k=4` 的 43 ms cuBLAS 退化；但同链
下一个 `m=4,n=3,200,k=4` 且 A 经过 pack 的步骤仍耗时约 35 ms，因此仍慢于通用路径。

初版审计曾把 topology 的 `status=2` 覆盖到 dedicated 的失败字段，造成“其余三项
dedicated 未执行”的误报。随后用独立进程隔离复现，四 dtype 的
`qcd_cuda_dedicated_apply` 均返回数值正确结果：

| dtype | 最大相对误差 | 结论 |
|---|---:|---|
| complex128 | 1.27e-15 | PASS |
| complex64 | 5.31e-7 | PASS |
| float64 | 6.77e-16 | PASS |
| float32 | 2.50e-7 | PASS |

complex128、complex64、float64 的 topology 入口仍返回 unavailable/status 2；这是
topology 路由能力问题，不得写入 dedicated 统计。

## 8. Python 特化热路径优化（2026-11-12）

计算节点作业 `1647706` 对完整 8638 条生产拓扑执行了缓存扫描：

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| 单条重复 `match()` 热调用 | 15.15 us | 仍为缓存命中路径 |
| 8638 条完整轮转 `match()` | 约 60.8 us/call | 约 **0.53 us/call** |
| 8638 条轮转首轮 | 全 miss | 约 0.54-0.57 s |
| 计划缓存热扫描 | 4096 项上限反复淘汰 | 约 **22.7 us/call** |
| 8638 条计划热扫描总时间 | 无法稳定覆盖全语料 | 约 **0.195 s** |
| 计划缓存占用 | 22.8 MiB / 4096 项 | 46.1 MiB / 8638 项 |

对应改动：

- `_specialize` 增加按规范化完整方程缓存的 `_match_of_normalized`，缓存容量覆盖全套
  8638 个唯一签名；
- dedicated 计划缓存容量提升到 16384 项、64 MiB，完整语料热扫描不再淘汰；
- `_run_specialized_kernel` 只尝试与输入 backend 对应的原生入口，避免 numpy 输入先
  尝试 CUDA、cupy 输入先尝试 KUPL。

后端定向分派 30 轮复测中，`seq=2584` 的原生调用链中位约降低 3.3%；`seq=866`
基本持平。该优化主要降低完整语料和跨后端无效探测开销，不应包装成算子内核加速。

## 9. CUDA cuBLAS 试验结论

已测试：

- 显式 cuBLAS workspace 32 MiB；
- `cublasGemmEx` / `GemmStridedBatchedEx`；
- 两者组合。

单步微基准中 `GemmEx` 曾把 `bmnk=1,409600,64,800` 从约 13.2 ms 降到约
9.8 ms，但在 `seq=2584/866/6585` 的完整生产 A/B 中未形成稳定端到端收益。
因此这些路径只保留为显式实验开关或已移除，不作为默认生产行为。

## 10. 冻结源码快照 A/B（2026-11-12 19:00-19:07）

为避免代理并行修改污染测量，已将当时工作树完整复制到
`data/runs/einsum-opt-20261112/prod-baseline-audit-20261112t1838/snapshot-current-20261112t1858/`，
并记录每个 Python/C++ 文件 SHA-256。CPU 与 GPU 作业均使用该快照。

### CPU / KUPL

作业 `1647747`，`7:24` 完成，峰值 RSS 约 `25.2 GB`：

| seq | dtype | generic | dedicated | 特化/通用 |
|---:|---|---:|---:|---:|
| 2584 | complex128 | 3.620 ms | 2.981 ms | 0.823x |
| 866 | complex128 | 94.853 ms | 89.213 ms | 0.941x |
| 855 | complex128 | 15.352 ms | 13.897 ms | 0.905x |
| 129 | complex128 | 3.314 s | 3.925 s | 1.184x |
| 1863 | complex128 | 106.862 ms | 100.840 ms | 0.944x |
| 6585 | complex128 | 12.637 s | 15.302 s | 1.211x |

### CUDA

作业 `1647748`，`3:18` 完成：

| seq | dtype | generic | dedicated | 特化/通用 |
|---:|---|---:|---:|---:|
| 2584 | complex128 | 1.496 ms | 0.301 ms | 0.202x |
| 866 | complex128 | 3.567 ms | 1.241 ms | 0.348x |
| 855 | complex128 | 2.415 ms | 0.610 ms | 0.253x |
| 129 | complex128 | 26.717 ms | 29.215 ms | 1.094x |
| 1863 | complex128 | 3.972 ms | 1.015 ms | 0.255x |
| 6585 | complex128 | 69.499 ms | 102.515 ms | 1.475x |

共同结论：

- 6/10/16 算子的轻中量路径在 GPU 上有 2.9-4.9x 加速；
- 14/16 算子重路径仍劣于通用路径，不能在拓扑命中后无条件执行；
- CPU 长链峰值内存约 28.3 GB，workspace/lifetime 优化仍是必要项；
- 通用路径对 `seq=6585` 明显更好，因此后续需要生产收益门控或更优计划/内存布局。

## 11. CUDA 生产收益门

在真实 A100 full-scene 调查中，complex128 计划只要包含单个超过
`40,000,000,000 MAC` 的 GEMM，`pack → cuBLAS → unpack` 执行器就明显退化；
低于边界的计划仍通常优于通用路径。因此在 CUDA dedicated 入口增加 fail-open 门：

- complex128 且任一步 MAC 超过阈值时返回 `UNSUPPORTED`，由现有通用路径执行；
- 其它 dtype 不受该门影响；
- 计算使用饱和乘法，溢出同样 fail-open；
- 数值结果不变，且该判定在 workspace 分配前完成。

最终 full-scene 复测又发现 `seq=129` 的单步仅 2.048G MAC，却因 3.41 GB 静态
arena 与 6.40 GB pack 流量使 dedicated 比 generic 慢约 27%。因此在 Python 计划层
增加第二道门：complex128 且 `plan_arena_bytes >= 3 GiB` 时直接返回
`UNSUPPORTED` 并写入负缓存。六序列中该门覆盖 `seq=129` 与 `seq=6585`，不会影响
`seq=2584/866/855/1863` 已验证的 dedicated 加速。该阈值只对当前 A100 full-scene
有实测依据，跨 GPU 架构需重新标定。

计算节点作业 `1647800`：

| seq | 门控前 dedicated | 门控后 top-level | generic reference | 结论 |
|---:|---:|---:|---:|---|
| 6585 | 约 71.42 ms | 50.57 ms | 45.70 ms | 改善 29.2%，仍有余量 |
| 3909 | 约 68.66 ms | 35.79 ms | 33.87 ms | 改善 47.9% |
| 3981 | - | 53.95 ms | 83.18 ms | 未触发门，继续专用并获胜 |

门控后 `seq=6585` 的 top-level 仍比直接 generic 多约 10.6%，原因是 Python 每次
仍先尝试 dedicated/topology 再回退。后续需要确定性 unsupported 负缓存。

## 12. CPU batch GEMM 并行

真实 trace 表明，`seq=6585` 的末三步均为大批量 CBLAS 调用。实现采用保守运行时门：

- 仅当 `batch > 1`、每批 MAC ≥ 32M、总 MAC ≥ 64M；
- 仅在 OpenBLAS 线程局部控制符号可用、当前不在并行区时启用；
- 宿主 OpenMP 按 batch 分片，每个 worker 将 BLAS 线程局部降到 1，避免
  OMP×BLAS oversubscription；
- 条件不满足时保持原有串行 CBLAS 路径。

作业 `1647913` 使用 baseline/optimized 两个已冻结 `.so` 做同进程 full-scene A/B，
`3` 轮中位数：

| seq | generic | optimized dedicated | 特化/通用 |
|---:|---:|---:|---:|
| 6585 | 22.306 s | **17.194 s** | **0.771x** |
| 2584 | 1.533 ms | 1.290 ms | 0.841x |
| 866 | 36.630 ms | 34.908 ms | 0.953x |
| 855 | 24.230 ms | 23.408 ms | 0.966x |
| 129 | 5.174 s | 4.145 s | 0.801x |
| 1863 | 192.661 ms | 185.651 ms | 0.964x |

六条生产路径全部不慢于 generic。`seq=6585` 数值最大相对误差 `1.84e-15`；
trace 中最重的 `batch=64,m=6400,n=400,k=256` 单步由约 11.27 s 降到约 6.05 s。

随后用同一作业、同线程配置对两个冻结 `.so` 做 5 轮重复（job `1647960`）：

| 实现 | 5 轮中位 |
|---|---:|
| baseline dedicated | 26.723 s |
| optimized dedicated | **18.606 s** |

同作业比值约 **0.696x**，峰值 RSS 18,834,816 KiB。另有保守回退验证
（job `1648034`，`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=4`）：trace 未出现
`batch-parallel`，generic 23.957 s、dedicated 24.025 s，比值 1.003x，
相对误差 `1.84e-15`。这说明加速来自大批量并行门，而不是改变了串行 CBLAS 语义。

## 13. CUDA 生命周期与 pack scratch 复用

当前 CUDA arena 先按 birth/last-use 对逻辑 slot 做区间着色，再为所有 step 使用
`max_i(pack_A_i + pack_B_i + pack_C_i)` 的单一 scratch 区间。物理复用只发生在旧
slot 的 last-use 严格早于新 slot 的 birth 时，并增加运行时输入/目的区间别名检查。

| seq | 原 arena | 新 arena | 降幅 |
|---:|---:|---:|---:|
| 2584 | 4.21 MB | 3.98 MB | 5.4% |
| 866 | 85.38 MB | 60.06 MB | 29.7% |
| 129 | 3.408 GB | 2.636 GB | 22.7% |
| 1863 | 112.42 MB | 79.29 MB | 29.5% |
| 6585 | 16.200 GB | **11.417 GB** | **29.5%** |

作业 `1647901` 中，四 dtype、非连续 fail-closed、compiled context、生产序列和
数值对照全部通过；full suite 仍只有环境缺少 `gvar` 的两个非 einsum 失败。

## 14. 确定性 unsupported 负缓存与 branch-2

对 dedicated/topology 的确定性 `status=4` 增加 Python 级、线程安全的负缓存；只缓存
`UNSUPPORTED`，OOM、内部错误和输入布局错误不缓存。缓存键包含方程、完整输入形状、
dtype、optimizer 和后端。

GPU full-scene `seq=6585`：

| 模式 | 中位 time |
|---|---:|
| 每次重新尝试 rejected paths | 91.846 ms |
| 命中负缓存 | **48.218 ms** |

改善约 47.5%，结果逐元素一致。

早先在未启用 CPU batch-parallel 的版本上，branch-2 没有稳定收益；该结论已被
2026-11-12 的六序列 full-scene 复测取代，见下一节。最新策略只对 CPU 的超大
arena 计划做条件提升；显式 `PYQCD_EINSUM_DEDICATED_PLAN=branch-2` 仍可用于实验。

## 15. branch-2 六序列 full-scene 审计与条件默认

为避免把单个有利样本写成全局结论，使用冻结源码与 `.so`，CPU/GPU 同时提交：

- CPU：job `1648042`，20 CPU，OMP=8/BLAS=4，`CUDA_VISIBLE_DEVICES=""`；
- GPU：job `1648043`，A100-PCIE-40GB，同一 full scene、complex128/float64 输入；
- 每序列 warmup=1、timed rounds=3，奇偶轮交换 auto/branch-2 顺序。

### CPU

| seq | auto | branch-2 | branch-2/auto | 判断 |
|---:|---:|---:|---:|---|
| 2584 | 1.367 ms | 1.265 ms | 0.925x | 快 7.5%，未达 10% 门槛 |
| 866 | 35.958 ms | 35.966 ms | 1.000x | 持平 |
| 855 | 23.991 ms | 24.079 ms | 1.004x | 持平 |
| 129 | 4.133 s | 5.271 s | **1.275x** | 明显变慢 |
| 1863 | 187.388 ms | 190.465 ms | 1.016x | 变慢 1.6% |
| 6585 | 17.537 s | **10.346 s** | **0.590x** | 快 41.0% |

`seq=129` 说明“arena 更小”不等于“执行更快”：branch-2 虽把 arena 从 3.41 GB
降到 2.73 GB，却把 FLOPs 从 16.94G 增到 30.05G，CPU 端到端慢 27.5%。因此不能
仅按 FLOPs 或仅按内存选计划。

### GPU

| seq | branch-2/auto | 路径结论 |
|---:|---:|---|
| 2584 | 0.948x | 约快 5.2%，但未达稳定门槛 |
| 866 | 1.003x | 持平 |
| 855 | 1.003x | 持平 |
| 129 | 0.970x | 噪声内，FLOPs 反而更高 |
| 1863 | 0.952x | 约快 4.8%，不足以改默认 |
| 6585 | 两条均 fallback | CUDA 40G MAC 收益门拒绝 dedicated |

### 最终自动策略

`_cheapest_dedicated_plan` 在 CPU `complex128` 上仅当以下条件同时成立时提升
branch-2：

1. 当前 auto/greedy 最优计划的静态 arena ≥ **8 GiB**；
2. branch-2 FLOPs ≤ 当前计划的 **80%**；
3. branch-2 arena ≤ 当前计划的 **65%**；
4. branch-2 pack bytes ≤ 当前计划的 **65%**。

当前六条 full-scene 中只有 `seq=6585` 命中；`seq=129`、`seq=1863` 不会进入
branch-2。CUDA 与其它 dtype 保持 auto/greedy，因为 40G MAC 门会让该重路径回退
generic，且尚无同口径稳定证据。条件是成本模型筛选，不是按 `seq` 或方程硬编码。

## 16. 最终冻结快照验证

最终 CPU 生产作业 `1648188` 使用冻结 KUPL `.so`
SHA-256 `78af9202...0a82`，full-scene、complex128、OMP=8/BLAS=4、20 CPU。
作业 `4:46` 完成，退出码 0，source-integrity 全部 `OK`。

| seq | generic | dedicated | dedicated/generic | 相对误差 |
|---:|---:|---:|---:|---:|
| 2584 | 1.592 ms | 1.363 ms | 0.856x | 0 |
| 866 | 37.921 ms | 35.480 ms | 0.936x | 1.47e-16 |
| 855 | 24.475 ms | 24.039 ms | 0.982x | 1.59e-15 |
| 129 | 5.194 s | 4.110 s | 0.791x | 2.71e-15 |
| 1863 | 190.745 ms | 185.880 ms | 0.974x | 1.71e-15 |
| 6585 | 22.445 s | **10.310 s** | **0.459x** | 3.05e-15 |

CPU 六条全部不快于 generic；`seq=6585` 快 54.1%，是本轮最大生产收益。
`verify_native` 16/16、热路径 4/4、plan-gate 10/10（CUDA 用例按环境跳过）、
`verify_compat` 980/980 全部通过。

最终 GPU 生产作业 `1648207` 使用冻结 CUDA `.so`，A100-PCIE-40GB、full-scene：

| seq | path | dedicated/generic | 结论 |
|---:|---|---:|---|
| 2584 | `cuda_dedicated` | 0.265x | 快约 3.8x |
| 866 | `cuda_dedicated` | 0.248x | 快约 4.0x |
| 855 | `cuda_dedicated` | 0.177x | 快约 5.7x |
| 129 | `plan-cache` | 1.017x | 3 GiB arena 门回退，与 generic 持平 |
| 1863 | `cuda_dedicated` | 0.157x | 快约 6.4x |
| 6585 | `plan-cache` | fallback | 40G MAC/arena 门回退，结果误差 0 |

GPU 所有生产结果 `correct=true`；`verify_native` 的 CUDA/KUPL 各 16/16、
`check_native --backend cuda`、热路径 4/4、plan-gate 11/11、`verify_compat`
980/980 均通过。冻结 v9 统一套件为 **117 passed / 2 skipped / 2 failed**；
两项失败均为环境缺少可选依赖 `gvar`（`test_analysis_safety_contracts`、
`test_bare_sumratio_contracts`），与 einsum 改动无关。v9 全套日志中拓扑语料
120/120 命中、native-install、plan-gate 与其余 einsum 契约全部通过。

关键可复现标识、完整 manifest SHA-256 与冻结库哈希见同目录运行记录：
`data/runs/einsum-opt-20261112/final-20261112t195557/FINAL.md`。

## 17. 扩展路径与全计划门控

主验收 6 条之外，再按 12 个 case 和元数 6/7/9/10/12/13/14/16 分层测试两批
共 43 条 full-scene 路径。详细逐路径数据见
`data/runs/einsum-opt-20261112/expanded-paths-20261112/EXPERIENCE.md`。

### 合并性能

| 设备 | 路径数 | 中位比 | 范围 | ≥10% 加速 | 不快于 generic |
|---|---:|---:|---:|---:|---:|
| CPU | 43 | 0.861x | 0.347–1.007x | 26 | 42/43 |
| GPU | 43 | 0.276x | 0.122–1.362x | 40 | 42/43 |

极少数反例：

- CPU `seq=856`：`1.0067x`，0.7% 反向，属于噪声级。
- GPU `seq=6911`：`1.3615x`，极小计划的固定 pack/launch 开销。

第一轮发现 `seq=7918` 的 pack 流量约 1.12 GiB、FLOPs 仅 0.25G，CPU 慢 1.93x；
新增通用 pack-heavy 门后，CPU/GPU 均回退，第二轮为 `0.996x/0.958x`。曾尝试的
CUDA small-plan 门会误伤正常加速的 `seq=2583/3431`，因此撤回，只保留该单例为
可接受反例。

### 全 8638 条计划审计

job `1648360` 使用完整生产形状审计全部 8638 条计划，不执行大张量，0 条计划不可
编码：

```text
CPU allowed=3833  blocked=4805
GPU allowed=2864  blocked=5774
```

CPU 拦截原因：arena ≥8 GiB `3651`、单 GEMM ≥100G MAC `953`、pack-heavy `201`。
GPU 拦截原因：arena ≥3 GiB `5191`、单 GEMM ≥40G MAC `382`、pack-heavy `201`。
计数按门控判定顺序归入首个原因。最大计划静态 arena 约 `5.04 PB`、最大单步约
`65.536P MAC`，证明必须在原生分配前做统一资源门。

当前源码最终复验：

- CPU job `1648387`：KUPL native 16/16、plan-gate 11/11（CUDA 用例跳过）、
  热路径 4/4，当前源码 16 个文件 SHA-256 全部 OK。
- GPU job `1648386`：plan-gate 12/12，统一套件
  `117 passed / 2 skipped / 2 failed`；两项失败仍仅为环境缺 `gvar`。
