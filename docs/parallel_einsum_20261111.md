# 批量独立收缩并行（`pyqcd.parallel.einsum.einsums`）实测报告

用途：把**相互独立**的一批 `pyqcd.einsum.contract` 任务按工作量分箱后分给多个
MPI 进程执行（任务级并行）。本文件记录 2026-11-11 在本机（4 物理核 / 8 线程、
16 GB、MPICH）的实测数字与可信性分析，便于后续复核。

## 1. 功能边界

* 只做「分发与汇总」：每个子任务仍是一次普通 `contract(...)`，参数与返回值语义不变，
  dtype 完全由操作数决定（本功能不做任何类型转换）。
* 适用：单队列里排着**大批量、小单量**的收缩——单次收缩并行度有限，只能任务级并行。
* 不适用：单个超大成片收缩（应改用库内 MPI/多 stream 的域分解路径）。

## 2. 两种运行模式

| 模式 | 触发条件 | 数据交换 | 结果返回 |
|---|---|---|---|
| collective | 已在 `mpirun -np N` 内，各 rank 调 `einsums` | 无（任务已在各 rank 内存里） | 默认 rank 0；`broadcast=True` 全体 |
| self-launched | 单进程直接调 `einsums` | 操作数落盘到临时目录，worker 读回 | 父进程 |
| serial | `max_procs=1` 或子任务数为 1 | 无 | 调用方 |

## 3. 工作量分配

* 工时估计：`contract_path(..., optimize="greedy").opt_cost`（优化 FLOP 数），
  并以「输入+输出元素数」兜底，覆盖无求和的逐元素方程。
* 装箱：LPT（longest-processing-time-first），按工时降序放入当前最空的箱，
  堆实现 O(m log m)。最慢进程 ≤ `(4/3 - 1/(3K)) × OPT`。
* 进程数：`min(子任务数, 上限)`；上限默认 `min(CPU 线程数, 可用内存/最重任务×2)`，
  可用 `max_procs=` 收紧。任务数少于上限时一对一，多于上限时均摊。

## 4. 实测（单线程公平基准）

条件：真实 sush 蒸馏方程 S1 = NN 2pt（`eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN`，
8 算子，形状照 `pyqcd/testing/einsum/bench_distillation.build_operands`），
`complex64`，`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`，Nvec=40，8 任务；
并行结果与串行 `contract` 逐位一致（`bit_exact=True`）。

| 模式 | 串行 | 并行 | 加速比 |
|---|---|---|---|
| collective，2 rank | 168.5 ms | 88.3 ms | 1.86× |
| collective，4 rank | 182.6 ms | 58.7 ms | **3.09×** |
| collective，4 rank，默认线程（未钉） | 138.3 ms | 53.2 ms | 2.28× |

自建进程池（同一方程，单线程基准）：

| 工作负载 | 串行 | 并行（4 进程） | 加速比 |
|---|---|---|---|
| S1，Nvec=40（约 0.22 s） | 216.9 ms | 390.7 ms | 0.56× |
| S1，Nvec=60（约 0.74 s） | 736.4 ms | 622.1 ms | 1.18× |
| S1，Nvec=100（约 5.3 s） | 5325.5 ms | 2160.4 ms | 2.47× |

## 5. 可信性分析（结论与代价）

1. **collective 是高效模式**：无数据搬运，4 rank 上 3.09×（理论线性上限 4×，
   缺口来自规划广播的 ~5 ms 与 BLAS 线程数下降）。
2. **自建进程池有固定成本**，实测约 0.2–0.3 s：
   MPICH `mpirun` 启动 ~150 ms；任务操作数落盘/回读约 1.5 GB/s（决定成本随操作数
   总量线性增长）。
3. 因此**串行耗时约 1 秒以上才开始划算**（0.74 s → 1.18×，5.3 s → 2.47×），
   小批次（<0.2 s）反而更慢。可信性分析必须包含这一条，不能用「任务数多」代替。
4. 线程配置影响很大：`pyqcd.einsum._threads` 会按 MPI rank 数分摊线程；父进程若把
   `OMP_NUM_THREADS` 写进环境，子进程会当成用户显式设置照搬 → K 进程 × 全线程超订
   （实测 4 进程 × 4 线程在 4 物理核上从 3.09× 掉到 2.28×，串行基准也被带快）。
   自建池已按 `物理核数 // 进程数` 重算这两个变量。

## 6. 复现方式

```bash
# 自建进程池 + 交错计时（串行/并行交替，取比值中位数）
python -m pyqcd.testing.parallel_einsum bench --equation S1 --tasks 8 --nvec 40 \
  --procs 4 --rounds 3 --precision complex64

# collective（必须在 mpirun 内）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  mpirun -np 4 python -m pyqcd.testing.parallel_einsum bench-collective \
  --equation S1 --tasks 8 --nvec 40 --rounds 3 --precision complex64

# 正确性契约（12 项）
python -m pyqcd.testing._parallel_einsum_contract
```

## 7. 已知限制

* 结果默认只回传 rank 0（collective）；需要全体持有请传 `broadcast=True`。
* 自建进程池要求同机可执行 `mpirun`；找不到时降级为串行并给 `RuntimeWarning`。
* 任务失败即整体失败，错误里点名任务键（list 用下标、dict 用键），不返回部分结果。
* GPU 张量会先转 numpy 落盘（自建池路径），显存优势在该路径上不成立。
