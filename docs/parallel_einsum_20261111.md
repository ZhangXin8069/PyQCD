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

## 4. 实测（单线程公平基准，两种口径）

条件：真实 sush 蒸馏方程 S1 = NN 2pt（`eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN`，
8 算子，形状照 `pyqcd/testing/einsum/bench_distillation.build_operands`），`complex64`，
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`；并行结果均与串行 `contract` 对照通过。

**口径定义**：`einsums(..., return_info=True)` 会带回各 worker 自报的纯计算时间
（`info["timings"]["compute_seconds"]`，取最慢进程）。因此自建进程池按两种方式计费：

* **计入初始化**（`×wall`）：调用方实际等待的墙钟时间，含 `mpirun` 启动与任务/结果落盘；
* **不计入初始化**（`×compute`）：只算 worker 实测的收缩时间，是任务拆分本身的上限。

集体模式（已在 `mpirun -np 4` 内，无数据搬运）：

| 场景 | 任务数 | 串行 | 并行 | `×wall` | `×compute` |
|---|---|---|---|---|---|
| 通用：大量小任务（128² 矩阵乘） | 400 | 42.5 ms | 25.7 ms | 1.65× | 3.81× |
| 通用：一对一（1200²） | 3 | 101.0 ms | 41.8 ms | 2.42× | 2.97× |
| 通用：偏斜（3 重 + 9 轻） | 12 | 103.8 ms | 37.3 ms | 2.78× | 3.21× |
| 核子 2pt（E1） | 8 | 184.4 ms | 46.1 ms | **4.00×** | **4.07×** |
| sush NN 2pt（S1） | 8 | 197.4 ms | 50.6 ms | **3.90×** | 3.96× |
| sush 3pt 主项（P1） | 8 | 134.7 ms | 39.7 ms | **3.39×** | 3.51× |
| sush NN 2pt，Nvec=100 | 8 | 6066.8 ms | 1536.7 ms | **3.95×** | 3.95× |
| γ 夹心（E5，约 1 ms/任务） | 8 | 0.5 ms | 0.7 ms | 0.73× | 1.59× |

自建进程池（单进程，自己拉起 `mpirun -np 4`）：

| 场景 | 任务数 | 串行 | 池 wall | 池 compute | `×wall` | `×compute` |
|---|---|---|---|---|---|---|
| 通用：大量小任务 | 400 | 38.7 ms | 283.3 ms | 14.1 ms | 0.14× | 2.74× |
| 通用：一对一 | 3 | 77.8 ms | 206.9 ms | 28.3 ms | 0.38× | 2.74× |
| 通用：偏斜 | 12 | 80.0 ms | 222.3 ms | 30.4 ms | 0.36× | 2.64× |
| 核子 2pt（E1） | 8 | 142.4 ms | 337.2 ms | 162.2 ms | 0.42× | 0.88× |
| sush NN 2pt（S1） | 8 | 140.0 ms | 218.7 ms | 57.2 ms | 0.64× | **2.45×** |
| sush 3pt 主项（P1） | 8 | 101.6 ms | 327.6 ms | 146.9 ms | 0.31× | 0.69× |
| sush NN 2pt，Nvec=100 | 8 | 4997.0 ms | 1879.7 ms | 1586.8 ms | **2.66×** | **3.15×** |
| γ 夹心（E5，约 1 ms/任务） | 8 | 0.6 ms | 156.7 ms | 6.4 ms | 0.00× | 0.10× |

## 5. 可信性分析（结论与代价）

1. **collective 是高效模式**：无数据搬运，两种口径几乎重合（4 rank 上 3.4–4.0×，
   线性上限 4×，缺口来自计划广播与 BLAS 线程降档）。
2. **自建进程池每次调用固定花约 0.15–0.3 s**（`wall − compute`）：MPICH `mpirun` 启动
   ~150 ms + 任务操作数落盘/回读约 1.5 GB/s（成本随操作数总量线性增长）。
3. **两种口径给出不同结论，必须分开看**：按端到端（`×wall`）计，只有秒级批次才赚
   （Nvec=100 → 2.66×；<0.2 s 的批次 0.14×）；按纯计算（`×compute`）计，同一批小任务能到
   2.6–2.7×——损失全部来自固定成本，装箱器本身是有效的。
4. **专用核方程在 `×compute` 上低于 1×**：每个新 worker 的首次调用要付一次性 kernel/计划
   预热（实测 E1 首调 140 ms vs 热调 17 ms），跑完一批就退出的池里这是真实成本。
5. 线程配置影响很大：`pyqcd.einsum._threads` 会按 MPI rank 数分摊线程；父进程若把
   `OMP_NUM_THREADS` 写进环境，子进程会当成用户显式设置照搬 → K 进程 × 全线程超订
   （实测 4 进程 × 4 线程在 4 物理核上把 collective 从 3.09× 打到 2.28×）。
   自建池已直接按 `物理核数 // 进程数` 给预算，不再依赖环境变量判断。

## 6. 复现方式

```bash
# 自建进程池（同时打印 init included 与 init excluded 两个倍率）
python -m pyqcd.testing.parallel_einsum bench --equation S1 --tasks 8 --nvec 40 --procs 4

# collective（必须在 mpirun 内）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  mpirun -np 4 python -m pyqcd.testing.parallel_einsum bench-collective \
  --equation S1 --tasks 8 --nvec 40 --rounds 3

# 正确性契约（14 项）
python -m pyqcd.testing._parallel_einsum_contract
```

## 7. 已知限制

* 结果默认只回传 rank 0（collective）；需要全体持有请传 `broadcast=True`。
* 自建进程池要求同机可执行 `mpirun`；找不到时降级为串行并给 `RuntimeWarning`。
* 任务失败即整体失败，错误里点名任务键（list 用下标、dict 用键），不返回部分结果。
* GPU 张量会先转 numpy 落盘（自建池路径），显存优势在该路径上不成立。
