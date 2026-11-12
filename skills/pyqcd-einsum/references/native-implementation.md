# 原生收缩实现与实测记录

更新日期：2026-11-06。本文只记录当前工作树证据；命令应在新会话定向复跑。

## 1. 文件与职责

| 层 | 路径 | 职责 |
|---|---|---|
| Python API/规划 | `pyqcd/einsum/` | opt_einsum 兼容入口、路径、表达式和回退 |
| Python 原生绑定 | `pyqcd/einsum/_native.py`、`_native_plan.py`、`kupl/einsum.py`、`cuda/einsum.pyx` | 计划编码、加载和指针传递 |
| KUPL C++ | `cpp/kupl/einsum/` | 共享计划核、CPU pack/reduce/GEMM 与 C ABI |
| CUDA C++ | `cpp/cuda/einsum/` | CUDA pack/reduce、cuBLAS batched GEMM 与 C ABI |
| 测试 | `pyqcd/testing/einsum/`、`cpp/kupl/einsum/tests/` | API、数值、ABI、蒸馏与性能门 |

## 2. 执行计划 v2

Header 为 `int32[9]`：

```text
magic=0x51434532, version=2, total_bytes, num_inputs, num_steps,
num_symbols, output_ndim, dtype_id, reserved
```

后续依次为：

1. 8-byte 对齐的 `int64[num_symbols]` 尺寸表；
2. 每个初始 operand 的 rank 与 `int64[rank]` shape；
3. 最终输出 labels；
4. 每步 arity、当前活动 operand positions、各输入 labels、输出 labels。

dtype id：`complex128=0`、`complex64=1`、`float64=2`、`float32=3`。混合精度和
单 operand 内重复 label 当前 fail closed。

## 3. K-lowest 布局

对任意二元步骤，将 label 分成：

- `B`：同时出现在两个输入和输出中的 batch；
- `M`：只属于左输入并保留到输出；
- `N`：只属于右输入并保留到输出；
- `K`：两个输入共有但本步消失的缩并轴。

原生规范布局为：

```text
left  = [B, M, K], K stride = 1
right = [B, N, K], K stride = 1
out   = [B, M, N]
result = left x right^T over K
```

当前 `cpp/kupl/einsum/tests/test_runtime.cpp` 已验证 label/permutation：

```text
left [a,b,c] -> [b,a,c]
right[b,d,c] -> [b,d,c]
out  [a,d,b] -> packed [b,a,d], then inverse output permutation
```

left/right 的 K 均为最低维；KUPL 调用 CBLAS 时对 right 使用 transpose-B。CUDA 镜像
必须保持同一语义。

## 4. 当前加速结构

- `ContractExpression` 与 `contract` 都缓存固定计划对应的原生 program；只缓存
  shape/dtype/path 和控制数组，不缓存输入内容或 output。
- KUPL compiled-context ABI：`applyCompileQcd`、`applyCompiledQcd`、
  `applyDestroyQcd`，避免每次调用重复解析执行计划。
- CUDA compiled context 首次分配并复用 pack/GEMM/中间 workspace；
  `PYQCD_EINSUM_CUDA_STREAMS` 控制 stream pool，batch 分片通过 event 在默认流汇合。
- `ab,...bce,cd->...dae` 识别为 gamma sandwich fast path：一次 e-lane 连续核完成
  两次 4x4 小矩阵夹心，不物化中间张量；按 work 大小选择串行或 OpenMP lane。
- `M=N=1` 的最终闭合走 batch-dot fast path，直接在原布局求和，不做三块 buffer
  分配、pack 或 1x1 GEMM。
- pack 使用增量 odometer、固定 16/8/4-byte copy，并按元素数限制 OpenMP 线程；
  避免每元素除法和 8 线程小任务过度并行。
- E1 类多小 GEMM workload 在 4 线程开发窗口中优于 8/16 线程；线程数必须与
  opt_einsum 对照保持一致并写入结果。

## 5. 当前验证快照

已测试：

- 共享 runtime 测试：`g++ -std=c++17 -Wall -Wextra -Werror ...` 后
  `test_runtime: PASS`。
- KUPL：四 dtype、batch、size-1 广播、输入/输出换序、一元 reduce/permute、外积、
  零维和多步活动槽位；代理记录 93 个构造计划及 UBSan 通过。
- Python：`python -m pyqcd.einsum._native_plan` 覆盖四 dtype；KUPL 真实执行与
  NumPy 对照通过。
- 全量：`python -m pyqcd.testing` 记录 `111 passed, 1 skipped, 0 failed`。
- 构建：KUPL/CUDA 动态库与 Cython 扩展均已产出；KUPL 导出面已包含冻结四入口与
  compiled-context 三入口，CUDA 三入口同步中。
- KUPL compiled context 与 sandwich/batch-dot fast path 已与 NumPy/opt_einsum T0
  示例逐元素一致。

未验证或失败：

- CUDA 一般 batched GEMM 当前返回 `CUBLAS_STATUS_INVALID_VALUE`，状态码 3；不能以
  逐元素 CUDA 路径正确性替代。
- CUDA K-last、一般 GEMM、compiled context、多 stream 与 E5 sandwich 已完成。
- MPI 支持（`applyContractMpiQcd`、`kupl/mpi.py`、`cuda/mpi.py`、`mpi_runtime.cpp`
  与运行时预加载）已于 2026-11-12 整体删除，不再有 rank 分片入口。
- KUPL 多进程 fallback 已实现并通过 25 项功能契约；32×64×64 小例为 `0.42x`
  串行速度，说明进程/shared-memory 开销占主导，不能写成性能加速。
- 多 CUDA stream、多 GPU 尚未完成数值与性能门。
- 固定 `opt_einsum` 子模块当前未初始化；两次 GitHub TLS/连接失败。开发对照使用本机
  `opt_einsum 3.4.0`，不能替代固定 `fccdd5aa` oracle 的最终证据。
- 指定 `beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg` 未在
  `/public/group/lqcd` 发现；现有近似 fixture 为
  `beta6.20_mu-0.2770_ms-0.2400_L24x72`，不得冒充目标系综结果。

## 6. 开发期性能快照

2026-11-06 开发窗口，T0 合成 operands、`complex128`、4 线程、相同 greedy path、
`contract` 入口，opt_einsum 3.4.0 站点包（非冻结 oracle）：

| 方程 | 开发窗口 qcd/opt 范围 | 状态 |
|---|---:|---|
| E1 8 算子 2pt | `3.0x-3.9x` | provisional pass，波动最高 |
| E2 8 算子 3pt | `9.9x-13.6x` | provisional pass |
| E4 peram×gauge×gamma | `4.0x-5.1x` | provisional pass |
| E5 gamma sandwich | `3.6x-4.3x` | provisional pass |

CUDA GPU/GPU T0/E5 开发窗口（opt_einsum 固定快照、ABBA）：

| GPU | opt_einsum | pyqcd | 倍率 |
|---|---:|---:|---:|
| V100 sm_70 | 1.119 ms | 0.275 ms | `4.07x` |
| P100 sm_60 | 0.922 ms | 0.264 ms | `3.49x` |

CUDA 其余 T0 开发窗口：E1 `3.14x`、E2 `4.12x`、E4 `3.10x`。这些仍是
provisional，需进程隔离、冻结 oracle 和目标/获准代理数据复验。

这些数字不是最终验收：必须换成冻结 `opt_einsum fccdd5aa`、目标/获准代理数据、
ABBA 顺序和至少三轮独立复测。CUDA 对 GPU 的同口径 3x 尚未完成。

## 7. 性能门

旧公共路径仅封装 NumPy/CuPy，实测约 `0.8x-1.08x`，不是新原生性能。目标为
CPU/CPU 和 GPU/GPU 均达到 opt_einsum 的 3x，并在至少 3 轮独立复测中保持。

每次测量至少记录：

```text
backend, dtype, device UUID, MPI ranks/grid, OpenMP threads, CUDA streams,
oracle commit/version, path hash, warmup, iterations,
median, IQR, min/max, peak bytes, max relative error, allclose status
```

性能 runner 必须把 `<3x` 计入非零退出码；必须同时保存 JSON 和 Markdown 摘要。
多传输或多 rank 的吞吐加速需与单 rank/单 stream 算法加速分开报告。

## 8. 多操作数拓扑入口（2026-11-11，KUPL / CUDA 两套）

用途：一次 ctypes 调用执行**整条** version-2 计划（6–16 个操作数），替代
Python 侧逐步调度。计划字节格式**不变**（第 2 节的 v2 格式）。

### 8.1 导出符号与 ABI

| 项 | KUPL | CUDA |
|---|---|---|
| 符号 | `qcd_kupl_topology_apply` | `qcd_cuda_topology_apply` |
| 声明处 | `cpp/kupl/einsum/include/einsum_topology.h:92-95` | `cpp/cuda/einsum/include/einsum_topology.h:92-100` |
| 实现处 | `cpp/kupl/einsum/src/topology_apply.cpp:132-145` | `cpp/cuda/einsum/src/apply_contract.cu:2047-2065` |
| version script | `cpp/kupl/einsum/make.sh:437` | `cpp/cuda/einsum/make-nv.sh:231` |
| 参数 | `(const void* plan, long long plan_len, void* const* operands, int n_operands, const long long* operand_offsets, void* out, int dtype_id)` | 同上 + 末尾 `void* stream` |
| 指针域 | **主机**指针 | **设备**指针（`cupy.data.ptr`） |
| stream | 无此参数 | 只接受 `nullptr`，非空返回 `1` |
| 可选偏移 | `operand_offsets` 为字节偏移；`NULL` = 全 0 | 同左 |
| 复核 | `nm -D --defined-only pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so \| grep topology` → `T qcd_kupl_topology_apply` | `...cuda/libpyqcdeinsum_cuda.so` → `T qcd_cuda_topology_apply` |

两侧参数顺序、`QCD_DTYPE_*` 枚举与返回码语义**逐项一致**，唯一差异是 CUDA 的 stream 尾参
（与既有 `qcd_cuda_pattern_3pt` 约定相同）。

### 8.2 失败码语义（fail closed：任何非 0 都不写 `out`）

| 码 | 含义 | Python 侧行为 |
|---|---|---|
| `0` | 成功 | 返回结果 |
| `1` | 参数非法：空指针、`plan_len` 与头部 `total_bytes` 不符、`n_operands` 与 `num_inputs` 不符、`dtype_id` 越界或与计划头不一致、地址加偏移溢出、操作数为空、CUDA 的 `stream != nullptr` | 回落，`LAST_FAILURE['status']=1` |
| `2` | 内存不足（`bad_alloc` / `length_error` 兜底） | 回落 |
| `3` | 其它内部错误（异常兜底） | 回落 |
| `4` | `QCD_TOPOLOGY_UNSUPPORTED`：**计划本身**超出本入口能力（单步 arity > 2、同一操作数内重复下标、无步骤、尾随字节、空输入） | 回落；文案与 `1` 分开（"计划不认" ≠ "参数不对"） |

`1` 与 `4` 必须分开诊断：前者是调用参数问题，后者退回通用执行器是**正常**行为。

CUDA 的 `4` 只在 `topology_entry` 之前的头部检查之后由 `TopologyProgram::load`
的解析失败产生；`topology_entry` 本身还会额外要求 `stream == nullptr`
（`apply_contract.cu:1959-1963`）。

### 8.3 两条执行路线（务必区分，这是本轮最大的坑）

| 路线 | 触发条件 | 证据 | 状态（2026-11-11） |
|---|---|---|---|
| **A. 转发通用执行器** | 入口收到计划就 `parse_execution_plan` → `execute_parsed_contract`（KUPL，`topology_apply.cpp:107-127` 提交版）/ → `execute_prepared`（CUDA，`apply_contract.cu:1992`） | `rc==0`，但无任何特化 trace | `84e68bb` 已完成并验证；数值与通用路径逐位一致 |
| **B. per-拓扑步表 + 参数化 GEMM** | 在 A 之前先 `run_specialized()`：要求所有轴等长、能构建规范签名、命中步表、`record.arity/out_rank` 匹配 | `PYQCD_EINSUM_TOPOLOGY_TRACE=1` 打出 `[pyqcd.einsum] topology-specialized: <sig> dim=… dtype=… steps=…` | **未打通**：`build_canonical_signature` 误用 `step.positions` 当输入下标（见 pitfalls T4），实测恒 `miss: signature build failed`；且 `topology_kernels.inc` 不进库（T5） |

判据：`rc==0` 不能证明走了 B。必须看 trace 或 `strings lib | grep topology-specialized`
（生产库 0、带步表的重建副本 1）。

### 8.4 步表格式（KUPL `topology_kernels.inc`）

每条拓扑一段 `uint8` 流，由 `pyqcd/testing/_topology_kernel_gen.py` 生成
（生成器 docstring 是权威定义，`:15-25`）：

```text
[0] int8  a_src      >=0 操作数下标；<0 表示中间槽 -1-slot
[1] int8  b_src
[2] int8  dst        >=0 中间槽；-1 表示最终输出缓冲区
[3] uint8 (rank_a<<4) | rank_b        （各 <= 15）
[4] uint8 rank_c                       （<= 15）
[5] uint8 (batch<<4) | inner           （各 <= 15；M/N 由秩复原）
[6..] nibble 流：a_axes[rank_a] + b_axes[rank_b] + c_axes[rank_c]，每轴 4 bit
```

轴语义：`a_axes[j]` = GEMM 规范序 `[batch…, M…, K…]` 第 j 轴在 A 源张量里的轴位置；
`b_axes` 对应 `[batch…, K…, N…]`；`c_axes[p]` = 目的张量第 p 轴在 `[batch…, M…, N…]`
里的下标。**只存轴次序、不存 stride**——每根轴长度恒为 `dim`，步长由 `dim` 的幂在运行期算。

记录表（`Record`，`topology_gemm.h:47-55`）：`hash`（FNV-1a 64）、`signature`（规范式
字符串）、`code_offset`/`code_bytes`、`arity`、`out_rank`、`step_count`。
查找 `topo_find(hash)` 为有序表二分，命中后再**逐字符比较 `signature`**（防哈希碰撞）。

KUPL 每步执行 = `pack`（按 `a_axes`/`b_axes` gather 成规范序）→ `cblas_{s,d}gemm`
（RowMajor，两侧 `NoTrans`）→ `unpack`（按 `c_axes` gather 回目标序）。

### 8.5 workspace / 块池

| 层 | 位置 | 策略 |
|---|---|---|
| KUPL 路线 A | `cpp/kupl/einsum/src/apply_contract.cpp` 匿名命名空间 `BufferPool` | 进程级；`acquire` 最佳适配、`release` 只回收 ≥512 KiB 且总量 ≤1 GiB 的块；只涨不缩 |
| KUPL 路线 B | `topology_gemm.h:60-121` `BlockPool` | 因 `apply_contract.cpp` 的池在匿名命名空间不可跨 TU，**另起最小实现**，参数口径同上（`kMinPooledBytes = 1<<19`、`kMaxPooledBytes = 1<<30`）；每步输出用完立刻归还，峰值 ≈ 相邻两步 |
| CUDA | `TopologyProgram`（`apply_contract.cu:1929-1932`）+ `CudaWorkspace` | 单槽缓存：按计划字节 `memcmp` + dtype 决定是否 `load`；换拓扑即覆盖（`release()` 后重分配）。`workspace.prepare()` 在 `execute_prepared` 内（`:1400-1407`） |

CUDA 侧 **arena 复用**是其相对 CUDA 通用路径唯一的结构性优势；KUPL 侧 `BufferPool`
跨调用保留，因此"逐步 malloc"不是主要成本项（实测 per-call 开销见 8.6）。

### 8.6 实测（本机，RTX 4060 Laptop sm_89 + OpenBLAS dlopen + CuPy 14.2）

| 项 | 数值 | 命令 |
|---|---|---|
| 匹配命中（语料） | 2582/2582；改名同命中 2582/2582；随机合法非语料误命中 **0/4000** | `python -B -c "from pyqcd.einsum import _specialize as S; ..."` |
| 匹配开销（单条重复） | 3.45 µs/call | 见 pitfalls 8.1 |
| 匹配开销（100 条轮转） | 5.02 µs/call | 同上 |
| 匹配开销（2582 条轮转） | 16.0 µs/call（512 项 LRU 抖动） | 同上 |
| 懒加载索引首次构建 | 0.87–0.90 ms（进程内一次） | `S._TOPOLOGY_DIGESTS=None; S.match(eq)` |
| KUPL 原生入口（自身，16 算子，dim=2） | 0.077 ms/call | 直接调 `kupl.topology_apply(plan, ops)` |
| 通用路径同题 | 0.182 ms/call | `contract(eq, *ops, specialize=False)` |
| `contract` 端到端比（float32, dim=4/8） | 1.02–1.50x（元数 6/10/13/16） | 循环 3 次取均值 |
| CUDA 正确性（dim=4/8，8 个元数 × 2 例） | 32/32 `correct`，`rel ≤ 4.1e-7` | `contract` 与 `specialize=False` 对照 |
| CUDA 正确性（dim=12） | arity13 `rel=5.8e-7`、arity16 `rel=2.3e-7` | 同上 |
| dtype（KUPL 直调） | float32 `2.6e-7`、float64 `4.5e-16`、complex64 `2.2e-7`、complex128 `6.3e-16` | `try_topology_apply` 四 dtype |
| dtype（CUDA） | 四 dtype 全通；`topology_dtypes()` 探测前 2 项、探测后 4 项 | `_probe_double` |

**不要**把这些数字读成加速结论：路线 A 的 1.0–1.5x 来自"少一次 Python 逐步调度"，
不是专用数值内核的收益；路线 B 尚未跑通。

### 8.7 新增一条拓扑内核的正确顺序

1. 生成步表：`python -m pyqcd.testing._topology_kernel_gen`（写
   `cpp/kupl/einsum/src/topology_kernels.inc`，`.gitignore:8` 会忽略它——**生成物不入库**）。
2. 重建 KUPL：`bash cpp/kupl/einsum/make.sh`。`make.sh` 只 glob `src/*.cpp`
   （`make.sh:400-413`），`.inc` 靠 `topology_apply.cpp:17` 的 `#include` 进入编译单元。
3. 验符号：`nm -D --defined-only cpp/kupl/einsum/build/libpyqcdeinsum_kupl.so | grep topology`；
   再 `strings … | grep -c topology-specialized`（接了步表应 ≥1）。
4. 验派发：`PYQCD_EINSUM_TOPOLOGY_TRACE=1 python -B …`，必须看到
   `topology-specialized:` 行；只有 `rc==0` 不算。
5. 验数值：同 dtype 的 `contract(..., specialize=False)` 当参考，**不要**在高元数
   `dim≥8` 上直接对 numpy（参考侧先算不完）。
6. 验回落：把 NativeUnavailable/非零状态注入，断言结果仍正确且只警告一次。
7. 客户端：`cp cpp/kupl/einsum/build/libpyqcdeinsum_kupl.so pyqcd/einsum/kupl/`（或在
   源码树用 `PYQCD_EINSUM_LIB=<path>` 指过去；`library_path()` 认这个变量，
   `pyqcd/einsum/kupl/einsum.py:94-104`）。

## 9. [P2] dedicated 步表入口的「dim=8 卡死」定位（2026-11-11）

**结论先说：`qcd_kupl_dedicated_apply` 的 dim=8 不存在卡死。** 报障方程
`cwdx,eufv,msnt,okpl,igjh,qarb,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHew`
（16 算子，F07 族）在本机实测：dim=4 原生 **0.016 ms**、dim=8 原生 **0.30 ms**、
dim=12 1.7 ms、dim=16 20 ms；dim=16 的 20 ms 对应 537 MFLOP，是真实工作量而不是缺陷。

### 9.1 定位手段：把每步耗时打出来（本轮加的开关）

`PYQCD_EINSUM_DEDICATED_TRACE=1` 时 `dedicated_apply.cpp` 向 stderr 打印
`parse` / `alloc` 耗时、每个中间槽字节数，以及每步 `pack / gemm / unpack / total`
和 `bmnk`（`dedicated_apply.cpp:42-54`、`:838-925`、`:999-1010`、`:1030-1040`）。
关闭时不进入任何计时分支。dim=8 的 15 步全表：最大两步是 step9
（256×64×64，gemm 0.022 ms）与 step11（512×64×64，pack 0.035 + gemm 0.073 ms），
**没有任何一步**接近秒级；`parse` 0.007–0.010 ms、14 个槽 `alloc` 0.006–0.012 ms。

配套最小 C++ 驱动：读 `_dedicated_plan` 落盘的步表字节流、按表头形状表自备操作数，
`-I cpp/kupl/einsum/include -L build -lpyqcdeinsum_kupl`。A/B 版用
`dlopen` 同时装载两个 `.so` 并**交错**计时同一步表——机器有别的负载时，这是唯一
能把两版代码比干净的办法（对比 Python 侧测量，见 9.4）。

### 9.2 踩到的三个假象（按坑的严重程度排序）

| 假象 | 症状 | 真相 |
|---|---|---|
| **机器被别的进程打满** | 同一调用从 0.30 ms 变 26 ms；单次 sgemm 0.022 ms → 10.9 ms（0.3 GFLOPS） | 当时 load average 12.45 / 8 核，numpy 自己的 `256×64×64` matmul 也从 0.027 ms 涨到 1.55 ms。**先做负载校准再报数**（`taskset -c 6` + `OPENBLAS_NUM_THREADS=1` + 同进程内 numpy matmul 基线） |
| **numpy `optimize="greedy"` 当参考** | dim=4 就要 **20.4 s**，dim=8 直接跑不完 | numpy 把末尾若干步合成一个多操作数 `c_einsum`，朴素工作集 4^10 / 8^10。参考一律用 `contract(..., specialize=False)` 或「按步表反解出的显式 einsum_path」，**不要** `np.einsum(..., optimize="greedy")`（与 8.7 第 5 条同源） |
| **装到 `pyqcd/einsum/kupl/` 的旧产物优先** | 改了源码不生效 | `library_path()` 先认 `pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so`（打包拷件），源码树里它常常比 `build/` 旧。测新代码用 `PYQCD_EINSUM_LIB=<build>/libpyqcdeinsum_kupl.so` |

### 9.3 本轮改的两处（都在 `cpp/kupl/einsum/src/dedicated_apply.cpp`）

1. **pack/unpack 连续后缀逐轴合并**（`copy_axes`，`:684-745`）：原来只在「最内轴
   两侧步长都为 1」时 memcpy，否则整块逐元素 gather；现在从最内轴往前把步长满足
   `step == 已合并内层乘积` 的轴并进同一次 memcpy。纯转置块从「每外层块一次
   memcpy」变成「整段一次」，真正交叉的步（末轴步长 ≠1）才逐元素。
2. **退化 GEMM 不进 BLAS**（`gemm_block`，`:764-819`，调用点 `:987`）：
   `M/N/K` 任一为 1 时该步是纯缩放（K=1，rank-1 更新）/ 广播归约（M=1 或 N=1），
   直接按定义算；分支在 batch 循环外，逐 batch 只是一次整型比较。K=1 时结果与
   BLAS 逐位相同（beta=0 不做加法），M/N=1 只差求和顺序。

**没做**的两项（实测不值得）：`alloc` 只占 0.006–0.012 ms（2–3%），不动块池阈值；
batch 循环（语料 dim=8 最多 85 次 GEMM 调用/方程）不折进 M。

### 9.4 实测（本机 8 核，单线程 BLAS，钉在单核；同进程交错 15 轮）

| 步表 | 改前 min/median | 改后 min/median | 两版输出 |
|---|---|---|---|
| dim=4 | 0.0177 / 0.0203 ms | **0.0159 / 0.0190 ms** | 逐位相同（dev 0.0） |
| dim=8 | 0.3173 / 0.3446 ms | **0.3038 / 0.3262 ms** | 逐位相同（dev 0.0） |

Python 侧 `try_dedicated` vs `contract(..., specialize=False)`（7 轮中位，dim=4/8，
float32/float64，同批 3 次重复）：比值 **0.20 / 0.20 / 0.68 / 0.76**，偏差全部
`0.0e+00`；重复 3 次结论不变，但单轮中位受负载影响可差 ±40%（0.38–0.66 ms 同库同题），
**报数值必须同时报校准与轮次**。

正确性回归：语料等距抽 150 条方程、dim=4，与 `contract(specialize=False)` 对照，
float64 最大偏差 **6.1e-14**（float32 侧 1.5e-5 是 16 算子连乘的固有舍入，改前同量级）；
新旧两库逐元素 A/B：float64 最大 1.3e-14、float32 最大 6.1e-6（仅求和顺序差异）。

### 9.5 若下次又收到「dim=8 卡死」

1. 先 `PYQCD_EINSUM_DEDICATED_TRACE=1` + 最小 C++ 驱动取逐步表；**先看 `parse`/`alloc`
   与最大步的 `bmnk`**，再谈优化。
2. 同时算该方程的总 FLOP（`2·Σ batch·m·n·k`）：dim=8 下语料最重的族（F01/F02，
   rank-6 输出）单条就要 **41 GFLOP ≈ 2.6 s**，把这类方程当「卡死」是误判；报障那条
   F07 方程只有 8.2 MFLOP。
3. 报数前跑基线校准；换库用 `PYQCD_EINSUM_LIB`；参考用 `specialize=False`。

## 10. [Q5] CUDA 侧 per-拓扑步表执行器（2026-11-11）

KUPL 的「固定步表 + 参数化执行器」已按同一份 v1 字节流镜像到 CUDA：
`cpp/cuda/einsum/src/dedicated_apply.cu`（契约头
`cpp/cuda/einsum/include/einsum_dedicated.h`），导出 `qcd_cuda_dedicated_apply`
（`make-nv.sh` version script 已加）。Python 侧 `_kernel_cuda.try_dedicated`
（fail-open），`_api._run_specialized_kernel` 的尝试链新增 `cuda_dedicated`，
位置在 `cuda_topology` 之前。

### 10.1 步表复用方式（不是「再写一份」）

- **格式与生成端都复用 KUPL 侧**：格式由
  `cpp/kupl/einsum/include/dedicated_apply.h` 固定；编码只在
  `pyqcd/einsum/_kernel_kupl.py::_dedicated_plan` 实现一次。
  CUDA 侧 `_kernel_cuda.try_dedicated` 直接调
  `_kernel_kupl._dedicated_cached(...)` 取 `(plan, output_shape, ctypes buffer)`，
  两个后端吃**同一份字节**、共用同一份缓存，因此槽位/轴序/可行性判定逐位一致。
  换后端不需要第二套 `contract_path` 规划代码。
- 原生侧只做 `pick 2 slots → pack → gemm → unpack`：pack/unpack 直接复用
  `src/pack.cu` 的 `qcd_cuda_permute(dtype, src, dst, qcd_cuda_copy_spec)`——
  `shape` 给目标连续形状（A: `a_extent`，B: `b_extent`，C: `c_destination_extents`），
  `src_stride` 给源步长，`elements` 给该步元素数。不新写搬运 kernel。
- 中间槽 + 3 块 pack 缓冲 = **一次 `cudaMalloc`**（单块 arena，各段 256 字节对齐），
  由本 TU 的进程级设备块池最佳适配复用（`[bytes, 2*bytes]` 命中即复用，
  <1 MiB 直接释放）；步内不再分配。cuBLAS 句柄按设备缓存，整次调用由互斥量串行化。
- 精度：`float32 → cublasSgemm`、`float64 → cublasDgemm`，batch>1 用同名
  `*StridedBatched`；complex 与其它 dtype 一律返回 4（unsupported），绝不降精度。

### 10.2 本轮最大的坑：cuBLAS 行主序的 `ldb >= k`

行主序的 `C(m×n)=A(m×k)·B(k×n)` 若直接按「cuBLAS 列主序 + OP_N、`lda=k`、
`ldb=n`」下发，在 **`n < k`** 时会被拒：
`** On entry to SGEMM parameter number 10 had an illegal value`（param 10 = `ldb`，
编号从 `transa` 起算、不含 handle——用 ctypes 单独打一次 `ldb=0` 即可校准）。
cuBLAS 对该操作数的要求是「至少等于其行数」，我们给成 `n` 就小了。
语料里 `n=1,k=16` 这类步非常常见，所以表现为「简单方程能跑、真实拓扑全挂」。

**解法与 `src/batched_gemm.cu` 同一条恒等式**：对调两侧、m/n 对调，
三个缓冲按同一基地址直传：

```
C_rm(m×n) = A_rm(m×k)·B_rm(k×n)  ⇔  C^T_col(n×m) = B^T_col(n×k)·A^T_col(k×m)
cublasSgemm(N, N, n, m, k, 1, B, lda=n, A, ldb=k, 0, C, ldc=n)
```

此时三个 leading dimension 恰好都等于各自矩阵行数（`n / k / n`），与 `n`、`k`
谁大无关。batch>1 的 strides 相应为 `k*n`（原 B）、`m*k`（原 A）、`m*n`（C）。

### 10.3 正确性（本机 RTX 4060 Laptop sm_89 + CuPy 14.2 + nvcc 12.4）

`_wick_tasklist.TASKS` 去重后等距取 20 条方程（元数 6/7/9/10/13/14/16），
dim=4 与 dim=8 各跑一遍，与 `contract(eq, *ops, specialize=False)` 对照：

| dtype | 命中 | 最大相对偏差 | 门槛 |
|---|---|---|---|
| float32 | 40/40 | **2.0e-6** | 1e-5 |
| float64 | 40/40 | **3.9e-15** | 1e-5 |

fail-open 契约同样实测：complex64 直接返回 `None`（原因写 `dtype`）、
形状不符的 `out=` 被拒且不写入、合法 `out=` 原地写入并 `LAST_INFO["path"] == "cuda_dedicated"`。

### 10.4 性能（同批 20 条方程，逐条 5 轮中位后求和；5 次独立进程取中位/范围）

| dtype | dim | dedicated (ms) | 通用 cupy 路径 (ms) | 加速 |
|---|---|---|---|---|
| float32 | 4 | 12.4（10.7 – 19.7） | 48.5（42.9 – 87.8） | 3.9x（3.9 – 5.6） |
| float32 | 8 | 13.5（13.3 – 14.0） | 44.7（39.9 – 96.6） | 3.3x（3.0 – 6.9） |
| float64 | 4 | 11.8（10.6 – 15.4） | 46.0（43.8 – 91.8） | 4.2x（3.8 – 7.4） |
| float64 | 8 | 28.9（28.0 – 43.4） | 55.5（46.2 – 110.7） | 1.9x（1.65 – 2.67） |

- dedicated 侧各轮稳定（±20%），通用侧波动 2x——本机 8 核笔记本 + 共享显存，
  **报数必须带轮次与负载**（与 9.2 同一条纪律）。
- 单条 dedicated 中位：dim=4 0.13–0.69 ms、dim=8 0.12–1.67 ms（含 Python 侧
  ctypes/指针数组开销，小方程由它主导）。
- float64/dim=8 收益最小（1.65–2.67x）：算力受限，pack/unpack 与 Python 开销占比上升。
- 抽样 20 条里 13 条含 batch>1 的步（最大 batch=16），因此 `*StridedBatched` 分支
  被上面的数值对照覆盖；全语料 8638 条可编码方程中 6048 条含批量步。

### 10.5 装配注意

- `make-nv.sh` 的 `src/*.cu` glob 会自动带上新文件；符号只经 version script 导出
  （`qcd_cuda_dedicated_apply`）；`qcd_cuda_permute` 仍是隐藏弱符号，缺实现时入口
  返回 3，不伪造成功。
- 与 KUPL 同一个坑（见 9.2 第 3 条）：`pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so`
  是打包拷件，源码树里常常比 `build/` 旧。改完必须
  `cp cpp/cuda/einsum/build/libpyqcdeinsum_cuda.so pyqcd/einsum/cuda/`，
  否则 `try_dedicated` 会以「符号不存在」静默回落。
- 诊断：`PYQCD_EINSUM_DEDICATED_TRACE=1` 打印 parse/alloc、每个槽偏移、每步
  `pack/gemm/unpack/total` 与 `bmnk`，以及实际下发 cuBLAS 的 `C^T`/leading
  dimension/指针（排查 10.2 那类问题就靠这一行）。



## 9. CUDA 侧 per-拓扑执行器（2026-11-12，`[Q5]` 阶段记录）

| 项 | 值 |
|---|---|
| 实现 | `cpp/cuda/einsum/src/dedicated_apply.cu`（1075 行）+ `include/einsum_dedicated.h`（111 行） |
| 导出 | `qcd_cuda_dedicated_apply`（`make-nv.sh:232` version script 已加） |
| Python | `_kernel_cuda.try_dedicated`（`_kernel_cuda.py:543`），`_api.py:557` 分派 `cuda_dedicated` |
| 步表 | 与 KUPL 同构（复用同一二进制格式），差异只在执行端用 cuBLAS |
| 精度 | float32→`cublasSgemm`；float64→`cublasDgemm`；**禁止隐式降精度** |

### 实测（RTX 4060 sm_89，双精度计时需 `cp.cuda.Device(0).synchronize()`）

| dim | 正确性 | cuda_dedicated | generic | 比值 |
|---:|---|---:|---:|---:|
| 4 | ok | 0.70 ms | 2.46 ms | **0.29×**（快 3.4×） |
| 8 | **FAIL** | 4.64 ms | 3.34 ms | 1.39× |

**未解决缺陷**：CUDA 侧 dim≥8 结果不正确（工作集 0.23 MB，远小于显存，
因此不是容量问题）。这是当前最严重的未修项，攻关中。

### 判据

走没走 CUDA 特化看 `_api` 的分派标签是否为 `cuda_dedicated`；
`try_dedicated` 返回 `None` 时读 `_kernel_cuda.LAST_FAILURE["reason"]`，
常见值：`operands must be cupy arrays`（宿主 numpy 数组）、
`topology entry point is missing from the native library`（`.so` 未重编）。

## 10. [K4] 25 族可套用模板入口（2026-11-11，KUPL）

用途：为 `pyqcd/einsum/_wick_tasklist.py` 的 8638 条收缩式提供**按实现特征分组**
的原生执行路径。严格拓扑签名下 8638 条两两互异（忽略算子顺序仍 6958 条），逐条
生成内核不可行；按

```text
(算子元数, 各操作数 rank 的多重集, 输出 rank)
```

分组后只有 **25 族**（`docs/wick_kernel_families_20261111.md`，F01..F25）。同族内
收缩结构相同，差异只在「哪个指标连到哪个槽位」——那是数据，不是代码。

### 10.1 三层职责（务必分清）

| 层 | 文件 | 编译期 / 运行期 |
|---|---|---|
| 族常量 | `src/family_templates.h` 的 `Family<ARITY, OUT_RANK, RANKS...>` | **编译期**：元数、输出 rank、rank 多重集 → 步数 `ARITY-1`、中间槽 `ARITY-2`、定长槽位/轴表 |
| 步表 | `src/family_apply.cpp::parse_plan` | **运行期**：每步两个槽、保留指标、轴长，以及预算好的源步长/紧凑步长/解包步长 |
| 数值 | `family_templates.h::run_family<T>` | pack → `cblas_{s,d}gemm` → unpack，步内不分配，中间体用完即还给块池 |

族的准入是**结构性**的：`check_family` 逐项比较元数/输出 rank/排序后的 rank 多重集，
不等就返回 1；`find_family` 找不到实例化就返回 4。两者都不会「看起来能跑」地算错。

### 10.2 导出符号与 ABI

| 项 | 值 |
|---|---|
| 符号 | `qcd_kupl_family_apply`（`nm -D --defined-only … \| grep family` → `T qcd_kupl_family_apply`） |
| 声明 | `cpp/kupl/einsum/include/family_apply.h` |
| 实现 | `cpp/kupl/einsum/src/family_apply.cpp`（解析 + 族派发）、`src/family_templates.h`（25 份实例化） |
| version script | `cpp/kupl/einsum/make.sh`（`qcd_kupl_family_apply;`） |
| 参数 | `(const void* plan, long long plan_len, void* const* operands, int n_operands, void* out, int dtype_id)` |
| Python | `_kernel_kupl.try_family_apply(subscripts, operands, out=None, *, optimize="greedy")` |

返回值（fail closed，任何非 0 都不写 `out`）：`0` 成功、`1` 参数非法、`2` 内存不足、
`3` 内部错误、`4` 计划不属于 25 族或 dtype 无本入口数值路径。

dtype：`float32`→`sgemm`、`float64`→`dgemm`，表头 `dtype_id` 必须与调用方一致；
`complex64/128` 在解析阶段直接返回 4 —— **不做任何隐式降精度**，也不抛给调用方
（`try_family_apply` 一律返回 `None` 并写 `LAST_FAILURE`）。

### 10.3 步表格式（v1，40 字节族表头 + dedicated v1 尾部）

```text
header: int32[10] = magic 0x31414651, version 1, total_bytes,
                    num_operands, num_slots, num_steps, out_rank, dtype_id,
                    reserved0=0, reserved1=0
形状表: num_operands 段 [int32 rank][int64 extent×rank]
        num_slots    段 同上
        输出        1 段（rank == out_rank）
步记录（num_steps 条）: int32 a_slot,b_slot,dst_slot
                        int32 rank_a,rank_b,rank_c
                        int32 batch_rank,m_rank,k_rank
                        int64 batch,m,n,k
                        int32 a_axes[rank_a] / b_axes[rank_b] / c_axes[rank_c]
```

强制二元收缩树：`num_steps == num_operands-1`、`num_slots == num_steps-1`、
只有最后一步 `dst_slot == -1`。任一条不满足即 `4`。

**Python 侧不重复实现收缩分解**：`_kernel_kupl._family_plan` 调用已有的
`_dedicated_plan`（同一套 greedy 分解 + 校验），把它的 32 字节表头换成 40 字节族
表头，尾部逐字节复用。改 `_dedicated_plan` 的步语义时必须同步改这里。

### 10.4 语料上界（决定定长数组尺寸）

8638 条全量扫过：操作数 rank ≤ 4、**中间体 rank ≤ 9**（`aibj,ckdl,esft,munv,owpx,gqhr,…`
一类 14/16 元式子）、输出 rank ≤ 6。`QCD_FAMILY_MAX_RANK = 12` 留了余量。

> 教训：最初把上界设成 8，全量抽检里 2740/3024/3113 三条 arity=14 式子以 status 1
> 回落。**先把语料的真实上界测出来再定常量**，不要按操作数 rank 猜中间体 rank。

### 10.5 正确性（本机实测，2026-11-11）

判据：`rel = max|kernel - ref| / max|ref|`（与 `pyqcd/testing/einsum/verify_topology`
同一口径）。参考为**同一分步顺序**的 numpy 数值，避免 numpy 自己的 greedy 选中不同
路径引入差异。

| 口径 | 用例数 | 结果 |
|---|---:|---|
| 每族 3 条 × f32/f64 × dim 2/3 | 300 | 0 回落；f32 最差 `7.41e-06`（F04），f64 最差 `1.04e-14` |
| 分层抽检（每族 12 条） | 262 | 0 失败，最差 `4.95e-06` |
| **全语料 f32/f64 混跑 dim 2/3** | **8638** | 1 条越 1e-5：`index=8026` `2.66e-05` |
| **全语料 f64 dim 2/3** | **8638** | 0 条越 1e-11；最差 `3.69e-13` |
| 全语料（f32，`np.einsum(greedy)` 参考） | 320 | 1 条 `2.13e-05`，同类条件数放大 |
| 并发（8 线程 × 40 次） | 40 | 全通，最差 `1.14e-13` |
| fail-open 契约 | 9 | 全通（complex/混合 dtype/非连续/arity 2/off-family/out 不符） |

那 1 条越线**不是内核错**：同一输入下 float64 路径 `1.10e-13`，而 numpy 自己的
float32 分步结果与 float64 真值差 `1.22e-04`（`|out|≈6.4`，条件数放大），内核算术
与 numpy 同序，误差来自 float32 本身。诊断脚本留在会话 scratchpad，未入库。

### 10.6 性能（本机，单线程 BLAS，min-of-9，loadavg ≈ 10.5）

`family` = `try_family_apply` 端到端；`native` = 同一份步表直接调 C ABI；
`contract_auto` = `contract(..., specialize=False)`；`contract_greedy` = 同一 greedy
计划（**与内核所用计划完全一致**，剥离路径选择差异）。

| 族 | 元数 | out | family ms | native ms | contract_auto ms | 比 auto | 比同计划 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F24 | 6 | 0 | 0.025 | 0.004 | 0.061 | 0.410 | 0.434 |
| F25 | 6 | 2 | 0.025 | 0.004 | 0.060 | 0.412 | 0.427 |
| F21 | 7 | 2 | 0.034 | 0.008 | 0.113 | 0.298 | 0.308 |
| F23 | 7 | 4 | 0.051 | 0.021 | 0.135 | 0.379 | 0.402 |
| F22 | 9 | 0 | 0.032 | 0.005 | 0.082 | 0.385 | 0.394 |
| F19 | 9 | 2 | 0.031 | 0.005 | 0.113 | 0.277 | 0.282 |
| F20 | 9 | 3 | 0.031 | 0.005 | 0.117 | 0.268 | 0.272 |
| F15 | 10 | 2 | 0.037 | 0.007 | 0.137 | 0.272 | 0.283 |
| F13 | 10 | 4 | 0.041 | 0.011 | 0.146 | 0.280 | 0.283 |
| F14 | 10 | 5 | 0.047 | 0.016 | 0.148 | 0.316 | 0.319 |
| F16 | 12 | 0 | 0.035 | 0.005 | 0.096 | 0.360 | 0.360 |
| F17 | 12 | 1 | 0.041 | 0.006 | 0.158 | 0.261 | 0.272 |
| F18 | 12 | 3 | 0.037 | 0.006 | 0.147 | 0.249 | 0.257 |
| F08 | 13 | 2 | 0.046 | 0.010 | 0.159 | 0.288 | 0.297 |
| F11 | 13 | 3 | 0.053 | 0.020 | 0.192 | 0.274 | 0.281 |
| F09 | 13 | 4 | 0.059 | 0.024 | 0.203 | 0.290 | 0.300 |
| F10 | 13 | 5 | 0.062 | 0.022 | 0.243 | 0.256 | 0.269 |
| F12 | 13 | 5 | 0.084 | 0.045 | 0.236 | 0.359 | 0.348 |
| F04 | 14 | 4 | 0.052 | 0.014 | 0.191 | 0.272 | 0.280 |
| F05 | 14 | 6 | 0.082 | 0.044 | 0.228 | 0.359 | 0.365 |
| F01 | 16 | 2 | 0.048 | 0.009 | 0.180 | 0.265 | 0.274 |
| F06 | 16 | 3 | 0.072 | 0.025 | 0.277 | 0.261 | 0.273 |
| F02 | 16 | 4 | 0.057 | 0.014 | 0.224 | 0.256 | 0.266 |
| F03 | 16 | 5 | 0.063 | 0.020 | 0.231 | 0.272 | 0.265 |
| F07 | 16 | 5 | 0.098 | 0.050 | 0.290 | 0.336 | 0.350 |

dim=4：**25/25 族都快**，family/auto 0.249–0.412（F25 最差），family/同计划
0.257–0.434（F24 最差）。Python 侧入口开销 ≈ 0.02–0.03 ms（`family - native`）。

dim=8（rounds=5，同一口径）：

| 族 | family ms | native ms | contract_auto ms | 比 auto | 比同计划 |
|---|---:|---:|---:|---:|---:|
| F23 | 1.457 | 1.212 | 1.691 | 0.861 | **1.006** |
| F09 | 1.352 | 1.104 | 1.450 | 0.932 | **1.018** |
| F05 | 2.864 | 2.520 | 2.990 | 0.958 | **1.019** |
| F11 | 0.963 | 0.882 | 1.214 | 0.793 | 0.943 |
| F12 | 6.975 | 6.285 | 4.645 | **1.501** | 0.993 |
| F07 | 6.479 | 6.323 | 7.089 | 0.914 | 0.985 |
| F14 | 1.581 | 1.225 | 0.969 | **1.632** | 0.819 |
| 其余 18 族 | — | — | — | 0.299–0.790 | ≤ 0.943 |

**如实标注的输族（dim=8）**：

| 族 | 现象 | 瓶颈 |
|---|---|---|
| F12 / F14（对 `contract_auto`） | 1.50× / 1.63× | **不是执行器慢**：同计划下 0.993 / 0.819（更快）。原因是内核固定用 `greedy` 路径，而 `contract` 默认 `optimize="auto"` 对这两族选了更优路径 —— F12 的 `largest_intermediate` greedy 2 097 152 vs auto 262 144，`opt_cost` 3.80e8 vs 3.50e8。修法：让 `_family_cached` 支持 `optimize="auto"`（可行性已在 `_dedicated_plan` 里验证过，只是当前只编码 greedy）。 |
| F23 / F09 / F05（同计划） | 1.006 / 1.018 / 1.019 | 纯噪声级（≈2%）。三族的算术强度 11–11.7 flop/byte，pack/unpack 流量与 GEMM 同量级，块池复用已到位，剩余差异在 BLAS 陈旧缓存与共享机器抖动内。 |

dim=8 的测量环境是共享 8 核机器、loadavg 9–13，`median` 会因抢占出现 7 倍抖动
（实测同一族 4.85 ms ↔ 0.70 ms），因此上表统一取 min-of-N；**不要**把 dim=8 的单次
读数当成结论。

### 10.7 复现

```bash
bash cpp/kupl/einsum/make.sh
nm -D --defined-only cpp/kupl/einsum/build/libpyqcdeinsum_kupl.so | grep family

# 端到端（源码树里必须显式指库，否则加载 site-packages 里的旧 .so）
export PYQCD_EINSUM_LIB=$PWD/cpp/kupl/einsum/build/libpyqcdeinsum_kupl.so
python -B -c "
from pyqcd.einsum import _kernel_kupl as K, _wick_tasklist as T
import numpy as np
eq = T.EQUATIONS[0]
ops = [np.ones((3,)*len(t), np.float32) for t in eq.split('->')[0].split(',')]
print(K.try_family_apply(eq, ops) is not None, K.LAST_INFO['path'])"
```

### 10.8 踩过的坑

1. **`.so` 陈旧**：`pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so` 是安装产物，重编只在
   `cpp/kupl/einsum/build/` 落盘。源码树里不设 `PYQCD_EINSUM_LIB` 会一直加载旧库，
   表现为「符号缺失，静默回落」。
2. **`static constexpr int ranks[]` 不能补零**：`{RANKS_..., 0, 0, …}` 会触发
   `too many initializers`（元素数已由 pack 给定）。用 `ranks[kRankCount] = {RANKS_...}`。
3. **定长上界要按语料实测**（见 10.4），不要按操作数 rank 推中间体 rank。
4. **`TASKS` 的字段形状会变**：实测期间 `_wick_tasklist.TASKS` 从 dict 变成
   `Task` 数据类，批量脚本改用 `EQUATIONS` 就不再受影响。
5. **numpy `optimize='greedy'` 会选到病态中间体**：14 元式子实测单条 98 s。全量
   抽检要用「同分步顺序」的参考，或先按中间体 rank 过滤。

## 11. [R5] 否决「CUDA dim≥8 错数 = cuBLAS 行/列主序转换」（2026-11-11）

**结论：布局不是原因，`cpp/cuda/einsum/src/dedicated_apply.cu` 无需改动。**
四个 GEMM 调用点用的就是标准「对调两侧 + m/n 对调」恒等式，`lda/ldb/ldc` 恰好
等于各 row-major 缓冲的**实际行宽**；GPU 实测 dim=4/6/8/10/12、全轴互异、含 size=1
退化轴三种形态全部正确（f32 ≤ 7.4e-7、f64 ≤ 1.3e-15）。

**最强的否决证据是反向的**：dim=4 的步表 15 步里已有 **14 步 m≠n**（如 `4×64×4`、
`8×64×4`、`128×8×4`、`256×4×4`、`64×32×16`），其中还有 1 步 batch=2。任何 trans/ld
混淆都会让 dim=4 先错，因此「dim=4 对、dim=8 错」与该机制**逻辑上不相容**。

### 11.1 调用点核对（行号即当前文件）

| 行 | 调用 | 实参（row-major 数据的解释） |
|---|---|---|
| `:680-684` | float 侧 `rows/cols/inner` 与两个 leading dimension | `rows=n, cols=m, inner=k, leading_b=n, leading_a=k` |
| `:695-696` | `cublasSgemm(N, N, …)` | A 操作数 = `b`（lda=**n**）、B 操作数 = `a`（ldb=**k**）、C（ldc=**n**） |
| `:701-704` | `cublasSgemmStridedBatched` | `stride_b=k*n`（即 B_rm 一块的大小）、`stride_a=m*k`、`stride_c=m*n` |
| `:716-720` | double 侧同上 | 同值 |
| `:723-724` / `:729-732` | `cublasDgemm` / `…StridedBatched` | 同值（**注意：double 分支没有 trace printf，只有 float 有**） |

三个 leading dimension 都等于对应 row-major 缓冲的行宽，而不是 `dim` 或 `m/n/k` 的猜测：
`b_rm` 是 `k×n` → 行宽 `n`；`a_rm` 是 `m×k` → 行宽 `k`；`c_rm` 是 `m×n` → 行宽 `n`。
运行时 trace 实测（dim=8, f32, 15 步）逐行符合，例：`bmnk=1,1024,16,8` 下发成
`C^T=16x1024 k=8 lda=16 ldb=8 ldc=16`。

### 11.2 CPU 判别实验（无需 GPU）

把「按当前实参解释一块 row-major 缓冲」写成 numpy 模型（`col_view(raw, rows, cols, ld)`，
`D[i,j] = raw[j*ld+i]`），再对 9 组 `(m,n,k)` 逐一复算：

| 参数组合 | m=n=k | m≠n 的矩形 |
|---|---|---|
| **当前（对调两侧 + m/n 对调）** | 0 | **0（全部 ≤1e-16）** |
| 朴素直传（不换序，lda=m） | 1.35 | 越界/1.4e0 |
| **只对调两侧、m/n 没对调** | **0** | **越界** |
| 对调两侧且 m/n 对调、但 lda/ldb 反了 | 0 | 越界 |
| 只对调 m/n、不换两侧 | 1.35 | 越界 |

「只对调两侧、m/n 没对调」这一格（m=n 时精确 0、m≠n 时崩）正是本假设期待的指纹形态；
当前代码**不在**这一格。batched 侧同理：`stride` 忘了对调的组合在 m≠n 时给 1.0–1.3
的相对误差，m==n 时谎报 0；当前 `k*n / m*k / m*n` 在 6 组 `(batch,m,n,k)` 上全 0。

### 11.3 GPU 实测（RTX 4060 Laptop sm_89，CuPy 14.2）

方程 `cudv,ewfx,msnt,okpl,iajb,qgrh,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHew`，
external=2，参考两条：numpy 显式路径（`optimize=['einsum_path', *pairs]`）与
`contract(…, specialize=False)`：

| dim | f32 / np-path | f32 / cupy | f64 / np-path | f64 / cupy |
|---:|---|---|---|---|
| 4 | 3.81e-07 | 3.81e-07 | 6.93e-16 | 7.70e-16 |
| 6 | 3.03e-07 | 2.65e-07 | 7.05e-16 | 5.79e-16 |
| 8 | 3.86e-07 | 3.17e-07 | 1.32e-15 | 1.14e-15 |
| 10 | 6.83e-07 | 6.83e-07 | 1.32e-15 | 1.13e-15 |
| 12 | 7.35e-07 | 5.41e-07 | 1.08e-15 | 1.32e-15 |

另外三种"最容易露馅"的形态也全过：

| 形态 | 结果 |
|---|---|
| 每个指标给不同轴长（3..25），15/15 步 m、n、k 两两不等 | f64 1.05e-15、f32 7.18e-07 |
| 大写 4 / 小写 5（7/15 步三量互异） | f64 5.77e-16、f32 4.11e-07 |
| 含 size=1 退化轴（5/15 与 15/15 步出现 m/n/k 之一为 1） | f64 ≤5.5e-16、f32 ≤3.4e-07 |
| 全轴=2（等长轴，唯一能跑 numpy greedy 的形态） | vs numpy-greedy f32 4.16e-07、f64 9.61e-16 |

公开 API 也确认走的就是这条路：`contract(eq, *ops)` 的 `LAST_INFO['path'] ==
'cuda_dedicated'`，dim=4/8/12 × f32/f64 相对通用路径 = 2.3e-7 / 4.6e-16 … 6.7e-7 / 1.5e-15。
语料扫描（12 条 × dim 4/8/12 × 2 dtype = 72 格；高元数 13..16 另 36 格）**0 失败**，
f32 最差 1.37e-06、f64 最差 1.05e-14。

### 11.4 为什么「dim≥8 错数」的报障会反复出现：参考侧先炸

同一条 16 元方程上 numpy **自己的** greedy 路径只有 6 步，末步是全合并的
`c_einsum`，朴素乘加量随 dim 暴涨：

| dim | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| greedy 步数 | 11 | 6 | 6 | 6 | 6 |
| 末步朴素乘加 | 6.6e4 | 3.1e9 | **5.5e11** | 8.1e14 | **1.4e17** |

dim=4 就要 5.5e11 次乘加（本机实测 >15 min 未返回），dim≥8 更不可能算完。
所以「dim≥8 与 `np.einsum(optimize='greedy')` 对照」这件事本身**不成立**：
对照侧不是慢，是不可计算。凡是这类报障，先问「参考是怎么算出来的」——
与 pitfalls `T3`、本文 9.2/[P3] 是同一个坑的第三次复现。

### 11.5 复现

```bash
export PYTHONPATH=$PWD                     # 否则 import 到 site-packages 的旧 pyqcd
export PYQCD_EINSUM_DEDICATED_TRACE=1      # float 侧打印每一步真实的 lda/ldb/ldc
python -B -c "
import numpy as np, cupy
from pyqcd.einsum import _kernel_cuda as K, contract
eq=('cudv,ewfx,msnt,okpl,iajb,qgrh,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHew')
terms=eq.split('->')[0].split(',')
shapes=[tuple(2 if c.isupper() else 8 for c in t) for t in terms]
ops=[cupy.asarray(np.random.default_rng(0).standard_normal(s)) for s in shapes]
got=K.try_dedicated(eq,ops); ref=contract(eq,*ops,specialize=False)
s=float(cupy.max(cupy.abs(ref))); print(float(cupy.max(cupy.abs(got-ref)))/s)"
```

### 11.6 若下次又收到「CUDA 某入口 dim≥8 错数」

1. 先确认**参考侧可算**：算它的朴素工作集（`np.einsum_path(..., einsum_call=True)` 的
   末步标签乘积），>1e9 就不是"kernel 错"的证据。
2. 用 `contract(..., specialize=False)` 或「同分步顺序」的 numpy 当参考。
3. 判定布局类假设最快的办法是**看 dim 小的一侧**：步表里 m≠n 的比例。dim=4 已大量
   m≠n，布局错不可能只在 dim≥8 出现。
4. 需要真实 lda/ldb/ldc 时：`PYQCD_EINSUM_DEDICATED_TRACE=1`（**只有 float 分支打印**；
   double 分支目前无 trace，若要用 f64 取证得先补一行）。

## 12. [R1] dim≥8 复核（2026-11-12）：执行器无错，受检的是三件别的事

复核结论与 §11 一致（CUDA 步表执行器在 dim=4…16 数值正确），本轮补的是
**可复算的判据**，以及三处真缺陷/口径问题。

### 12.1 判定"错数"之前先分三档：算法 / 精度 / 参考

| 档 | 判据（都要跑） | 本机结果（报障方程，dim=8） |
|---|---|---|
| 算法 | 同一**输入**换成 f64 再跑同一条步表 | rel 1.34e-14 → 步表/轴序/GEMM 无误 |
| 精度 | f32 结果 vs 同输入 f64 精确值；numpy f32 路径 vs 同一精确值 | kernel 5.47e-6、numpy 9.33e-6 → **kernel 比 numpy 的更准** |
| 参考 | f32 结果 vs numpy f32 路径（两条都带固有误差） | 1.48e-5（=两条 f32 链的误差之和） |

所以「f32 与 f32 对照、门槛 1e-5」在 16 元 dim=8 上**处在误差地板上**：
等距抽 100 条 dim=8 只有 0/100 超阈（worst 3.1e-6），30 条那组出现 1 条例外
（seq 7148，1.48e-5），它同时满足上表第 2 行（kernel 更接近精确值）。
报"通过率"必须写清 dtype：f64 全过（max 1.34e-14），f32 是 29/30 或 100/100 看抽样。

### 12.2 布局假设的定量反驳：`n<k` 是常态，不是 dim≥8 的反应

抽 200 条 dim=8 语料方程、解码全部步表（2802 步）：`n<k` 448 步（16.0%）、
其中 84 步还带 `batch>1`；`m≠n` 2323 步（82.9%）。旧写法 `ldb=n` 在这些步上会被
cuBLAS 直接拒（`parameter 10 had an illegal value`），而 dim=4 同样大量 `m≠n`——
**布局错不可能"只在 dim≥8"出现**，这条经验足以提前否掉"轴序/leading dimension"假设。

实测下发的三个 leading dimension（`PYQCD_EINSUM_DEDICATED_TRACE=1`，
`_kernel_cuda` 侧 `C^T=rows×cols`，rows=n、cols=m）：

```
m=2,k=8,n=3 → C^T=2x3 k=8 lda=2 ldb=8 ldc=2     # n=2<k=8，ldb=k 才是合法的
dim=8 步0   → C^T=512x8 k=8 lda=512 ldb=8 ldc=512
```

### 12.3 本轮修的一处：计划口径与 KUPL 侧不一致（默认 `"greedy"` → `"auto"`)

`_kernel_cuda.try_dedicated` 的 `optimize` 默认值曾是 `"greedy"`，而
`_kernel_kupl.try_dedicated` 是 `"auto"`（= auto/greedy 两条计划按步表 FLOPs 择优），
后果有两个：同一条方程两个后端**跑的不是同一个计划**，且缓存键（`optimize` 进键）
错开、"共用同一份缓存"实际不成立。
实测（dim=8 随机 100 条语料）8/100 条的 greedy 计划更贵，单条最大 **4.48×** FLOPs，
全表合计 **1.86×**；30 条那组是 3/30 不同、最大 3.14×、合计 1.10×。
配对计时（同进程同方程，5 轮中位）：seq 596 `1.81 → 0.68 ms`、seq 5064 `1.14 → 0.68 ms`，
计划相同的条目无差别。改后 `LAST_INFO` 多记 `plan_flops` / `optimize`，
判定"这条走了哪条计划"不用再解码步表。

### 12.4 另一个真缺陷：缓存项是 4 元组

`_kernel_kupl._dedicated_cached` 返回 `(plan, output_shape, plan_buffer, plan_flops)`。
CUDA 侧曾按 3 元组解包 → `ValueError`。注意它**不在** `try` 内，不会被吞成回落，
而是直接抛给调用方（§10.1 曾说"取三元组"，按那句写就会踩）。现在按 4 元组解包。

### 12.5 验收口径（本轮实跑，可在任何有 cupy 的机器复算）

```bash
python - <<'PY'
import numpy as np, cupy, opt_einsum
from pyqcd.einsum import _kernel_cuda as K
eq=('cudv,ewfx,msnt,okpl,iajb,qgrh,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHew')
def ref(eq,a):
    p,_=opt_einsum.contract_path(eq,*[x.shape for x in a],shapes=True,optimize='greedy')
    return np.einsum(eq,*a,optimize=['einsum_path',*list(p)])   # 唯一可算的 numpy 参考
for dim in (4,6,8,12,16):
    a=[(lambda s: np.random.default_rng(0).standard_normal(s))(tuple(2 if c.isupper() else dim for c in t))
       for t in eq.split('->')[0].split(',')]
    for dt in (np.float32, np.float64):
        g=K.try_dedicated(eq,[cupy.asarray(x.astype(dt)) for x in a])
        r=ref(eq,[x.astype(dt) for x in a]); s=np.max(np.abs(r))
        print(dim, dt.__name__, float(np.max(np.abs(cupy.asnumpy(g)-r))/s))
PY
```

本机（RTX 4060 Laptop、CuPy 14.2、loadavg 24/8 核）：
f32 2.5e-7…7.5e-7、f64 2.3e-16…1.4e-15（dim=4/6/8/12/16）；
语料等距 30 条 dim=4 30/30（max 7.0e-7）、dim=8 29/30（例外见 12.1）或 f64 30/30（max 1.3e-14）；
100 条 dim=8 抽检 0/100 超 1e-5（worst 3.1e-6）。

### 12.6 性能（同步计时，20 条语料方程，3 次重复各 5 轮中位后取最小值）

| dim | cuda_dedicated | generic `specialize=False` | 加速 |
|---:|---:|---:|---:|
| 4 | 10.02 ms | 65.53 ms | 6.54x |
| 8 | 21.02 ms | 52.16 ms | 2.48x |

单条 dedicated 0.09–7.98 ms；dim=8 仅 1/20 条比通用慢。**报数纪律照旧**：
loadavg 24 时同一条方程的中位可差 ±40%，上表是 min-of-N，别拿单次值下结论。

### 12.7 仍未做（留档）

- f32 的 1e-5 门槛事实上是误差地板：要么按 f64 判，要么门槛随链长放宽
  （实测 16 元 f32 链的 kernel 自身误差就到 5.5e-6）。
- `double` 分支的 `GemmDispatch` 没有 trace 行，f64 取证目前只能看数值、看不到 ldd。
- 设备块池（`DeviceBlockPool`）不记块所属 device：同进程切卡时可能把 A 卡的
  指针发给 B 卡（单卡生产路径不受影响，多卡场景未验证）。

## 13. [R6] 否决「CUDA dim≥8 错数 = pack/unpack 轴序与 stride」（2026-11-12）

**结论先说：pack/unpack 不是根因。** 把「pack → GEMM → unpack」三段分别隔离、
逐字节对照后，`dim=4` 与 `dim=8` **都精确**（差异恒为 0）；当前构件上
`dim≥8` 也复现不出错数。本节顺带把 §9「CUDA 侧 per-拓扑执行器」表里那条
`dim=8 FAIL` 判为**过期记录**：同一文件的 §10.3（40/40）、§11、§12 与本轮
8638 条全语料结果互相一致，只有该行不同。

### 13.1 判别实验：用「恒等 B」把 pack 输出整块取回来（不改源码）

构造**单步**步表（两操作数、无中间槽、`dst_slot=-1`）：A 按 `a_axes=P` 任意置换，
`batch_rank=0/m_rank=1/k_rank=r-1`；B 取 `K×K` 单位阵（形状 `(k1..k_t, N=K)`）。
于是 `out[m,j] = Σ_p packed_A[m,p]·δ[p,j] = packed_A[m,j]`——**一次调用取回整个
pack 缓冲**，与 `np.transpose(A, P)` 逐元素比即可，不需要 dump、不需要改 CUDA 代码。
unpack 用对偶构造（A=单位阵 ⇒ `C_canon=B`，`out_phys` 必等于 `np.transpose(B, c_axes)`）。

| 口径 | 用例 | 结果 |
|---|---|---|
| pack，dim=4（rank 2–4，5 种置换） | 9 | 9/9 逐字节相同（`maxabs=0`） |
| pack，dim=8（rank 2–4，5 种置换） | 9 | 9/9 逐字节相同 |
| pack，带 batch 轴 / size-1 轴 / rank 5–8（`batch_rank` 1–3） | 12 | 12/12 逐字节相同 |
| unpack，rank 3–6 置换（含量 1 轴） | 9 | 9/9 逐字节相同 |
| 纯批量 GEMM（无 pack），`batch` 1–16、`m≠n`、`k` 1–16，f32/f64 | 14 | 全部 ≤1.5e-7（f64 ≤3.6e-16） |

**负对照（证明上面的 0 不是空断言）**：同样的比较换成「反向置换」或「翻转 B 的 K 轴」，
偏差立刻变成 **3.29–4.59**（pack）与 **5.48–19.14**（GEMM）。所以这三个 0 有分辨力。

### 13.2 逐条核对（行号为当前文件）

| 问题 | 答案 | 证据 |
|---|---|---|
| 轴 stride 是运行期算还是按 dim 幂次 | **运行期**：后缀连乘 + 按 `a_axes/b_axes/c_axes` 逐轴取真实步长 | `dedicated_apply.cu:211-224`（`suffix_steps`/`shape_strides`）、`:398-411`、`:446-467`、`:740-755`；`pack.cu:61-71`（`source_offset` 现场分解行主序）、`pack.cu:91-114`（tile 快路径的目的步长也是 `spec.shape` 后缀连乘） |
| 越界检查用 dim 还是真实 extents | **真实 extents** | `pack.cu:122-123`、`:138-139` 比 `spec.shape[axes.inner/outer]`；`pack.cu:210-224` 校验 `prod(shape)==elements`；`pack.cu:327-328` 网格超 65535 自动退回通用 kernel；`dedicated_apply.cu:885-890` 拒绝 >`INT_MAX` 而非截断 |
| 中间槽跨调用是否残留 | **会残留字节，但无可读路径**：写一次（`:332-334`）+ 存活位（`:873-875`、`:998-1007`）+ 每步整面写（GEMM `beta=0`，`:679/:733`；`k==0` 走 `zero_block`，`:943-952`） | 8638 条方程在同一进程内串行跑（arena 反复回收复用、尺寸互不相同），无一条读到陈旧数据；同一输入连跑 5 次逐位相同 |

### 13.3 全语料实测（当前构件：`build/libpyqcdeinsum_cuda.so` 06:53 晚于两个源文件，且与 `pyqcd/einsum/cuda/` 的拷件 `cmp` 相同）

参考用**同语义 numpy 逐步重放**（按步表解码后逐 step 做 pack/GEMM/unpack），
排除 numpy 自己另选路径带来的差异；另用 f64 作真值区分「算法错」与「精度地」。

| 口径 | 结果 |
|---|---|
| dim=8，全 8638 条语料（f32） | 8638/8638 原生入口返回 0；12 条超 1e-5，逐条复核后**全部**是 f32 精度地板（同输入 numpy f32 与 f64 真值偏 1.2e-5…3.97e-4，kernel f64 为 1.7e-15…8.2e-12） |
| dim=4，全 8638 条（f32） | 6 条超 1e-5，同样是精度地板（例：line 7597 np_f32 1.75e-4 / kernel f32 1.20e-4 / kernel f64 2.9e-13） |
| dim=4/6/8/10/12 × 20 条（f32，门槛 1e-5） | 100/100 通过，worst 5.1e-7 / 6.3e-7 / 1.0e-6 / 1.4e-6 / 1.8e-6 |
| 同上（f64，门槛 1e-11） | 100/100 通过，worst 3.1e-15 |
| **非均匀 extents**（每个指标独立取 1/2/3/5/7，786 条） | f64 0 条超 1e-11；f32 vs f64 真值 worst **4.9e-6** |
| 步表覆盖度（1728 条方程、24347 步，dim=8） | 需要 `pack_a` 66.5%、`pack_b` 72.9%、`pack_c` 8.9%、`batch>1` 9.4% |

「非均匀 extents」这一行是本节的关键补强：语料的 `dim/external` 是逐条统一的，
恰好会给 m≡n（4=2·2、8=2·2·2）这类**偶然相等**留出遮蔽；一旦每个指标独立取长度，
任何按 `dim^k`/对称性写的 stride 都会立刻暴露——实测 f64 仍精确到 1e-13。

### 13.4 若下次又收到「pack/unpack 错序」

1. 先按 13.1 的「恒等 B」单步表把 pack 缓冲整块取回来比 `np.transpose`——**0 与非 0
   之间没有中间态**，一次就能定案，比读 1075 行源码快。
2. 判 f32 的 1e-5 超阈之前，先算同一输入的 f64 结果；f64 干净 ⇒ 是链长 + 条件数的
   精度地板，不是轴序（轴序错在 f64 下同样是 O(1) 偏差）。
3. `pack_c` 只占步数的 8.9%，最容易漏测——单步表能直接构造 rank 3–6 的输出置换。

## 14. [S1] 固定步表执行器接入 complex64 / complex128（2026-11-12）

### 14.1 实现口径

| 层 | 做法 |
|---|---|
| 元素布局 | numpy/cupy 的复数就是交错 `[re, im]` 对，与 `std::complex<T>`、`cuFloatComplex/cuDoubleComplex` 同布局。两个入口各加一组 `static_assert(sizeof(...) == 2*sizeof(T))`，把前提钉在编译期而不是注释里 |
| 搬运 | pack/unpack 本来就是逐元素字节拷贝（CPU `memcpy`、CUDA `qcd_cuda_permute`），复数不需要新 kernel。CUDA 侧 `qcd_cuda_permute` 早已实例化四种 dtype，唯一要改的是把原来「`dtype==FLOAT32 ? FLOAT32 : FLOAT64`」的二选一换成 `table.dtype_id` 直传 |
| GEMM | CPU 走 `cblas_cgemm` / `cblas_zgemm`：alpha/beta 是指向**本精度复常量**的 `const void*`，其余参数（RowMajor、transA/B、lda/ldb/ldc）与实数版逐项相同。CUDA 走 `cublasCgemm` / `cublasZgemm`（batch>1 用同名 `*StridedBatched`），alpha/beta 为 host 复常量；行主序→列主序仍用「两侧操作数对调、m/n 对调」那条恒等式，复数照搬 |
| 手工归约短路径 | k==1 / n==1 / m==1 三条不进 BLAS 的分支改用 `Accumulator<T>` trait：`float→double`、`complex<float>→complex<double>`、其余原样。complex64 因此按双精度复数累加后再舍回，与 `cblas_cgemm` 同档，长链不劣于 BLAS |
| dtype 分派与守卫 | `parse_table` 放开四档；复数且 BLAS 缺 `cgemm/zgemm`（`complex_complete()` 为 false）时返回 4（UNSUPPORTED）让上层回落，不硬错、不降精度。执行期 `BlasGate<T>` 按精度只查自己那一个符号；入口 `switch` 四档分派（CUDA 用 `cuFloatComplex/cuDoubleComplex` 作 `T`，本 TU 对复数只做指针算术） |
| Python 白名单 | `_DEDICATED_DTYPES`（准入闸门）与 `_DEDICATED_DTYPE_IDS`（下发 id：complex128=0/ complex64=1/ float64=2/ float32=3）**两处都要改**；family 入口另立 `_FAMILY_DTYPES`，不复用 dedicated 的元组 |

### 14.2 实测（CPU: OpenBLAS via `libcblas.so.3`；GPU: RTX 4060 Laptop sm_89 / cuBLAS 12.4）

| 口径 | 结果 |
|---|---|
| `python -m pyqcd.testing._double_precision_contract --dtype complex128 --sample 100` | 特化真跑 **100/100**，失败 **0**，最大偏差 **2.42e-14**（中位 2.44e-16，门槛 1e-12） |
| complex128 全语料 8638 条 | 8638/8638 跑完；2 条只超「特化 vs 通用 ≤1e-13」这一项（1.20e-13、1.11e-13），对参考的偏差仍 ≤1.20e-13 |
| complex64 自建同形口径（rtol 1e-5，参考升 complex128）`--sample 100` | **100/100**，失败 **0**，最大偏差 **6.74e-07** |
| complex64 全语料 8638 条 | 8638/8638；1 条 specialized-vs-generic 1.62e-5 超阈，但该条对参考只有 6.4e-7，**比通用路径的 1.6e-5 更准** |
| float64 全语料（未改动路径的对照） | 同样在 `seq=1973` 超阈（4.18e-13）⇒ 超阈来自 16 算子链的求和顺序，与复数无关 |
| 边界形态：k==1 / n==1 / m==1 / batch / transB / packA / batch+trans+pack / 三算子链 / k==0 | CPU 9/9、CUDA 8/8（k==0 在 CUDA 上是既有零尺寸指针限制，实数同样），四种 dtype 全过；complex64 worst 7.6e-8、complex128 worst 3.5e-16 |
| CUDA 语料抽 30 条（dim=2） | complex128 worst **1.85e-15**、complex64 worst **6.89e-7** |
| 缺复数符号回落 | 用只导出 `cblas_sgemm/dgemm` 的假 BLAS 指向 `PYQCD_EINSUM_CBLAS_LIB`：complex128 收到 **status 4**，实数照常 |

### 14.3 四个坑

1. **包内副本优先于仓库构建产物**：`library_path()` 先看 `pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so`，再看 `cpp/kupl/einsum/...`。`make.sh` 只更新后者，不拷进包内就会跑**旧库**——现象极具误导性：**实数全对、复数一律 status 4**（旧库的 dtype 白名单只到实数）。CUDA 侧同理（`pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so`）。
2. **只改 `_DEDICATED_DTYPE_IDS` 不够**：`try_dedicated` 先用 `_DEDICATED_DTYPES` 拦 dtype，字典在拦过之后才查；两处必须同时加，否则复数永远走不到下发。
3. **family 入口不要跟着放开**：`try_family_apply` 原本复用 `_DEDICATED_DTYPES`，复数放开后会把 complex 下发给只实例化了实数的族模板（白跑一次原生、留 status 4）。现独立为 `_FAMILY_DTYPES`，`family_dtypes()` 也只报实数两档。
4. **dispatch 顺序的连带效应**：`_api` 里 `kupl_dedicated` 排在 `kupl_topology` 之前，complex64 一旦被 dedicated 接下，语料里就**不再有**「只由拓扑入口服务」的式子。`_topology_kernel_contract.test_native_failure_falls_back_with_one_warning` 正是靠这个前提挑方程（断言"原生入口必须真的被调用过"），因此转红。数值本身正确（该式 dedicated 6.2e-8、通用 1.3e-7，参考为 complex128 精确值），修法是在该测试里把 dedicated 入口一并 patch 掉——属测试侧改动，不在本内核任务范围内。

## 15. [S2] KUPL 复数特化的独立验收（2026-11-12）

§14 的 `[S1]` 是**实现侧**记录；本节是**验收侧**复核：不改内核，只复跑判据、补两条此前没测过的
证据，并修正 §14.2 里两处会被误读的表述。

### 15.1 四项交付逐条核对（源码行号对应当前工作树）

| 任务项 | 落点 | 复核结果 |
|---|---|---|
| `run_typed<std::complex<T>>` 数值路径 | `dedicated_apply.cpp:1206`（`run_typed` 是模板，四档共用同一份代码） | 已有；`sizeof(std::complex<T>) == 2*sizeof(T)` 的 `static_assert` 在 `:276-279`（不是模板内——两档各写一条，编译期钉死） |
| 归约分支双精度累加 | `Accumulator<T>` trait `:284-295`，用在 `n==1`（`:1149`）与 `m==1`（`:1166`）两处 | 已有；`complex<float> → complex<double>` |
| GEMM 走 `cgemm`/`zgemm`，alpha/beta 为**指针** | `gemm_call` 复数重载 `:1094-1116`；`CgemmFn`/`ZgemmFn` `:305-308` | 已有；`const std::complex<float> alpha(1.0F,0.0F)` 取地址下发 |
| `parse_table` 白名单放开四档 + 缺符号 fail closed | `:571-581` | 已有；缺 `cgemm`/`zgemm` 返回 `QCD_DEDICATED_UNSUPPORTED`(4) |
| `apply_dedicated` 四档分派 | `:1449-1460` | 已有 |
| `_DEDICATED_DTYPE_IDS` 加两档 | `_kernel_kupl.py:468-471`（complex128=0 / complex64=1 / float64=2 / float32=3） | 已有，且与 `einsum_define.h:31-34` 逐项一致 |

复核时的**前置确认**（不做这步会误判）：包内副本
`pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so` 与 `cpp/kupl/einsum/build/` 的新构建产物
`cmp` **逐字节相同**——否则测的是旧库（§14.3 第 1 条）。

### 15.2 判据实跑（本机 conda OpenBLAS `libcblas.so.3`，`cblas_cgemm`/`cblas_zgemm` 均在）

| 口径 | 结果 |
|---|---|
| `python -m pyqcd.testing._double_precision_contract --dtype complex128 --sample 100` | 特化真跑 **100/100**、失败 **0**、最大偏差 **2.42e-14**（中位 2.44e-16，门槛 1e-12） |
| 同上 `--sample 0`（全 8638 条） | 跑完 **8638/8638**，仅 2 条超「特化 vs 通用 ≤1e-13」辅助项（`seq=1973` 1.20e-13、`seq=7998` 1.11e-13），**对参考的偏差仍 ≤1.20e-13、无一条超主判据** |
| complex64 抽检 **120** 条（自建脚本，参考升 complex128 + 固定 path，rtol 1e-5） | 命中 **120/120**、失败 **0**、最大偏差 **6.59e-07**、中位 1.74e-07；`LAST_INFO['dtype_id']` 恒为 **1** |
| complex64 结果 dtype | 逐条仍是 `complex64`（未被悄悄升精） |
| `python -m pyqcd.testing._complex_precision_contract` | **9 passed / 0 pending / 1 failed**；失败项是契约 4「公开入口拒绝混合精度」，属 `_api.contract` 的语义，不在内核范围（内核层 `test_mixed_precision_kernel_fails_closed` 通过） |

`seq=1973` 的超阈不是复数问题：**同一方程用 float64 跑同一路径，偏差 4.18e-13，比 complex128 的
1.20e-13 更大**。它是 16 算子链的求和顺序差异，§14.2 已有同结论。

### 15.3 补测一：complex64 确实按**双精度**累加（§12 结论对复数的落点）

只报「偏差小」不足以证明走了升档累加——必须让升档与不升档**分出差距**。做法：把归约轴拉到
`k=20000`（长链放大 float 累加误差），与「顺序 `complex64` 累加」这一错误口径对照：

| 分支 | kernel 偏差 | 顺序 float 累加 | 倍数 |
|---|---:|---:|---:|
| `n==1`（`ij,j->i`） | **6.30e-07** | 2.868e-06 | kernel 准 **4.6x** |
| `m==1`（`j,kj->k`） | **1.31e-08** | 8.191e-07 | kernel 准 **62.5x** |

更强的判据是**随链长的增长斜率**（`k` 从 1e3 → 1e5）：

| k | kernel 偏差 | 顺序 float 累加 |
|---:|---:|---:|
| 1 000 | 3.40e-07 | 4.19e-07 |
| 10 000 | 4.51e-07 | 1.06e-06 |
| 100 000 | **5.54e-07**（仅 1.6x） | **4.31e-06**（10.3x） |

双精度累加的误差只随**存储精度**（`complex64` 的 1.2e-7 表示误差 × √k）走，与累加位数无关；
float 累加则随 k 线性放大。kernel 的曲线是前者 ⇒ 升档累加生效。

### 15.4 补测二：`transA`/`transB`/`batch>1` 的复数组合（此前只测过 transB 单点）

用 `PYQCD_EINSUM_DEDICATED_TRACE=1` 同时确认**真的下发了 trans 位**，而不是靠 pack 绕过去：

| 形态 | trace 里的 trans | 偏差（vs opt_einsum，complex128） |
|---|---|---:|
| `ji,jk->ik`（A 为 `[K][M]`） | `trans=10` | **0.00e+00** |
| `ij,kj->ik`（B 为 `[N][K]`） | `trans=01` | **0.00e+00** |
| `bij,bjk->bik`（batch=4） | `trans=11`（两侧均直通） | 1.24e-16 |
| `bji,bjk->bik`（batch+transA） | `trans=10` | 2.35e-16 |

另外三项落地契约也实测：`out=` 原地写入（返回同一对象、`dtype=complex64`、偏差 3.78e-08）、
`k==0` 复数走 zero-fill（形状 `(2,0)`、dtype 保留、全零）、`try_dedicated` 的 `LAST_INFO['path']
== 'kupl_dedicated'`。

### 15.5 fail-closed 的证据强度：假 BLAS 是唯一可靠办法

`cblas_api()` 按 `PYQCD_EINSUM_CBLAS_LIB` → `RTLD_DEFAULT` → 候选名 依次探测，本机 conda 的
OpenBLAS 一定命中，**没法靠"这台机器没有复数 BLAS"来测回落**。必须编一个只导出部分符号的
假 `.so` 指过去：

| 假 BLAS 导出 | complex64 | complex128 | float64/float32 |
|---|---|---|---|
| 只有 `sgemm/dgemm` | status **4** | status **4** | 正常出结果 |
| `sgemm/dgemm/cgemm`（**缺 zgemm**） | status **4** | status **4** | 正常出结果 |

第二行是本轮新发现的口径细节：`parse_table:576` 用的是 `complex_complete()`（= `cgemm && zgemm`
**两者齐备**），而执行期 `BlasGate<T>` 只查自己那一个符号。所以「只有 `cgemm` 的 BLAS」会把
**complex64 也一起拒掉**。这是**偏保守的 fail closed**（回落通用路径、结果仍正确、绝不硬错、
绝不降精度），符合任务要求，故不改；但要知道它不是"按精度分别放行"。若将来遇到真的只带
`cgemm` 的精简 BLAS 并希望 complex64 仍走特化，把 `:576` 的判据换成 `BlasGate` 同款按精度检查
即可（一处改动，执行期守卫已就绪）。

### 15.6 实数不回归

`python -B -m pyqcd.testing` → **126 passed, 1 skipped, 1 failed**。该失败项是
`test_contract_edge_cases`，报
`ValueError: Number of einsum subscripts, 15, must be equal to the number of operands, 16`：
用例给 `…,ax,sy->MNGLxy`（**15** 个项）画了 **16** 个操作数。判据无关 dtype——
把操作数换成 `float64` 同样报同一句，因此与复数特化无关（属测试侧多画一个算子的笔误）。
`_topology_kernel_contract` 在 dedicated 接管 complex64 后一度转红（§14.3 第 4 条），
已在测试侧 patch 掉 dedicated 入口修复，现单跑 PASS。

## 16. [S3] CUDA 步表入口的复数四精度实测（2026-11-12）

范围：**只 CUDA**（`cpp/cuda/einsum`），KUPL 侧由 §15 记录。本机 RTX 4060 Laptop
sm_89 + CuPy 14.2 + nvcc 12.4，`baseline=03210c3`；被测 `.so`
`md5=948ebc80…`（加上 §16.5 的环境变量接缝后 `a42e74f6…`）。

结论先说：`complex64/complex128` 的数值路径**已在 HEAD 里**（`GemmDispatch<cuFloatComplex/
cuDoubleComplex>` + `AccumTraits` 升档），本轮工作是把它**测出来**并补上一处可测性接缝，
不是重写。四 dtype × 20 条语料全部命中，复数精度分别落在 1e-15 / 1e-7 档。

### 16.1 四 dtype 语料实测（20 条 `_wick_tasklist.TASKS`，dim 3/4 交替）

语料取 `seq` 等距 20 条（1, 455, …, 8638），arity 6–16，参考值用**升精度后
`contract(specialize=False)`** 重算（complex64→complex128、float32→float64）：

| dtype | 命中 | 最大相对偏差 | 判据 | `try_dedicated` 中位（首轮含初始化） |
|---|---|---:|---:|---:|
| complex128 | 20/20 | **1.10e-15** | 1e-12 | 2.11 ms |
| complex64 | 20/20 | **4.31e-07** | 1e-05 | 2.00 ms |
| float64 | 20/20 | **8.66e-16** | 1e-12 | 1.91 ms |
| float32 | 20/20 | **3.88e-07** | 1e-05 | 1.99 ms |

`LAST_INFO['path'] == 'cuda_dedicated'`、结果 dtype 与形状逐条与参考一致。

### 16.2 语料覆盖不到三条短路径，必须单独造形状

16 算子方程基本都落在 GEMM 分支；`k==1 / n==1 / m==1` 与 `*StridedBatched` 要**按形状
逼出来**，并用 `PYQCD_EINSUM_DEDICATED_TRACE=1` 的 trace 证明走的是哪条
（trace 打印 `bmnk=` 与 `pack_a/b/c`；worktree 版本另有 `short path: scale (k==1)` /
`reduce n==1 …` 行，按它一眼能看出进了短路径）：

| 目标分支 | 造法（equation + 形状） | trace 证据 | complex64 偏差 |
|---|---|---|---:|
| `k==1` 纯缩放 | `ab,cb->ac`，(128,1)×(128,1) | `k=1` | 4.83e-08 |
| `n==1` 归约 | `ab,cb->ac`，(1,4096)×(128,4096) | `bmnk=1,128,1,4096` | 3.30e-08 |
| `m==1` 归约 | `ab,cb->ac`，(128,4096)×(1,4096) | `bmnk=1,1,128,4096` | 3.77e-08 |
| StridedBatched | `abc,abd->acd`，(3,64,5)×(3,64,7) | batch=3 | 2.15e-07 |
| batched+pack_c | `abc,abd->cad`，(4,32,5)×(4,32,7) | `pack_c=1` | 1.89e-07 |

坑：`abc,bd->acd` 这类**只有一个操作数带 batch 轴**的式子，生成端会把 `a` 并进 `n`
（trace 里 `bmnk=1,7,15,64`），batch 恒为 1 —— 想测 batched 分支必须让两侧**都**带
该轴且它进输出（`abc,abd->…`）。只看"命中了 dedicated"会误以为 batch 路径被测过。

### 16.3 双精度累加怎么证明（不要靠读代码）

`complex64` 的归约分支升到 `cuDoubleComplex` 累加，与 KUPL 同口径。要**实测**它，
构造单步长归约（`ab,cb->ac`，`m=n=1`，`k=65536`），再和"按存储精度累加"的模拟值对比：

| dtype | CUDA 实测偏差 | 同精度累加模拟 | 分离度 |
|---|---:|---:|---:|
| complex64 | **1.15e-08** | 6.14e-06 | **534×** |
| complex128 | 8.69e-15 | 8.33e-15 | 1.0（已在地板） |
| float32 | **2.34e-08** | 2.26e-06 | **96×** |
| float64 | 1.63e-14 | 1.79e-14 | 1.1（已在地板） |

分离度 ≫ 10 就是"升档确实生效"的判据；若哪天有人把 `AccumTraits` 改回同精度，
这两行会立刻塌到 1× 附近（而 20 条语料**照样全绿**——这就是必须单造长 k 的原因）。

### 16.4 dtype 白名单 = 运行期 cuBLAS 符号，且**按 dtype 分别放行**

`qcd_cuda_dedicated_dtypes()` 对 `cgemm/zgemm` 缺符号的处理比 KUPL 更细：
KUPL 用 `complex_complete()`（c+z 必须**同时**齐备）保守地一起拒掉，
CUDA 侧 `gemm_available(dtype)` 只看**自己那一个** dtype 的 base+StridedBatched。
两档都是 fail open，差别只在"只剩 cgemm 的精简 BLAS"上：KUPL 拒 complex64，
CUDA 放行 complex64。**这不是 bug，是口径差异**，改动时别互相抄。

### 16.5 本轮唯一的代码改动：`PYQCD_EINSUM_CUBLAS_LIB` 接缝

想测"缺符号 → status 4 → Python 返回 None"必须有反例库，但**用 `LD_LIBRARY_PATH`
换不掉 cuBLAS**：`cpp/cuda/einsum` 链出来的 `.so` 带 `DT_RPATH=/…/miniforge3/lib`
（`readelf -d | grep RPATH`），RPATH 优先于 `LD_LIBRARY_PATH`，塞空壳库不会生效。
可行办法只有两条：patchelf 去掉 RPATH（本机没有），或在探测处开一个显式句柄口。

故照抄 KUPL `load_cblas()` 的既有约定，加 15 行：`gemm_symbol_scope()` 读
`PYQCD_EINSUM_CUBLAS_LIB`，命中就 `dlopen(RTLD_NOW|RTLD_LOCAL)` 并**只查该句柄**，
否则退回 `RTLD_DEFAULT`（= 原行为，env 不设时逐位不变）。实测（三个手搓空壳库）：

| 假 cuBLAS 导出 | mask | 白名单 | complex128 | complex64 | float32 |
|---|---:|---|---|---|---|
| 仅 `sgemm/dgemm`（无 `*StridedBatched`） | 0 | `[]` | 4 / None | 4 / None | 4 / None |
| `+*StridedBatched` | 12 | `float64, float32` | **4 / None** | **4 / None** | 0 正常 |
| `+cgemm(+batched)`，缺 `zgemm` | 14 | `complex64, float64, float32` | **4 / None** | 0 正常 | 0 正常 |

第一行同时验证了头文件里"base 与 StridedBatched 两种变体都在才算可用"这条声明：
只导出 base 时 mask=0。手工复现：

```bash
printf 'void cublasSgemm(void){}void cublasDgemm(void){}' > /tmp/fake_sd.c
gcc -shared -fPIC -o /tmp/fake_sd.so /tmp/fake_sd.c
# 只导出部分符号即可：探测只做 dlsym，不会真的调用这些函数
PYQCD_EINSUM_CUBLAS_LIB=/tmp/fake_sd.so python -c \
  'from pyqcd.einsum import _kernel_cuda as k; print(k.dedicated_dtypes())'
```

### 16.6 装配坑：`cpp/` 里编出来 ≠ Python 加载的那份

`_cuda_library_path()` **优先**返回 `pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so`，
`cpp/cuda/einsum/libpyqcdeinsum_cuda.so` 只是 symlink→`build/`。`make-nv.sh` 只写
`cpp/`，不发包内那份；只编 `cpp/` 会让 Python 继续吃旧的包内 `.so`（本轮踩到：
`build/` 已 948ebc80、包内还是更早的版本）。要么走 `setup.py` 安装流程，要么
`cp build/libpyqcdeinsum_cuda.so pyqcd/einsum/cuda/`，然后 `md5sum` 两边对齐再测。
另一个陷阱：本仓库可能**有别的 agent 同时在编译**（`ps -eo cmd | grep nvcc`），
所以报告里的数字必须绑定被测 `.so` 的 md5，否则复现不出来。

### 16.7 计时口径

* 首次进 GEMM 的调用含 cuBLAS handle/context 初始化：本机实测 **35–45 ms**（复数）/
  20–60 ms（实数，视上下文），第二发起回 ~2 ms。不预热就报"中位"会把初始化算进去。
* dim 3/4 的小张量上，预热后仍只有 **0.39–0.43 ms**，慢于通用 CPU 路径的
  **0.24–0.25 ms**（GPU 启动开销主导）。**dim 3/4 不是加速证据**，
  不要把这里的数字写成"CUDA 更快"；`_dedicated_allowlist` 是按 dim 4/6/8 实测的
  另一套表，勿混用。
* 计时后必须 `cupy.cuda.get_current_stream().synchronize()`，否则测的是入队时间。

### 16.8 观察到但未改的两处

1. **零尺寸操作数**（如 `(0,4)×(4,3)`）走 native 会返回 status **1**（`INVALID`）而不是 4
   （`UNSUPPORTED`）。功能上仍是 fail open（Python 拿到非 0 即回落、结果正确），
   只是语义上"零维"不是非法参数。真实语料没有零维，故不改，仅记录。
2. `qcd_cuda_permute/hadamard/batched_gemm` 在 `.so` 里是 **weak** 符号、被 version
   script 隐藏（`nm -D` 看不到，`nm --defined-only` 才有）。`make-nv.sh` 用后者检查，
   不要因为 `nm -D` 缺符号就以为 provider 没编进去。

## 17. [S2] KUPL 复数门改为**按 dtype 分别放行**（2026-11-12）

§15 复核时结论是"四项交付源码里都已具备"，本轮把"已具备"逐条**实跑**了一遍，
唯一发现需要改的是 §15.5 记下的那处口径不一致——`parse_table` 用
`complex_complete()`（cgemm **且** zgemm）一把闸，而执行期 `BlasGate<T>` 只查自己
那一个符号。任务书第 2 项写的是"BLAS 缺复数符号时返回 UNSUPPORTED"（按 dtype 语义），
不是"两档齐备才放行"，故本轮把它对齐成按 dtype 放行。

### 17.1 改动（`cpp/kupl/einsum/src/dedicated_apply.cpp`）

| 位置 | 改动 |
|---|---|
| `CblasApi` | 删除 `complex_complete()`（唯一调用点已改，留着就是第二套口径） |
| `parse_table` 复数门 | 拆成两条：`COMPLEX64 && cgemm == nullptr` → 4；`COMPLEX128 && zgemm == nullptr` → 4 |
| `bind_pair` 注释 | 不再引用已删的 `complex_complete()` |

`_kernel_kupl.py` 本轮**零改动**：`_DEDICATED_DTYPE_IDS` 的四档
（complex128=0 / complex64=1 / float64=2 / float32=3）已在位，与
`einsum_define.h:31-34`、`_native_plan.py::_DTYPE_IDS` 三方一致。

### 17.2 判据：假 BLAS 必须**部分**导出才测得出差别

`cblas_api()` 在本机一定命中 conda OpenBLAS（c/z 都齐备），所以"缺符号"只能用假 `.so`
经 `PYQCD_EINSUM_CBLAS_LIB` 制造。两种反例库都要编：

| 假 BLAS 导出 | complex64 | complex128 | float32/float64 |
|---|---|---|---|
| `sgemm,dgemm`（无复数） | 4（回落） | 4（回落） | 正常特化 |
| `sgemm,dgemm,cgemm`（**缺 zgemm**） | **特化且数值正确**（改动前是 4） | 4（回落） | 正常特化 |

第二行就是本轮的判据：改成按 dtype 放行后，只带 cgemm 的精简 BLAS 上 complex64
能真跑（自写朴素 cgemm 与 `np.einsum` 逐元素 `allclose`），complex128 仍 fail closed。
改动前同一构件下 complex64 也是 4 —— 没有这一行就证明不了改动生效。

### 17.3 定向分支实跑（此前只测过命中路径，没测分支选择）

用 `PYQCD_EINSUM_DEDICATED_TRACE=1` 取原生 step 行的 `bmnk/pack/trans`，确认**真的
下发了目标分支**再比数值（complex128 vs opt_einsum 固定 path）：

| 形态 | trace（bmnk / trans） | 偏差 |
|---|---|---:|
| `n==1`（`ab,ab->a`） | `5,1,1,900 / 00` | 4.99e-16 |
| `m==1`（`a,ab->b`） | `1,5,1,900 / 10` | 2.23e-15 |
| `k==1` 缩放（`ab,bc->ac`） | `1,6,4,1 / 11` | 0.00e+00 |
| `transA`（`ba,bc->ac`） | `1,6,5,900 / 10` | 0.00e+00 |
| `transB`（`ab,cb->ac`） | `1,6,5,900 / 01` | 0.00e+00 |
| `batch>1`（`bij,bjk->bik`） | `3,6,5,7 / 11` | 1.89e-16 |
| 非恒等输出轴序（`abc,acd->bd`） | `1,5,3,8 / 10`，pack=011 | 0.00e+00 |
| 三算子链（`ab,bc,cd->ad`） | 两步 | 0.00e+00 |
| 标量内积（`i,i->`） | `1,1,1,7 / 00` | 9.63e-17 |

`out=` 原地写入（返回同一对象 + `LAST_INFO['out']=='provided'`）、`k==0` zero-fill
（形状 `(2,0)`、dtype 保留、全零）在 complex64/128 两档均符合。

### 17.4 双精度累加：用"注入灾难性抵消"而不是"看偏差小"

§15.3 用长链放大误差，本轮换成更硬的判据：`1.0` 后跟 `k-1` 个 `1e-8`，**float32
下 `1 + 1e-8` 直接舍回 1**，所以同精度顺序累加会把小项全丢掉。k=4000：

| dtype | 分支 | kernel 偏差 | 同精度逐项累加 | 倍数 |
|---|---|---:|---:|---:|
| complex64 | `n==1` | **5.489e-08** | 3.999e-05 | **728.6x** |
| complex64 | `m==1` | **3.393e-06** | 3.999e-05 | 11.8x |
| complex128 | `n==1` | 2.420e-13 | 2.420e-13 | 1.0x（同精度，预期） |
| complex128 | `m==1` | **8.881e-15** | 2.420e-13 | 27.2x |

complex128 的 `1.0x` 不是没升档：`Accumulator<complex<double>>` 本来就是自身，
同精度累加**就是**升档累加，两者必须相同。判据要按 dtype 分开写，否则会把正确行为
判成失败（本轮先踩了一次：拿 complex128 要求 >10x，直接假 FAIL）。

### 17.5 验收数字（本机，被测 `.so` md5 `6066867f…`）

* `bash cpp/kupl/einsum/make.sh` → **SUCCESS**，`build.log` 里 **0 warning**；
  `pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so` 与 `build/` 逐字节 `cmp` 相同。
* `python -m pyqcd.testing._double_precision_contract --dtype complex128 --sample 100`
  → 特化真跑 **100/100**、失败 **0**、最大偏差 **2.42e-14**（中位 2.44e-16，门槛 1e-12）。
* complex64 抽检 **120** 条（rtol 1e-5）→ 命中 **120/120**、超阈 **0**、最大
  **5.06e-07**、中位 2.28e-07，结果 dtype 逐条仍是 complex64。
* `python -B -m pyqcd.testing` → **118 passed, 1 skipped, 0 failed**（0 FAIL）。
  任务书写的"当前 127 passed"与实测不符：`pyqcd/testing` 的汇总口径按 TESTS 元组
  计数，本轮工作树 HEAD 下实测就是 118（同一命令连跑两次一致），不是本轮改动导致
  ——本轮只动了一个 `.cpp` 的闸门判据与注释，Python 侧零改动。

### 17.6 两条不属本轮范围的既有事实（别改错地方）

1. `python -m pyqcd.testing._complex_precision_contract` 恒有 **1 failed**：
   `test_mixed_complex_and_real_inputs_are_rejected` 要求公共入口 `contract` 显式
   拒绝混合 dtype，而 `_api.py` 沿用 numpy 提升语义。该文件不在本任务授权范围
   （任务书明确"严禁改 `_api.py`"），内核层 `test_mixed_precision_kernel_fails_closed`
   已 PASS，故只记录不动。
2. 并发注意：本轮运行期间**有别的 agent 同时在提交**，我的两处改动被
   `05e75a7`（"十轮巡查脚本 + 复数验收脚本"）一并收进它自己的 commit。核对改动是否
   还在，用 `grep -n "按 dtype 分别放行" cpp/kupl/einsum/src/dedicated_apply.cpp`
   而不是看 `git status`（工作树干净≠改动丢了）。
