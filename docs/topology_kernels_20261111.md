# 拓扑特化内核：`wick_12cases` 2582 条真实收缩的口径与机制

日期：2026-11-11。对应实现：`84e68bb`（拓扑匹配 + 原生拓扑执行入口）、
`d278a9f`（阶段 2 进度归档）。本文记录**语料口径、匹配机制、执行机制、精度策略、
已知限制**；性能数字见第 5 节，全部为 `TBD`，在实现侧补齐前不得引用任何
"拓扑 kernel 更快"的结论。

## 1. 语料来源与计数口径

### 1.1 来源

| 项 | 值 |
|---|---|
| 文件 | `refer/sush/wick_12cases.md` |
| 生成工具 | lqcddb `run_wick_analysis` + `identify_equivalent_diagrams` |
| 调用口径 | `ignore_dis=False`、`use_equivalence=False`、`Projection=False` |
| gamma 处理 | 任意 gamma 名称统一为 `gamma_G`，保留缩并指标，不提供 `gamma_transpose_signs` |
| 情形编号 | 保留原编号 `1–7、9、10、13、14`（原 8/11/12 已删除，不重编号） |
| 每行内容 | `拓扑类`、`图号`、`result_indx`（einsum 方程）、`result_name`（张量名列表）四列 |

`result_indx` 就是该拓扑的收缩方程（操作数指标串 `->` 输出指标串），也是
`pyqcd.einsum.contract` 的输入；`result_name` 给出同名次的操作数类型
（peram / gamma / VVV / VDV）。

### 1.2 汇总与"720"的含义

| 情形 | 原始 Wick 图数 | 拓扑类数 |
|---:|---:|---:|
| 1 | 2 | 2 |
| 2 | 6 | 6 |
| 3 | 120 | 120 |
| 4 | **720** | **720** |
| 5 | 6 | 6 |
| 6 | 24 | 24 |
| 7 | **720** | **720** |
| 9 | 24 | 24 |
| 10 | 120 | 120 |
| 13 | 120 | 120 |
| 14 | **720** | **720** |
| **合计** | **2582** | **2582** |

要点：

* 每个情形都是**单元素拓扑类**（图数 = 拓扑类数），因此 2582 张图就是
  **2582 条唯一拓扑**。实测：2582 条字面方程、2582 个拓扑签名，重复 0 条。
* **720 不是独立来源**：它是情形 4（六夸克多粒子两点）、7（五夸克多粒子三点）、
  14（五夸克到三夸克四点）各自的规模。三者合计 `3 × 720 = 2160` 条，
  占全部 2582 条的 **83.7%**。
* 元数（操作数个数）范围 **6–16**，分布为
  `6:2, 7:6, 9:6, 10:24, 12:24, 13:360, 14:720, 16:1440`；
  没有 8、11、15 元方程。13/14/16 元方程合计 2520 条（97.6%）。
* 输出秩分布为 `0:32, 2:1830, 4:720`（标量、二阶、四阶输出）。

### 1.3 文档里哪些表算语料

`wick_12cases.md` 末尾的 `## 指标说明` 与 `## 动量指标与流分量指标` 是同一批拓扑的
**展开表**（带动量与流分量指标），不是新拓扑。整个文件有 8638 条 `| T...` 记录，
其中前 2582 条是语料，附录 6056 条不计入。两个解析器的切法：

* `pyqcd/testing/_topology_equations_contract.py`：在
  `## 动量指标与流分量指标` 处截断；
* `pyqcd/testing/einsum/topology_corpus.py`：只认 `## <数字>. <标题>` 情形段落，
  遇到附录标记或其它 `## ` 标题即停止。

运行时不读 `refer/`（不随 sdist 发布）。2582 条签名摘要与代表写法固化在生成物
`pyqcd/einsum/_topology_registry.py`（334,725 字节，`N_ENTRIES = 2582`）中。

## 2. 匹配机制

### 2.1 拓扑签名：全式首次出现重编号

`pyqcd.einsum._specialize.canonical_signature(subscripts)` 的定义：

1. 去掉空白；有 `->` 时用右侧字面写法作为输出。无 `->` 时按 einsum 规则补全：
   `...`（若任一操作数含省略号，放最前）+ 只出现一次的指标（按字母序）。
2. 从头到尾扫描**全部操作数**（操作数顺序不变，每个操作数内部的指标顺序也不变），
   再扫描输出；把指标按"全式首次出现顺序"重编号为 `0,1,2,…`。
3. 广播位 `...` 固定记作 `-1`，不参与编号。
4. 输出指标不得重复；非法方程（多重 `->`、空操作数、非法字符、输出含输入之外的
   指标等）返回 `None`，**不抛异常**——调用方据此保持旧的未命中行为。
5. 返回值是 `(操作数0, 操作数1, …, 输出)`，每项是编号元组。

等价与不等价示例（本机实测）：

| 方程 | 签名 | 说明 |
|---|---|---|
| `ij,jk->ik` | `((0,1),(1,2),(0,2))` | 基准 |
| `ij,jb->ib` | `((0,1),(1,2),(0,2))` | 整体改名，同签名 |
| `ab,bc->ac` | `((0,1),(1,2),(0,2))` | 整体改名，同签名 |
| `ij, jk -> ik` | `((0,1),(1,2),(0,2))` | 空白不敏感 |
| `ij,jk` | `((0,1),(1,2),(0,2))` | 隐含输出补全后同签名 |
| `ij,kj->ik` | `((0,1),(2,1),(0,2))` | 第二操作数指标换位，**不同** |
| `ij,jk->ki` | `((0,1),(1,2),(2,0))` | 输出顺序不同，**不同** |
| `jk,ij->ik` | `((0,1),(2,0),(2,1))` | 交换操作数，**不同** |
| `i,j->ij` | `((0,),(1,),(0,1))` | 外积，**不同** |

**边界**：签名只消除"字母命名"和"空白/省略输出"的差异，**不规范化操作数顺序，
也不规范化操作数内部的指标顺序**。`wick_12cases.md` 的方程按固定顺序发射，因此这个
口径足够；但调用方若先交换两个操作数，就不会命中同一条拓扑（见第 6 节）。

### 2.2 三段式 match

`_specialize.match(subscripts)` 依次尝试：

| 段 | 数据 | 规模 | 命中开销与行为 |
|---|---|---:|---|
| 1. 字面 | `REGISTRY` | 9 条 | 去空白后逐字符比较；与历史行为一字不差 |
| 2. 拓扑等价类 | `FAMILIES` | 9 个 | 由 `REGISTRY` 的 9 条字面写法按签名折叠而成 |
| 3. 语料拓扑 | `_topology_registry.TOPOLOGY_DIGESTS` | 2582 条 | 计算签名后取 blake2b-128 摘要查集合 |

命中第 3 段时构造条目：

| 字段 | 值 |
|---|---|
| `name` | `topology:<摘要前 8 位>` |
| `kind` | `kernel(topology)` |
| `kernels` | `("kupl_topology", "cuda_topology")` |
| `note` | 真实语料拓扑（整条计划交给原生拓扑入口；不支持时回落计划缓存） |
| `subscripts` | 该摘要的代表写法（用于"为什么 kernel 没跑"的报错） |

三段都未命中时返回 `None`，`contract()` 照旧走计划缓存路径，行为与旧实现一致。

### 2.3 惰性索引与缓存

| 机制 | 位置 | 作用 |
|---|---|---|
| 签名 LRU 缓存 512 项 | `_signature_of_normalized` | 重复方程不重复解析 |
| 摘要集合惰性导入 | `_specialize._topology_index()` | 首次第 3 段命中才 import `_topology_registry`，避免拖慢 `import pyqcd.einsum` |
| 条目缓存 ≤4096 | `_TOPOLOGY_ENTRY_CACHE` | 摘要 → 条目只构造一次 |
| 代表写法按需读取 | `REPRESENTATIVES` | 报错信息只在需要时取 |

摘要口径（必须与生成物一致）：

```python
payload = "|".join(",".join(map(str, term)) for term in signature)
digest = hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()
```

即：先 `canonical_signature`，再按上式做 blake2b-128 十六进制。

## 3. 执行机制

### 3.1 从 `contract()` 到原生入口

1. `contract()` 调 `_specialize.match(subscripts)`；只有 `out is None` 且专用路径
   启用时才尝试 kernel（`out=` 非空只打印 `[plan-cache]`）。
2. `_api._run_specialized_kernel` 按**固定顺序**尝试，第一个非 `None` 的结果胜出：

   | 顺序 | 标签 | 形态 |
   |---:|---|---|
   | 1 | `kupl_nucleon_2pt` | 核子 2pt（KUPL） |
   | 2 | `kupl_3pt` | sush 3pt 主项（KUPL） |
   | 3 | `cuda_nucleon_2pt` | 核子 2pt（CUDA） |
   | 4 | `cuda_3pt` | sush 3pt 主项（CUDA） |
   | 5 | `kupl_topology` | **整条语料拓扑计划（KUPL）** |
   | 6 | `cuda_topology` | **整条语料拓扑计划（CUDA）** |

   失败不抛异常：各 kernel 返回 `None` 并写自己的 `LAST_FAILURE`，上层汇总原因后
   回落。命中提示里的标签是**实际执行成功**的那个。
3. Python 侧计划：`_kernel_kupl._topology_plan` / `_kernel_cuda._topology_plan`
   调 `contract_path(subscripts, *operands)`，再用 `encode_execution_plan` 序列化为
   **version-2 计划字节流**。缓存键是 `(去空白方程, 各操作数形状, dtype.str)`，
   每个后端模块各 32 项 FIFO；同形状的第二次调用不再规划。
4. 原生入口：
   * KUPL：`qcd_kupl_topology_apply`（`cpp/kupl/einsum/src/topology_apply.cpp`）；
   * CUDA：`qcd_cuda_topology_apply`（`cpp/cuda/einsum/src/apply_contract.cu`）。
5. 入口先校验头部（9 个 int32，共 36 字节）：
   `magic / version / total_bytes / num_inputs / num_steps / num_symbols / output_ndim /
   dtype_id / reserved`。`total_bytes` 必须等于实际字节数，`num_inputs` 必须等于
   `n_operands`，`dtype_id` 必须与调用方一致，`reserved` 必须为 0；任一不符即
   `QCD_STATUS_INVALID`。
6. 解析计划：单步 arity 只能是 1 或 2；同一操作数内不得重复指标；不得有尾随字节。
   超出该模型（arity > 2、重复指标、尾随字节等）返回 `QCD_TOPOLOGY_UNSUPPORTED`
   （fail closed，不猜测）；空输入/空步骤同样归入"本入口不支持"。
7. 执行步表：按步做一元（copy / permute / reduce）或二元（pack + GEMM），
   擦除已消费的输入槽、压入中间体，最后把唯一存活槽 `memmove` 到输出。

### 3.2 per-拓扑预编译步表：目标与现状

**目标形态**：每条拓扑只解析/编译一次，得到一张步表——每步记录输入槽位置、
输入/输出标签、`sizes`、打包 stride 与 GEMM 的 `(batch, m, n, k)`。执行器只接收
指针，按同一张步表跑任意数值；步表里没有符号查找，也没有计划解析。

**现状（必须区分）**：

| 后端 | 现状 | 证据 |
|---|---|---|
| CUDA | 已按计划字节 + dtype 缓存 `TopologyProgram`（parsed plan + resolved steps + prepared steps + workspace）；命中即跳过解析与 prepare | `apply_contract.cu:1878-1994` |
| KUPL | **每次调用都重新** `parse_execution_plan` 后执行，没有缓存层 | `topology_apply.cpp:107-127` |

CUDA 的缓存是**单槽**：`topology_program()` 是静态单例，计划字节一变就覆盖
（`matches()` 用 `memcmp` 比较整份计划）。因此扫描 2582 条拓扑会反复 prepare，
不能把这种扫描当稳态。

另有一条更激进的路线：`pyqcd/testing/_topology_kernel_gen.py` 为每条拓扑生成
**静态展开的 C++ 函数**（形状、索引映射、收缩顺序全部编译期常量），输出
`cpp/kupl/einsum/src/topology_kernels.inc`（2582 个 `topo_XXXX`，约 13 MB，
已被 `.gitignore` 忽略）。目前只有开发期自验驱动
`cpp/kupl/einsum/tests/topology_kernels_verify.cpp` 消费它，**未编入生产库**；
其生成物只展开 `float32` 变体。它不代表已落地的性能路径。

### 3.3 参数化 GEMM 执行器

二元步统一规范化为行主序的

```text
A[batch, M, K] × B[batch, N, K]^T -> C[batch, M, N]
```

* **KUPL**：动态绑定 `cblas_sgemm / dgemm / cgemm / zgemm`（优先 ILP64 的
  `scipy_cblas_*gemm64_` 变体），按 dtype 选变体，按 batch 循环；打包与转置由步表
  决定。见 `apply_contract.cpp:328-385`（绑定）与 `:883-976`（四种 dtype 的分派）。
* **CUDA**：`qcd_cuda_batched_gemm`（cuBLAS），按 stream 分片；一元步走各自的
  provider（permute / reduce / hadamard）。入口先用 `providers_available` 检查所需
  provider 是否齐全，缺一个就返回 `QCD_STATUS_INTERNAL`，不会跑到一半才发现。
  见 `apply_contract.cu:1850-1876`（provider 检查）、`:1541`（batched GEMM 调用）。

工作树中正在落地的 per-拓扑参数化执行器（`cpp/kupl/einsum/src/topology_gemm.h`，
尚未接线）按同一 `[batch, M, K] × [batch, K, N]` 口径做打包 / GEMM / 解包，
只实例化 `float` 与 `double` 两条路径；complex 一律交回通用路径，**不隐式降精度**。

### 3.4 命中提示

| 提示 | 含义 |
|---|---|
| `[kernel: kupl_topology]` | 整条计划由 KUPL 拓扑入口执行成功 |
| `[kernel: cuda_topology]` | 整条计划由 CUDA 拓扑入口执行成功 |
| `[kernel: kupl_nucleon_2pt]` / `[kernel: kupl_3pt]` 等 | 命中的是排在前面的形态专用 kernel，不是拓扑入口 |
| `[plan-cache]` | 命中了形态，但没有专用 kernel 成功执行（`out=`、缺库/缺符号、混合 dtype、计划超出 v2 模型、kernel 内部报错）；只复用计划，数值不变 |

默认每个形态名每个进程只打印一次；`PYQCD_EINSUM_SPECIALIZE_VERBOSE=1` 可每次打印。

## 4. 精度策略

1. **默认单精度，且只有显式声明的 dtype 才下发**。两个适配模块各自声明
   `TOPOLOGY_DTYPES = (complex64, float32)` 与
   `TOPOLOGY_DOUBLE_DTYPES = (complex128, float64)`：
   * 单精度恒可下发（默认与首选）；
   * 双精度需原生**实测通过**：`_probe_double` 用固定 2×2 两算子矩阵乘跑一次，
     形状、dtype、`1e-9` 相对误差三项都对才置为可用（每进程每 dtype 只探测一次）；
     否则 `try_topology_apply` 返回 `None`，上层回落通用路径；
   * 其它 dtype 直接返回 `None`。
2. **混合 dtype 一律拒绝**。`encode_execution_plan` 在
   `len(set(operand_dtypes)) != 1` 时抛
   `ValueError: mixed precision is not supported`；两个 `try_topology_apply` 也在
   进入原生前检查所有操作数 dtype 完全相同。
3. **`out` 与 `offsets` 也有 dtype/布局契约**：`out` 必须与操作数同 dtype、C 连续
   （CUDA 还要求同设备）；`offsets` 必须为 `None`——numpy/cupy 视图的数据指针已经
   带字节偏移，再叠加会双重偏移并静默读错数据，因此非 `None` 一律 fail closed。
4. **原生头按 dtype 校验**。`header_matches` 要求 `header.dtype_id == dtype_id`，
   不一致返回 `QCD_STATUS_INVALID`，不存在"按 double 读 float"的隐式转换。
5. **GEMM 按 dtype 选变体**：`complex128 → zgemm`、`complex64 → cgemm`、
   `float64 → dgemm`、`float32 → sgemm`（KUPL）；CUDA 同四档。
6. 结论：double 要么走 double 变体（探测通过），要么回落通用路径——
   **绝不隐式降精度**。每次成功下发都会在 `LAST_INFO` 里记录实际交给原生的参数
   （plan 字节数/步骤数、操作数个数、输出形状、`out` 是否调用方提供、哪些操作数被
   复制成连续、offsets）；回落时该字典为空，可据此区分"跑过原生"与"根本没跑"。
7. 未接线的两条实验路线都不参与上述 dtype 契约：静态生成内核
   （`topology_kernels.inc`）只展开 `float32`；per-拓扑参数化执行器
   （`topology_gemm.h`）只实例化 `float` / `double`，complex 交回通用路径。

## 5. 实测（占位，待补）

> 本节所有单元格都是 `TBD`。数字由实现代理在修复 KUPL 执行器后补齐；补齐前
> 不得在文档、发布说明或报告中引用任何拓扑 kernel 的倍率。

### 5.1 口径

| 项 | 要求 |
|---|---|
| 对比对象 | ① 拓扑 kernel（默认 `contract`，命中 `[kernel: kupl_topology]`）② 通用路径（`specialize=False`）③ `opt_einsum.contract(..., optimize="dp")` |
| dtype | 单精度与双精度分开报；同一行三方必须同 dtype |
| 线程 | 固定 `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS`；三方同档 |
| 计时 | 预热 ≥1 次；3 轮交错（A/B/C/B/A…），取每轮比值中位数后相除 |
| 正确性 | 每个用例先与 `numpy.einsum` 对照（相对最大范数），再计时 |
| 必须记录 | 方程、元数、逻辑维度 `dim`、dtype、实际命中标签；只写 `dim` 不写方程等于没有口径 |
| 规模限制 | 16 元拓扑在 `dim=16` 下元素数与 FLOPs 都会爆炸，通常无法在单机完成；能跑的组合必须显式注明 `dim` 与方程 |

`specialize=False` 同时关闭该调用的程序缓存，因此通用路径列**包含规划开销**；
若要把规划剥离，另记 `contract_path` 的单独耗时并在表注中说明。

### 5.2 表（待补）

| dim | 拓扑 / 元数 | kernel | 通用路径 | `opt_einsum(dp)` | kernel / 通用 | kernel / opt |
|---:|---|---:|---:|---:|---:|---:|
| 4 | TBD | TBD | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD | TBD | TBD |
| 16 | TBD | TBD | TBD | TBD | TBD | TBD |

### 5.3 语料级基线命令

```bash
# 正确性 + 每情形耗时（--bench 时打印每情形中位数与总量）
python -m pyqcd.testing.einsum.verify_topology --case 1 --limit 1 --size 2 --bench --jobs 1

# 全语料正确性（默认 complex64，尺寸 2/3/4 轮换；耗时较长）
python -m pyqcd.testing.einsum.verify_topology --size "2 3"
```

`verify_topology` 的默认参考是"把操作数升到 double 后 numpy 的结果"（`--reference
upcast`），同 dtype 的 numpy 结果也会打印；14–16 元拓扑的 float32 累加误差可达
`~1e-5`（相对最大输出元素），这是参考实现自身的误差，不是被测路径的错误。

## 6. 已知限制与未完成项

| # | 项 | 现状与证据 |
|---:|---|---|
| 1 | **KUPL 尺寸断崖（方程相关）** | 语料 #194（16 元）在 `size=4`：通用路径 **2 ms**、拓扑 kernel **>90 s（超时）**；`size=3` 两者均约 4 ms。计划只有 13 步、`opt_cost` 4.0e5，理论微秒级。见 `logs/v202611112059.txt`。**不是所有 16 元方程都慢**：内核契约按 21 条等距抽样的 120 条 `dim=4` 方程全部命中，最大相对偏差 `2.1e-6`（2026-11-11 本机），#194 属个例。 |
| 2 | 断崖位置 | TRACE 显示 13 个 step 均 0.01–0.1 ms（累计约 0.2 ms），耗时在步循环之外；已把嫌疑缩到"每次调用重新 `parse_execution_plan`"，**尚未最终证实**。见 `logs/v202611112110.txt`。 |
| 3 | 修复方向 | 按计划哈希缓存 `ExecutionPlan`，再复用 CUDA 的 resolved/prepared 步表设计；修复后用 `size=4/8` 对齐 2 ms 基线。未完成。 |
| 4 | CUDA 单槽缓存 | 换一条拓扑就覆盖上一份已编译计划；整语料扫描会反复 prepare。未做多拓扑表。 |
| 5 | 静态生成内核未接线 | `_topology_kernel_gen.py` → `topology_kernels.inc`（2582 个 `float32` 静态函数）只被开发期 `topology_kernels_verify.cpp` 使用，未进生产库，未做性能测量。 |
| 6 | 摘要再生成脚本缺失 | `_topology_registry.py` 的 docstring 指向 `-m pyqcd.testing._topology_registry_regen`，该模块**不在仓库中**；目前只能按 2.3 节的摘要口径重算。 |
| 7 | 签名不规范化顺序 | 操作数顺序、操作数内指标顺序都进签名；`jk,ij->ik` ≠ `ij,jk->ik`。语料按固定顺序发射，故不影响 2582 条命中。 |
| 8 | `out=` 绕过专用 kernel | `contract(..., out=buf)` 只打印 `[plan-cache]`，不会调用拓扑入口。 |
| 9 | 计划缓存按字面拼写 | `_topology_plan` 的键是调用方写的方程，不是拓扑签名；同一拓扑的改名写法各占一个 32 项 FIFO 槽。 |
| 10 | `contract_expression` 不走拓扑入口 | 与既有专用 kernel 的限制一致。 |
| 11 | CUDA 入口不接受 stream | `topology_entry` 在 `stream != nullptr` 时返回 `QCD_STATUS_INVALID`；多流拓扑执行未实现。 |
| 12 | 未验证项 | 多 rank/多流拓扑基准、CUDA 实机数字、double 全语料扫描、静态内核正确性全量比对。 |

## 7. 复现

```bash
# 1) 拓扑匹配与 2582 条数值契约（改名/结构负例、空白/省略输出）
python -m pyqcd.testing._topology_equations_contract

# 2) 内核级契约：dtype 声明与四类 dtype 矩阵、plan/指针/out/offsets、
#    120 条 dim=4 语料抽检（对 specialize=False）、失败回落只告警一次
python -m pyqcd.testing._topology_kernel_contract

# 3) 真实语料抽样：正确性（可加 --bench 出每情形耗时）
python -m pyqcd.testing.einsum.verify_topology --case 1 --limit 2 --size 2

# 4) 语料统计与签名唯一性
python -c "from pyqcd.testing.einsum import topology_corpus as c; r=c.load_corpus(); print(c.format_statistics(c.corpus_statistics(r))); print(c.corpus_problems(r))"

# 5) 单条方程的命中标签（每次调用都打印）
PYQCD_EINSUM_SPECIALIZE_VERBOSE=1 python -c "import numpy as np, pyqcd.einsum as pe; r=np.random.default_rng(0); eq='cadb,menf,ac,em,bd,fn->'; ops=[r.standard_normal((2,)*len(t)) for t in eq.split('->')[0].split(',')]; print(pe.contract(eq, *ops).shape)"
```

第 5 条应打印
`[pyqcd.einsum] 专用路径: topology:350a4b59 [kernel: kupl_topology] ...`
（双精度探测通过时；若该机探测未通过会打印 `[plan-cache]`，把操作数改为
`astype(np.float32)` 可复验单精度路径）。把 `PYQCD_EINSUM_SPECIALIZE_VERBOSE`
去掉则每个形态每进程只打印一次。成功下发时 `_kernel_kupl.LAST_INFO` 会给出
`plan_bytes / steps / n_operands / output_shape / out / copied / offsets`。

## 8. 相关文件

| 角色 | 文件 |
|---|---|
| 语料 | `refer/sush/wick_12cases.md` |
| 签名与三段匹配 | `pyqcd/einsum/_specialize.py` |
| 语料摘要索引（生成物） | `pyqcd/einsum/_topology_registry.py` |
| Python 侧原生适配 | `pyqcd/einsum/_kernel_kupl.py`、`pyqcd/einsum/_kernel_cuda.py` |
| 分派顺序 | `pyqcd/einsum/_api.py` |
| KUPL 原生入口 | `cpp/kupl/einsum/src/topology_apply.cpp` |
| CUDA 原生入口与计划缓存 | `cpp/cuda/einsum/src/apply_contract.cu` |
| 契约 | `pyqcd/testing/_topology_equations_contract.py`、`_topology_kernel_contract.py` |
| 语料工具与校验 | `pyqcd/testing/einsum/topology_corpus.py`、`verify_topology.py` |
| 静态内核生成器（未接线） | `pyqcd/testing/_topology_kernel_gen.py` |
