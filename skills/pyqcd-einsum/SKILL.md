---
name: pyqcd-einsum
description: Use when implementing, debugging, or benchmarking pyqcd.einsum against opt_einsum, especially native CUDA/KUPL execution, contraction-plan ABI, distillation workloads, vectorization, or CUDA streams.
---

# pyqcd-einsum - 原生兼容收缩引擎

## 目的与边界

本技能维护 `pyqcd.einsum` 的兼容 API、收缩规划、原生 ABI、CUDA/KUPL 执行、
蒸馏实战基准与性能门禁。它不定义 LapH/Wick 物理语义，也不替代 `pyqcd-infra`
对 backend、显存和计时平台的统一契约。

状态词：

- `implemented`：接口或文件存在。
- `tested`：有当前命令、数值断言或 ABI 输出。
- `performance-validated`：同后端、同 dtype、同输入下稳定超过目标倍率。
- `unverified`：没有本次可复现证据，不得由其他后端的成功外推。

## 按需参考

- 修改原生计划、ABI、CUDA/KUPL 或性能时，读
  [references/native-implementation.md](references/native-implementation.md)。
- **动构建、性能测量或接线专用 kernel 之前必读**
  [references/measured-pitfalls.md](references/measured-pitfalls.md)：
  记录了 2026-11-08 会话中实测踩到的构建坑（CUDA 架构默认值、同名头文件冲突）、
  测量坑（预热不足、对照变成 kernel 比 kernel、用现编 .so 冒充生产库、逐元素误差误判）、
  正确性坑（输出全 1 维未写、设备 kernel 缺连续性校验、异常不回退、开关未进缓存键）
  与已知限制（`contract_expression` 不走 kernel、E4 融合仅极小规模生效）。
- 只做 API/物理缩并语义时，沿用 `pyqcd-distillation` 与
  `pyqcd-infra`，不要复制其完整规则。

## 强制工作流

1. **先固定兼容面**：以 `pyqcd/einsum/AGENTS.md` 和固定 opt_einsum 版本为
   oracle。规划路径、代价、`PathInfo`、`contract_expression` 与错误行为分别验证。
2. **先规划后执行**：生产路径必须复用 contraction list；禁止每次调用重新在 Python
   中展开所有步骤。固定形状优先使用 `contract_expression` 或等价原生计划缓存。
3. **K 轴最低维**：二元收缩统一 pack/transpose 为
   `A[B,M,K] x B[B,N,K]^T -> C[B,M,N]`。K 必须连续，pack/reduce/output kernel
   必须保证相邻线程或 SIMD lane 访问相邻 K 元素。
4. **一次原生调用**：Python 只负责生成一次计划、校验 dtype/连续性/设备和传入指针；
   多步收缩在 C++/CUDA 内完成。中间结果与工作区必须复用，避免逐步 Python dispatch。
5. **后端独立**：KUPL 不链接 CUDA，CUDA 不调用 PyQCU 库；两边共享 ABI 与计划格式，
   数值实现保持独立。生产代码不得 import `refer/` 或 `pyqcd.testing`。
6. **并行分层**：CUDA 为单 device context → 多 stream 批任务；KUPL 为多进程/OpenMP →
   SIMD。跨进程拆分与结果聚合由调用方负责，引擎本身不建通信域、不做 collective。
7. **性能门禁**：CPU 对 CPU、GPU 对 GPU，固定 dtype、线程、设备、输入、规划与
   stream；预热后取至少 10 次中位数。目标为 opt_einsum 的 3x，且结论需多轮复测。
   低于目标必须返回非零，不能只打印 warning。
8. **失败关闭**：重复轴、混合精度、非连续输入、越界尺寸、缺 kernel、CUDA/cuBLAS
   错误必须显式失败或安全回退；不得把零输出或缺步骤当作成功。

## 当前已验证经验

- 共享执行计划 v2 已 `implemented, tested`：header、对齐尺寸表、初始 shape、
  逐步 positions/labels 与最终输出共同描述 contraction list。
- K-lowest 布局分析已由
  `cpp/kupl/einsum/tests/test_runtime.cpp` 的 C++ 合约测试覆盖。
- KUPL 四 dtype、batch、广播、一元 reduce/permute 和多步收缩已 `tested`；
  全量计数以 `python -B -m pyqcd.testing` 的实际输出为准。
- 2026-11-12 新增并接入统一套件：
  `python -B -m pyqcd.testing.einsum.verify_python_hotpath`（全语料匹配/计划缓存）、
  `python -B -m pyqcd.testing.einsum.verify_plan_gate`（负缓存、计划统计、
  CPU branch-2 条件提升）。
- 最终 full-scene 实测：CPU 六条 dedicated/generic 为 `0.459–0.982x`，其中
  `seq=6585` 22.445 s→10.310 s；GPU 四条轻中量路径为 `0.157–0.265x`，
  `seq=129/6585` 由超大计划门安全回退 generic。冻结库哈希、快照与完整证据见
  `docs/einsum_production_optimization_20261112.md`。
- 额外 43 条分层 full-scene 路径：CPU 中位 `0.861x`、42/43 不快于 generic；
  GPU 中位 `0.276x`、42/43 不快于 generic。确认普遍加速，同时允许
  CPU `seq=856`（1.007x）与 GPU 极小计划 `seq=6911`（1.362x）两个低收益反例。
- CUDA 编译、动态导出与 Cython 加载已 `implemented`，但一般 cuBLAS GEMM 路径仍
  `unverified`，不得用逐元素路径通过代替 batch GEMM 通过。
- 开发窗口 CPU T0 四条方程约 `3.0x-18x`，CUDA V100/P100 E5 约
  `3.49x-4.07x`；但固定 oracle、进程隔离、MP 与目标系综仍未全部闭合，
  因此只能标为 provisional，不能升级为 `performance-validated`。

## 性能调试与路径选择（2026-11-08 实测）

`contract()` 用五道门决定"谁在算"。走错门**不改数值、只改耗时**，症状永远是"结果对但慢"。
排错顺序固定：先 `--gpu-probe`，再用分支探针确认方程落在哪条门，最后才看计时。

### 委派与门限规则（四条委派 + 一条融合体积门）

门顺序（`pyqcd/einsum/_api.py::contract`）：专用 2pt/3pt kernel → `cupy` 委派 →
无求和标签 → 计划 + 原生执行器（`_try_native_contract`）→ Python 分步。
两道委派都在**规划之前**返回，连契约解析的记账也一起省掉。

| # | 形态 | 判据 | 谁执行 | 实测依据 |
|---|---|---|---|---|
| 1 | 无求和标签：逐元素/转置/对角/迹/外积 | `_is_elementwise_equation(eq)` | backend 原生 `einsum` | Hadamard 4000² 89 → 30 ms（0.33× → 1.00×） |
| 2 | `numpy` + batch 标签 | `_has_batch_index(eq)` | KUPL 原生计划执行器 | 批量 32×1024² **11.6×**、32×256² **10.6×**（对 opt_einsum） |
| 3 | `numpy` 无 batch 标签 | 上条为 False | numpy 分步（tensordot） | 原生分步慢 6–10%，故主动不启用 |
| 4 | `cupy` 非融合形态 | `_cupy_operands` 且 `not _cuda_fused_shape` | `cupy.einsum`（cuBLAS） | 16×1024² 原生批量 GEMM **0.21×** → 委派后 **0.98×** |
| 5 | `cupy` γ 夹心 | `_cuda_fused_shape`（只认 `ab,...bce,cd->...dae`） | CUDA 原生融合 kernel | 512×512 16.2×；8192×1024 2.0×（2.1 GB 工作区带宽封顶） |

**一条命令看清某方程落在哪条分支**（`_backend_einsum` 与 `_try_native_contract` 都是模块级
名字，可直接包一层；只在调试时这么做，别进生产代码）：

```bash
cd /home/zhangxin/PyQCD
python - <<'PY'
import numpy as np, pyqcd.einsum._api as api
log = []
be, native = api._backend_einsum, api._try_native_contract
api._backend_einsum = lambda n, s, o, out=None: (log.append(("backend", n)), be(n, s, o, out))[1]
def spy(o, c, b, out, kw, **extra):
    handled, res = native(o, c, b, out, kw, **extra)
    log.append(("native", b, extra.get("batch_like"), handled)); return handled, res
api._try_native_contract = spy
rng = np.random.default_rng(0); r = lambda *s: rng.standard_normal(s)
for label, eq, ops in [
        ("hadamard", "ij,ij->ij", [r(200, 200), r(200, 200)]),
        ("numpy batch", "bij,bjk->bik", [r(4, 64, 64), r(4, 64, 64)]),
        ("numpy 2d chain", "ij,jk->ik", [r(256, 256), r(256, 256)]),
        ("numpy double contract", "abc,bcd->ad", [r(8, 16, 24), r(16, 24, 32)])]:
    log.clear(); api.contract(eq, *ops); print(f"{label:22s} {log}")
PY
```

期望（逐行，本机实跑）：

```text
hadamard               [('backend', 'numpy')]
numpy batch            [('native', 'numpy', True, True)]
numpy 2d chain         [('native', 'numpy', False, False)]
numpy double contract  [('native', 'numpy', False, False)]
```

读法：`backend` = 委派给 backend 原生 `einsum`；`native … True` = 原生执行器接手；
`native … False` = 该门拒绝、落到 Python 分步。把操作数换成 `cupy` 后，批量形态应打印
`[('backend', 'cupy')]`。

### 规则 1：无求和标签 → backend 原生 einsum

- **判据**：`_is_elementwise_equation(eq)` 为真——有显式 `->`、无 `...`，且
  `set(输入标签) - set(输出标签)` 为空；单操作数一律算（转置/对角/迹/缩放）。
  `i,j->ij`（外积）同样命中，它也没有求和标签。
- **命令**：`python -c "import pyqcd.einsum._api as a; print(a._is_elementwise_equation('ij,ij->ij'), a._is_elementwise_equation('ii->i'), a._is_elementwise_equation('ij,jk->ik'), a._is_elementwise_equation('i,j->ij'))"`
- **期望**：`True True False True`。若第一项是 `False`，说明 `->` 被吞掉或用了隐式输出，
  委派不会发生（隐式输出刻意留给规划器）。
- **数值口径**：Hadamard 4000² 从 89 ms（分步执行）降到 30 ms；本机 float64 复测
  `pyqcd 29.3 ms vs numpy 28.2 ms`（0.95×）。**一旦出现 0.33×，就是这条规则被回退了。**

### 规则 2：numpy + batch 标签 → KUPL 原生计划执行器

- **判据**：`_has_batch_index(eq)` 为真——某标签被 **≥2 个操作数共享且出现在输出**；
  含 `...` 恒为真；隐含输出（无 `->`）为假。**不要用 `ndim>=3` 代替**：`abc,bcd->ad`
  是三维却没有 batch 轴，那里原生执行器反而慢（`_batched_operands` 只在方程文本缺失时兜底）。
- **命令**：`python -c "import pyqcd.einsum._api as a; print(a._has_batch_index('bij,bjk->bik'), a._has_batch_index('abc,bcd->ad'), a._has_batch_index('...ij,...jk->...ik'))"`
- **期望**：`True False True`；分支探针同时给出 `('native', 'numpy', True, True)`。
- **实测**：批量 32×1024² 11.6×、32×256² 10.6×、ellipsis 7.1×（均对 `opt_einsum(dp)`）。
- **注意对手**：同一方程对 `np.einsum(..., optimize='optimal')` 本机只有 0.61×（32×256²）
  和 0.89×（16×1024²）——numpy 自己会逐 batch 调 BLAS。倍率必须写清是跟谁比。

### 规则 3：numpy 无 batch 标签 → numpy 分步（不启用原生）

- **判据**：`_has_batch_index` 为假且未设 `PYQCD_EINSUM_KUPL_PLAN=1` 时，`_try_native_contract`
  直接返回 `(False, None)`；分支探针打印 `('native','numpy',False,False)`。
- **凭据**：CPU 计划执行器在纯 2D 链/矩阵乘上比 numpy 的 tensordot 阶梯慢 6–10%（打包/置换
  没有额外收益）；生产尺度 matmul 4600² 1.01×、chain 0.98×、complex 3100² 0.96×——持平但
  不回退，就是这条门的收益。
- **强制启用**（只用于对照实验）：`PYQCD_EINSUM_KUPL_PLAN=1`，此时同一条探针应打印
  `('native','numpy',False,True)`；跑完必须撤掉，否则 6–10% 的税会回到所有 2D 方程上。

### 规则 4：cupy 非融合形态 → `cupy.einsum`（cuBLAS）

- **判据**：`_cupy_operands(operands) and not _cuda_fused_shape(eq)`，且 `out is None`、无 kwargs。
  `python -c "import pyqcd.einsum._api as a; print(a._cuda_fused_shape('bij,bjk->bik'))"` → `False`
  才会委派；委派统一带 `optimize=True`（backend 不支持时自动去掉重试）。
- **凭据**：原生批量 GEMM 在 16×1024² 上比 cuBLAS 慢 5 倍（**0.21×**）；委派后 **0.98×**（R33）。
- **反作用**：`out=` 或任何 kwargs 会**绕过委派**回到原生执行器——这也正是 E4 融合 kernel 现在
  唯一的可达路径（见"已知限制"）。

### 规则 5：cupy γ 夹心 → CUDA 原生融合 kernel（带体积门）

- **判据**：`_cuda_fused_shape(eq)` 只认 `_CUDA_FUSED_EQUATIONS = ("ab,...bce,cd->...dae",)`；
  `equation=None` 时保守返回 `True`。C++ 侧另有体积门 `kE4FusedMaxSpatialVolume = 1`（E4）：
  `z·y·x > 1` 直接不匹配。
- **命令**（trace 是唯一证据，`[plan-cache]` 不算）：
  ```bash
  PYQCD_EINSUM_TRACE=1 python - <<'PY'
  import cupy as cp, pyqcd.einsum as pe
  rng = cp.random.default_rng(0)
  g = lambda *s: (rng.standard_normal(s) + 1j*rng.standard_normal(s)).astype(cp.complex128)
  print(pe.contract("ab,...bce,cd->...dae", g(4, 4), g(2, 4, 4, 8), g(4, 4),
                    optimize="dp", backend="cupy").shape)
  cp.cuda.Device(0).synchronize()
  PY
  ```
- **期望**：
  ```text
  [pyqcd.einsum] 专用路径: gamma-sandwich(E5) [plan-cache] ...
  PYQCD_EINSUM_CUDA_WORKSPACE mode=sandwich elements=256 streams=1 mallocs=1
  PYQCD_EINSUM_CUDA_SANDWICH prefix=2 left=4 middle=4 right=4 lane=8 kernel=1
  (2, 4, 4, 8)
  ```
  只有 `mode=sandwich` + `kernel=1` 才算融合；出现 `PYQCD_EINSUM_CUDA_GEMM ...` 而无
  `..._SANDWICH`，说明该形态没进融合、走的是多步 GEMM。trace 走 stderr，与 stdout 的
  打印顺序可能交错，别按行号对齐判断。
- **凭据**：512×512 16.2×；到 8192×1024 只剩 **2.0×**——2.1 GB 工作区带宽封顶，不是算术瓶颈。
  所以"融合 kernel 有 16×"只在中小体积成立，别写成生产结论。
- **E4 的体积门代价**（`z·y·x` = 8/27/64/512 时分别慢 1.4/9.7/22/26×，L24 的 13824 慢 33×）
  说明门槛必须留在 C++ 里，不能靠"形态命中就融合"。

### 拓扑匹配：基础表 2582 条到当前语料 8638 条（2026-11-12 更新）

**匹配口径**：`_specialize.canonical_signature` 按"全式首次出现顺序"把指标重编号为
`0,1,2,…`（`...` 固定 `-1`）：先扫描全部操作数（操作数顺序、操作数内指标顺序都
不变），再扫描输出；无 `->` 时按 einsum 规则补隐含输出；非法方程返回 `None` 而不抛
异常。因此 `ij,jk->ik`、`ij,jb->ib`、`ab,bc->ac` 同签名
`((0,1),(1,2),(0,2))`；`ij,kj->ik`、`ij,jk->ki`、`jk,ij->ik` 都不同。

**`match()` 只有一段：拓扑签名查语料索引**（`pyqcd/einsum/_specialize.py`）。
旧的「字面 `REGISTRY`（9 条）→ `FAMILIES` → 语料」三段式**已删除**——那张字面表
（`gauge-propagator(E4)`、`gamma-sandwich(E5)`、`sush-2pt(S1)`、`sush-3pt-munu(S2)`、
`sush-2pt-mu(S3)`）是退役专用内核的遗留项，**不在** wick_12cases 语料内，留着会让
人误以为支持的特化路径只有 5 条。当前任务表共 8638 条路径，全部由拓扑入口服务，
未命中即语料之外的方程。

摘要口径必须与生成物一致：先 `canonical_signature`，再对
`"|".join(",".join(map(str, term)) for term in signature)` 做 blake2b-128 十六进制。

**命中/回落判据**（只认实际执行分支，不认注册表）：

| 提示 | 含义 |
|---|---|
| `[kernel: kupl_topology]` | 整条 v2 计划由 `qcd_kupl_topology_apply` 执行成功 |
| `[kernel: cuda_topology]` | 同一入口的 CUDA 实现执行成功 |
| `[kernel: kupl_nucleon_2pt]` / `[kernel: kupl_3pt]` 等 | 实际跑的是分派顺序更靠前的形态专用 kernel，不是拓扑入口 |
| `[plan-cache]` | 命中了形态但没有专用 kernel 成功执行：`out=` 非空、缺库/缺符号、操作数混合 dtype、计划超出 v2 单步 arity≤2 模型、或 kernel 内部报错；只复用计划，数值不变 |

复核方式：`PYQCD_EINSUM_SPECIALIZE_VERBOSE=1` 看每次调用的标签；或先调用再读
`_kernel_kupl.LAST_FAILURE`（**必须在调用之后读**，早读是上一次的残留值；且要按
`_api._kernel_labels_for` 过滤，否则会把别的形态的原因报给当前方程）。成功下发时
`LAST_INFO` 记录实际交给原生的参数（plan 字节数/步骤数、操作数个数、输出形状、
`out` 来源、连续性复制、offsets），回落时为空——用它区分"跑过原生"与"根本没跑"。

**新增一条拓扑内核**：

1. 语料侧：在 `refer/sush/wick_12cases.md` 增加 `result_indx` 行，然后重跑
   `_wick_tasklist` / `_dedicated_step_tables` / `_topology_registry` 三个生成器。
   不再有字面注册表可加——新增路径只能通过语料进入。
2. 语料摘要：`_topology_registry.TOPOLOGY_DIGESTS` 必须含该签名的摘要，
   `REPRESENTATIVES` 必须含一条代表写法；按上面的摘要口径重算。**注意**：registry
   的 docstring 指向 `-m pyqcd.testing._topology_registry_regen`，该模块目前不在仓库
   中（见"常见坑"），只能用同一口径手工重算。
3. 声明标签：`_KERNEL_LABELS`（按形态名关键词）或条目的 `kernels` 字段；
   `_api._kernel_labels_for` 必须与之一致，`_specialize._selfcheck` 会检查防漂移。
4. 原生实现：KUPL 走 `_kernel_kupl.try_topology_apply`，CUDA 走
   `_kernel_cuda.try_topology_apply`。契约是"成功返回结果 / 失败返回 `None` 并写
   `LAST_FAILURE`"，绝不把异常抛给调用方；dtype 必须全局一致。
5. 验证：`_topology_equations_contract`（基础表 2582 条 + 改名同签名 + 结构负例）、
   `_topology_kernel_contract`（dtype 声明与四类 dtype 矩阵、plan/指针/out/offsets、
   120 条 `dim=4` 语料抽检、失败回落只告警一次）、
   `python -m pyqcd.testing.einsum.verify_topology --case N --limit M`（真实语料抽样
   对 numpy）。

**常见坑**：

- **交换操作数不是同一拓扑**：`jk,ij->ik` ≠ `ij,jk->ik`；语料按固定顺序发射，调用前
  不要重排操作数。
- **交换一个操作数内的两个指标是改结构**（语义是转置），签名必变；不要当"改名"。
- 改名字母、加空白、省略输出可命中；省略输出只在 numpy 语义一致时等价。
- `[plan-cache]` 不代表没命中，只代表没有 kernel 执行；反过来，`[kernel: ...]` 的标签
  也可能是 2pt/3pt 而不是 topology。
- `out=` 会绕过**全部**专用 kernel；要验证拓扑入口就不要传 `out=`。
- 混合 dtype 一律回落（`encode_execution_plan` 直接报 mixed precision），不会隐式转换。
  单精度 `complex64/float32` 恒可下发；双精度 `complex128/float64` 必须先在原生侧
  实测通过（`_probe_double`，每进程每 dtype 一次）才启用，否则回落——
  见 `TOPOLOGY_DTYPES` / `TOPOLOGY_DOUBLE_DTYPES` / `topology_dtypes()`。
- 静态生成内核只展开 `float32`，不能服务 double/complex；它是未接线的实验路线。
- `_topology_registry.py` 的再生成入口 `_topology_registry_regen` **不在仓库中**；
  改语料后只能按摘要口径手工重算，别照着 docstring 去跑一个不存在的模块。
- Python 计划缓存只有 32 项 FIFO（`_kernel_kupl` / `_kernel_cuda`），键是**调用方写的
  方程**而不是拓扑签名；整语料扫描会反复建计划，单遍扫描耗时不能当稳态。
- KUPL 拓扑入口每次调用都重新 `parse_execution_plan`：16 元方程 `size=4` 实测 >90 s
  （通用路径 2 ms），`size=3` 约 4 ms。修复前不要写"拓扑 kernel 更快"；证据见
  `logs/v202611112059.txt` 与 `logs/v202611112110.txt`。
- CUDA 的 `TopologyProgram` 是**单槽**缓存（按计划字节 `memcmp`），换拓扑即覆盖；
  CUDA 拓扑入口还要求 `stream == nullptr`。

### 线程档：kernel / generic 双档

`pyqcd/einsum/_threads.py` 维护两套档位，`contract()` **按是否命中注册形态逐调用切换**；
两组线程要相反的配比，所以不能只留一套。

| 档 | 何时用 | OMP | BLAS | 为谁调 |
|---|---|---:|---:|---|
| `kernel` | `_specialize.match(eq)` 命中注册形态（E4/E5/S1–S3）或语料拓扑 | 物理核数（≤8） | ≤4 | 拓扑/步表 kernel |
| `generic` | 其余所有调用（GEMM 链、打包、分步） | 核数//2（≥1） | 2–4 | BLAS |

优先级（高 → 低）：`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` → `PYQCD_EINSUM_OMP_THREADS`/
`PYQCD_EINSUM_BLAS_THREADS` → 自动（`OMP=min(cores,8)`、`BLAS=clamp(cores//2,1,4)`）→
`PYQCD_EINSUM_THREADS=off` 全关。自己导出的变量记在 `_state["exported_env"]`，
`_pinned()` 会忽略它们，否则档位切换会把自己"钉死"。

- **命令**：`python -c "from pyqcd.einsum import _threads; print(_threads.profiles()); print(_threads.describe())"`
- **期望**（4 物理核/8 线程本机）：`{'kernel': (4, 2), 'generic': (2, 4)}` 与
  `threads: omp=4 blas=2 [auto kernel (4 physical cores); runtime: blas]`。
  `runtime:` 里没有 `gomp` 只表示 `libgomp` 还没映射（按 `/proc/self/maps` 探测），不是错误，
  但要写进报告。
- **关闭**：`PYQCD_EINSUM_THREADS=off python -c "from pyqcd.einsum import _threads; print(_threads.describe())"`
  → `threads: untouched (disabled by PYQCD_EINSUM_THREADS)`；`resolve()` 同时返回
  `(None, None, 'disabled by PYQCD_EINSUM_THREADS')`。对照 opt_einsum 时两边必须同档。

### 线程档陷阱：幂等键（曾把小收缩拖慢 10×）

`ensure_runtime()` 在原生库真正加载后补写一次线程，靠"`runtime_applied` 规范键 + 最多重试
3 次"判定完成。历史上该完成标志与 `runtime_applied` 的键名不一致 → 完成判断永假 →
**每次 `contract()` 都重设一次线程**，线程创建开销吃掉小收缩：每 10 次调用重设 10 次，
E5 从 4.88× 掉到 0.25×。修好后重设次数与调用次数无关。

```bash
python - <<'PY'
import numpy as np
from pyqcd.einsum import _threads, contract
n = [0]; orig = _threads._set_runtime_threads
_threads._set_runtime_threads = lambda *a, **k: (n.__setitem__(0, n[0] + 1), orig(*a, **k))[1]
a = np.random.default_rng(0).standard_normal((8, 8))
for _ in range(10): contract("ij,jk->ik", a, a)
print("resets:", n[0])
PY
```

- **期望**：`resets: 2`（首调一次 `ensure_runtime` + 一次档位切换），其后 9 次为 0；
  R26 只数 `ensure_runtime` 那一路时记录为"10 → 1"。
- **判据**：`resets` 随调用次数线性增长（≈10）即为该缺陷回归，先查 `_threads._state` 的键名，
  不要先去查收缩算法。

### 测量方法论：交错协议 + 生产尺度 + 不可外推

| 项 | 口径 | 反例 |
|---|---|---|
| 顺序 | 两边在**同一循环内交替**计时（A/B/A/B…），取每轮比值的中位数 | 分两段各测一遍，机器漂移直接变成"倍率" |
| 预热 | CPU 1 次；CUDA 至少 4 次（前 3–4 次受显存池增长影响慢 3–4 倍） | `warmup=2` 把中位数拖到非稳态 |
| 重复 | 3 轮，取中位数；单次测量不下结论 | |
| 尺度 | 单次 **0.4–2.7 s**；< 100 ms 的单元格标 `too_short`，不参与倍率判定 | 十几毫秒的"3×"是计时器噪声 |
| 对手 | 写清是 `opt_einsum(dp)` 还是 `np.einsum(optimize='optimal')` | 同一份代码两个倍率（见规则 2） |
| 正确性 | 每个用例先对 `numpy.einsum` 断言，再计时 | 快而不对的数字最贵 |

- **小尺寸不可外推**：cupy 批量 GEMM 32×512² 只有 0.94×（看着持平），到 16×1024² 是 **0.21×**
  ——固定开销掩盖了 5 倍差距。反过来，小尺寸的"快"也不能写成生产结论。
- **交错模板**（换方程与形状即可复用）：

```python
import statistics, time
import numpy as np, opt_einsum as oe, pyqcd.einsum as pe

def one(f):
    t = time.perf_counter(); f(); return time.perf_counter() - t

def compare(ours, theirs, rounds=3):
    ours(); theirs()                      # 预热
    L, R, Q = [], [], []
    for _ in range(rounds):
        a, b = one(ours), one(theirs)
        L.append(a); R.append(b); Q.append(b / a)
    return statistics.median(L), statistics.median(R), statistics.median(Q)

rng = np.random.default_rng(0)
a = rng.standard_normal((4600, 4600)); b = rng.standard_normal((4600, 4600))
eq = "ij,jk->ik"
assert np.allclose(np.asarray(pe.contract(eq, a, b)),
                   np.einsum(eq, a, b, optimize="optimal"))
print(compare(lambda: pe.contract(eq, a, b),
              lambda: oe.contract(eq, a, b, optimize="dp")))
```

- **期望**：`pyqcd` 与 `opt_einsum` 同量级（本机 matmul 1000² 1.09×、`abc,bcd->ad` 1.02×；
  文档生产口径 matmul 4600² 1.01×）；对 `numpy` 的 `optimize='optimal'` 也应在 0.9–1.1× 内，
  **出现 0.33×/0.21× 这类数字先怀疑走错门，而不是硬件**。

### 诊断与告警：先问"跑没跑"，再问"快不快"

| 手段 | 回答什么 | 命令 | 期望 |
|---|---|---|---|
| GPU 探针 | 构建/设备/算力/kernel 状态/命中提示 | `python -m pyqcd --gpu-probe` | 见下 |
| 跳过告警 | 该跑 kernel 却没跑（含原因） | 见下 | 1 条 `RuntimeWarning` |
| `LAST_FAILURE` | 为什么没跑 | 见下 | 具体原因串 |
| `PYQCD_EINSUM_TRACE=1` | C++ 内部到底跑了哪个 kernel | 见规则 5 | `... kernel=1` |
| 分支探针 | 委派还是原生执行器 | 见开头 spy | 见上表 |

- **`--gpu-probe` 必须永不抛异常**：无设备时走 `device_count == 0` 分支，输出
  `verdict: no GPU available — pyqcd.einsum runs the CPU/native path…` 并**退出码 0**（诊断只给结论，
  不抛异常）。本机（RTX 4060，compute 8.9）实测 `kernel status: ok`、`verdict: CUDA fast path is active`
  ——**GPU 数字异常时先跑它**，旧安装/错误架构会把每个数字都变成"PyQCD 很慢"。
- **"注册了 kernel 却没跑"立即告警**：命中注册形态但 kernel 返回 `None` 时发
  `RuntimeWarning`，同（形态, 原因）只报一次（`_KERNEL_SKIP_WARNED`）。复现：用 float64 操作数
  跑 E1 方程 `gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M`，期望提示 `[plan-cache]` 且告警
  `pyqcd.einsum: specialised kernel did not run for nucleon-2pt(E1) (operand shapes do not satisfy the kernel contract …)`。
  判据：**告警文本里必须带原因**，只报"慢了"等于没说。
- **`LAST_FAILURE` 必须在调用之后读**，早读拿到的是上一次的残留值；且要按形态过滤
  （`_kernel_labels_for`：E1 → 2pt，P1/P2 → 3pt），否则会把 3pt 的原因报给 E1。

```bash
python - <<'PY'
import numpy as np
from pyqcd.einsum import _kernel_kupl as k
E1 = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"; n = 4
rng = np.random.default_rng(0)
ops = [rng.standard_normal(s) for s in
       [(4,4,n,n), (4,4,n,n), (4,4,n,n), (4,4), (4,4), (4,n,n,n), (4,n,n,n), (4,4)]]
print("return:", k.try_nucleon_2pt(E1, ops))     # 必须先调用
print("LAST_FAILURE:", k.LAST_FAILURE)           # 再读原因
PY
```

期望：`return: None` 与
`LAST_FAILURE: {'reason': 'operand shapes do not satisfy the kernel contract', 'status': None}`。

- **`probe()` 不得改动全局提示状态**：它保存/恢复 `_specialize._announced`，所以诊断不会
  吃掉调用方本来该看到的一次性提示。

```bash
python - <<'PY'
from pyqcd.einsum import _probe, _specialize
_specialize._announced.clear(); before = set(_specialize._announced)
print(_probe.probe()["verdict"])
print("announce set unchanged:", before == set(_specialize._announced))
PY
```

期望：`CUDA fast path is active` 与 `announce set unchanged: True`；若为 `False`，说明有人又在
`probe()` 里 `clear()` 了提示集合（R31 修过一次）。

### 已否决方向：direct 模式（跳过规划器）

0.2.5/0.2.6 曾加过 `PYQCD_EINSUM_DIRECT` / `pe.configure_direct()`：命中专用 kernel 就用
kernel，否则一次 n 元调 `backend.einsum`，跳过规划与内存重划分。**实测判定失败，0.2.7 全量
回滚，不要重做。** 默认模式为基准（本机 float64/complex128）：

| 形态 | 默认 | direct | 代价 |
|---|---|---|---|
| 核子 2pt nvec=32（有 Python 级 kernel） | 16.7 ms | 14.4 ms | 0.86×（唯一不亏） |
| 逐元素 4000² | 21.0 ms | 28.0 ms | 1.33× 慢 |
| 批量 8×256² | 3.5 ms | 25.3 ms | 7.2× 慢 |
| 矩阵乘 1024² / 2048² | 18.3 / 79.6 ms | 199 ms / 3.48 s | 10.9× / 43.7× 慢 |
| γ 夹心 512×512（注册态但无 Python kernel） | 6.3 ms | 187 ms | 29.9× 慢 |
| 三算子链 3×256 | 0.6 ms | 2.21 s | 3900× 慢 |
| 四算子链 4×2048 | 0.27 s | 10 分钟未返回 | — |

三条判据（任一条都足以否决同类提案）：

1. **"跳过规划"不是加速项，是放弃优化项**：`numpy/cupy.einsum(optimize=False)` 是朴素 n 元
   循环，不调 BLAS、不切中间体，代价随算子数指数恶化；规划的收益（选顺序 + 走 BLAS）远大于
   它自身开销（小/中型图 31–84%，已由 0.2.4 的计划缓存压掉）。
2. **"注册态"≠"有 kernel"**：`_specialize` 的 9 个形态里只有 2pt/3pt 有 Python 级 kernel；
   γ 夹心是 `[plan-cache]`，融合发生在**原生执行器内部**，跳过执行器即退化为 backend.einsum。
   判断是否命中 kernel 只能看实际执行分支（分支探针 / `PYQCD_EINSUM_TRACE=1`），不能看注册表。
3. **单/双算子测不出灾难**：direct 只在单算子与小形态上"持平"，一上三算子链就是 3 个数量级。
   任何执行路径改动都必须覆盖 2/3/4 算子递增形态，否则会把崩溃性退化误判为"可能慢几倍"。

复盘全文：`docs/report_direct_mode_postmortem_20261111.md`。

### 已知限制（不要声称已支持）

| 项 | 现状与判据 |
|---|---|
| E4/E5 融合 kernel 在 `.so` 里，但 `nm -D` 看不到 | 版本脚本 `local: *` + `-fvisibility=hidden` 藏内部符号。`nm -D --defined-only pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so \| grep -c 'e4_fused\|sandwich'` → **0**，而 `nm pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so \| grep -c …` → **98**（本机构建）。"看不到"≠"没编进去"。 |
| E4 体积门 | `kE4FusedMaxSpatialVolume = 1`，只在 `z·y·x == 1` 命中；超出即回落通用多步（真产量级会慢 33×，故主动关闭）。 |
| E4 融合在公开入口基本不可达 | R33 的 cupy 委派在规划前就把 E4 交给 `cupy.einsum`（`_cuda_fused_shape` 只认 E5）。实测 `contract(E4, *cupy_ops)` 分支探针只有 `[('backend','cupy')]`；**传 `out=` 绕过委派后**才回到原生执行器，此时 trace 才有 `PYQCD_EINSUM_CUDA_E4 … kernel=1`。要验证 E4 融合就必须带 `out=`。 |
| `[plan-cache]` 不代表 C++ 没融合 | Python 层只判断"命中注册形态"，E4/E5 在 Python 没有专用 kernel，恒打 `[plan-cache]`。是否真融合看 `PYQCD_EINSUM_TRACE=1` 的 `… kernel=1`。 |
| 批量形态内存约 3× opt | `8×512²`：`opt_einsum` 16 MB vs `pyqcd` 52 MB（打包工作区）。长批量作业按此估 RSS/显存。 |
| 其余 | `contract_expression` 不走专用 kernel；细节见 `references/measured-pitfalls.md`。 |

## 验收门

1. `nm -D --defined-only` 只暴露冻结四入口及 compiled-context 三入口；KUPL/CUDA
   计划结果逐位一致，内部 kernel 符号不得导出。
2. 固定小方程和四条 distillation 方程分别与 NumPy/opt_einsum 对照；complex128
   相对误差 `<= 1e-12`，同时检查 shape、dtype、NaN/Inf 和 `out` 身份。
3. 非连续输入、`size-1` 广播、隐式输出、trace/Hadamard、多步中间结果和空归约
   逐类覆盖；不支持项必须走显式回退。
4. CUDA 同 plan 在 `sm_60` 与 `sm_70` 实机执行；检查 device synchronize、stream
   依赖、错误码与中间 buffer 释放。
5. KUPL 检查 OpenMP 线程数、SIMD 连续 K 和工作区复用。
6. 性能 JSON 必须保存 oracle commit/version、设备 UUID、线程、stream、warmup、
   iterations、path hash、median/IQR、峰值内存和正确性摘要；所有 3x 项 PASS 才可
   声明 `performance-validated`。
7. **拓扑三条线必须分开验收**（2026-11-11 补）：
   - 匹配线：`python -m pyqcd.testing._topology_equations_contract` → `6 passed`；
     另查当前任务表命中 8638/8638；基础表契约覆盖 2582 条改名同签名与结构负例。
   - 执行线：`PYQCD_EINSUM_TOPOLOGY_TRACE=1` 必须出现 `topology-specialized:`；
     只有 `[kernel: kupl_topology]` 标签或 `rc==0` **不算**（那可能是转发通用执行器）。
   - 数值线：参考必须用**同 dtype 的通用路径** `contract(..., specialize=False)`；
     高元数 `dim≥8` 上直接用 numpy 当参考会先超时，会把"参考算不动"误判成"kernel 错"。
8. 构建线：动过 `topology_kernels.inc` / `topology_gemm.h` 后必须重建并复核
   `nm -D … | grep topology` 与 `strings … | grep -c topology-specialized`
   （生产库未接步表时为 0）。

## 常见错误

- 仅把 NumPy/CuPy `einsum` 包一层就宣称原生加速。
- 只优化规划，不接通 `applyContractQcd`，公共 `contract` 仍走旧执行层。
- 将 K 放在高维或依赖隐式跨步，导致 SIMD lane 和 CUDA warp 访存离散。
- 用逐元素 CUDA 正确性替代 cuBLAS batched GEMM 正确性。
- benchmark 固定执行顺序、无预热、无同步、混用 `optimal/dp` 或低倍率仍返回 0。
- 把多 stream 的吞吐加速冒充算法单流加速；两者必须分别报告。

## 交接

- 物理方程与 LapH 轴序：`pyqcd-distillation`、`pyqcd-physics-correlator`。
- backend、设备、显存和计时基础：`pyqcd-infra`。
- CUDA/KUPL 构建参考经验：PyQCU 原理吸收后自包含实现，不在运行时引用。
- 报告或 PDF 验收：`pyqcd-docs`。

## 拓扑特化内核：基础表阶段（2026-11-11）

本节保留 `refer/sush/wick_12cases.md` 基础表 2582 条阶段的匹配与执行证据；当前任务
表已扩展到 8638 条，执行层也已由后文“执行层已打通”一节更正。任何“已实现特化
内核”的说法仍必须附实际执行证据（见“命中/回落判据”）。

细分内容：
- 匹配口径与语料统计 → 本节 §1–§2
- 执行机制（两条路线，务必区分） → 本节 §3
- 精度策略 → 本节 §4
- 命中/回落判据 → 本节 §5
- 验收门与新增拓扑的顺序 → 本节 §6
- 已知限制 → 本节 §7
- 步骤表格式、workspace、失败码细节 → `references/native-implementation.md` §8
- 陷阱表（含四条本轮实踩） → `references/measured-pitfalls.md` §8

> 与上文 `### 拓扑匹配：基础表 2582 条到当前语料 8638 条（2026-11-12 更新）` 的关系：那一节讲
> **匹配与调用契约**，本节讲**执行层现状与实测数字**。两者不冲突；唯"KUPL 拓扑入口
> 每次重新 parse 导致 size=4 > 90 s"一条，本轮在当前构建上**复现不出来**（见 §7）。

### §1 语料与匹配口径

| 项 | 值 | 证据 |
|---|---|---|
| 基础表条数 | **2582**（情形 1/2/3/4/5/6/7/9/10/13/14 = 2/6/120/720/6/24/720/24/120/120/720） | `_parse_equations()` 长度；`logs/v202611112000.txt` |
| 基础表去重签名 | **2582**（两两互异，无跨情形复用） | `len({canonical_signature(e) for e in eqs}) == 2582` |
| 当前任务表 | **8638**（基础表 2582 + 附录 6056） | `_wick_tasklist.TASKS` |
| 当前唯一签名 | **8638/8638** | `len({canonical_signature(t.equation) for t in TASKS})` |
| 元数分布 | 6:2, 7:6, 9:6, 10:24, 12:24, 13:360, 14:720, 16:1440 | 同上 |
| 输出秩分布 | 0:32, 2:1830, 4:720 | `logs/v202611112000.txt` |
| 注册表 | **8638** 条 96 位 blake2b 摘要，按 `(元数, 槽位总数)` 分 24 桶 | `_topology_registry.N_ENTRIES` |
| 当前任务命中 | **8638/8638** | 2026-11-12 全任务 `match()` 实跑 |
| 基础表契约 | 2582 条数值/改名/结构负例 | `_topology_equations_contract` |
| 随机合法非语料误命中 | **0/4000** | 2026-11-11 实测；本轮未复测 |

`canonical_signature` 口径：按"全式首次出现顺序"把指标重编号为 `0,1,2,…`
（`...` 固定 `-1`）；先扫描全部操作数（操作数顺序与操作数内指标顺序都不变），
再扫描输出；无 `->` 时按 einsum 规则补隐含输出；无法解析返回 `None` 不抛异常。
因此 `ij,jk->ik` ≡ `ij,jb->ib` ≡ `ab,bc->ac` = `((0,1),(1,2),(0,2))`；
`ij,kj->ik`、`ij,jk->ki`、`jk,ij->ik` 均不同。

`match()` 单段式：拓扑签名 → 语料索引（`_topology_registry` 的
`(元数, 槽位总数)` 二级分叉，惰性加载，首次 0.87–0.90 ms）。字面 `REGISTRY` 与
`FAMILIES` 已随旧专用内核一并删除。

摘要口径（改动语料时必须一致）：`"|".join(",".join(map(str, term)) for term in signature)`
→ `blake2b(digest_size=16).hexdigest()`（`_specialize.py:243-246`）。

最终匹配结果由 `_match_of_normalized` 缓存，`_match_of_normalized` 与
`_signature_of_normalized` 的上限均为 `_CORPUS_CACHE_LIMIT = 16384`，可覆盖当前
8638 条唯一方程。只扩大签名缓存仍会重复摘要分桶，不能替代最终结果缓存。

下表是 2026-11-12 对当前完整任务表的 full-shape 实跑，不涉及缩放维度或冒烟场景：

| 工作集 | 实测（当前 8638 条语料） |
|---|---|
| 首轮冷扫描（含惰性索引导入与 8638 项填充） | 约 0.54–0.57 s |
| **8638 条热轮转** | **0.532 µs/call**，`currsize=8638` |
| 8638 条 dedicated 计划热扫描 | **22.66 µs/call**，总约 0.195 s |

测量环境为 48 个逻辑 CPU，8638 条轮转取 5 轮中位数。复现时先
`source "$HOME/env.sh"`，再设 `PYQCD_EINSUM_NOTICE=0` 读取
`_wick_tasklist.TASKS`，记录耗时与 `_match_of_normalized.cache_info()`。

**性能数据门（2026-11-12）**：性能结论只接受生产场景或完整生产语料上的实跑。
`smoke`、`--scale 16`、缩小维度和其他代理尺寸只能用于正确性/路径验证，必须丢弃
其中的耗时与倍率，不能改写成生产性能。生产规模装不下时，正确状态是
`unverified`，不是用较小场景替代。

### §2 语料统计速查命令

```bash
python -m pyqcd.testing._topology_equations_contract          # 6 passed / 0 failed
python -B -m pyqcd.testing                                    # 实际计数以命令输出为准
```

### §3 执行机制：两条路线，务必区分

| | 路线 A（已提交，可用） | 路线 B（未打通） |
|---|---|---|
| 名称 | 拓扑入口转发通用执行器 | per-拓扑步表 + 参数化 GEMM |
| KUPL 入口 | `qcd_kupl_topology_apply` → `parse_execution_plan` → `execute_parsed_contract` | 同上，但在转发前先 `run_specialized()` |
| CUDA 入口 | `qcd_cuda_topology_apply` → `topology_entry` → `execute_prepared` | 同上，CUDA 侧本轮**无改动** |
| 加速来源 | 少一次 Python 逐步调度 | 预编译步表 + cblas GEMM 装箱 |
| 实测 | 端到端 1.02–1.50x（float32, dim=4/8, 元数 6/10/13/16）；原生入口自身 0.077 ms vs 通用 0.182 ms（16 算子, dim=2） | **未跑通**：`build_canonical_signature` 恒失败，实际总是落回 A |

路线 B 的步表格式（`uint8` 流，每轴 4 bit）见
`references/native-implementation.md` §8.4；执行器在
`cpp/kupl/einsum/src/topology_gemm.h`（`BlockPool` + 模板 `run_typed<float|double>`）。

**为什么必须换掉朴素嵌套循环**：旧形态每条方程展开成"每符号一层"循环，实测循环深度
= 全式不同符号数，元数 13→20 层、14/16→24 层；dim=4 时 `4^24 = 2.8e14` 次乘加，
物理上不可用。现行生成器已改为步表（`.inc` 从 13 MB 降到 2.5 MB）。

### §4 精度策略

| 项 | 规则 |
|---|---|
| 默认/首选 | 单精度（`float32` / `complex64`） |
| 声明集合 | `TOPOLOGY_DTYPES = (complex64, float32)`，见 `_kernel_kupl.py:275`、`_kernel_cuda.py:269` |
| 双精度 | `TOPOLOGY_DOUBLE_DTYPES = (complex128, float64)`，须经 `_probe_double` **实测通过**才启用（每进程每 dtype 一次），否则返回 `None` 回落 |
| 降级 | **绝不隐式降精度**；float64/complex128 不支持时回落通用路径，不转 float32 |
| 混合 dtype | 一律回落（`encode_execution_plan` 直接拒绝 mixed precision） |
| 实测（KUPL 直调） | float32 `2.6e-7`、float64 `4.5e-16`、complex64 `2.2e-7`、complex128 `6.3e-16` |
| 实测（`topology_dtypes()`） | 探测前 2 项；探测后 4 项（CUDA 同样全通） |

### §5 命中/回落判据（只认实际执行分支）

| 观察 | 含义 |
|---|---|
| `[kernel: kupl_topology]` / `[kernel: cuda_topology]` | 原生入口**返回成功**；仍不能区分路线 A/B |
| `PYQCD_EINSUM_TOPOLOGY_TRACE=1` 打出 `topology-specialized:` | 真的走了路线 B（步表） |
| `[plan-cache]` | 形态命中但无 kernel 成功执行：`out=` 非空、缺库/缺符号、混合 dtype、计划超出 v2 arity≤2、kernel 报错 |
| `_kernel_kupl.LAST_INFO['path']` | `'kupl_topology'` = 实际下发给原生；空 = 根本没跑。**必须在调用之后读** |
| `_kernel_*.LAST_FAILURE` | 回落原因与状态码；同样必须在调用之后读，并按 `_api._kernel_labels_for` 过滤 |
| `strings lib \| grep -c topology-specialized` | 生产库 **0**（未接步表）；带步表的重建副本 **1** |

`out=` 会绕过**全部**专用 kernel；要验证拓扑入口就不要传 `out=`。

### §6 新增一条拓扑内核的顺序

1. 生成步表：`python -m pyqcd.testing._topology_kernel_gen`
   （写 `cpp/kupl/einsum/src/topology_kernels.inc`；`.gitignore:8` 忽略它，**生成物不入库**）。
2. 重建：`bash cpp/kupl/einsum/make.sh`（只 glob `src/*.cpp`，`.inc` 靠
   `topology_apply.cpp:17` 的 `#include` 进编译单元）。
3. 验符号：`nm -D --defined-only cpp/kupl/einsum/build/libpyqcdeinsum_kupl.so | grep topology`。
4. 验派发：`PYQCD_EINSUM_TOPOLOGY_TRACE=1`，必须出现 `topology-specialized:`。
5. 验数值：参考用同 dtype 的 `contract(..., specialize=False)`。
6. 验回落：注入非零状态，断言结果仍正确且只警告一次。
7. 指库：源码树里可用 `PYQCD_EINSUM_LIB=<path>`（`kupl/einsum.py:94-104`）。

### §7 已知限制（不要声称已支持）

| 项 | 现状 |
|---|---|
| 特化数值路径（路线 B） | **未打通**。`build_canonical_signature` 把 `step.positions` 当输入下标，而它是**活动槽**语义；第 5 步后活动槽含中间体，签名构建失败。插桩实测：`[dbg] sig: slot conflict at 4: had 4 [20,21,22,23,] now 2 [8,12,]` → `miss: signature build failed`，`rc=0`（回落通用路径仍算对，所以纯正确性测试抓不到） |
| 性能目标 | 未达成，也未出现"断崖"。`logs/v202611112059.txt`/`v202611112110.txt` 记的 `size=4 > 90 s` 在**当前构建复现不出来**（同一方程 `size=4` 实测 generic 1.88 ms / kernel 2.54 ms）。那些数字属于旧构建，不能当现状引用 |
| CUDA dim≥8 "错数" | **当前树无法复现**：32 组 `dim∈{4,8}` + `dim=12` 全部 `correct`，`rel ≤ 4.1e-7`；交替拓扑 × 3 轮 0 失败 |
| CUDA 步表/GEMM 重写 | 本轮 `cpp/cuda/**` 无改动；只有既有转发入口 |
| 静态生成内核只展开 float32 | 不服务 double/complex；且当前 `.inc` 是步表形态，不是静态展开 |
| `_topology_registry_regen` | docstring 指向该模块，但**仓库中不存在**；改语料后只能按摘要口径手工重算 |
| 计划缓存 | Python 侧 `_TOPOLOGY_PLAN_CACHE` 只有 32 项 FIFO，键是**调用方写的方程**而非拓扑签名；全语料扫描会反复建计划，单遍扫描耗时不能当稳态 |
| CUDA `TopologyProgram` | **单槽**缓存（按计划字节 `memcmp`），换拓扑即覆盖；入口要求 `stream == nullptr` |


## 拓扑特化内核：执行层已打通（2026-11-12 更正）

> **更正上一节**：`## 拓扑特化内核（2026-11-11）` 写"执行层只做到转发、特化数值路径
> 尚未打通"——该结论在 2026-11-12 已**失效**。本节给出打通后的实测状态。
> 读技能时以本节为准，上一节仅保留其匹配层与契约部分。

### 实测状态

| 项 | 值 | 证据 |
|---|---|---|
| 内核代码 | `cpp/kupl/einsum/src/dedicated_apply.cpp` **1064 行** + `include/dedicated_apply.h` 157 行 | `wc -l` |
| 导出符号 | `qcd_kupl_dedicated_apply` | `nm -D --defined-only libpyqcdeinsum_kupl.so` |
| Python 入口 | `_kernel_kupl.try_dedicated(subscripts, operands)` | — |
| API 接线 | `_api.py` 分派链新增 `("kupl_dedicated", ...)`，**优先于** `kupl_topology` | `grep kupl_dedicated pyqcd/einsum/_api.py` |
| 全量正确性 | **8638 / 8638**（dim=4，偏差 ≤1e-5，0 失败） | 本轮实跑 |
| 命中日志 | `[pyqcd.einsum] 专用路径: topology:xxxxxxxx [kernel: kupl_dedicated]` | — |

### 性能（本机，5 轮中位，预热后；比值 = 特化/通用，<1 为更快）

| dim | 中位比 | 最快 | 最慢 | 慢于通用条数 |
|---:|---:|---:|---:|---:|
| 4 | **0.19×**（快 5×） | 0.01× | 0.87× | 0 / 400 |
| 8 | 0.76× | 0.33× | 3.84× | 12 / 40 |
| 12 | 0.96× | 0.21× | 4.32× | 18 / 40 |
| 16 | 0.91× | 0.17× | 9.10× | 16 / 40 |

**冷热差异极大**：首次调用含步表生成，命中缓存后热调用可快约 100 倍。
`_DEDICATED_CACHE_LIMIT = 16384`、字节预算 64 MiB，当前 8638 条 full-shape
计划实测约占 46.1 MiB，可完整常驻；若同时轮转多 dtype/多形状族，有界 LRU 仍会淘汰。

**CPU complex128 的 `optimize="auto"` 包含一个保守的 branch-2 条件提升**：仅当基础
auto/greedy 最优计划的静态 arena ≥ 8 GiB，且 branch-2 同时把 FLOPs 压到 ≤80%、
arena 与 pack bytes 压到 ≤65% 时才替换。六序列 full-scene 中只有 `seq=6585`
命中，CPU 从 17.537 s 降到 10.346 s（0.590x）；`seq=129` 虽 arena 更小，但 FLOPs
从 16.94G 增到 30.05G，端到端慢 27.5%，因此不会被该门选中。CUDA 保持
auto/greedy；complex128 先受静态 arena ≥3 GiB 的超大计划门约束，再由原生
单步 >40G MAC 门兜底，二者都 fail-open 到 generic。不要按 seq/方程硬编码，也不要
只按 FLOPs 或只按 arena 选计划。

dedicated/topology 的确定性 `status=4` 可由线程安全负缓存复用，键必须包含
normalized equation、完整 operand shapes、dtype、optimizer、backend；只缓存
`UNSUPPORTED`，不得缓存 OOM/内部错误/可修复布局。

### 抽样复核：固定 15 条之外的语料（2026-11-12）

`python -m pyqcd.einsum.bench_specialized --sample N --seed S` 按元数分层从 8638 条语料
随机抽 N 条（排除固定 15 条），同 seed 结果稳定——固定 15 条只证明"挑出来的能跑"，
抽样才证明语料整体落同一执行路径。

冒烟档实跑 8 条（seq 2584/2590/3436/866/1606/2650/3186/6585，元数 6–16）：
**8/8 命中 `[kernel: kupl_dedicated]`，0 WRONG**；其中 5 条双方均 <1 ms 标
`too_short` 不计入，3 条可用样本中位比值 **0.14×**。抽样只用于验证执行路径与数值，
性能结论仍须 scene/full 档多轮复测。

**抽样必须在装得下的档位跑**（2026-11-12 实测，本机 15 GB RAM）：

| 档 | 单张 peram（complex128） | 6 传播子路径光操作数 | 结果 |
|---|---:|---:|---|
| `scene`（默认） | **1.53 GiB**（102,400,000 元素） | ~9 GiB | 16 算子实测 RSS **14.4 GiB** + 换出 5.4 GiB、CPU 仅 13% → thrashing，无可用数字 |
| `--scale 16` | 312 MiB | ~1.9 GiB | 正常跑完 |

`_bench_scene.py` 原写「本档 peram = 82 MB」「scene < 1 GB」是错的（差 20 倍，
把 1.024e8 元素当成了 1.024e7），已按实测更正。**15 GB 机器跑 16 算子必须降到
`--scale 16` 或 `smoke`**；换页后耗时不再反映内核质量。

`--scale 16 --sample 8 --seed 0` 实跑（表头 `Nx=16 Nev=50 Lt=32 Mom=4`）：
7 条出数 + 1 条 REF-FAIL，**7/7 命中 `kupl_dedicated`、0 WRONG**；可用 4 条
比值 0.91/0.45/0.16/0.22×，中位 **0.33×**（3 条 `too_short` 不计入）；
16 算子 seq 6585 为 0.069 s vs 0.323 s = **0.22×**。

**参考实现的病态会打断整条基准**：`--scale 16` 下 seq 3186（14 算子）让
`opt_einsum(dp)` 要 `(32,32,50,50,50,50)` complex128 的中间体 = **95.4 GiB**，
直接 `ArrayMemoryError`，原脚本整体退出、后面用例全丢。dp 只按 flops 选路，不管内存。
现改为逐行 `REF-FAIL` 降级：打印 `REF-FAIL <异常>`、不给倍率、继续下一条。
**判据：一行 `REF-FAIL` 是参考算不动，不是内核错**，不要据此改内核。

**表头必须带 `scale`**：`run()` 曾调用 `describe_scene(scene)` 而漏传 `scale`，
`--scale 16` 时表头仍打印 `scene=Nx=64…`，照日志引用就会把数字贴到错误维度。
已改 `describe_scene(scene, scale=scale)`；读历史日志先核对表头与 `--scale` 是否一致。

### 关键教训：大尺寸慢不是内核缺陷

先看 `references/local-sizing.md`：本机 L2 1.25 MB/核、L3 24 MB、DRAM 带宽 31.8 GB/s。
16 算子方程的**总工作集**：dim=4→0.01 MB(L1)、dim=8→0.23 MB(L2)、dim=12→1.44 MB(L3)、
dim=16→5.54 MB(L3)、dim=24→38 MB(DRAM)。
因此 **dim≥12 的吞吐由带宽决定，与内核质量无关**；本地性能结论只在 **dim≤8** 有效，
更大尺寸留待其他计算环境。

### 判据：这次到底走没走特化

1. 日志出现 `[kernel: kupl_dedicated]`（不是 `[plan-cache]`、不是 `[kernel: kupl_topology]`）；
2. `pe.build_info()["native_available"] is True`；
3. 热调用耗时 ≈ 通用路径的 0.2–0.8×；若 ≈1.0× 说明回落了。

### 本轮实踩的三个错（写下来防止重犯）

| 错 | 现象 | 正确做法 |
|---|---|---|
| 手推自旋/色扇区结构 | 连错 6 次，生成的子方程收缩量 4^12，比通用慢 1700× | **不要推结构**：式子、角色、维度全部从图纸逐行读，或直接从数组 `.shape` 取 |
| 用朴素嵌套循环当"专用内核" | 2582 个函数、13 MB，dim=4 需 4^24 次乘加（估算单核 26 年） | 每个求和指标恰好出现两次 ⇒ 成对 GEMM；固定步表 + 参数化执行器 |
| 把参考实现的病态当内核故障 | 报"dim=8 卡死 >60 s"，实际是 `np.einsum(optimize='greedy')` 在 16 算子上病态 | 参考一律用 `contract(..., specialize=False)`；对比前先分别计时各环节 |

### 任务清单的主键：seq，不是 T 号也不是行号

`pyqcd/einsum/_wick_tasklist.py` 的 **主键是连续序号 `seq`（1…8638）**。原因（实测）：

| 候选键 | 为什么不行 |
|---|---|
| T 号 | 图纸 11 个情形各自从 `T00001` 重新编号，**720 个 T 号重复**（`T00720` 有 10 个实例） |
| 文件行号 | 表格间夹着标题/说明/空行，**最后一条任务在 9142 行，但只有 8638 条** |
| `(case, subtable, tid)` | 唯一，但冗长 |

接口：`T.lookup(8638)`（int→seq）、`T.lookup('T00720')`（str→首个同名）、
`T.by_tid(case, subtable, tid)`、`T.by_seq()`、`T.by_case()`、`T.base_table_count()`。
`Task` 用 `__slots__`，含 `seq/tid/line/case/subtable/equation/roles` 与派生 `.arity`。

**踩坑**：基础表的 `subtable` 为空字符串，附录才带 `动量/3.2` 之类；
按 `(case, subtable)` 查时两者别混。

### [T1] 彻底删除 MPI 与 2pt/3pt 专用 kernel（2026-11-12 实测）

用户要求 `qcd-einsum` 不再有任何 MPI 相关内容，同时保留「通用内核 + 新版特化内核」。
已删除：`nucleon2pt`/`pattern_3pt`（KUPL/CUDA 的 .cpp/.cu/.h 共 8 个文件）、
`kupl/mpi.py`、`cuda/mpi.py`、`kupl/src/mpi_runtime.cpp`、`kupl/include/einsum_mpi.h`、
`pyqcd/parallel/{_mpi,_mpi_transport,einsum}.py`、`pyqcd/testing/{parallel_einsum,_parallel_einsum_contract,_mpi_*_contract,_native_loader_contract}.py`、
`pyqcd/testing/einsum/{verify_mpi,verify_cuda_mpi,bench_cuda_mpi}.py` 与
`pyqcd/testing/comparisons/lqcddb/mpi_test.py`。保留 `apply_contract`/`runtime`/`plan_core`/
`pack`/`apply_plan`/`e4_fused`/`sandwich`/`batched_gemm` 与 `dedicated_apply`/`topology_apply`/
`topology_gemm.h`/`einsum_topology.h`/`einsum_dedicated.h`。

踩坑与修法（都可用一条命令复现）：

1. **`multiprocess.py` 隐性依赖被删模块**：它从 `kupl/mpi.py` import
   `_initial_labels` / `_validate_plan` / `MpiConfig`。删掉 `mpi.py` 后
   `import pyqcd.einsum` 立刻 `ModuleNotFoundError`。这两个函数只是纯 plan 校验，
   内联到 `multiprocess.py` 并改用本文件的 `MpConfigError`；`verify_multiprocess.py`
   里“MPI/MP 互斥”一条随 `MpiConfig` 一起删除。**先跑 `python -c "import pyqcd.einsum"`
   再继续删**，否则错误会拖到测试阶段。
2. **旧测试用被删形态当夹具**：`_contract_edge_cases.test_kernel_skip_warns_immediately`
   用 P1 方程 + 16 个操作数（方程只有 15 项），原先靠专用 kernel 提前拦截才不报错；
   删除注册后 parser 直接抛 `Number of einsum subscripts, 15, must be equal to ... 16`。
   改为直接驱动 `_api._warn_kernel_skipped_once(entry, reasons)` 并断言只告警一次。
   `_threads_contract.test_profile_switching_is_observable` 用 E1 命中 `kernel` 档，
   删除 E1 后恒为 `generic`；换成仍注册的 S3（`manb,cedf,bd,Mfn,ac,MFem->MF`）即可。
3. **`build_info()` 去字段要同步所有读者**：删掉 `mpi`/`kupl_mpi_build`/`mpi_label`
   后，`__main__.py` 的打印、`_release_metadata_contract` 的键集合、
   `_api._native_compile_warning` 的 `report['mpi_preload']` 都要一起改；
   漏一处就是 `KeyError` 或断言失败。
4. **C++ 双侧必须逐字节同步**：`cpp/{kupl,cuda}/einsum/include/einsum_define.h`
   被 `make-nv.sh` 用 `cmp` 比较；改 `QCD_PARAM_MPI_*` / `QCD_MPI_MODE_*` /
   `applyContractMpiQcd` 声明时两份要同样改。另需清 `make.sh` 里的 MPI 探测块
   （否则它仍去编已删除的 `mpi_runtime.cpp`）、version script 的
   `qcd_kupl_nucleon_2pt*` / `qcd_cuda_pattern_3pt*` / `applyContractMpiQcd`、
   `apply_contract.cpp` 的 `#include "einsum_mpi.h"` 与 `qcd_kupl_mpi_enabled()`。
5. **残留 grep 有假阳性**：`pyqcd/renorm/**` 的 `mpi` / `MPI_PHYSICAL` 是 pion mass
   （m_π）数学符号，不是 MPI；`compiled`/`compile` 含子串 `mpi`。按
   `grep -rn "mpi4py\|MPI\." pyqcd/ --include="*.py"` 判据，删除后只应命中
   明确写“已删除/不提供”的说明文档字符串。

验证（本机 2026-11-12）：`bash cpp/kupl/einsum/make.sh` → SUCCESS（7 源，
导出 `applyPlanQcd/applyContractQcd/applyCompileQcd/applyCompiledQcd/applyDestroyQcd/
applyInitQcd/applyEndQcd` + `qcd_kupl_{topology,dedicated,family}_apply`）；
`bash cpp/cuda/einsum/make-nv.sh` → SUCCESS（9 源，sm_89）；
`python -B -m pyqcd.testing` → **118 passed, 1 skipped, 0 failed**
（skip 是 `physics.sty` 缺失，与本改动无关）。
