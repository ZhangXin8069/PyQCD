---
name: pyqcd-einsum
description: Use when implementing, debugging, or benchmarking pyqcd.einsum against opt_einsum, especially native CUDA/KUPL execution, contraction-plan ABI, distillation workloads, vectorization, MPI, or CUDA streams.
---

# pyqcd-einsum - 原生兼容收缩引擎

## 目的与边界

本技能维护 `pyqcd.einsum` 的兼容 API、收缩规划、原生 ABI、CUDA/KUPL 执行、
蒸馏实战基准与性能门禁。它不定义 LapH/Wick 物理语义，也不替代 `pyqcd-infra`
对 backend、显存、MPI 和计时平台的统一契约。

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
  记录了 2026-11-08 会话中实测踩到的构建坑（CUDA 架构默认值、同名头文件冲突、MPI 探测误判）、
  测量坑（预热不足、对照变成 kernel 比 kernel、用现编 .so 冒充生产库、逐元素误差误判）、
  正确性坑（输出全 1 维未写、设备 kernel 缺连续性校验、异常不回退、开关未进缓存键）
  与已知限制（`contract_expression` 不走 kernel、CUDA MPI 是 stub、E4 融合仅极小规模生效）。
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
6. **并行分层**：CUDA 为 MPI rank → 单 device context → 多 stream 批任务；KUPL 为
   MPI rank → 多进程/OpenMP → SIMD。多 rank 只切独立 batch，任何跨 rank 求和必须用
   对称 collective；不得让 rank 提前返回。
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
  当前全量入口记录为 `111 passed, 1 skipped`。
- CUDA 编译、动态导出与 Cython 加载已 `implemented`，但一般 cuBLAS GEMM 路径仍
  `unverified`，不得用逐元素路径通过代替 batch GEMM 通过。
- 开发窗口 CPU T0 四条方程约 `3.0x-18x`，CUDA V100/P100 E5 约
  `3.49x-4.07x`；但固定 oracle、进程隔离、MPI/MP 与目标系综仍未全部闭合，
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

### 线程档：kernel / generic 双档

`pyqcd/einsum/_threads.py` 维护两套档位，`contract()` **按是否命中注册形态逐调用切换**；
两组线程要相反的配比，所以不能只留一套。

| 档 | 何时用 | OMP | BLAS | 为谁调 |
|---|---|---:|---:|---|
| `kernel` | `_specialize.match(eq)` 命中 2pt/3pt 等注册形态 | 物理核数（≤8） | ≤4 | 专用 kernel |
| `generic` | 其余所有调用（GEMM 链、打包、分步） | 核数//2（≥1） | 2–4 | BLAS |

优先级（高 → 低）：`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` → `PYQCD_EINSUM_OMP_THREADS`/
`PYQCD_EINSUM_BLAS_THREADS` → 自动（`OMP=min(cores,8)`、`BLAS=clamp(cores//2,1,4)`，按 MPI rank
均分）→ `PYQCD_EINSUM_THREADS=off` 全关。自己导出的变量记在 `_state["exported_env"]`，
`_pinned()` 会忽略它们，否则档位切换会把自己"钉死"。

- **命令**：`python -c "from pyqcd.einsum import _threads; print(_threads.profiles()); print(_threads.describe())"`
- **期望**（4 物理核/8 线程本机）：`{'kernel': (4, 2), 'generic': (2, 4)}` 与
  `threads: omp=4 blas=2 [auto kernel (4 physical cores, 1 rank); runtime: blas]`。
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

### 任务级并行：`pyqcd.parallel.einsum` 的 `init` + `run`（2026-11-11 实测）

单次收缩的并行度有限，真正能榨干硬件的是**把一批相互独立的收缩分给多个 MPI 进程**。
入口是 `init(config, ...)` + `run(state)`：`init` 只接受列式字典
`{task_id, equation, operands, results}` 并规划，`run` 无返回值、结果原地写入调用方
缓冲区；`plan_tasks` 只规划。不提供一体化包装（避免与第三方 Einsums 库混淆）。实测（本机 4 物理核/8 线程，真实 sush S1 方程，
`complex64`，单线程基准，结果与串行逐位一致）：

| 模式 | 工作负载 | 串行 | plan（`init`） | exec（`run`） | `×exec` |
|---|---|---|---|---|---|
| 自建池（4 进程） | S1 Nvec=40 × 8 | 165 ms | 1 ms | 255 ms | 0.65× |
| 自建池（4 进程） | S1 Nvec=100 × 8 | 5310 ms | 1 ms | 2241 ms | **2.37×** |

四条硬结论：

1. **`init` 与 `run` 必须分开计时**：`init` 里跑 `contract_path` 做规划，`run` 才是真实执行。
   合成一个计时桶会把规划算进"计算时间"；harness 现在同时打印 `plan`/`exec`/`total`。
2. **`collective` 已弃用且不再纳测**：它曾是最快模式（4 rank 3–4×，零序列化），但每个 rank
   各持一份整批输入，内存随 rank 数线性增长；不要再引用它的基准，也不要为它写新测试。
3. **工时估计用 `contract_path(optimize='greedy').opt_cost`**，逐元素方程以元素数兜底，
   再 LPT 装箱；不要用「任务数均分」——那会产生短板。
4. **线程超订是这里最容易踩的坑**：父进程若把 `OMP_NUM_THREADS` 写进环境，worker 会当成
   用户显式设置照搬，K 进程 × 全线程在 4 物理核上把并行收益吃掉一大半。自建池已按
   `物理核数 // 进程数` 重算；手工 `mpirun` 时请自行导出单线程变量。

复现：`python -m pyqcd.testing.parallel_einsum bench --equation S1 --nvec 40 --procs 4`；
契约 16 项 `python -m pyqcd.testing._parallel_einsum_contract`。
报告：`docs/parallel_einsum_20261111.md`。

### 已知限制（不要声称已支持）

| 项 | 现状与判据 |
|---|---|
| E4/E5 融合 kernel 在 `.so` 里，但 `nm -D` 看不到 | 版本脚本 `local: *` + `-fvisibility=hidden` 藏内部符号。`nm -D --defined-only pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so \| grep -c 'e4_fused\|sandwich'` → **0**，而 `nm pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so \| grep -c …` → **98**（本机构建）。"看不到"≠"没编进去"。 |
| E4 体积门 | `kE4FusedMaxSpatialVolume = 1`，只在 `z·y·x == 1` 命中；超出即回落通用多步（真产量级会慢 33×，故主动关闭）。 |
| E4 融合在公开入口基本不可达 | R33 的 cupy 委派在规划前就把 E4 交给 `cupy.einsum`（`_cuda_fused_shape` 只认 E5）。实测 `contract(E4, *cupy_ops)` 分支探针只有 `[('backend','cupy')]`；**传 `out=` 绕过委派后**才回到原生执行器，此时 trace 才有 `PYQCD_EINSUM_CUDA_E4 … kernel=1`。要验证 E4 融合就必须带 `out=`。 |
| `[plan-cache]` 不代表 C++ 没融合 | Python 层只判断"命中注册形态"，E4/E5 在 Python 没有专用 kernel，恒打 `[plan-cache]`。是否真融合看 `PYQCD_EINSUM_TRACE=1` 的 `… kernel=1`。 |
| 批量形态内存约 3× opt | `8×512²`：`opt_einsum` 16 MB vs `pyqcd` 52 MB（打包工作区）。长批量作业按此估 RSS/显存。 |
| 其余 | `contract_expression` 不走专用 kernel；CUDA MPI 是 fail-closed stub；细节见 `references/measured-pitfalls.md`。 |

## 验收门

1. `nm -D --defined-only` 只暴露冻结四入口及 compiled-context 三入口；KUPL/CUDA
   计划结果逐位一致，内部 kernel 符号不得导出。
2. 固定小方程和四条 distillation 方程分别与 NumPy/opt_einsum 对照；complex128
   相对误差 `<= 1e-12`，同时检查 shape、dtype、NaN/Inf 和 `out` 身份。
3. 非连续输入、`size-1` 广播、隐式输出、trace/Hadamard、多步中间结果和空归约
   逐类覆盖；不支持项必须走显式回退。
4. CUDA 同 plan 在 `sm_60` 与 `sm_70` 实机执行；检查 device synchronize、stream
   依赖、错误码与中间 buffer 释放。
5. KUPL 检查 OpenMP 线程数、SIMD 连续 K、MPI rank 对称性和工作区复用。
6. 性能 JSON 必须保存 oracle commit/version、设备 UUID、线程、stream、warmup、
   iterations、path hash、median/IQR、峰值内存和正确性摘要；所有 3x 项 PASS 才可
   声明 `performance-validated`。

## 常见错误

- 仅把 NumPy/CuPy `einsum` 包一层就宣称原生加速。
- 只优化规划，不接通 `applyContractQcd`，公共 `contract` 仍走旧执行层。
- 将 K 放在高维或依赖隐式跨步，导致 SIMD lane 和 CUDA warp 访存离散。
- 用逐元素 CUDA 正确性替代 cuBLAS batched GEMM 正确性。
- benchmark 固定执行顺序、无预热、无同步、混用 `optimal/dp` 或低倍率仍返回 0。
- 把 MPI/多 stream 的吞吐加速冒充算法单流加速；两者必须分别报告。

## 交接

- 物理方程与 LapH 轴序：`pyqcd-distillation`、`pyqcd-physics-correlator`。
- backend、设备、MPI、显存和计时基础：`pyqcd-infra`。
- CUDA/KUPL 构建参考经验：PyQCU 原理吸收后自包含实现，不在运行时引用。
- 报告或 PDF 验收：`pyqcd-docs`。
