# pyqcd/einsum — 内部契约（子代理唯一真源）

本文件是 `pyqcd/einsum` 子系统的**冻结契约**。任何实现必须与本文档一致；
与 `refer/git-rep/opt_einsum` 冲突时，以 opt_einsum 3.4 的公开行为为准，并在此登记差异。

## 0. 稳定版本冻结（stab31 起生效）

**当前状态：qcd-einsum 的关键稳定版本自 `stab31` 起冻结（首个冻结版为 0.4.1，当前补丁版 0.4.2 仅修正文档措辞、代码与数值行为不变）。**

生效期间，任何**影响本版本 qcd-einsum 行为**的改动默认禁止，包括但不限于：数值结果、
公共 API 签名与语义、分派链（elementwise → cupy 委派 → 原生执行器 → Python 分步）、
拓扑匹配口径与语料索引、原生 ABI、提示文案、缓存与线程档语义、精度策略与回落行为。

三条例外，缺一不可：

1. 用户**明确强制**要求改动（明确点名该改动与本冻结冲突并接受）；或
2. 修复**本版本自身**的正确性缺陷（数值错、崩溃、静默错答），且改动范围不超过缺陷本身；或
3. 新增**纯增量**能力——不改动任何既有符号与执行路径，且新增行为可由 `stab31` 基线
   逐位复核（例如本次的 `verify_specialized` 纯查询入口：不调 `contract()`、不改全局状态）。

解除条件：出现下一个 qcd-einsum 关键稳定版本（新的 `stab<N>`）时自动失效，届时在
该标签的说明里重新界定冻结范围。**任何声称"不影响原有功能"的改动，都必须附基线对比证据**
（测试计数一致、关键数值逐位一致、行为探针输出一致），不得仅凭推理声明。

## 1. 分层与文件归属

| 文件 | 职责 | 唯一负责人 |
|---|---|---|
| `__init__.py` | 公共导出（只做 re-export） | API 组 |
| `_api.py` | `contract` / `contract_path` | API 组 |
| `_pathinfo.py` | `PathInfo`（字段与 repr） | API 组 |
| `_expr.py` | `ContractExpression` / `contract_expression` | API 组 |
| `paths.py` | `register_path_fn` / `get_path_fn` / `PATH_OPTIONS` | API 组 |
| `backends/` | `dispatch.get_func/has_einsum/has_tensordot/infer_backend` | API 组 |
| `_sharing.py` | `shared_intermediates` | API 组 |
| `verify_specialized.py` | 单条方程的特化路径**纯查询**（命中/签名/摘要/语料出处，不做数值） | API 组 |
| `_plan_py.py` | 纯 Python 规划回退（greedy/optimal/branch/dp） | 规划组 |
| `helpers.py` | `compute_size_by_dict` / `flop_count` / `find_contraction` | 规划组 |
| `parser.py` | `get_symbol` / `parse_einsum_input` / ellipsis 展开 | 规划组 |
| `blas.py` | `can_blas` 判定 | 规划组 |
| `_execute.py` | 执行层：BLAS 直派 + 缓冲池 | 执行组 |
| `_native.py` | ctypes 绑定（RTLD_LOCAL + 签名表 + 惰性加载） | 原生组 |
| `../../cpp/einsum/**` | C ABI + 规划核心 + CPU 数值域 + CUDA kernel | 原生组 |

**禁止跨组改文件**；需要对方文件变更时，在提交说明中提出，由主代理统一改。

## 2. 冻结的内部类型

```python
# 一步收缩（与 opt_einsum contraction_list 完全一致）
Step = tuple[
    tuple[int, ...],          # inds：本步消耗的操作数下标，升序
    frozenset[str],           # idx_removed：本步被求和掉的索引
    str,                      # einsum_str：本步等价的 einsum 字符串
    tuple[str, ...] | None,   # remaining：剩余张量下标；>20 个时为 None
    str | bool,               # do_blas: False | "DOT" | "GEMM" | "TDOT" | "OUTER"
]
```

## 3. 冻结的公共 API 签名

```python
contract(subscripts, *operands, out=None, use_blas=True, optimize="auto",
         memory_limit=None, backend="auto", **kwargs)

contract_path(subscripts, *operands, use_blas=True, optimize="auto",
              memory_limit=None, shapes=False, **kwargs) -> tuple[list, PathInfo]

contract_expression(subscripts, *shapes, constants=None, use_blas=True,
                    optimize="auto", memory_limit=None, **kwargs) -> ContractExpression

get_symbol(i: int) -> str
shared_intermediates(cache: dict | None = None)   # contextmanager

paths.register_path_fn(name: str, fn) -> None
paths.get_path_fn(name: str) -> PathOptimizer | Callable
behaviors: paths.PATH_OPTIONS 为已注册名字集合

backends.get_func(func: str, backend: str = "numpy", default=None)
backends.has_einsum(backend: str) -> bool
backends.has_tensordot(backend: str) -> bool
backends.infer_backend(x) -> str
```

`PathInfo` 必含字段（名字逐字一致）：
`contraction_list, input_subscripts, output_subscript, path, indices,
scale_list, naive_cost, opt_cost, speedup, size_list, size_dict, shapes, eq,
largest_intermediate`；
`naive_cost/opt_cost/speedup/largest_intermediate` 用 `decimal.Decimal`。

## 4. 规划器内部接口

```python
# _plan_py.py
def optimize(eq: str, shapes: Sequence[Sequence[int]],
             optimize: str | list | bool = "auto",
             memory_limit: int | None = None) -> list[Step]

# helpers.py
def flop_count(idx_contraction, inner: bool, num_terms: int, size_dict) -> int
def compute_size_by_dict(indices, size_dict) -> int
def find_contraction(positions, input_sets, output_set) -> tuple[frozenset, frozenset, frozenset]
```

## 5. 原生 ABI（`cpp/einsum/include/einsum_define.h` ↔ `pyqcd/einsum/_native.py`）

```c
/* params: int32[32] */
[0] num_ops      [1] max_ndim     [2] planner_id   /* 0 greedy,1 optimal,2 dp,3 branch */
[3] mem_hi       [4] mem_lo       [5] seed         [6] verbose
[7] use_blas     [8] dtype_id     [9] layout_id
/* set_ptrs: int64[16] */
[0] inputs_flat  [1] input_offsets [2] output      [3] sizes
[4] out_path     [5] out_cost      [6] scratch     [7] scratch_size

int  applyPlanQcd(long long set_ptrs, long long params);
int  applyContractQcd(long long plan, long long tensors, long long out, long long params);
int  applyCompileQcd(long long plan, long long params, long long out_context);
int  applyCompiledQcd(long long context, long long tensors, long long out, long long params);
void applyDestroyQcd(long long context);
void applyInitQcd(long long set_ptrs, long long params);
void applyEndQcd(long long set_ptrs, long long params);
```

前三个数值入口是 compiled-context 扩展：`applyCompileQcd` 解析一次 version-2 计划并
返回不透明 `int64` 上下文，`applyCompiledQcd` 复用计划和后端 workspace，
`applyDestroyQcd` 释放上下文。上下文只由对应 backend 解释，禁止跨 KUPL/CUDA 传递。

返回值：`0` 成功；`1` 参数非法（fail-closed）；`2` 内存超限；`3` 内部错误。
**NUL 必须导出冻结四入口与 compiled-context 三入口**；构建脚本用
`-Wl,--version-script` 收敛导出面，内部 kernel 符号不得进入动态导出表。

## 6. 精度与数值口径

- dtype：`complex128` / `complex64` / `float64` / `float32` **四类完全平权**，
  不设主精度或默认精度；混合精度 **fail closed**，任何情况下不得把一类 dtype
  静默转成另一类精度。每类的启用条件相同：由原生 2×2 实测自证正确（每进程每
  dtype 一次），没通过只回落该类；拒绝原因必须区分“契约外 dtype”与“探测没过”。
- 数值验收：与 `numpy.einsum` / `opt_einsum` 逐元素 `rel ≤ 1e-12`
  （`complex128`/`float64`）与 `rel ≤ 1e-4`（`complex64`/`float32`，含 13–16
  算子链的固有误差）。
- 性能验收：同后端、同 dtype、同线程数，`median(≥10 次)`；CPU 对 CPU、GPU 对 GPU。
- **性能依据只认生产尺寸（强制，2026-11-14 依 `refer/sush/sush-einsum.py` 重写）**：
  生产尺寸的唯一来源是该基准脚本的实际张量（详见 §6.1）：
  `Nev_src=100`/`Nev_curr=400`、`Mom_src=1`/`Mom_curr=3`、`Ns=4`、`Nc=3`。
  该脚本张量没有空间轴（`Nx`/`Lt`/`lattice` 全文匹配数为 0），因此**两条口径互不
  替代**，任何性能结论必须写明用的是哪一条：
  `distill`（`distill`/`distill_curr` 档，主口径，蒸馏收缩）与
  `spatial`（`scene`/`full` 档，仅"空间×本征模展平"的矩阵乘）。
  小尺寸（`dim ≤ 8`、`n ≤ 8`、`smoke`、`--scale`、超立方体 `dim` 扫描）**只能用于
  路径与正确性证据，其倍率一律不得写成性能结论**，也不得用于选路门限的标定；
  把 `spatial` 的倍率写成蒸馏结论、或用 `_label_sizes` 反推的 `distill` 形状
  （它表达不出蒸馏轴）当生产形状，同样作废。违反此条的产物必须在报告里显式标
  `unverified`，**不允许**用缩小档替代。
  可执行闸门：`python -m pyqcd.testing._production_perf_gate_contract`。
- 二元收缩规范布局：左 `[B,M,K]`、右 `[B,N,K]`，K 必须是两侧最低连续维；GEMM
  执行语义为 `out[B,M,N] = left[B,M,K] @ right[B,N,K]^T`。
- 稳定基准入口应缓存计划/compiled context，不缓存输入内容；同一 plan 的重复调用必须
  使用本次输入指针且不能返回旧数据。

### 6.1 生产场景的来源与各维度依据（2026-11-14 增补，只读核对）

两套生产参数并存，**不要混用**，也不要把一档的值搬到另一档：

* `PRODUCTION` / `DEFAULT_SCENE` / `SMOKE_SCENE`（`Nx`/`Lt` 为主尺度）来自
  `refer/sush/_distillation/contraction.py` 的 peram 展平描述；
* `PRODUCTION_DISTILL` / `DISTILL_SCENE`（`distill`） / `DISTILL_CURR_SCENE`
  （`distill_curr`）**逐字**取自基准脚本 `refer/sush/sush-einsum.py`，规模只由
  `Nev`（100/400）、`Mom`（1/3）、`Ns=4` 决定。该脚本里 `Nx`、`Lt`、`lattice`
  的 grep 匹配数为 **0** —— 蒸馏张量没有空间轴，`Lt` 也不是张量轴（peram 按
  `(tq,ta)` 逐对注册，`peram` 本身没有时间维）。

| 维度 | 值 | 出处（`refer/sush/sush-einsum.py`） | 为什么 |
|---|---|---|---|
| `Nev_src` | 100 | `:170` `nev_src = 100` | sink/source 端的本征模数 |
| `Nev_curr` | 400 | `:171` `nev_curr = 400` | 每个 current 端点（tcur0/tcur1）用 400；**Nev 是两个值，不能合成一个** |
| `Mom_src` | 1 | `:172` `n_mom_src = 1` | 源端只取一个动量 |
| `Mom_curr` | 3 | `:173` `n_mom_curr = 3` | current 端三个动量（分配 `(3,400,400)`，`:224`） |
| `Ns` | 4 | `:200` peram `(4,4,nev,nev)`；`:248` `gamma(4)` 4×4 | Dirac 旋量维 |
| `Nc` | 3 | `refer/sush/_distillation/constant.py:1` | 色数；**不是**本脚本的张量轴（色已在 peram 内收缩） |
| `Nt_labels` | 4 | `:179` `times = (tsink,tsrc,tcur0,tcur1)` | 注册表键个数，不是张量轴 |
| `Nx` | 无 | 该文件无匹配 | 蒸馏张量没有空间轴；`dims("distill")` 用 `Nx=1` 作"无此轴"占位 |
| `Lt` | 无 | 该文件无匹配 | 同上；`Lt=1` 为占位 |

`nev_by_time = {tsink:100, tsrc:100, tcur0:400, tcur1:400}`（`:184-189`），因此
`peram[tq,ta] = (4, 4, nev[tq], nev[ta])`（`:200`）在同一张图里出现
`(4,4,100,100)`、`(4,4,100,400)`、`(4,4,400,100)`、`(4,4,400,400)` **四种异形轴长**。
`distill_tensor_shapes()` / `distill_memory()` 给的就是这组真实形状与内存估算
（`3pt` 段 0.327 GiB、`6peram` 段 0.194 GiB；最坏可信 6pt 峰值下界 5.16 GiB < 15 GiB）。

已知口径差异（**不要**在没有依据时"顺手修正"）：

1. `bench_specialized._label_sizes` 把 `VVV` 当 `(Lt, Mom, Nev, Nev, Nev)`、`VDV` 当
   `(Lt, Mom, Nev, Nev)`，并且全方程只有一个 `Nev` —— 这两点都与参考脚本不符
   （无 `Lt` 轴、两个 `Nev`）。实测后果：拿 `dims("distill")` 喂 `_label_sizes`，
   `Nx/Lt=1` 的占位与 `Mom`（1 或 3）会作为**真实轴长**漏进 peram 的轴
   （seq 3 得到 `(4,100,100,100)`、`(100,4,1,1)` 之类），与参考脚本的
   `(4,4,100,400)` 无关。因此 **`--scene distill` 经 `_label_sizes` 的形状只能当
   path-only 证据，不得写成蒸馏生产结论**；真实形状只认 `distill_tensor_shapes()`。
   该限制由 `_bench_scene_distill_contract.test_label_sizes_cannot_express_distill_axes`
   锁定；修 `_label_sizes` 归规划/语料组，改了会让该契约失败并提醒同步本节。
2. `VVV` 分配 `(1,100,100,100)`（`:209`）但注册时用 `host_vvv_sink[0]` 切片（`:241`），
   进 einsum 的是 `(100,100,100)`；`VDV` 保留动量轴（`:243-245`）→ `(1,100,100)` /
   `(3,400,400)`。两者不对称，照抄参考脚本即可。
3. 语料里 `VDV` 的 term 只有 2 个标签（seq 1/855），而参考脚本注册的 `VDV` 是 3 轴；
   这需要在 `lqcddb` 侧核对，本库不擅自决定。

## 7. 回退不变量

1. 缺 `.so` → 自动使用 `_plan_py.py`，**行为与结果不变**（仅性能下降）。
2. `_native.py` 只做 `ctypes`，import 时不加载、不编译、不弹错误。
3. 任何原生路径失败 → 记录一次 warning，回退纯 Python，不抛给调用方。

## 8. 命名与禁忌

- Python：`snake_case`；私有前缀 `_`；类型大驼峰。
- **生产代码禁止 import `refer/` 与 `pyqcd.testing`**；oracle 只在 `pyqcd/testing/einsum/` 内。
- 不修改 `refer/git-rep/**` 子模块。
- 不加注释解释显而易见的代码；只在非显然处写一句原因。

## 9. 当前验证入口

```bash
python -m pyqcd.testing.einsum.verify_native --backend kupl
python -m pyqcd.testing.einsum.verify_native --backend cuda
PYQCD_EINSUM_ORACLE=installed PYQCD_EINSUM_NATIVE=kupl \
  python -m pyqcd.testing.einsum.bench_native --profiles T1 \
  --backends cpu --isolation process
```

`T0/T0n` 是 overhead 诊断 profile；`T1/T2` 是性能验收 profile。安装 oracle 模式必须
在产物中保留 `frozen=false`，不能替代冻结子模块证据。

#### CUDA 用例必须用 mamba 环境跑（2026-11-14 记录，踩过一次）

凡是会触发 **nvrtc JIT** 的用例（`_einsum_semantics_contract` 的
`test_gpu_side_concurrency_matches_host` 等），必须用：

```bash
mamba run -n pyqcd-einsum-cu124 python -B -m pyqcd.testing
```

用系统 `python` 会得到
`CompileException: nvrtc: error: invalid value for --gpu-architecture (-arch)`：
系统 `cupy 14.2.0` 配的是 **nvrtc 13.2**（CUDA 13 已移除 `sm_60`/`sm_70`），而
mamba 环境配的是 **nvrtc 12.9**，与 `cupy.cuda.runtime` 的 `12090` 一致。同一份代码
在 mamba 环境 `13 passed / 0 failed`，在系统 python 会 1 个 ERROR。

这**不是**产品缺陷，也不是"必须降 arch"——不要为此改 `CUDAARCHS` 或
`CUPY_NVCC_GENERATE_CODE`，换环境即可。

### 6.2 已知缺陷：v3 步表的收缩序按 `PROBE_DIM=2` 定型（2026-11-14 记录，未修）

`pyqcd/testing/_topology_kernel_gen.py:43` 用 `PROBE_DIM = 2` 且 `optimize="greedy"`
生成步表，于是**收缩序由 dim=2 的超立方体决定**，与生产异形轴无关。实测在验收用的
6 个 seq 上是病态序：

| seq | 探针序 MAC | 生产形状 MAC | 倍数 |
|---|---:|---:|---:|
| 500 | 2.287e9 | 1.912e8 | **11.97×** |
| 1700 | 1.490e7 | 1.444e6 | 10.31× |
| 1000 | 3.051e6 | 3.833e5 | 7.96× |

同样 13 步做约 12 倍的乘加，`M=15625 K=1600 N=25` 这类巨中间体即由此而来
（`dedicated` 的同规模步是 `625×625×400`）。后果：实数 dtype 的 topology 路径在
seq=500 / scene scale=8 上是 **246.8 ms vs dedicated 4.7 ms（52.9×）**。

**已采取的措施（方案 B）**：`_kernel_kupl._TOPOLOGY_DECLINED_DTYPES` 在**分派层**
拒绝 `float64`/`float32`（`status=4`，原因含 `declined by policy`），实数交给
`kupl_dedicated`（它在 `_api` 的分派顺序里更靠前）。设
`PYQCD_EINSUM_TOPOLOGY_REAL=1` 可强制放开做 A/B 对照。复数是"能力支持但原生未实现"
的另一层问题，见下条。

**不要**把这当成"v3 已经修好"。以下三条是**已被实测推翻**的说法，禁止再写进报告：

1. "complex 走 v3 步表且与 dedicated 持平" —— `topology_apply.cpp:232` 对非
   float32/float64 直接 `return kSpecializedMiss`，随后 `:346` 仍调
   `execute_parsed_contract`（**v2 通用执行器**）。所以复数测到的 12.7 ms 是 v2 的
   耗时，与 v3 无关；`try_topology_apply` 对复数返回非 None 只是 v2 兜底命中。
2. "v3 缺 batch 并行，补上就能提速" —— 验收用的 105 步里 `batch_rank > 0` 为 **0 步**
   （全语料 121724 步中有 11990 步带 batch，但不在这批 seq），补了是死代码；且
   `packed_b` 是 `[batch][K][N]`，折平成 `[K, batch*N]` 的 memcpy 前提不成立。
3. "步表已按真实轴长还原" —— 轴长还原（v3 的 `label_sizes`）是对的，但**收缩序**
   仍来自 dim=2 探针，两者是不同层面的问题。

正解（需单独授权，会改"步表是静态生成产物"这个设计前提）：让步表按生产 label 尺寸
定型，或运行期按真实 `label_sizes` 重排收缩序（等价于用 dedicated 的 `auto` 计划）。
改动面：`_topology_kernel_gen.py` + 重生成 `topology_kernels.inc` + 全语料对拍。
