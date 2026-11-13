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

- dtype：`complex128` 为主，`complex64`、`float64` 必须支持；混合精度 **fail closed**。
- 数值验收：与 `numpy.einsum` / `opt_einsum` 逐元素 `rel ≤ 1e-12`（complex128）。
- 性能验收：同后端、同 dtype、同线程数，`median(≥10 次)`；CPU 对 CPU、GPU 对 GPU。
- 二元收缩规范布局：左 `[B,M,K]`、右 `[B,N,K]`，K 必须是两侧最低连续维；GEMM
  执行语义为 `out[B,M,N] = left[B,M,K] @ right[B,N,K]^T`。
- 稳定基准入口应缓存计划/compiled context，不缓存输入内容；同一 plan 的重复调用必须
  使用本次输入指针且不能返回旧数据。

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
