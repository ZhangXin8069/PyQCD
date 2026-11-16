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
         memory_limit=None, backend="auto", specialize=None,
         kernel_force=None, **kwargs)

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
  `Nev_src=100`/`Nev_curr=400`、`Mom_src=1`/`Mom_curr=3/9`（**可变量**，生产里两者
  都在用，本档取 9）、`Ns=4`、`Nc=3`。
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
| `Mom_curr` | 3/9（可变量） | `:173` `n_mom_curr = 3` | current 端动量数**可变**，生产里 3 与 9 都在用；本档取 9（分配 `(9,400,400)`，`:224`）。引用时须写明用的哪个 |
| `Ns` | 4 | `:200` peram `(4,4,nev,nev)`；`:248` `gamma(4)` 4×4 | Dirac 旋量维 |
| `Nc` | 3 | `refer/sush/_distillation/constant.py:1` | 色数；**不是**本脚本的张量轴（色已在 peram 内收缩） |
| `Nt_labels` | 4 | `:179` `times = (tsink,tsrc,tcur0,tcur1)` | 注册表键个数，不是张量轴 |
| `Nx` | 无 | 该文件无匹配 | 蒸馏张量没有空间轴；`dims("distill")` 用 `Nx=1` 作"无此轴"占位 |
| `Lt` | 无 | 该文件无匹配 | 同上；`Lt=1` 为占位 |

`nev_by_time = {tsink:100, tsrc:100, tcur0:400, tcur1:400}`（`:184-189`），因此
`peram[tq,ta] = (4, 4, nev[tq], nev[ta])`（`:200`）在同一张图里出现
`(4,4,100,100)`、`(4,4,100,400)`、`(4,4,400,100)`、`(4,4,400,400)` **四种异形轴长**。
`distill_tensor_shapes()` / `distill_memory()` 给的就是这组真实形状与内存估算
（`3pt` 段 0.384 GiB、`6peram` 段 0.194 GiB；最坏可信 6pt 峰值下界 5.16 GiB < 15 GiB）。
`3pt` 段总量随可变量 `Mom_curr` 线性变化，上值对应本档 `Mom_curr=9`。

已知口径差异（**不要**在没有依据时"顺手修正"）：

1. `bench_specialized._label_sizes` 的形状口径有**两种形态**（附录形态的错位绑法已于
   2026-11-15 修正，见 §6.2 之后的「附录动量轴」小节）：
   * **基础表**（seq 1..2582，指标段纯小写）：`VVV` 当 `(Lt, Mom, Nev, Nev, Nev)`、
     `VDV` 当 `(Lt, Mom, Nev, Nev)` —— 参考脚本里**没有** `Lt` 轴，这是历史约定，
     逐位未改；
   * **附录形态**（seq ≥ 2583，指标段 `<大写动量标签><小写>...`）：`(Mom, Nev, ...)`，
     与参考脚本注册的 `VDV = (Mom_curr, 400, 400)`、`VVV = (Mom, Nev, Nev, Nev)` 一致。
   仍然表达不出、也不该由它表达的只有一条：**同一方程里只有一个 `Nev`**（源端 100 与
   current 端 400 无法共存）。实测后果：拿 `dims("distill")` 喂**基础表**的 seq 3，
   `Nx/Lt=1` 的占位与 `Mom`（1 或 `Mom_curr`）会作为**真实轴长**漏进 peram 的轴
   （得到 `(4,100,100,100)`、`(100,4,1,1)` 之类），与参考脚本的
   `(4,4,100,400)` 无关。因此 **`--scene distill` 经 `_label_sizes` 的基础表形状只能当
   path-only 证据，不得写成蒸馏生产结论**；真实形状只认 `distill_tensor_shapes()`。
   该限制由 `_bench_scene_distill_contract.test_label_sizes_cannot_express_distill_axes`
   锁定；附录形态的绑法由 `_label_sizes_contract` 锁定。
2. `VVV` 分配 `(1,100,100,100)`（`:209`）但注册时用 `host_vvv_sink[0]` 切片（`:241`），
   进 einsum 的是 `(100,100,100)`；`VDV` 保留动量轴（`:243-245`）→ `(1,100,100)` /
   `(Mom_curr,400,400)`（本档 `Mom_curr=9`，即 `(9,400,400)`）。两者不对称，照抄参考脚本即可。
3. 语料里**基础表**的 `VDV` term 只有 2 个标签（seq 1/855 的 `bd`/`fn`），附录形态则是
   3 个（`Nbd`，动量标签打头）——后者才与参考脚本注册的 3 轴 `VDV` 对齐；基础表那两个
   标签就是「无动量轴」的旧约定（首轴 `Lt`、次轴 `Mom`）。两形态的判据与逐轴长度见
   `pyqcd/testing/_label_sizes_contract.py`。

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

`python -m pyqcd.einsum.verify_specialized "<eq>"` 支持 `--language {auto,en,zh-cn}`
（缺省 `auto`：只在 `PYQCD_EINSUM_LANG` 显式设了时跟随，未设时保持中文，**不跟随
系统 locale**）。它只改人读文案，不改判定、退出码或 `--json` 的结构化字段
（`--json` 仍是 `query()` 的原样序列化）。契约：
`pyqcd/testing/_einsum_verify_specialized_contract.py`。

同一个入口支持 `--equivalence {auto,true,false}`（缺省 `auto`：跟随
`PYQCD_EINSUM_EQUIVALENCE`，未设即开启；`true`/`false` 强制开关并**压过**环境变量，
`true` 连 `_equivalence.enabled()` 那道内部开关也压）。开启时未命中的方程多一段
「等价写法 + `张量调整`（输入侧）+ `结果调整`（输出侧）」——**两行恒在**，某侧无需
调整就写 `-`，输出侧不需要回填时写 `-（改写后输出轴序已一致）`；关闭时该段整段消失、
MISS 原因逐字回到引入前的原文。`--json` 的 `equivalence` 字段开启时**总是**写
（没有改写也写 `rewritten=false`，用来区分"查过"与"没查"），关闭时**缺该字段**
（与 `test.py` 同口径）。它只是附加信息：不改 `matched`、不改退出码、不改执行路径。

### 9.1 等价映射 `_equivalence.py` 的开关 `PYQCD_EINSUM_EQUIVALENCE`（2026-11-16）

`pyqcd/einsum/_equivalence.py` 把"同一个收缩换了个写法"映射回语料规范形式，覆盖
三族改写：**操作数位置 / 单操作数轴序 / 输出段轴序**；三族**可任意复合**（如
`cadb,menf,db,em,ac,fn->` ↔ `cadb,menf,ac,em,bd,fn->`），`kind` 用 `+` 连接
（`operand-order+axes` / `operand-order+axes+output-axes` 等，见 `_kind`）。
`_kind` 的完整取值共 8 个：`identity` / `operand-order` / `operand-axes` /
`output-axes`，加三族的两两复合与三层复合共四种；`find_equivalents()` 另有
`normal-form`，不由 `_kind` 产生。
字母重命名不归本层，由 `canonical_signature` 负责。

| 项 | 值 |
|---|---|
| 环境变量 | `PYQCD_EINSUM_EQUIVALENCE`（`_equivalence.EQUIVALENCE_ENV`） |
| 缺省 | `"1"` —— **开启** |
| 关闭 | `0` / `false` / `off` / `no`（大小写不敏感，去首尾空白） |
| 求值时机 | **调用时**读 `os.environ`（与 `_specialize.notice_enabled()` 同风格），同进程内改环境变量即时生效 |
| 影响面 | **只影响查询报告，不影响任何数值计算**：`contract()` 仍按调用方给的方程走 |

关闭后行为与"没有这个功能"完全一致：`normalize_equation()` 返回 `None`（零枚举、
零 `probe` 调用）、`find_equivalents()` 返回 `[]`、`explain()` 返回
`{"rewritten": False, "reason": "等价映射已关闭（PYQCD_EINSUM_EQUIVALENCE=0）"}`，
`verify_specialized` 的 MISS 报告回到原措辞
（"既不在 wick_12cases 语料里，也没有形状特化路径，走通用分步路径（无特化内核）"），
报告里**不出现** `equivalence` 字段。契约用例
`test_disabled_verify_specialized_output_matches_the_pre_change_original` 用
`fd26357e` 实跑抓取的原文逐字符比对钉住这一点。

**默认路径零枚举**（2026-11-16 重写）：结构不变量分桶（`_Shape`）+ 桶内构造式同构
（`_match`），复杂度 O(incidence + Σ Σ|term|²)，产出三个纯置换字段。位置次序取两两
互换、同时改轴序的位置只取 1 或 2 这类**叠加限制在默认路径上已不存在**——三族任意
复合都走同一条同构搜索（`probe=` 兼容路径仍保留 `changed∈{0,1,2}` 这类有界枚举）。
`_MAX_SEARCH_NODES=4096` / `_MAX_LABEL_OPTIONS=512` 是**同构搜索**上限（不是覆盖面
上限），触发只会漏判、不会误判；`512` 同时用于 `probe=` 路径的排列截断。
旧实现的 `_candidate_equations` /
`_MAX_CANDIDATES` / `orders[:96]` 已全部删除；只有调用方显式传 `probe=` 时才退回
有界改写枚举（`_MAX_PROBE_CANDIDATES=2048`、`_PROBE_FULL_PERM_LIMIT=6`），两个 CLI
都不传。契约：
`pyqcd/testing/_einsum_equivalence_contract.py`（**44 项**，含按
`_label_sizes`+`dims("distill")` 取的 seq=1 形状数值对拍、4 个置换方向判别用例，以及
"零代价优先 / 退化到旧选择 / 上界小于最大桶"三条选择策略用例）。

> 形状口径提醒：`_label_sizes` 反推的形状**不是**蒸馏生产形状（见根 `AGENTS.md`
> 硬性禁止 #3；真实形状只认 `_bench_scene.distill_tensor_shapes()`），这里只当
> "一组真实量级的形状"用。

**同桶多条同构写法怎么挑**（2026-11-16）：桶里常有多条结构同构的语料条目。先按
`_alignment`（逐位置同集合 / 逐位置同文本 / 共有字母数）排序，再在前
`_MAX_ZERO_COST_SCAN=64` 条里用轴序敏感指纹 `_Shape.axis_key` 预筛，取**第一条零操作数
轴置换且零输出置换**的映射；找不到才退回"按对齐序取第一条成功者"。`operand_order`
不计代价——它只是换实参次序，张量仍 C 连续。修前 8638 条 order 变体里 772 条带轴置换
且桶内均有零代价替代，修后为 0。

映射给出 `operand_permutations` 时注意它是 `transpose(perm)` 语义、返回**视图**而
通常非 C 连续，而 `_kernel_kupl.try_dedicated` 的连续门（`_kernel_kupl.py:1452`，
位于 enabled/类型/dtype/optimize/shapes 五项轻量检查**之后**）会拒非连续操作数：
目标是原生特化路径就先落一份 `np.ascontiguousarray`（代价：一份额外内存 + 一次拷贝）。
不做副本时的落点按 dtype 二选一：**复数**档被拒后落 `kupl_topology`（该层没有操作数
连续性门，在原生边界静默复制一次并记 `LAST_INFO["copied"]`）；**实数**档（float64 /
float32）直接落通用分步路径（`_TOPOLOGY_DECLINED_DTYPES`）。

### 9.1b 强制指定专用内核 `kernel_force=` / `PYQCD_EINSUM_KERNEL_FORCE`（2026-11-16）

强制内核是**纯选路**语义：数值永远等价于通用路径，所以普通数值测试抓不到它的
缺陷（标签解析错了只是"强制没生效"，分派错了只是"跑成了别的内核"）。约定与踩坑
记在这里；用法与排查见 `skills/pyqcd-einsum/SKILL.md`，用户视角见
`docs/einsum_user_guide_20261116.md` §11。

**三态取值**（`_kernel_force.parse`，`:97`）：

| 取值 | `forced_labels()` | 分派行为 |
|---|---|---|
| `auto` / `None` / 空串 | `None` | 原次序（`*_dedicated` → `*_topology`），失败**只** `RuntimeWarning` + 回落 |
| `generic` | `()` | 完全不调 `_run_specialized_kernel`、不发 warning，直接通用路径 |
| `<label>[,<label>]` | `(label, ...)` | **只**试这些、按书写次序；一个都没跑成 → `KernelUnavailableError` |

`KERNEL_LABELS`（`:57`）恰为 `_api._specialized_kernel_attempts` 的四个标签：
`cuda_dedicated` / `cuda_topology` / `kupl_dedicated` / `kupl_topology`。解析细节：
去空白、大小写不敏感、保序去重、`auto` 在列表里被忽略（`auto,kupl_dedicated`
等价于 `kupl_dedicated`）；`generic` 与具体标签同时出现抛 `ValueError`；未知标签抛
`ValueError` 且**列出合法集合**——静默忽略拼写错误会让用户以为强制生效了。

**优先级**：显式 `kernel_force=` > `PYQCD_EINSUM_KERNEL_FORCE` > 未设（= `auto`），
由 `_kernel_force.resolve`（`:133`）判定。取值在**调用时**解析，同进程内改环境变量
即时生效。注意 `--kernel-force auto` 与"不给"不同：显式 `auto` 会把环境变量**压掉**
（`resolve("auto")` → `()`，而 `resolve(None)` 仍读环境变量）。

**接线**（`_api.contract`）：`forced = _kernel_force.forced_labels(kernel_force)`
（`:857`）之后分三条互斥分支——

- `forced == ()`：命中特化形态只 `_specialize.announce`，不进专用分支（`:894`）；
- `forced` 非空：`_run_forced_kernel`（`:628`）按 `requested` 次序真试，失败即抛
  （`:897`、`:926` 两处，后者覆盖"语料没命中却强制了标签"）；
- `forced is None`：原有分支逐字不变（`:903`）。

**与既有开关的关系**（不是互斥，是不同层）：

| 开关 | 管什么 | 与强制的交互 |
|---|---|---|
| `PYQCD_EINSUM_KERNEL_KUPL=0` / `_CUDA=0` | **关掉**该后端原生入口 | 能同时给；此时强制该后端的标签 → `KernelUnavailableError`，原因 `kernel disabled (PYQCD_EINSUM_KERNEL_* = 0, or library missing)` |
| `specialize=False` / `PYQCD_EINSUM_SPECIALIZE=0` | **整体跳过**专用路径 | 与具体标签**冲突**，`ValueError` 早失败（`:932`） |
| `kernel_force=...` | **只准跑这个** | —— |
| `out=` | 禁用专用路径 | 与具体标签冲突，`ValueError` 早失败（`:864`） |

三条早失败都是"一次内核都没试就抛"（`_LAST_KERNEL` 未被写），由
`test_kernel_force_conflicts_fail_early` 钉住。`kernel_force="generic"` 与
`out=` / `specialize=False` **不冲突**（generic 本就不走专用路径）。

**负缓存必须旁路**：`PYQCD_EINSUM_NEGATIVE_CACHE`（`_kernel_kupl.py:59`，缺省开）
会把 status=4 之类的确定性拒绝记下来，于是第二次调用连试都不试、原因变成
`negative-cache hit: ...`。强制的语义是"强制即真试一次"，所以
`_run_specialized_kernel` 在每个 attempt 前套上模块自己的 `bypass_negative_cache()`
（`_kernel_kupl.py:88`；`_kernel_cuda.py:43` 只是把 KUPL 侧那个转出去——CUDA
**没有**自己的负缓存，键里的 backend 已把两者分开）。旁路由**模块内计数器**实现、
不碰 `os.environ`（环境变量是调用方与并发线程共享的状态），进入 +1/退出 −1 支持嵌套，
旁路期间**既不读也不写**缓存，退出后 `negative_cache_info()` 与进入前逐项相同。

**可观测记录**：`_LAST_KERNEL` + `last_kernel()`（`_api.py:541`），字段
`requested`（`None`=自动 / `()`=只走通用 / 标签元组）/ `ran` / `attempted` /
`reasons`。每次走进专用分支**整体覆盖写**（`:890`），所以读到的永远是最近一次调用，
不会残留上一次的命中。注意它**没有**在包级 `__all__` 导出，要
`from pyqcd.einsum._api import last_kernel`。

**报告行**（两个 CLI 共用同一格式，机器可读、**不随** `--language` 翻译）：

```
kernel : requested=<auto|generic|label,...>  ran=<label|none>  attempted=<label,...>
```

- `requested` 恒有值（来自本次解析，`auto` 时反映那一刻环境变量的生效值）；
- `ran` / `attempted` 取不到写 `unknown`（"没观测到"，不是"没有"）；
- 与既有 `kernel   :` 行（注意三空格对齐）**是两回事**：那个是 `kernel_info` 的
  dict（`LAST_INFO` 汇总），`ran` 才是"这次真跑了哪个"。

**踩坑**：`verify_specialized --kernel-force` **不做收缩**，所以它的 `ran` 恒为
`unknown`——`--kernel-force` 在那儿只决定报告里出不出这一行、以及 `requested=`
显示什么。要真跑内核只能用 `python -m pyqcd.einsum.test --kernel-force <label>`。
契约：`pyqcd/testing/_einsum_kernel_force_contract.py`（**9 项**，纯选路断言、
零计时）。

### 9.1c 自动施加等价写法 + 缓存（2026-11-16）

**自动施加**（`as_equivalence`）：`contract(..., as_equivalence=None)` 缺省跟随
`PYQCD_EINSUM_AS_EQUIVALENCE`（缺省开启）；`--as-equivalence` 是它的 CLI 三态。
开启且方程未命中特化形态时，`_contract_via_equivalence` 会：按 `operand_order`
重排操作数 → 按 `operand_permutations` 转置并落连续副本（视图会被 dedicated 的
连续门拒）→ 递归 `contract(..., as_equivalence=False)` → 按 `output_permutation`
把结果转回。**返回的永远是原方程的答案**，并有一道独立输出形状守卫
（`_expected_output_shape`），算得出原形状却对不上时抛 `RuntimeError` 而不是交出
错结果。两道硬闸不可被显式 `True` 压过：`PYQCD_EINSUM_EQUIVALENCE=0` 与
`specialize=False`。不施加：非字符串方程、`out=`、额外 `kwargs`、逐元素方程、
cupy 通用委派、`optimize` 为 callable 或显式 `list/tuple`（字符串/布尔优化器**仍会**
施加）。**代价**：小形状上改写+副本开销占主导（实测 2.92 ms vs 0.28 ms），批量小
收缩建议关掉。

**缓存**：`_cache.LRUCache` 提供有界线程安全 LRU + `hits/misses/stores/evictions`，
实例化即注册。当前三处：`specialize.match`（键=原始方程串，**含未命中**）、
`equivalence.normalize`（键=原始方程串，**含 `None` 负结果**；`probe=` 路径不缓存）、
`api.find_path`（键含 optimize/输入集合/输出集合/size_dict/memory_limit/eq/形状/
use_blas；callable 与显式路径不缓存，返回值按**只读**复用）。总开关
`PYQCD_EINSUM_CACHE`；`contract(..., cache=)` / `contract_path(..., cache=)` 三态；
`cache=False` 时 `LRUCache.get/put` 既不读也不写（统计零变化）。
`pyqcd.einsum.cache_stats()` / `cache_clear()` 是公开入口；CLI `--cache` 只在显式
给出时打印 `cache    : enabled=... hits=... misses=... hit_rate=...`。

### 9.0 单方程测试的唯一接口：`python -m pyqcd.einsum.test`（强制）

> **硬性规定（2026-11-15 起）**：`qcd-einsum` 的**单方程验证/复现/性能对比**只有
> 一个合法入口 —— `python -m pyqcd.einsum.test`。
> 不得新增 `/tmp/*.py`、`scratchpad/*.py`、一次性 `python -c`、或
> `pyqcd/testing/` 下的临时脚本去替代它。违反此规定的验证结论**不予采信**，
> 也不得写进报告、文档或 tag 消息。

强制的原因不是形式主义，是三次实测教训：

1. **warmup 口径不一**：临时脚本普遍测单次，而首调含 plan 缓存 + 原生库绑定 + JIT，
   实测 seq=2583 首调 45 ms vs 热态 0.56 ms（**80×**），单次测量完全失真。
2. **容差口径不一**：`np.allclose` 的逐元素绝对阈值会误判接近零的元素；本入口统一用
   相对容差（按 `max|expected|` 归一）。
3. **参照不固定**：有人比 `np.einsum`、有人比 `opt_einsum`、有人不比，结论无法互比。

```bash
python -m pyqcd.einsum.test --help      # 全部选项与缺省值（唯一权威来源）

python -m pyqcd.einsum.test --device gpu --dtype complex128 \
    --equation 'ambn,cgdh,ekfl,oipj,ce,mo,gi,bdf,np,hjl->ak' \
    --shape '[400,4,1,1],[4,4,9,1],[4,400,400,400],[4,4,9,9],[4,4],[4,4],[4,4],[1,9,400],[1,9],[1,9,400]->[400,400]' \
    --atol 1e-12 --backend cupy --verbose true --compare opt_einsum
```

**唯一例外**：整库回归仍用 `python -m pyqcd.testing`（它答"整库有没有退化"，
本入口答"这一条方程对不对、多快"）。两者互补，不可互相替代；除此之外没有第三个入口。

**违规如何被发现**：本规定由契约
`pyqcd/testing/_einsum_test_cli_contract.py` 护航（形状解析、参数校验、相对容差口径、
缺省值、`--help` 完整性、退出码语义、内存采样不在计时循环内）。新增/修改本入口必须
同步该契约，否则套件红。

**“这是唯一规范端”的意义**：本入口是"单条方程、这组形状、这个 dtype、这块设备，
算得对不对、多快"的唯一规范端——不要再另立第二套口径。

缺省值（`--help` 同源）：

| 选项 | 缺省 | 说明 |
|---|---|---|
| `--device` | `cpu` | `cpu`→numpy，`gpu`/`cuda`→cupy |
| `--dtype` | `complex128` | 四类平权：complex128/complex64/float64/float32 |
| `--backend` | `auto` | 原样传给 `contract(backend=...)`；可选 auto/numpy/cupy/torch/kupl/cuda |
| `--compare` | `numpy` | 参照：numpy / opt_einsum / **backend**（跨后端互证）/ none |
| `--compare-optimize` | `auto` | **只看参照**的收缩序优化器：auto / greedy / dp / optimal / branch-all / branch-2 / branch-1。`auto` = 沿用旧行为（numpy→`greedy`，opt_einsum→`greedy` 或跟随 `--optimize`）。两库名字不通用、**不做跨库重映射**：numpy 只认 `greedy`/`optimal`（`dp`/`branch-*` 报用法错误，退出码 2）；opt_einsum 按 `oe.paths` 注册表校验。`--compare backend`/`none` 下无效（verbose 提示，不报错） |
| `--optimize` | `auto` | 同 `contract(optimize=...)`，**只管主路径** |
| `--specialize` | `auto` | 专用内核三态：`auto`=跟随 `PYQCD_EINSUM_SPECIALIZE`；`true`=强制特化路径；`false`=**强制通用路径**（排查原生内核问题的第一手段）。主路径与 `--compare backend` 的参照路径同时生效 |
| `--equivalence` | `auto` | 等价写法报告三态：`auto`=跟随 `PYQCD_EINSUM_EQUIVALENCE`（**缺省开启**，只有 `0/false/off/no` 才关闭）；`true`=总是查；`false`=不查。方程未命中任何特化路径、但存在能命中的等价写法时，`--verbose true` 的明细多一段 `equiv`：改写后的方程 + **输入侧张量调整**（`transpose(perm)` / `order=[...]`，无调整写 `-`）+ **输出侧结果回填**（`transpose(perm)`，无则写 `-`）。已命中特化的方程不出该段；没有等价写法时明写「无」。`--json` 多一个 `equivalence` 字段（`_equivalence.explain()` 原样），关闭时缺该字段。**只做报告**：不改 `contract()` 行为、不进计时区域、不改任何数值与耗时 |
| `--atol` | `1e-12` | **相对**容差（按 `max\|expected\|` 归一，不是逐元素绝对阈值） |
| `--rtol` | `0` | 附加相对余量 |
| `--rounds` | `5` | 计时轮数（中位） |
| `--warmup` | `1` | 丢弃的预热次数，**不允许 0** |
| `--seed` | `0` | 操作数随机种子（各路径同一批数据） |
| `--verbose` | `false` | true 时把逐项明细打到 stderr |
| `--language` | `auto` | 本入口的**输出语言**：`auto` / `en` / `zh-cn`。`auto` 只在 `PYQCD_EINSUM_LANG` **显式设了**时才跟随它（`en*`→英文，`zh*`→中文），未设时保持历史行为**中文**；因此缺省行为零变化。它**不跟随系统 locale**（本机 `LANG=C.UTF-8`，跟随 locale 会让不传选项的输出从中文变英文），全程复用 `_specialize.notice_language()` 的那一个环境变量，不另立开关。只影响文案：不改任何计算结果、不改 `--compare` 的选择、**不写**任何环境变量（同进程后续调用不受影响）。缺 key 时 `translate` 直接抛 `KeyError`，不做静默兜底 |
| `--shape` | 省略 | 省略时按 `--scene`（缺省 `distill`）从语料 `roles` 反推生产尺寸 |
| `--json` | 无 | 完整结果落盘 |

`--compare backend` 是**跨后端互证**：主路径跑 `--device`，参照在另一个后端上跑
同一批张量（CPU↔GPU）。这是唯一能同时验证"两条原生路径都给对答案"的模式。

`--compare-optimize` 是**参照侧**的收缩序开关，与主路径的 `--optimize` 完全独立：
改它绝不改主路径的收缩序。为什么必须分开——两条参照路径的名字空间不同，且 opt_einsum
内部 greedy 与 dp 的差距是实打实的（seq=856、`--rounds 5` 中位，同一台共享机上两组
配对样本：`greedy` 1848.2 / 1578.0 ms vs `dp` 1591.2 / 1274.1 ms，即 dp 快
**14–24%**；绝对值随机器负载漂移，**比值方向稳定**），而旧入口在 `--optimize auto`
下会把参照也按 `greedy` 跑，无法单独指定 `dp`。**不做跨库重映射**：

* `--compare numpy`：`np.einsum` 源码只认 `greedy`/`optimal`，其余一律
  `KeyError('Path name %s not found')`（numpy 2.5.4 实测，`dp`/`branch-*`/`auto`
  全部抛错）。所以传 `dp` 是**用法错误**（退出码 2），不是"退回 greedy"；
* `--compare opt_einsum`：按 `oe.paths._PATH_OPTIONS` 注册表校验，`dp`/`greedy`/
  `optimal`/`branch-all`/`branch-2`/`branch-1`/`auto` 均可用；
* `--compare backend`/`none`：没有可设的优化器，显式非 `auto` 值只在 stderr 提示
  无效（不报错）。

报告里有 `compare_optimize`（要什么）与 `compare_optimize_effective`（实际用了什么）；
verbose 的 `ref opt` 行与 `reference` 串都打**参照实际用的**，一行式 PASS 带
`ref …ms@dp`。缺省 `auto` 逐字复现旧行为，契约
`test_compare_optimize_default_reproduces_the_old_reference_behaviour` 钉住这一点——
缺省悄悄改动就会改掉所有历史对照结论。

`--specialize` 是**专用路径 vs 通用路径**的对照开关，三态直传
`contract(specialize=...)`（`pyqcd/einsum/_api.py`）。实测用例：seq=856 走 dedicated
要 8662.7 ms，`--specialize false` 的通用路径只要 2330 ms——没有这个开关就只能靠改
环境变量，且 `--compare backend` 无法与主路径保持同一策略。`--verbose true` 会打印
`specialize: <选择>（实际 <bool>）`，且 `false` 时 `kernel` 行必须是
`无原生命中（通用路径）`；若仍出现 `kupl_dedicated`/`cuda_topology`，说明该参数没
传到底层，属于必须修的 bug（契约
`test_specialize_false_really_disables_the_native_path` 钉住）。

**两条硬性口径**（改它就是改规范）：

1. `--atol` 是相对容差。`np.allclose` 那种逐元素绝对阈值会误判接近零的元素。
2. **warmup 不可为 0**。实测 seq=2583 首调 45 ms、热态 0.56 ms（差 80×），
   单次测量会被 plan 缓存 + 原生库绑定 + JIT 完全支配。

典型输出（本条方程 10 操作数 / 2.56 亿输入元素）：

```
# CPU (--device cpu --backend numpy)，默认一行式
PASS ... rel=3.45e-15 vs opt_einsum.contract(optimize='greedy')  9858.038ms
       (ref 1350.031ms, 7.30x)  rss+4004MiB

# GPU (--device gpu --backend cupy --verbose true)
equation : ambn,cgdh,ekfl,oipj,ce,mo,gi,bdf,np,hjl->ak
device   : gpu  dtype=complex128  backend=cupy
operands : 10 个，共 256,010,297 元素；输出 [400, 400] (160,000 元素)
time     : 20.943 ms （warmup=1，rounds=5，中位）
reference: opt_einsum.contract(optimize='greedy')
ref time : 1489.477 ms  （pyqcd/参照 = 0.01×）
rel      : 1.075e-15  (atol=1e-12)
memory   : 输入张量 +3999 MiB RSS，+3906 MiB 显存
peak     : RSS +4065 MiB，显存 +3909 MiB，总显存 32768 MiB
pyqcd    : 0.4.2  /home/zhangxin/PyQCD/pyqcd/__init__.py
kernel   : {'cuda': {'path': 'cuda_topology', 'steps': 9, 'copied': ()}}
PASS ... 20.943ms (ref 1489.477ms, 0.01x)  vram+3909MiB
```

三处读数各有用处：

| 行 | 含义 |
|---|---|
| `ref time` | 参照实现的耗时 + `pyqcd/参照` 比值。**这条方程上 CPU 与 GPU 结论相反**：CPU 侧 pyqcd 9858 ms vs opt_einsum 1350 ms（**慢 7.3×**），GPU 侧 pyqcd 20.9 ms vs 1489 ms（**快 71×**）。所以比值必须连同 `device` 一起读 |
| `memory` | 构造输入张量带来的增量（`RSS` 与 cupy 显存池 `used_bytes`） |
| `peak` | 跑完收缩后的总增量，含中间体；GPU 侧另给池保留量与设备总显存，便于判断会不会 OOM |

内存采样点只有三个（构造前 / 构造后 / 收尾），**都不在 `_timed` 的计时循环内**，
因此不污染 `--rounds` 的中位。契约
`test_memory_snapshot_is_outside_the_timed_region` 用"同一命令连跑两次、耗时差异
<2×"把这个结构性要求钉住。

`kernel` 字段直接暴露走的哪条原生路线（`kupl_dedicated` / `cuda_topology` /
`kupl_topology` / `kupl_topology_v2`），省去再翻 `LAST_INFO`。

**分工**：本入口负责"单点排查"；`python -m pyqcd.testing` 负责"全量回归"。
两者不可互相替代——前者答一条方程对不对、多快，后者答整库有没有退化。

其余验证入口（整库级）：

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

**根因（2026-11-15 查清）**：base 环境的 **pip 侧**装了 `cuda-toolkit 13.2.2` 与
`nvidia-*` 系列，它们优先于 conda 的 12.4，于是 `nvrtc.getVersion()` 返回
`(13, 2)`，而 `cupy.cuda.runtime` 是 `12090`、V100 是 `sm_70`——CUDA 13 已移除
`sm_70`，任何 cupy 逐元素 kernel（含 `_probe_dtype`）都报
`invalid value for --gpu-architecture`。检查命令：

```bash
python -c "from cupy_backends.cuda.libs import nvrtc; print(nvrtc.getVersion())"
# 期望 (12, x)；若打印 (13, x) 就是这个坑
```

#### 用户实测专用环境 `pyqcd-test`（2026-11-15 建）

```bash
conda create -y -n pyqcd-test python=3.12
conda install -y -n pyqcd-test -c conda-forge "numpy=2" openblas gxx_linux-64=12 \
    cupy opt_einsum "cuda-version=12"          # 锁 CUDA 12，避开 pip 侧 13.x
conda run -n pyqcd-test python -m pip install -e /path/to/PyQCD \
    --no-deps --no-build-isolation
```

要点：
- **用 `conda` 而不是 `mamba`**：2026-11-15 一次 `mamba create` 在本机 segfault，
  留下一个没有 `python` 二进制的残缺环境（只能 `conda env remove -n` 清掉重建）。
- **必须 `cuda-version=12`**：不锁的话 conda 可能配到 13.x，重现上面那条 nvrtc
  `(13, 2)` 的错误。装完用上面那条 `nvrtc.getVersion()` 自检。
- **必须装 `openblas`**（含 `cblas.h` 与 `libopenblas.so` 软链）：KUPL 的每个 GEMM
  都走 cblas，缺开发包会让 `pip install` 在原生构建阶段直接失败（exit 3）。
  只有运行时 `libopenblas.so.0` 不够。
- 实测结果（V100，`--device gpu --backend cupy`）：
  `ambn,cgdh,ekfl,oipj,ce,mo,gi,bdf,np,hjl->ak` 走 `cuda_topology` 9 步，
  **20.3 ms**，rel 1.08e-15。

`--verbose true` 现在会额外打印实际加载的 pyqcd 版本与包路径
（`pyqcd : 0.4.2  /home/zhangxin/PyQCD/pyqcd/__init__.py`）。这是为了对付
"装的到底是哪个 pyqcd"——base 环境曾同时存在 `site-packages/pyqcd 0.1.3` 与仓库源码，
在仓库目录外跑 `-m pyqcd.einsum.test` 会直接 `No module named`，而 verbose 一行
就能看出用的是哪个。

#### 测试不得假设"本机看不见 GPU"（2026-11-15 记录）

`CUDA_VISIBLE_DEVICES=""` **只在 CUDA 运行时首次初始化前生效**。套件里只要有任何
更早的用例碰过 cupy，把它设成空串就没有任何效果，`_probe.probe()` 仍会看到真实
设备并给出 `ok` / `fallback` / `call-failed` / `mismatch`。因此
`_contract_edge_cases_contract.test_gpu_probe_survives_missing_device` 断言的是
"**给出可读结论而不是 traceback**"这一契约，允许集合必须覆盖全部合法状态
（含 `fallback`：本机没构建 CUDA 原生库时 `qcd_cuda_topology_apply` 不导出，
探测如实报 `fallback`）。

同理，`_TOPOLOGY_DECLINED_DTYPES` 的实数回落是**分派选路**而非能力缺失，所以
`_topology_kernel_contract` 里两个"四 dtype 都要命中原生"的用例在作用域内临时
打开 `PYQCD_EINSUM_TOPOLOGY_REAL=1`，验完即恢复——它们要钉的是原生步表对四类
dtype 的正确性，选路偏置由 `test_dedicated_routing_contracts` 负责。

**不要**把具体 GPU 型号/UUID 写进代码、测试默认值或文档结论：本机实测用的
`CUDA_VISIBLE_DEVICES` 只是运行的临时环境变量，发布物保持设备无关。

### 6.2 已修复：v3 步表的收缩序按 `PROBE_DIM=2` 定型（2026-11-15 修复）

**原缺陷**：`_topology_kernel_gen.py` 曾用 `PROBE_DIM = 2` + `optimize="greedy"`
生成步表，于是**收缩序由 dim=2 的超立方体决定**，与生产异形轴无关。等长时全部轴
同长，路径代价与轴长无关，48/48 全绿恰好掩盖了它；一到生产异形轴
（`Ns=4`/`Nev=100`/`Mom`/`Lt`）就暴露。

**修复**（`_topology_kernel_gen.py`）：

1. `plan_steps(equation, sizes=None)` 接受真实 label 长度；`main()` 用
   `_production_sizes()` 从语料自带的 `roles` + `distill` 生产 dims 反推每个 label
   的尺寸，直接复用 `pyqcd.einsum.bench_specialized._label_sizes`（本库唯一的
   「角色→物理尺寸」权威实现，另写一份必然漂移）。
2. 优化器从 `"greedy"` 改为 `"auto"`（`_OPTIMIZE`），与 `try_dedicated` 同口径：
   实测 seq=646 上 greedy 4.08e8 MAC vs auto 1.08e8 MAC（26%）。

**收益**（抽样 21 条、distill 尺寸、**MAC 口径**——这是收缩序本身的改善，
与执行器实现无关，因此不受下面 v3/v2 路线混算的影响）：

| 项 | 值 |
|---|---:|
| 更优 / 持平 / 更差 | **17 / 4 / 0** |
| 中位改善 | **222×** |
| 均值改善 | 4558× |
| 最坏改善 | 42738×（seq=259） |

端到端：seq=500 float32 由 **169.5 ms vs dedicated 155.0 ms（1.09×）**，修复前是
53.1×。小规模 seq（1291/1936/2581）仍有 6–9×，但绝对差只有 ~1.8 ms——那是拓扑
入口的固定开销（签名哈希 + plan 缓存查找），**不是** GEMM 形状退化。

**同批修复的第二处**：topology 路径原先**没有** CPU pack-bound 守卫，而
`_cpu_plan_fallback_reason` 只服务 dedicated。seq=646（`amwq` 四轴各 100，distill
尺寸）输出本身 1e8 元素、只做 2.16e8 FLOPs，topology 在中间体上做 1.67 GB
pack/unpack，实测 **5.2 s**。新增 `_kernel_kupl._topology_pack_bound_reason()`
（用 `_native_plan._simulate_plan` 的步序列与中间槽位形状算 pack 流量，判据与
dedicated 相同：pack ≥ 1 GiB 且 FLOPs ≤ 1e9 即劝回落），于是该例如实
`status=4` + `topology declined: pack-bound on CPU`，交给通用路径。

**dedicated 侧新增 `pack/flops` 比值判据（2026-11-15 补）**：上面那条是**绝对**字节
门槛，"活干得也多"的病态计划从它下面漏过去——seq=856
（`ambn,cgdh,ekfl,oipj,ce,mo,gi,bdf,np,hjl->ak`，形状
`[400,4,1,1],[4,4,9,1],[4,400,400,400],[4,4,9,9],[4,4],[4,4],[4,4],[1,9,400],[1,9],[1,9,400]`）
`pack_bytes = 8,281,198,592`、`flops = 2,096,413,216`，`pack >= 1 GiB` 成立但
`flops > 1e9`，因此放行；dedicated 实测 **8662.7 ms**（step 0 就要把 `ekfl` 的
3.9 GiB 整块重排成扁平 GEMM 布局），而 `specialize=False` 的通用路径 **2330 ms**
（干净进程 5 轮中位，样本 2116–2538）、改后实际回落的 `kupl_topology` **2675 ms**。

现加 `_HEAVY_PACK_RATIO_LIMIT = 1.0`（每 1 次 FLOP 至少搬 1 字节布局流量即视为
带宽支配），**仅在 `flops > 0` 时比较**；拒绝理由写明 `pack/flops` 比值与
`pack_bytes`/`flops` 两个原始数。标定（complex128，`distill`/`distill_curr` 各
32 格语料）：比值双峰，`distill` 放行格最高 0.313 / 被拒格最低 4.976，
`distill_curr` 放行格最高 0.937 / 被拒格最低 1.748，1.0 落在两处空档内。该判据
只作用于 dedicated；topology 侧的比值阈值按它自己的口径单独标定，见下面
「topology 侧跟进」一段（两条口径量的不是同一份账，数值不能互相搬）。

**已知副作用（实测，未修）**：比值不看绝对规模，`distill` 下 pack 只有 KB–MB 的小
计划也会超限——32 格里新拒 18 格。默认路径 `contract()` 实测改前→改后：
`distill` 10 个小格变慢 **1.9–4.1×**（0.11→0.38、0.13→0.44、0.15→0.60、0.33→0.88
ms 等，绝对差 ≤0.55 ms），回落目标 `kupl_topology` 的入口开销所致；`distill_curr`
侧 seq=855 **41.7→5.2 ms（8.0× 快）**、seq=3431 34.9→30.3 ms（1.15×）、seq=2583
持平，但 seq=880（比值 1.748）4.4→10.3 ms（**2.3× 慢**）。要去掉这类误伤，把比值
判据与 `pack_bytes >= _HEAVY_PACK_MIN_BYTES` 并列即可：新拒降到 `distill` 0 格 /
`distill_curr` 3 格（全是 7.9 GiB 级），seq=856 仍被拒。

**topology 侧跟进：`pack/flops` 比值判据（2026-11-15 补）**。`_topology_pack_bound_reason`
现在也有两条判据，形态与 dedicated 逐字对齐：比值**与**
`pack_bytes >= _HEAVY_PACK_MIN_BYTES`（1 GiB）并列、`flops <= 0` 时跳过、拒绝理由
写明 `pack/flops` / `pack_bytes` / `flops` 三个数。阈值不照搬 1.0，按本口径重新标定为
`_TOPOLOGY_PACK_RATIO_LIMIT = 2.5`。

标定（全语料 8638 条 × 两档，complex128；只统计 `pack >= 1 GiB` 且 `flops > 1e9`
的格子，绝对判据已拒的那一侧到不了这条门）：

| 档 | 格数 | 比值范围 | 空档 | 1.0 / 2.0 / 2.5 / 4.0 新拒 |
|---|---:|---|---|---:|
| `distill` | 3802 | 0.0016..0.3222（绝对判据拒的 688 格全 ≥ 14.69） | 0.322..14.69 | 0 / 0 / **0** / 0 |
| `distill_curr` | 5873 | 2.5e-5..15.93 | 4.07..11.57（最宽，但会放过 3.95） | 991 / 215 / **213** / 131 |

`distill_curr` 没有单一空档，2.5 落在次宽空档 2.15..2.77 内、对 seq=856 的 3.950
留 1.45 余量（3.9 只有 0.011），且不碰 `[1.0,2.15)` 那 776 格。全语料放行计数：
`distill` 3802/3802（新拒 0）、`distill_curr` 5660/5873（新拒 213，全部
`pack >= 7.65 GiB`）。契约：
`_topology_kernel_contract.test_pack_ratio_guard_declines_the_pathological_plan`
（含"抬高绝对底后必须放行"的并列性断言 + 三条不误杀反例 + 两条变异验证）。

**这一步的已知代价（已实测，根因在原生侧）**：`distill_curr` 上被新拒的那一族
（比值 ≈ 3.95、pack 8.28 GB、FLOPs 2.10e9）**不齐性**。同一份原始代码配对测
（`python -m pyqcd.einsum.test`，`--rounds 5` 中位；kernel 开 = topology vs
`PYQCD_EINSUM_KERNEL_KUPL=0` = 通用）：拒掉 seq=856/14/20 分别省 **4.3–9.0× / 4.1×
/ 3.3×**，而 seq=9/3/5 分别亏 **2.3× / 5.0× / 5.3×** —— 3 快 3 慢、量级相当。
根因不在比值：**非 `distill` 尺寸下原生步表的收缩序与运行期 plan 不是同一条路径**
（`PYQCD_EINSUM_TOPOLOGY_TRACE=1` 实测 seq=3 原生 `M=9 K=400 N=640000`、Python 推演
`M=4 K=640000 N=400`；seq=9/856 同样对不上）——步表在生成期按 `distill` 定型，尺寸
一变 `auto` 就换序。所以该比值在 `distill_curr` 上量的是"另一条计划"，只有
`distill` 口径（seq 3/855/2 逐步与原生对账通过）才是它的有效域，而那里这条门一格
都不拒。真正的修法在原生侧：按尺寸重生成步表，或像 CUDA 侧 §6.3 那样加一道
「收缩序可接受」守卫。

**CUDA 侧同一条比值判据（2026-11-15 补）**：`_kernel_cuda._plan_fallback_reason` 原先
只有绝对那一条，现加第四条 `_CUDA_HEAVY_PACK_RATIO_LIMIT`，形态与 CPU 侧逐字对齐
（`flops > 0` 且 `pack >= _CUDA_HEAVY_PACK_MIN_BYTES` 且 `pack >= limit * flops`），
阈值**按 CUDA 口径重标、不照搬 CPU 的 1.0**：complex128、语料 arity 6..16、
`distill`/`distill_curr` 各 8638 条，在"旧三条门放行且 pack ≥ 1 GiB"的 1734 条里
`distill` 最高 0.617、`distill_curr` 中位 1.728 / p90 1.813 / p99 2.101 / 最高 2.388，
靶子 3.950 → 取 **3.0**（空档 2.388–3.950 内，两侧留 1.26×/1.32×）。降到 1.0 会新拒
`distill_curr` 500 条生产计划（pack 2.1–3.1 GB）、2.0 仍新拒 30 条；这两族里 GPU 实测
过的两格都是**好计划**（seq=8513 比值 2.388 → `cuda_dedicated` **1.249 ms** vs
opt_einsum 531.2 ms；seq=7900 比值 2.101 → **1.133 ms** vs 338.0 ms），所以阈值必须
高于 2.388，1.0/2.0 不取。**与 CPU 侧的一处实测差异（勿照抄 CPU 用例的断言）**：CUDA
的 arena 底只有 3 GiB，seq=856 的 CUDA 计划 arena 是 4,219,857,152 B，所以它在这一条
判据加进来**之前就已经被 arena 门拒掉**（GPU 实测走 `cuda_topology`，改前 21.2 ms /
改后 21.0–27.4 ms 三样本，rel 1.08e-15，路径未变）；比值判据的价值是让结论不再依赖
"arena 恰好超了"——归零 arena/单步 MACs 后只剩它能拒。语料层面这条门新拒 **0 格**
（`distill` 0/8638、`distill_curr` 0/8638），即它今天只对"arena 底以下但 pack/flops
超限"的计划生效，而这一类在生产语料里尚未出现。
契约：`pyqcd.testing._topology_kernel_contract.test_cuda_plan_gate_rejects_pack_traffic_per_flop`
（纯计划统计，无设备也跑）。

**第三处：输入侧缺 direct 直连（2026-11-15 补，这是 32× 的真实根因）**

上一条说"小规模 6–9× 是拓扑入口固定开销"**只对中位数成立，对离群格是错的**，
而且那个 6–9× 本身混入了附录格（见下面的口径警告）。
逐层追 seq=3（`ambn,codp,egfh,ce,mo,bdf,nph->ag`，distill 尺寸）发现：该式 6 步里
有 **7 个输入侧非恒等**，第 2 步 `nph,egfh->npegf` 把 4,000,000 元素的中间体整块
gather，实测 **topology 49 ms vs dedicated 1.49 ms（32×）**，同一路径同一输入。

根因：`dedicated_apply.cpp` 有 `kDirectPlain`（源即规范序）与 `kDirectTrans`
（组交换，交给 BLAS 的 trans 吸收），两者都**零拷贝**；而 `topology_gemm.h` 的
`v3_run_step` 原先**只有输出侧**的 `step_output_is_canonical`，输入侧无条件
`gather_extents`。

修法（`topology_gemm.h`，**运行期判定，步表格式不变**）：`v3_run_step` 内检查
`a_axes`/`b_axes` 是否等于恒等排列或组交换排列，命中则走新增的
`gemm_call_trans`（`kCblasTrans=112`）直接读源张量。首版保守限制
`count_batch == 1`，其余仍走 pack（下一条把这条限制放开）。

效果（**只统计 route == `kupl_topology` 的真 v3 命中**，13 条基础表 seq × 4 dtype，
distill 尺寸，热态 3 轮中位；产物见 `bench_topology_vs_dedicated`）：

| dtype | n | 中位 | 最优 | 最差 | 快于 dedicated |
|---|---:|---:|---:|---:|---:|
| complex128 | 12 | 2.71× | 0.12× | 4.72× | 3/12 |
| complex64 | 12 | 2.89× | 0.04× | 4.70× | 3/12 |
| float64 | 12 | 2.88× | 0.14× | 4.29× | 3/12 |
| float32 | 13 | 3.03× | 0.20× | 4.40× | 3/13 |

**口径警告（2026-11-15 更正）**：此前本节写的是「中位 4.89–8.18×、最差 12.7–15.0×」，
那是用 20 条 seq 的抽样算的，而其中 **13 条是附录（seq>2582）**——它们**不在** v3
编译表内，实际走 `kupl_topology_v2`（v2 通用执行器）。把两类混在一起取中位会同时
抬高中心与尾部。现在 `try_topology_apply` 会写 `LAST_INFO["specialized"]` 与
`path=kupl_topology|kupl_topology_v2`，引用该 bench 的数字**必须按 route 分组**。

按 seq 看差异来源（真 v3）：seq=9 只有 0.045–0.195×（v3 大幅胜出），
seq=1700/2000 是 0.26–0.79×，其余 9 条是 2.0–4.7×（v3 慢于 dedicated）。
即「v3 胜在特定大规模拓扑，常规规模仍走 dedicated」。

**第四处：批直连（2026-11-15 补，放开上面的 `count_batch == 1`）**

判据（仍然运行期判定、步表格式不变）：

1. A/B 两侧都必须命中恒等排列或组交换排列。这两种排列把 batch 轴固定在源轴位
   `[0, batch_rank)`（组交换对 `axis < batch_rank` 是恒等映射），M/K（或 K/M）轴
   全部排在它们之后；
2. 于是源里每个 batch 元素的块是**恰好** `count_rows*count_inner`（A）/
   `count_inner*count_cols`（B）个元素的连续板，且按 batch 的 row-major 线性下标
   递增（源是 C 连续的），逐 batch 取 `base + batch * block` 即可 —— 与
   `dedicated_apply.cpp` 的批循环偏移公式逐字一致，`lda`/`ldb` 与 batch 无关；
3. 越界兜底：`count_batch * block` 超过该操作数的元素数（或 block ≤ 0）就 fail
   closed 走 pack，绝不让 BLAS 读到源之外。

不满足的布局（batch 轴与 M/K 交错、或轴序既非恒等也非组交换）**仍走 pack**：
那时块在源里不连续，`lda`/`ldb` 表达不出来。诊断开关
`PYQCD_EINSUM_TOPOLOGY_TRACE=1` 现在会逐步打印
`topology-gemm-step batch=… M=… K=… N=… a=plain|trans|pack b=… path=direct|pack`
（2026-11-15 起 `path` 变成三态 `direct|direct-a|direct-b|pack`，见第五处）。

**编译表已扩到全语料（2026-11-15）**：`topology_kernels.inc` 现在含 **8638 条**
（`MAX_KERNELS` = 基础表 2582 + 附录 6056），不再是 seq 1..2582。因此
`batch_rank > 0` 的步（全语料 121724 步里 11858 步，全部在 seq ≥ 2583）**已经
在表内可跑**——附录形态的 `count_batch` 正是 `Mom` 的幂（`distill_curr` 下
seq=2591/3437 为 9、seq=2711 为 81，见 `_label_sizes_contract`）。

扩表前的证据链仍保留如下，作为"与编译表无关"的独立验证（它手工编步表、不依赖
`.inc`，所以即便表再变也不失效）：

* `cpp/kupl/einsum/tests/topology_batch_direct.cpp`（dev-only 驱动，手工编单步 v3
  段，不需要编译表）：
  `g++ -O2 -std=c++17 -I ../src -I ../include topology_batch_direct.cpp -o /tmp/tbd -ldl`
  → 9 用例 × 4 dtype 全绿（double 类 ≤1.2e-16、single 类 ≤1.2e-7），覆盖
  plain/plain、trans/plain、plain/trans、trans/trans、`count_batch`=1/2/3/6
  （含双 batch 轴 2×3）、M 组两轴，以及一类非直连轴序 `[b,M1,K,M0]`（回落 pack）。
  变异测试（把 A 的 batch 偏移写死成 0）立刻在该驱动上报 `rel=1.02` 而 pack 用例
  不变 —— 用例有牙齿。
* 端到端（真值 `np.einsum(..., optimize="greedy")`）：用 `_topology_kernel_gen`
  临时生成只含 batch 方程的步表，语料 seq 3437/4206/6953/6933/4295/3461/2955/6836
  八个方程 × 四类 dtype 全部在容差内（双精度 ≤2.3e-15、单精度 ≤3.2e-6），trace 里
  `batch=2/3/6 … path=direct` 与 `batch=3/6 … path=pack` 两类都出现。
  **该表是临时产物**：验证后已用标准生成器恢复，逐字节一致（`cmp` 通过）。

**性能：distill 口径无可分辨变化**（符合预期，该口径没有 `count_batch > 1` 的步）。
5 次 A/B（改前 2 次、改后 3 次，20 seq × 2 复 dtype，`--rounds 3`）：
同配置重复运行 |Δ|中位 3.9–10.9%、最大 37%；改前/改后配对 |Δ|中位 3.3–14.1%、
最大 37%。首轮改前样本系统性偏慢（`before → before2` 自身就差 −10.9%），
以 warmed 的 `before2` 为基准时改前→改后中位 −0.7%/−1.2%/−1.1%，
整体中位比值 5.33–6.22×、最差 11.72–14.86× 在噪声内。
实数档仍被 `_TOPOLOGY_DECLINED_DTYPES` 拒绝，四 dtype 对照需
`PYQCD_EINSUM_TOPOLOGY_REAL=1`。

**上面那句「distill 口径没有 `count_batch > 1` 的步」的适用范围（2026-11-15 补）**

它成立**只因为 `distill` 档 `Mom_src=1`**：附录形态里每个 batch 轴都是动量标签，
故 `count_batch = Mom ** batch_rank`；`Mom=1` 时恒为 1，与"有没有 batch 步"无关。
早前 `_label_sizes` 把附录形态（seq ≥ 2583）的大写动量标签错绑成 `Lt`（`distill*`
档 `Lt=1`），于是 `distill_curr` 档（`Mom_curr=9`）也被算成 1 —— 「附录语料没有真正
可批的步」是那个错绑的**口径产物**。修正后实测：`distill_curr` 档 seq=2591/3437 出现
`count_batch=9`、seq=2711 出现 `count_batch=81`（双 batch 轴 M×N）。编译表
（seq 1..2582）的 `batch_rank > 0` 步数仍是 **0**，所以「批直连在编译表里跑不到」
这一条不变。互证见 `pyqcd/testing/_label_sizes_contract.py`。

**第五处：单侧直连（2026-11-15 补，不再让两侧一起 pack）**

`v3_run_step` 原先的 `use_direct` 要求**两侧**都命中恒等/组交换排列，否则两侧一起
`gather_extents`。现在拆成两个独立判定 `a_direct` / `b_direct`：

1. 可直连的一侧零拷贝用源张量，另一侧 gather 到 `packed_a`/`packed_b`；
2. `trans_a`/`trans_b` 与 `lda`/`ldb` 按**各自**的直连方式算：直连 plain → 不转置、
   `lda=K`（A）/ `ldb=N`（B）；直连 trans → 转置、`lda=M` / `ldb=K`；pack 侧已落成
   规范序 → 不转置、`lda=K`、`ldb=N`；
3. batch 偏移逐侧：直连侧 `base + batch*block`（源里的连续板），pack 侧
   `packed_x + batch*block`（规范序缓冲里的板），两侧 block 同值
   （`count_rows*count_inner` / `count_inner*count_cols`）；
4. 越界兜底（`block>0 && block<=total && count_batch<=total/block`）**逐侧**只作用于
   真正直连的那一侧：某侧不满足就只把该侧转 pack；两侧都不满足时与旧版完全一致
   （一次 BLAS 调用、`lda=K`、`ldb=N`、不转置）；
5. 归还只针对真正 acquire 过的一侧；B 侧分配失败时先归还已分配的 A 侧，任何早退
   路径都不泄漏。

诊断开关打印**三态**路径：`path=direct`（双侧直连）/ `direct-a`（仅 A）/
`direct-b`（仅 B）/ `pack`（两侧都 pack）；`a=`/`b=` 仍是该侧自己的轴序标签。

**正确性**（`tests/topology_batch_direct.cpp` 扩到 16 用例）：手工编单步 v3 段，
覆盖 plain/pack、pack/plain、trans/pack、pack/trans 四种单侧组合（共用几何
M=2×3、K=4、N=2×5）、`count_batch`=1/2/3/6、双 batch 轴 2×3 上的单侧步、双侧都
不可直连的 pack/pack，以及原有双侧组合与 M 组两轴。64 格全绿（double 类 ≤2.9e-16、
single 类 ≤1.3e-7）。驱动内置一份与执行器同口径的路径分类并**断言**每个用例真的
落在预期档位——混合档不会悄悄退化成双侧直连。变异测试（逐个单独注入、驱动当场
报错，恢复即全绿）：A 直连 batch 偏移写死 0 → `rel=1.02`；pack 侧 A 误用
`count_rows` 当 lda → `rel=1.42`；pack 侧 B 误用 `count_inner` 当 ldb →
`rel=1.00`；B 的 batch 偏移误用 `a_block` → `rel=1.49`。

端到端（真值 `np.einsum(optimize="greedy")`，容差 1e-9 / 1e-4）：语料 seq
1/3/4/5/6/7/8/9/10/11/12/13 × 四 dtype 共 48 格 0 超容差（complex128 ≤1.3e-15、
complex64 ≤7.3e-7、float64 ≤8.3e-16、float32 ≤8.8e-7），route 全为
`kupl_topology`；trace 里四种 path 与四种单侧组合都出现（plain/pack 136 步、
pack/plain 28、trans/pack 16、pack/trans 52）。

**语料静态账**（`plan_steps` + `decompose`，distill 尺寸；逐侧判定已与运行期 trace
在 seq 3/7/9/855/1599 的 44 步上逐行核对一致）：全语料 121724 步中 **65053 步**
是「恰好一侧可直连」（编译表内 17542、附录 47511）。但**基线本来就在两侧都可直连
时跳过 pack**，所以真实节省比步数小得多：

| 范围 | pack 元素数 改前 → 改后 | 省 |
|---|---|---|
| 编译表 seq ≤ 2582（v3 实际执行） | 2,697,418,831 → 2,693,526,513 | **0.14%** |
| 附录 seq 2583..8638（表里没有，走 v2） | 30,532,268,767 → 29,471,359,333 | 3.47% |

表内单 seq 最大节省 20,232 元素（seq=433）；bench 里 8 个 v3 活跃 seq 只省
32–532 元素。

**性能：distill 口径无可分辨变化**（与上表一致）。交错 A/B（base/new `.so` 每轮
互换、5 对、`--rounds 9`、8 个 v3 活跃 seq × 4 dtype = 32 格）：中位比值 1.025、
最快 0.835、最慢 1.120；同一次测量的噪声地板为 base/base 中位 0.988（p10 0.731、
p90 1.290）、new/new 0.980（0.685/1.106）、base/new 0.984（0.698/1.344）——跨变体
分布与同变体分布重合，**不得**把这 +2.5% 读成退化。真实收益要等步表覆盖到
seq > 2582（附录那 3.47%）才会显形。

**顺带核实的 bench 覆盖口径**：`bench_topology_vs_dedicated` 的 `DEFAULT_SEQS`
20 个 seq 里 13 个 > 2582（步表只到 2582），对 v3 恒 miss；distill 口径的 20 格里
只有 6 格真正命中 v3（1/3/9/849/855/1599），seq=129 被 pack-bound 守卫拒绝，其余
13 格的 route 标签是 `kupl_topology_v2`。引用该 bench 的"四 dtype 中位比值"时必须
按 route 区分，不能只看 `LAST_INFO['path']` 是否以 `kupl_topology` 开头。

**三条必须记住的事实**（禁止写成相反结论）：

1. "complex **曾经**完全没进 v3 步表" —— 2026-11-15 之前的 `topology_apply.cpp`
   把 `elem_kind` 只赋给 `QCD_DTYPE_FLOAT32` / `QCD_DTYPE_FLOAT64`，复数直接
   `return kSpecializedMiss`，随后仍调 `execute_parsed_contract`（**v2 通用执行
   器**）。回落对上层不可见：`LAST_INFO['path']` 是 Python 侧无条件写的
   `kupl_topology`，数值也正确，只是全部耗时都来自 v2。
   **现已修复**：`run_specialized` 对四类 dtype 都给出 `elem_kind`（与
   `QCD_DTYPE_*` 逐项同值），复数另过一道 `cblas_api().complex_complete()`——cgemm/
   zgemm 在 `bind_pair` 里是可选符号，缺符号时在分派层就 fail closed 回落 v2，
   绝不隐式降精度。v3 的 pack/unpack 按 `ComplexElement<T>`
   （`static_assert(sizeof == 2 * sizeof(T))`，与 numpy 的交错实虚布局同构）整元素
   搬运，GEMM 侧 lda/ldb/ldc 以**复数元素**为单位直传 `cgemm` / `zgemm`。
   证据：`PYQCD_EINSUM_TOPOLOGY_TRACE=1` 下复数打印
   `topology-specialized: 0123,...-> dtype=0 steps=5 sizes=8`（dtype 0/1 =
   complex128/complex64）；契约用例
   `_topology_kernel_contract.test_complex_dtypes_reach_the_v3_step_table` 把这条
   钉住（子进程抓 trace：`trace_enabled()` 的结果被静态缓存，同进程改环境变量已晚）。
2. "v3 缺 batch 并行，补上就能提速" —— 验收用的 105 步里 `batch_rank > 0` 为 **0 步**
   （全语料 121724 步中有 11990 步带 batch，但不在这批 seq），补了是死代码；且
   `packed_b` 是 `[batch][K][N]`，折平成 `[K, batch*N]` 的 memcpy 前提不成立。
   **2026-11-15 补充**：批**并行**仍然没有（也不该有）；新增的只是批**直连**
   （零拷贝喂 BLAS，逐 batch 调一次），它同样不在这张编译表里 —— 见上面「第四处」，
   证据来自 dev-only 驱动与一次临时步表的端到端对拍，**不是**来自生产语料。
3. "实数档不再被分派层拒绝" —— 仍被 `_TOPOLOGY_DECLINED_DTYPES` 拒绝；**原因已变**：
   不再因为步表病态，而是 dedicated 没有那 ~2 ms 固定开销（见
   `_kernel_kupl` 的模块注释）。

**重新生成步表**：`python -m pyqcd.testing._topology_kernel_gen`（约 4 s，产出
`cpp/kupl/einsum/src/topology_kernels.inc`，**该文件被 gitignore，是构建产物**；
`topology_apply.cpp:25` 无条件 include 它，所以干净树必须先跑生成器再编原生库）。

### 6.3 已修复：CUDA 拓扑描述符的摘要位宽错配（2026-11-15 修复）

CUDA 拓扑入口是**另一套实现**（`cpp/cuda/einsum/src/apply_contract.cu`，与 KUPL
`topology_gemm.h` 无关）：描述符只带结构，收缩序取编译期常量表
`cpp/cuda/einsum/src/topology_steps.h`（2582 条 ↔ 1052 条不同步序）。

**原缺陷**：`_kernel_cuda._descriptor_words` 用
`_specialize._signature_digest(...).ljust(16, b"\0")` 拼描述符摘要，而那张表的 key
是 blake2b-**128**（`digest_size=16`）的前 8 字节。96 位摘要零填充到 128 位后前
12 字节相同、后 4 字节不同，`topology_key_from_digest` 于是算出另一个 key，
`find_topology_entry` 恒查不到 → 落到 `topology_plan_chain` 的「左到右链式」兜底，
而该兜底只接受 `num_inputs <= 6`。语料的元数分布是 13/14/16 占 99.9%
（7..16 全部 > 6），因此**除 arity=6 的 2 条外整张表都取不到**，一律 `status=4`
回落。旧契约用例恰好只用 `ij,jk->ik`（arity=2）与 arity=6 抽样，所以没抓到。

**修法**：描述符里按 128 位原地重算摘要（payload 口径与 `_signature_digest` 完全
一致），不碰索引侧的 96 位口径，也不改 C++ 格式。

**证据**（V100、`CUDA_VISIBLE_DEVICES` 临时指定、`PYQCD_CUDA_ARCHITECTURES=7.0`）：

* 全语料 2582 条 × 四类 dtype（dim=2，真值 `np.einsum(..., optimize="greedy")`）：
  **每类 dtype 都是 2582/2582 = 100% 命中**（`path=cuda_topology`），miss=0、bad=0；
  最坏相对误差 complex128 1.296e-14 / complex64 6.547e-06 / float64 5.341e-14 /
  float32 9.581e-05，全部在 1e-9 / 1e-4 / 1e-9 / 1e-4 容差内。修复前只有 arity≤6
  的 2 条能命中（其余 `status=4`）。
* 无隐式升降精度：同一个 arity=7 方程、把首个操作数放大 1e8 制造量级差后，
  float64 rel=2.616e-15 而 float32 rel=1.000e-06（差 9 个量级）——两者确实各走
  自己的 BLAS 精度，没有互相冒充。
* 复数确实走 cuBLAS 的 c/z gemm：`batched_gemm.cu` 的 `qcd_cuda_batched_gemm`
  按 `QCD_DTYPE_*` 分派到 `cublasZgemmStridedBatched` / `Cgemm` / `Dgemm` / `Sgemm`
  （`cublasLt` 只在 cuBLAS 返回 `NOT_SUPPORTED` 时兜底，且其 compute type 同样按
  dtype 选 `COMPUTE_64F`/`32F`，不做降精度）。
* 回归：`python -B -m pyqcd.testing` → **128 passed / 1 skipped / 0 failed**
  （新增 1 条 `test_cuda_descriptor_reaches_the_step_table_for_high_arity`；
  变异验证：把 `_descriptor_words` 换回 96 位零填充后该用例立刻
  `AssertionError: seq=3 arity=7 ... failed with status 4`，恢复后 PASS）。

**另一项：CUDA 侧已有 pack 恒等直连，但不需要 KUPL 的 `kDirectTrans`**。
`prepare_binary` 早就设了 `left_pack_identity` / `right_pack_identity`，执行时整段
跳过 `qcd_cuda_permute`（trace 的 `pack_skipped` 计数）。按真实步表统计（语料每 7
条抽样、5121 步）：恒等已覆盖左脚 20.1% / 右脚 21.1%，而**组交换只有左 0.1% /
右 0.0%** —— CUDA 的 pack 目标是 `[batch][free][reduce]` 三级分组，不是 KUPL 的两级
`[M][K]`，所以「组交换」这个类别在这里几乎不存在（92.2% 的步还带非空 batch，
更不能像 KUPL 那样用 `count_batch == 1` 放开 trans）。再补 `kDirectTrans` 的收益
上限约 0.1%，不值得动。

**收缩序：确认病态，本轮用「病态即 fail closed」守卫止损（2026-11-15）**。
`topology_steps.h` 的步序由均匀 dim=4 的 greedy 定型，而生产轴长是 `{1,4,100}`
混杂。全语料 2582 条 × `dims("distill")`、两种序用**同一套 MAC 记账**（每步「两侧
标签并集」元素数，与执行器的 pack + GEMM 一一对应）实跑
`python -m pyqcd.testing.einsum.analyze_cuda_topology_order`（产物
`pyqcd/testing/einsum/artifacts/cuda_topology_order_<时间戳>/`）：

* 表内序 ÷ 生产 `optimize="auto"` 序：**中位 59.9×、p90 9521×、最差 3.81e4×**
  （seq=510/654，arity=14：`4.004e12` vs `1.051e8` MAC；单步中间体 **4e8 元素**
  = complex128 6.4 GB，而 `kTopologyMaxIntermediateElements` 只挡 2^31 —— 挡得住
  越界、挡不住病态）；更差/更好/持平 = 2388/178/16，>2× 1836、>10× 1724、
  >100× 1138 条；
* 交叉核对（`contract_path` 的 `opt_cost` 口径）同为中位 59.9×，即差异来自收缩序
  而不是记账口径；上一轮 63 条抽样（中位 0.020）的方向与量级一致；
* **成因坐实**：表内序与 `contract_path(optimize="greedy")` 在**均匀 dim=4** 下的
  路径逐位相同 **2582/2582 = 100%**，而它与生产序相同只有 1/2582。

**修法（不碰 4724 行常量表，也不改 Python 生成端）**：`apply_contract.cu` 的
`build_topology_plan` 增加一道**只用描述符自带 sizes** 的体检
`topology_order_acceptable` —— 用「最小并集乘加优先」贪心算出参考序（每步 O(n²)
候选、n ≤ 16，每个计划只算一次），表内序劣化超过 `kTopologyOrderTolerance = 8` 倍
即返回 `QCD_TOPOLOGY_UNSUPPORTED`（status 4，trace 记 `reject=order-pathological`），
由上层回落带真实 path 搜索的通用路径并进负缓存。阈值 8 与 16 在语料上落在同一
空档（被拒条数同为 1540/2582），判据对取值不敏感；参考序相对生产
`optimize="auto"` 中位只差 1.1%，足够当尺子。

**证据**（V100、`CUDA_VISIBLE_DEVICES` 临时指定、`PYQCD_CUDA_ARCHITECTURES=7.0`）：

* `python -m pyqcd.testing.einsum.verify_cuda_topology_order`（distill 尺寸）：
  complex128 全 2582 条放行 1042 / 拒绝 1540，与 Python 侧预测**逐条一致**
  （mismatch=0），放行格子对 `np.einsum(optimize="greedy")` 最差相对误差
  **1.537e-14**（≤1e-9）；complex64/float64/float32 各 80 格（40 放行 + 40 拒绝）
  0 失败，最差 4.87e-06 / 5.14e-15 / 3.76e-06（限 1e-4 / 1e-9 / 1e-4）。
* **不误伤等长探针**：`dim=2` 等长时守卫一条都不拒（表内序 ÷ 贪心 中位 0.385、
  最差 2.87 < 8），上一轮「dim=2 命中 2582/2582」的结论不变。
* **真实暴露面**（distill、complex128，逐条先试 dedicated）：2582 条里 dedicated
  命中 1902，**topology 入口只被触达 680 条（全部 arity=14）**，其中守卫放行 332 /
  拒绝 348 —— 本次行为变化面是 348 条，不是 1540 条。
* **端到端收益**（`bench_cuda_topology_order_guard`，distill、complex128、首调
  丢弃、3 轮中位；`--label` 区分同一脚本在两次构建上的测量）：seq=510 守卫关
  **5.0999 s** → 守卫开 **0.0063 s**（806×）；被 dedicated 接住的 5 条对照格
  （2565/869/1859/3/853，0.4–1.3 ms）两次测量持平。
* 回归：`python -B -m pyqcd.testing` → **127 passed / 1 skipped / 0 failed**。

**仍未解（后续单独立项）**：整表重定型需要先补一个能写出 `topology_steps.h` 的
生成器（blake2b-128 key + 规范标签流 FNV 指纹 + `kQcdTopoOrders` 索引三件套；仓库
里**没有**这份脚本，`_topology_kernel_gen.TARGET` 只写 KUPL 的 `.inc`）。守卫已经
在二进制里算出生产尺寸的贪心序，把它从「只做判据」改成「直接作为执行序」是两行
改动，可把被拒的 348 条也留在原生路径上（贪心序相对 auto 中位 0.99×）；是否改由
后续单独决策。

### 6.4 编译表扩到全语料 8638 条，步表以 gzip 入库（2026-11-15）

`topology_kernels.inc` 此前只含**基础表 2582 条**（`MAX_KERNELS`），附录 6056 条
（动量标签打头，如 `Mbd`/`Nfnp`）不在表内 —— 它们命中 `_topology_signature` 后
`run_specialized` 返回 miss，静默回落 v2 通用执行器（route 记为
`kupl_topology_v2`）。

**改动**：`MAX_KERNELS` 2582 → 8638，`main()` 改用
`_parse_equations(include_momentum=True)`；该函数默认仍只读基础表，加参数才追加
文件末尾"动量指标与流分量指标"一节。实测 8638 条与 `_wick_tasklist` **逐条零失配**。
`kTopoRecordOfEquation` 用 `uint16`，8638 < 65535 安全。

**入库形式是 gzip**（`src/topology_kernels.inc.gz`，1794207 字节）。原因：裸 `.inc`
是 13835244 字节，超过本库 10 MB 单文件限制（`form-large-files-check` 会拦截），
而 `topology_apply.cpp` 无条件 `#include` 它 —— 裸检出必须能拿到，不能只依赖生成器。

* `.gitignore` 只豁免 `src/topology_kernels.inc.gz` 这一条精确路径，裸 `.inc` 仍忽略；
* `cpp/kupl/einsum/make.sh` 在编译前解压（仅当 `.inc` 缺失或比 `.gz` 旧），
  解压结果与生成器输出 md5 一致（`b4e107ca…`）；裸检出实测打印
  `unpacking topology_kernels.inc.gz` 后构建成功（34 s）；
* 重新生成仍是 `python -m pyqcd.testing._topology_kernel_gen`（12 s / 8638 kernels /
  2645582 步表字节），生成后需 `gzip -9 -c` 覆盖 `.inc.gz`。

**收益（热态中位，distill，complex128；单次测量会被首调开销误导）**：

| seq | arity | v3 | generic | 比值 |
|---|---:|---:|---:|---:|
| 2583 | 6 | 0.56 ms | 1.58 ms | **0.35×** |
| 2584 | 6 | 4.05 ms | 8.26 ms | **0.49×** |
| 3431 | 9 | 0.68 ms | 1.16 ms | **0.58×** |
| 4301 | 9 | 0.70 ms | 1.18 ms | **0.59×** |

即附录方程现在比 v2 回落路径快 1.7–2.9×。**但 dedicated 仍更快**（0.27–0.37 ms），
与"常规规模走 dedicated"的既有结论一致 —— 扩表解决的是"附录根本没有 v3"，不是
"v3 该取代 dedicated"。

**注意单次测量的陷阱**：seq=2583 的**首次**调用是 45 ms（plan 缓存 + 签名哈希），
而 generic 的 5 ms 是热路径；只测一次会得出"v3 慢 9×"的错误结论。任何对比都必须
先丢弃一次 warmup 再取中位。

**回归**：`python -B -m pyqcd.testing` → 128 passed / 1 skipped / 0 failed；
`form-large-files-check` 通过（5 个暂存文件，无超限）。
