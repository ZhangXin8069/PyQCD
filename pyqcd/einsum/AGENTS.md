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

1. `bench_specialized._label_sizes` 把 `VVV` 当 `(Lt, Mom, Nev, Nev, Nev)`、`VDV` 当
   `(Lt, Mom, Nev, Nev)`，并且全方程只有一个 `Nev` —— 这两点都与参考脚本不符
   （无 `Lt` 轴、两个 `Nev`）。实测后果：拿 `dims("distill")` 喂 `_label_sizes`，
   `Nx/Lt=1` 的占位与 `Mom`（1 或 `Mom_curr`）会作为**真实轴长**漏进 peram 的轴
   （seq 3 得到 `(4,100,100,100)`、`(100,4,1,1)` 之类），与参考脚本的
   `(4,4,100,400)` 无关。因此 **`--scene distill` 经 `_label_sizes` 的形状只能当
   path-only 证据，不得写成蒸馏生产结论**；真实形状只认 `distill_tensor_shapes()`。
   该限制由 `_bench_scene_distill_contract.test_label_sizes_cannot_express_distill_axes`
   锁定；修 `_label_sizes` 归规划/语料组，改了会让该契约失败并提醒同步本节。
2. `VVV` 分配 `(1,100,100,100)`（`:209`）但注册时用 `host_vvv_sink[0]` 切片（`:241`），
   进 einsum 的是 `(100,100,100)`；`VDV` 保留动量轴（`:243-245`）→ `(1,100,100)` /
   `(Mom_curr,400,400)`（本档 `Mom_curr=9`，即 `(9,400,400)`）。两者不对称，照抄参考脚本即可。
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

**收益**（抽样 21 条、distill 尺寸、MAC 口径）：

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

**第三处：输入侧缺 direct 直连（2026-11-15 补，这是 32× 的真实根因）**

上一条说"小规模 6–9× 是拓扑入口固定开销"**只对中位数成立，对离群格是错的**。
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

效果（同口径 20 seq × 4 dtype，产物 `topo_vs_dedicated_distill_20261115132117`）：

| dtype | 中位 前→后 | **最差 前→后** | 快于 dedicated |
|---|---|---:|---:|
| complex128 | 8.18 → **4.89** | 24.63 → **15.04** | 5/19 → 6/19 |
| complex64 | 6.56 → 6.47 | 48.95 → **13.24** | 6/19 → 6/19 |
| float64 | 6.80 → 6.94 | 56.22 → **12.73** | 5/19 → 6/19 |
| float32 | 6.73 → 6.67 | 62.19 → **13.79** | 5/20 → 6/20 |

**最差比值全面收敛到 ~13×**（原 49–62×），病态格子消失；中位数变化小是因为多数
格子本来就没有非恒等输入侧。seq=3 从 49 ms 降到 5.52 ms（3.84×）。

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
`topology-gemm-step batch=… M=… K=… N=… a=plain|trans|pack b=… path=direct|pack`。

**这条路径在编译表里跑不到**：`topology_kernels.inc` 只含语料 seq 1..2582
（`MAX_KERNELS`），其中 `batch_rank > 0` 的步为 **0**；全语料 121724 步里有 11858
步带 batch，但都在 seq 2583 以后。因此批直连的正确性由两条互补证据给出：

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

**收缩序：确认病态，但本轮不改（无生成器）**。`topology_steps.h` 的步序由均匀
dim=4 的 greedy 定型，而生产轴长是 `{1,4,100}` 混杂。按真实步表逐条复算
（语料每 41 条抽样，63 条，distill 尺寸，`contract_path` 的 `opt_cost` 口径）：
**生产序更优 55 / 持平 2 / 更差 6，中位比值 0.020**（即生产序便宜约 50×），
最差样本 seq=489 表内序 `2.0010e+12` vs 生产序 `8.1811e+08`。V100 实测：
seq=489 topology 中位 **149.79 ms**，其中单步 `m=n=k=10000` 的 sgemm 参考耗时
**141.94 ms** —— 病态步几乎吃掉全部时间（对照 seq=3 为 1.21 ms、seq=1700 1.31 ms）。
**不能本轮硬改**：仓库里**没有**生成 `topology_steps.h` 的脚本
（`_topology_kernel_gen.TARGET` 只写 `cpp/kupl/einsum/src/topology_kernels.inc`，
且它不含 blake2b-128 key 与 CUDA 的 FNV 指纹口径；该文件本身也禁止改）。重定型
需要先补一个生成器（复用 `plan_steps(equation, sizes=…)` +
`_label_sizes(roles, dims("distill"))`），再重生成 1052 条常量表。
