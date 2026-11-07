# pyqcd.einsum 的 opt_einsum 兼容层（2026-11-08）

> **事实边界**：本文每条行为陈述都有下列之一作为依据——(a) 本文给出的命令 + 实跑输出；
> (b) 直接引用源码位置。没有实测或源码依据的推测一律集中在 §8「未验证 / 不作结论」，
> 不与结论混写。
>
> 采集时间 2026-11-08 06:30–06:50。工作树当时正被其它代理并行修改
> （`pyqcd/einsum/_specialize.py` mtime 06:24、`pyqcd/einsum/_api.py` 06:27、
> `cpp/cuda/einsum/src/apply_contract.cu` 06:35），**行号以采集时刻为准**，
> 引用时优先用函数名/符号名定位。

## 0. 结论摘要

| 项 | 结论 |
|---|---|
| 能否无缝替换 opt_einsum 3.4 | **能**：`import pyqcd.einsum as opt_einsum` 后，本文 §1 列出的 API 与 opt_einsum 3.4 的 path / 数值 / 类型行为一致（`check_compat` 39/39 PASS） |
| 唯一新增的调用参数 | `contract(..., specialize=True)`，**keyword-only**；`contract_path` / `contract_expression` 签名未变 |
| 专用路径是否默认开启 | 是；命中时**库自己**打印 `[pyqcd.einsum] 专用路径: <name> [<kind>] <note>`（每形态每进程一次） |
| 关闭方式 | 单次：`contract(..., specialize=False)`；进程级：`specialize=None` + `PYQCD_EINSUM_SPECIALIZE=0`（见 §2 的实测优先级） |
| A/B 工具 | `python -m pyqcd.testing.einsum.simple.specialize_ab`（§4） |
| 已知差异 | 5 条，逐条列在 §6（其中只有 1 条影响可观测行为：`shared_intermediates` 在 `optimize≠False` 时不填充缓存） |

## 1. 兼容面清单

### 1.1 逐项对照

| API | opt_einsum 3.4 的行为 | pyqcd.einsum 现状 | 证据 |
|---|---|---|---|
| `contract(subscripts, *operands, out=None, use_blas=True, optimize=True, memory_limit=None, backend='auto', **kwargs)` | 解析 + 规划 + 逐步收缩；`optimize=True/None` 解析为 `auto` | 签名逐字相同（多 `specialize`，见 §2）；`optimize` 默认写 `'auto'`，`True`/`None` 同样解析为 `auto` | 签名对拍；E5（T0n）下 `greedy/optimal/dp/branch-2/auto` 五种 optimizer 的 path **逐项相等**；`check_compat` PASS |
| `contract_path(...) -> (path, PathInfo)` | 只规划不计算；支持 `shapes=True`、`memory_limit`、`use_blas` | 同；`shapes=True` 的 path 与 opt_einsum 相等；`memory_limit='max_input'` 可用 | 实测 §7 命令 3 |
| `contract_expression(...) -> ContractExpression` | 预编译表达式；`constants=[i]` 时该位放**数组**而非 shape | 同；`constants=[0]`（传数组）结果与 opt_einsum 逐元素一致 | 实测 §7 命令 3 |
| `shared_intermediates(cache=None)` | 上下文管理器，`yield` 传入的那个 dict；上下文内中间量按对象 id 记忆化 | 上下文协议、`yield` 的对象**同一性**一致；`optimize=False` 路径下 `count_cached_ops` 与 opt_einsum 完全一致（`{'tensor': 2, 'einsum': 1}`）。**但** `optimize≠False` 时我们的缓存为空（opt_einsum 同例 5–6 条）→ 差异 D1 | 实测 §7 命令 3 |
| `get_symbol(i)` | `a..zA..Z` 后接其它单字符 | `get_symbol(0,1,25,26,51)` = `a,b,z,A,Z`，与 opt_einsum 相同 | 实测 §7 命令 3 |
| `paths.register_path_fn(name, fn)` | 注册自定义路径函数 | 同：注册后出现在 `PATH_OPTIONS`，可作为 `optimize='<name>'` 使用 | 实测：注册 `my_doc_test` 后 `contract_path(..., optimize='my_doc_test')` 返回 `[(0, 1)]` |
| `paths.get_path_fn(name)` | 取回路径函数，未知名字抛 `KeyError` 并列出合法选项 | 同（`KeyError: "Path optimizer 'nope' not found, valid options are [...]"`）；返回**我们自己的**函数对象，不是 opt_einsum 的对象 | 实测 §7 命令 3 |
| `paths.PATH_OPTIONS` | 3.4 中为私有 `_PATH_OPTIONS`（公开可用性见 D2） | 公开名；是 opt_einsum `_PATH_OPTIONS` 的**超集**（多一个 `'branch'` 别名） | 集合对拍 |
| `backends.get_func(fn, backend='numpy', default=None)` | 取后端实现 | 同；`get_func('einsum','numpy')` 返回可调用的 einsum 包装函数（不是 `np.einsum` 本身） | 实测 |
| `backends.has_einsum(backend)` / `has_tensordot(backend)` | 后端能力探测 | 同；`has_einsum('numpy')` 两侧均为 `True` | 实测 |
| `backends.infer_backend(x)` | opt_einsum 3.4 的 `backends` 命名空间**没有**这个名字（实测 `dir()`） | pyqcd 提供；numpy 数组返回 `'numpy'` | 差异 D3（超集，不破坏旧调用方） |
| `PathInfo` | `.contraction_list/.input_subscripts/.output_subscript/.path/.indices/.scale_list/.naive_cost/.opt_cost/.speedup/.size_list/.size_dict/.shapes/.eq/.largest_intermediate`，代价字段是 `decimal.Decimal` | 14 字段齐全、四个代价字段是 `Decimal`、`opt_cost` 与 opt_einsum 整数值相等 | `check_compat`：`pathinfo/*` 全部 PASS |
| 通用 einsum 语义 | 隐式输出、`...`、trace、Hadamard、`optimize=False` | 同 | `check_compat` 的 `behavior/*` PASS（含 matmul / implicit-output / trace / hadamard / ellipsis） |

`check_compat` 实跑结果（本文采集时）：

```
compat: 39 passed, 0 skipped, 0 failed
```

### 1.2 可直接运行的等价性示例

把下面整段存成任意 `.py` 在本仓库根目录运行（`PYTHONPATH` 指向仓库根，或直接
`cd /home/zhangxin/PyQCD` 后 `python <文件>`）；实测输出为
`compat layer OK：6 组断言全部通过`。

```python
import decimal

import numpy as np
import opt_einsum
import pyqcd.einsum as pe

rng = np.random.default_rng(0)
a = rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))
b = rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))

# 1) contract：数值一致
assert np.allclose(pe.contract('ij,jk->ik', a, b, optimize='dp'),
                   opt_einsum.contract('ij,jk->ik', a, b, optimize='dp'))

# 2) contract_path：path 与代价一致；代价字段是 decimal.Decimal
path_ours, info_ours = pe.contract_path('ij,jk->ik', a, b, optimize='dp')
path_ref, info_ref = opt_einsum.contract_path('ij,jk->ik', a, b, optimize='dp')
assert list(path_ours) == list(path_ref)
assert int(info_ours.opt_cost) == int(info_ref.opt_cost)
assert isinstance(info_ours.opt_cost, decimal.Decimal)

# 3) contract_expression：预编译表达式
expr = pe.contract_expression('ij,jk->ik', (3, 4), (4, 5))
assert np.allclose(expr(a, b), a @ b)

# 4) shared_intermediates：上下文协议 + 显式 cache 复用
cache = {}
with pe.shared_intermediates(cache) as shared:
    assert shared is cache
    pe.contract('ij,jk->ik', a, b, optimize=False)
assert pe.count_cached_ops(cache)['einsum'] == 1

# 5) paths / backends / get_symbol
assert 'dp' in pe.paths.PATH_OPTIONS
assert callable(pe.paths.get_path_fn('greedy'))
assert pe.backends.infer_backend(a) == 'numpy'
assert pe.backends.has_einsum('numpy') and pe.backends.has_tensordot('numpy')
assert pe.get_symbol(0) == opt_einsum.get_symbol(0) == 'a'

# 6) PathInfo 14 个字段
FIELDS = ('contraction_list', 'input_subscripts', 'output_subscript', 'path',
          'indices', 'scale_list', 'naive_cost', 'opt_cost', 'speedup',
          'size_list', 'size_dict', 'shapes', 'eq', 'largest_intermediate')
assert all(hasattr(info_ours, name) for name in FIELDS)
print('compat layer OK：6 组断言全部通过')
```

## 2. 额外参数 `specialize` 的语义

### 2.1 位置与为什么它不破坏兼容性

```python
# pyqcd/einsum/_api.py（采集时 line 523-533）
def contract(subscripts, *operands, out=None, use_blas=True, optimize="auto",
             memory_limit=None, backend="auto", specialize=True, **kwargs) -> Any
```

- **keyword-only**：`specialize` 位于 `*operands` 之后，与 `out` / `optimize` 同类，
  因此不可能被旧代码按位置传入，也不会改变任何位置参数的含义。
- **默认 `True`**：不传就是"开启专用路径"，即现有 opt_einsum 调用方的默认体验。
- **opt_einsum 的调用方不会传它**：opt_einsum 3.4 的 `contract` 签名里没有这个名字
  （实测签名对拍），因此 `TypeError: unexpected keyword argument` 不会被触发；
  反过来，把 `pyqcd.einsum` 换回 opt_einsum 时，只有显式写了 `specialize=` 的代码才需要改。
- **`contract_path` 与 `contract_expression` 的签名里没有 `specialize`**（实测
  `inspect.signature`），专用路径的开关只挂在 `contract()` 上。

### 2.2 与两个环境变量的优先级（实测）

判定逻辑在 `pyqcd/einsum/_specialize.py::enabled`（采集时 line 85-90）：

```python
def enabled(specialize=None):
    if specialize is not None:
        return bool(specialize)          # 调用参数优先
    return os.environ.get("PYQCD_EINSUM_SPECIALIZE", "1").strip().lower() not in (
        "0", "false", "off", "no")       # 只有 specialize=None 时才看环境变量
```

实测矩阵（E5、T0n、CPU；每行一个独立进程，观察是否打印提示行与
`PYQCD_EINSUM_DISABLE_FUSED` 的值）：

| 调用方式 | `PYQCD_EINSUM_SPECIALIZE` | 是否走专用路径 | 调用后 `PYQCD_EINSUM_DISABLE_FUSED` |
|---|---|---|---|
| 默认（`specialize=True`） | 未设置 | **是**（打印提示） | `None` |
| 默认（`specialize=True`） | `0` | **是**（仍然打印） | `None` |
| `specialize=None` | 未设置 | 是 | `None` |
| `specialize=None` | `0` | 否 | `1` |
| `specialize=False` | 任意 | 否 | `1` |

**优先级（高 → 低）：显式 `specialize=` 参数 > `PYQCD_EINSUM_SPECIALIZE` > 默认开启。**
由此得到一条容易踩的事实：**单靠 `PYQCD_EINSUM_SPECIALIZE=0` 关不掉通过
`contract()` 默认调用触发的专用路径**，因为默认值 `True` 不是 `None`；要进程级关闭
必须写成 `specialize=None`（或在所有调用点写 `specialize=False`）。
`_specialize` 模块 docstring 把环境变量描述为"进程级关闭"，实测口径应按本节修正。

### 2.3 关闭时到底关了什么

1. `_api.contract` 把 `PYQCD_EINSUM_DISABLE_FUSED=1` 写进进程环境
   （`_api.py` 采集时 line 554-556），并在下一次 `specialize=True` 调用时把
   **本模块自己写的**该变量弹掉（`_FUSED_ENV_OWNED`，line 544-550）。
   实测：`specialize=False` 后环境为 `'1'`，紧接着一次默认调用后回到 `None`。
   **边界情况（实测）**：若调用方自己设了 `PYQCD_EINSUM_DISABLE_FUSED=1`，期间又出现过
   一次 `specialize=False`，那么随后一次默认调用会把该变量一并弹掉——
   `_FUSED_ENV_OWNED` 只记录"本模块设过"，不区分数值是谁写的。
2. CUDA 侧：`initialize_context()`（`cpp/cuda/einsum/src/apply_contract.cu`，
   采集时 line 1771）在该变量存在时**跳过** `match_sandwich` / `match_e4_fused`，
   于是走通用多步 GEMM。T1（Nvec=100、spatial=1）E4 的 `PYQCD_EINSUM_TRACE=1`
   实测对比见 §5.4。
3. Python 侧：`contract()` 的编译程序缓存 `cache_enabled` 要求 `specialize is not False`
   （`_api.py` 采集时 line 568-576）。所以 `specialize=False` **同时**关掉了
   "同一 plan 只编译一次" 的快速通道，每次调用都会重新规划。CPU A/B 的差异主要来自
   这一条，见 §5.3。

只有一个 `PYQCD_EINSUM_SPECIALIZE_VERBOSE=1` 是纯输出开关：命中时每次都打印提示
（默认每形态每进程只打印一次，`_specialize.py::announce`）。

## 3. 已注册的方程形态表

下表**由脚本从 `pyqcd/einsum/_specialize.py::REGISTRY` 直接生成**（不是手抄），
复核命令：

```bash
cd /home/zhangxin/PyQCD
python -m pyqcd.testing.einsum.simple.specialize_ab --list
```

采集时 `REGISTRY` 共 **9** 条：

| # | name | kind | subscripts（REGISTRY 键） | note |
|---|---|---|---|---|
| 1 | `nucleon-2pt(E1)` | `fused-plan` | `gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M` | 8 算子核子 2pt；γ₇×2 + VVV×2 + 宇称投影 |
| 2 | `nucleon-3pt(E2)` | `fused-plan` | `Geifj,ckdl,agbh,ce,gi,Mbdf,Mhjl,Gka->GM` | 8 算子核子 3pt |
| 3 | `gauge-propagator(E4)` | `fused` | `zyxcTabn,zyxgTdef,Gae,Tzyxgc->TGdbfn` | peram × peram × γ × 规范场，CUDA E4 融合 kernel |
| 4 | `gamma-sandwich(E5)` | `fused` | `ab,...bce,cd->...dae` | γ₅ 夹心，CUDA sandwich kernel |
| 5 | `sush-2pt(S1)` | `fused-plan` | `eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN` | sush NN 2pt 生产方程 |
| 6 | `sush-3pt-munu(S2)` | `fused-plan` | `gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG` | sush 3pt 含 μν 插入 |
| 7 | `sush-2pt-mu(S3)` | `fused-plan` | `manb,cedf,bd,Mfn,ac,MFem->MF` | sush 2pt 含 μ 插入 |
| 8 | `sush-3pt-main(P1)` | `multi-step` | `agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy` | sush 3pt 主项，15 算子 14 步（生产热点） |
| 9 | `sush-3pt-main(P2)` | `multi-step` | `agbh,eqfr,ospt,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy` | sush 3pt 主项（置换型），15 算子 14 步 |

匹配规则：`REGISTRY` 的键是**去掉空白后**的 subscripts 全串
（`_specialize.normalize`），所以调用时写成 `'ab, ...bce, cd->...dae'` 同样命中；
未命中则不打印、走通用路径。

## 4. 性能对照：方法、口径与注意事项

### 4.1 工具

```bash
cd /home/zhangxin/PyQCD

# 默认：方程表中命中 REGISTRY 的全部方程，T0n、CPU、10 次中位、预热 2
python -m pyqcd.testing.einsum.simple.specialize_ab

# 指定 profile / 后端 / 方程
python -m pyqcd.testing.einsum.simple.specialize_ab --profile T1 --backend cuda --equations E4,E5

# 换测量先后顺序（默认 onoff；用来检查顺序依赖）
python -m pyqcd.testing.einsum.simple.specialize_ab --equations E5 --order offon --iters 20 --warmup 5

# 只看形态表
python -m pyqcd.testing.einsum.simple.specialize_ab --list
```

实现要点（`pyqcd/testing/einsum/simple/specialize_ab.py`）：
复用 `_common` 的方程表 / 操作数 / `median_seconds`，对每条方程分别测
`specialize=True` 与 `specialize=False`，并与 opt_einsum 的同方程中位耗时对比。

输出列：

| 列 | 含义 |
|---|---|
| `opt_ms` | opt_einsum 的同方程中位耗时（倍率参照） |
| `on_ms` / `off_ms` | `specialize=True` / `False` 的中位耗时 |
| `x_opt.on` / `x_opt.off` | `opt_ms / on_ms`、`opt_ms / off_ms`（>1 表示我们更快） |
| `gain` | `off_ms / on_ms`（**专用路径收益**，>1 表示专用路径更快） |
| `rel.on` / `rel.off` | 与 opt_einsum 结果的逐元素相对误差（应 ≤ 1e-12） |
| `note` | `too_short(仅诊断)`、`FAIL(...)`、`NUMERIC-FAIL(...)` 等 |

**提示行的归属**：`specialize=True` 命中形态时，库会自己打印
`[pyqcd.einsum] 专用路径: ...`。工具把这些行以 **`[库自身输出]`** 前缀原样转发，
并在表头下方打印一句说明——它们是**库**的日志，不是本工具的调试输出。
`specialize=False` 那一侧不会打印这类行（未命中）。

### 4.2 口径（缺一条结论就不可比）

- **同 profile**：profile 决定 `nvec / spatial / n_t / t_sink`，即全部操作数形状。
  跨 profile 的毫秒数不可比。
- **同 dtype**：统一 `complex128`（`_common.DTYPE`）。
- **同线程**：CPU 侧必须钉住 BLAS/OpenMP 线程数（仓库既有口径
  `PYQCD_BENCH_THREADS=4`，见 `docs/pyqcd_einsum_performance_status_20261108.md`）。
  **本文 §5 的 CPU 数字没有钉线程**，属诊断值，见 §8。
- **同 device**：CPU 对 CPU、单卡对单卡；CPU 与 CUDA 的倍率不能混用。
- **同 planner**：`--optimize`（默认 `dp`），两个变体各自规划一次，只计稳态执行。
- **每个变体重新编译**：工具在测每个变体前清空 `pyqcd.einsum` 的编译程序缓存与原生
  上下文缓存。原因很实在：`_NATIVE_CONTEXT_CACHE` 的键里**不含**
  `PYQCD_EINSUM_DISABLE_FUSED`，不清空的话 `specialize=False` 会直接复用
  `specialize=True` 编译出来的（已匹配融合 kernel 的）上下文，CUDA A/B 就变成
  "自己跟自己比"。
- **预热 + 中位**：默认 `--warmup 2`、`--iters 10`，取中位数。
- **<100 ms 只作诊断**：`spatial=1` 的 E4/E5 常在毫秒乃至微秒量级，双方中位耗时
  任一低于 100 ms 就标 `too_short(仅诊断)`，**不得**当作生产性能结论。
- **opt_einsum 自身会失真**：某些 path 上 opt_einsum 不走 BLAS（既有报告记录 E2 在 T1
  单次 2.18 s），此时 `x_opt` 会虚高，必须同时看绝对耗时。

## 5. 实测数据

> 全部数字来自本次实跑，标注了 profile / 后端 / 迭代次数 / 预热次数 / 测量顺序。
> 采集时机器为 8 核、load average ≈ 6.3–6.8（其它代理在并行跑任务），
> 绝对毫秒数受此影响，只能作为同一轮内的相对比较。

### 5.1 前序给定数据（本文未复现，仅引用）

E4（`zyxcTabn,zyxgTdef,Gae,Tzyxgc->TGdbfn`，`spatial=1`、`Nvec=100`）：

| 变体 | 中位耗时 |
|---|---|
| `specialize=True` | **0.685 ms** |
| `specialize=False` | **1.326 ms** |

≈ **1.9x**。来源：本次任务的既有实测记录（未标注后端与迭代次数）。
本机复跑同 profile（T1、CUDA）得到 1.702 ms vs 1.871 ms（§5.2），
**未能复现这组数字**，原因未查明，见 §8。

### 5.2 本次实跑：CUDA

```bash
python -m pyqcd.testing.einsum.simple.specialize_ab --profile T0n --backend cuda --equations E4,E5
python -m pyqcd.testing.einsum.simple.specialize_ab --profile T1  --backend cuda --equations E4,E5
```

`iters=10, warmup=2, optimize=dp, order=onoff, dtype=complex128`：

| profile（nvec, spatial） | 方程 | opt_ms | on_ms | off_ms | x_opt.on | x_opt.off | gain | rel.on / rel.off | note |
|---|---|---|---|---|---|---|---|---|---|
| T0n（6, 1） | E4 | 1.527 | 1.501 | 1.541 | 1.018 | 0.991 | 1.027 | 4.86e-16 / 3.24e-16 | too_short |
| T0n（6, 1） | E5 | 1.512 | 1.534 | 1.529 | 0.985 | 0.989 | 0.996 | 0 / 0 | too_short |
| T1（100, 1） | E4 | 1.824 | 1.702 | 1.871 | 1.072 | 0.975 | 1.099 | 5.01e-16 / 4.01e-16 | too_short |
| T1（100, 1） | E5 | 1.577 | 1.662 | 1.594 | 0.949 | 0.989 | 0.959 | 0 / 0 | too_short |

这一轮里两个变体都在 1.5–1.9 ms 量级（CUDA 启动/同步开销占比很大），
`gain` 落在 0.96–1.10，**不足以支撑"专用路径更快"的结论**。

### 5.3 本次实跑：CPU（默认验收命令）

```bash
python -m pyqcd.testing.einsum.simple.specialize_ab --profile T0n --backend cpu --equations E4,E5
```

同一命令连跑 5 次（`iters=10, warmup=2`），五次的区间：

| 方程 | opt_ms | on_ms | off_ms | x_opt.on | x_opt.off | gain |
|---|---|---|---|---|---|---|
| E4 | 0.240 – 0.302 | 0.036 – 0.045 | 0.269 – 0.396 | 5.81 – 7.90 | 0.73 – 1.10 | **7.19 – 8.81** |
| E5 | 0.117 – 0.129 | 0.015 – 0.017 | 0.156 – 0.182 | 6.97 – 8.63 | 0.69 – 0.83 | **9.46 – 10.99** |

全部带 `too_short(仅诊断)`：E4/E5 在 T0n 下是几十微秒的游戏，**不能**外推。
另外这 7–11x 的 `gain` **不是** CUDA 融合 kernel 的功劳（CPU 侧根本不读
`PYQCD_EINSUM_DISABLE_FUSED`，见 §2.3 第 2 条）。同 profile 下单独测量（E4、T0n、CPU、
`iters=10`、`warmup=2`）：

| 量 | 中位耗时 |
|---|---|
| `specialize=True`（编译程序缓存命中） | 0.0446 ms |
| `specialize=False`（每次重新规划 + 执行） | 0.3074 ms |
| 其中 `contract_path` 单独耗时 | 0.1889 ms |

即 `off` 的耗时里约六成是每次调用的重新规划，与 §2.3 第 3 条的源码结论一致。
该次探针同时确认 `_NATIVE_DISABLED` 与 `_NATIVE_CONTEXT_UNAVAILABLE` 都是空集
（没有静默退化到纯 Python），两侧比的是真实执行。

CPU 生产规模（T1 = Nvec 100）同一命令的结果，**受机器负载影响，仅作诊断**：

| 方程 | opt_ms | on_ms | off_ms | x_opt.on | x_opt.off | gain | note |
|---|---|---|---|---|---|---|---|
| E4 | 10.957 | 49.909 | 37.698 | 0.220 | 0.291 | 0.755 | too_short |
| E5 | 0.173 | 0.100 | 0.198 | 1.735 | 0.875 | 1.983 | too_short |

单跑 E4 复测：opt 7.999、on 39.993、off 39.767 ms（gain ≈ 0.99）。
这组数字与 `docs/pyqcd_einsum_performance_status_20261108.md` 记录的
"E4 T1 CPU 4.07 ms / 1.82x" 差距很大；本次**不判断**是我方回归、还是
并行负载导致的失真（§8）。

### 5.4 机制证据：关闭后 CUDA 确实换了执行路径

T1（Nvec=100、spatial=1）E4，单次调用，`PYQCD_EINSUM_TRACE=1`：

```
# specialize=True
[pyqcd.einsum] 专用路径: gauge-propagator(E4) [fused] ...
PYQCD_EINSUM_CUDA_E4 z=1 y=1 x=1 c=3 t=2 a=4 b=4 n=100 g=3 G=1 d=4 e=4 f=100 kernel=1

# specialize=False
PYQCD_EINSUM_FUSED_DISABLED
PYQCD_EINSUM_CUDA_GEMM batch=2 m=3 n=1600 k=3 streams=1
PYQCD_EINSUM_CUDA_GEMM batch=1 m=2400 n=4 k=4 streams=1
PYQCD_EINSUM_CUDA_GEMM batch=2 m=400 n=400 k=12 streams=1
```

即：`specialize=False` → `initialize_context` 跳过融合匹配 → 3 步通用 GEMM。

### 5.5 全方程扫描（T0n、CPU、一遍）

```bash
python -m pyqcd.testing.einsum.simple.specialize_ab --profile T0n --backend cpu   # 9 条方程，约 3 s
```

| 方程 | opt_ms | on_ms | off_ms | gain |
|---|---|---|---|---|
| E1 | 2.873 | 0.071 | 3.407 | 48.28 |
| E2 | 3.431 | 0.115 | 3.433 | 29.90 |
| E4 | 0.264 | 0.047 | 0.303 | 6.48 |
| E5 | 0.134 | 0.018 | 0.169 | 9.56 |
| S1 | 2.896 | 0.086 | 3.013 | 35.00 |
| S2 | 0.803 | 0.053 | 1.063 | 19.98 |
| S3 | 0.634 | 0.039 | 0.732 | 18.60 |
| P1 | 9.334 | 8.734 | 8.932 | 1.02 |
| P2 | 6.051 | 6.208 | 6.290 | 1.01 |

全部 `too_short(仅诊断)`、全部 `rel ≤ 1.3e-15`。这是**一遍快照**：重跑会出现
10% 级抖动（尤其 P1/P2，两侧都在 6–11 ms 且受并行负载影响）。

## 6. 已知差异清单

| 编号 | 差异 | 影响 | 依据 |
|---|---|---|---|
| D1 | `shared_intermediates` 在 `optimize≠False` 时不填充缓存（`len(cache)==0`，opt_einsum 同例 5–6 条）；`optimize=False` 时两侧内容一致 | 结果不变；跨调用复用尚未生效，`count_cached_ops` 读数为空 | 实测 §7 命令 3 |
| D2 | `paths.PATH_OPTIONS` 是 opt_einsum 3.4 `_PATH_OPTIONS` 的超集（多 `'branch'`）；opt_einsum 3.4 的 `paths` 命名空间没有公开的 `PATH_OPTIONS` | 超集，不破坏旧调用方 | 集合对拍 |
| D3 | `backends.infer_backend` 是 pyqcd 新增名字（opt_einsum 3.4 的 `backends` 里没有） | 超集，不破坏旧调用方 | `dir(opt_einsum.backends)` 实测 |
| D4 | `contract` 的 `optimize` 默认值写作 `'auto'`，opt_einsum 3.4 写作 `True` | 等价（两侧都把 `True`/`None` 解析成 `auto`；五种 optimizer 的 path 逐项相等） | 签名对拍 + path 对拍 |
| D5 | `contract` 多一个 keyword-only 参数 `specialize` | 见 §2；`contract_path`/`contract_expression` 不受影响 | 签名对拍 |

## 7. 复现命令汇总

```bash
cd /home/zhangxin/PyQCD

# 命令 1：API 兼容面（39 项）
python -m pyqcd.testing.einsum.simple.check_compat

# 命令 2：专用路径 A/B（本文 §5 的全部表格）
python -m pyqcd.testing.einsum.simple.specialize_ab --profile T0n --backend cpu --equations E4,E5
python -m pyqcd.testing.einsum.simple.specialize_ab --profile T0n --backend cuda --equations E4,E5
python -m pyqcd.testing.einsum.simple.specialize_ab --profile T1  --backend cuda --equations E4,E5
python -m pyqcd.testing.einsum.simple.specialize_ab --profile T1  --backend cpu  --equations E4,E5
python -m pyqcd.testing.einsum.simple.specialize_ab --profile T0n --backend cpu

# 命令 3：§1/§6 里的零散对拍（在交互式解释器里逐段执行即可）
#   contract_path: greedy/optimal/dp/branch-2/auto 的 path 相等；shapes=True 相等
#   contract_expression: constants=[0]（传数组）与 opt_einsum 相等
#   shared_intermediates: optimize=False 时 count_cached_ops 一致
#   paths.PATH_OPTIONS / backends.infer_backend / get_symbol / 未知 path 名的 KeyError

# 命令 4：CUDA 执行路径证据（§5.4，本段已实跑）
PYQCD_EINSUM_TRACE=1 PYQCD_EINSUM_NATIVE=cuda python - <<'PY'
from pyqcd.testing.einsum.simple import _common
from pyqcd.testing.einsum.simple.specialize_ab import reset_target_state
import pyqcd.einsum as pe

cfg = _common.profile_config('T1')                 # Nvec=100, spatial=1
eq = _common.EQUATION_BY_NAME['E4']['eq']
arrays = _common.to_backend(_common.operands('E4', cfg), 'cuda')
for flag in (True, False):
    pe.contract(eq, *arrays, optimize='dp', backend='cupy', specialize=flag)
    _common.sync('cuda')
    reset_target_state()      # 不重置会复用上一个变体编译出的上下文
PY
```

## 8. 未验证 / 不作结论

1. **未使用真实蒸馏数据**：`/public/group/lqcd` 在本机不存在，全部操作数由固定种子
   合成（形状与真实蒸馏一致），只保证"同形状同 dtype"下的相对对比。
2. **未钉线程**：§5 的 CPU 数字没有固定 BLAS/OpenMP 线程数，且采集时 load average
   ≈6.3–6.8（8 核，其它代理并行运行）。因此 CPU 的绝对毫秒数**不能**当作验收值。
3. **未复现给定的 0.685 / 1.326 ms**（§5.1）。本机同 profile（T1、CUDA）得到
   1.702 / 1.871 ms。差异原因（机器、kernel 版本、计时口径、迭代次数）未查明，
   本文不给结论。
4. **未复现** `docs/pyqcd_einsum_performance_status_20261108.md` 的
   "E4 T1 CPU 4.07 ms / 1.82x"；本次测得 49.9 ms（on）与 8.0 ms（opt）。
   是回归还是负载失真未判定。
5. **未测 L24 / L24n**（spatial=24）：显存与耗时都未采样。
6. **未测多卡 / MPI / 多进程**；不把任何多卡结果与单卡 opt_einsum 对比。
7. **未验证 `specialize` 对 `ContractExpression` 的可观测影响**：
   `contract_expression` 签名里没有该参数（实测），因此本文不对"表达式路径是否
   命中融合 kernel"下结论；§5.4 的 trace 只覆盖 `contract()`。
8. **不作"专用路径总是更快"的结论**：§5.2 的 CUDA 小形状 gain 0.96–1.10、
   §5.3 的 T1 CPU E4 gain 0.755 都说明收益与形态、规模、后端强相关。
