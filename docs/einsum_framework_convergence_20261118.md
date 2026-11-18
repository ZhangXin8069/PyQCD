# qcd-einsum 收敛重构的判据与流程（2026-11-18）

面向**下一次做同类清理/收敛重构的人**。这里只写"怎么判、怎么做、怎么证明"，
不写流水账；时间线与原始输出见 `logs/einsum-framework-convergence-20261118.txt`。

| 项 | 值 |
|---|---|
| 范围 | `pyqcd/einsum`（Python 分派层 + 画像/缓存层）与 `cpp/cuda/einsum`（融合核删除） |
| 结论口径 | 只认"可运行环境实测 + 改前/改后同环境对比"；单次测量、跨机基线一律不作结论 |
| 不覆盖 | 具体 API 参考（见 `--help` 与 `pyqcd/einsum/AGENTS.md`）、装机与构建（见 `skills/pyqcd-einsum`） |

---

## 1. 三条通用判据（先背下来，能挡掉大半误判）

1. **"看起来没人用"不是死代码的判据**。判据是**沿调用链走到入口，且入口可达**。
   一个函数只在 `PYQCD_*` 环境开关打开时才被调用，只要那个开关在源码里真实存在且
   可以被打开，它就是**活代码**。反例见 §2.1。
2. **弱符号 / `QCD_CUDA_LOCAL` 符号在 Python 层 grep 为 0 是正常的**。provider 模式
   （`qcd_cuda_permute` 等）本来就不导出，它由 C++ 内部消费。判死要查 **C++ 侧调用点**，
   不是 Python 侧字符串。
3. **任何"删 C++ 零回归"的声明，必须有两棵 Python 完全相同的树**。只用 `LD_PRELOAD`
   换 `.so` 仍可能因并行改动而混淆——见 §3.1 的 `git archive` 隔离 A/B。

## 2. 识别"活代码"的四个实测动作

### 2.1 先跑入口，再谈结构

`e4_fused` / `sandwich` 曾被判"结构不可达"，实测在 `PYQCD_CUDA_FUSED=1` 下
`[fused-hit] sandwich` 正常打印。判"不可达"之前必须至少做一次：

```bash
# 1) 结构上：谁调它（AST/grep，两处都要看）
grep -rn 'match_e4_fused\|execute_e4_fused' cpp/cuda/einsum/src/apply_contract.cu
# 2) 运行期：把门打开，看它是否真的被调用
PYQCD_CUDA_FUSED=1 python -m pyqcd.einsum.test --equation "..." --device gpu ...
```

**教训**：结构不可达是"入口本身不存在"，而不是"入口藏在环境变量后面"。

### 2.2 原生符号：查 C++ 调用点，不是 Python 字符串

```bash
nm --defined-only pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so | grep qcd_cuda_   # t = 本地定义
grep -rn '\bqcd_cuda_permute\b' cpp/cuda/einsum/src/dedicated_apply.cu         # 真实消费点
grep -rn '\bqcd_cuda_permute\b' pyqcd/ --include=*.py | wc -l                  # 常态为 0
```

本库实测：`qcd_cuda_{permute,hadamard,reduce,batched_gemm}` 在 Python 侧命中 **0**，
在 `dedicated_apply.cu` 内被 `run_typed` 的打包/解包路径消费；`apply_contract.cu`
另有 15 处跨 TU 引用。**Python 侧 0 命中 → 判死 = 误删。**

`pack.cu` 尤其不能动：`dedicated` 的打包/解包全部依赖 `qcd_cuda_permute`，删掉它
**不会崩溃**，只会让 `dedicated` 每次都静默回落（症状是"结果对但永远不加速"）。
凡是"删了以后不报错、只是变慢"的候选，一律按活代码处理。

### 2.3 源码与产物不一致时，先重建再判

若 `.so` 比源码**旧**，或产物里出现源码中不存在的字符串（`strings | grep`），
任何"产物里有什么分支"的结论都不可靠。判死之前先重建，并把 md5/mtime 记进报告。

```bash
ls -l --time-style=full-iso cpp/cuda/einsum/src/apply_contract.cu pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so
strings pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so | grep -cE 'e4_fused|sandwich|PYQCD_CUDA_FUSED'
```

### 2.4 三态语义：签名默认值 ≠ 语义默认值

`kernel_force` 的签名默认是 `None`，但 `_kernel_force.parse(None)` 解析成
`("generic",)`，`forced_labels(None)` 返回 `()`。于是"未指定"和"明确要求 generic"
在**取值上完全一样**——只看取值就判"用户显式要求了"，命中提示与画像 `force` 槽
整段变成死代码。

**正确做法**：在解析与任何默认填充**之前**取一个显式布尔位。

```python
caller_force_given = kernel_force is not None   # 只回答"用户给过没有"
forced = _kernel_force.forced_labels(kernel_force)
```

`_api.py:1805` 现在就是这一行；凡是要区分"用户真的给过"的地方都必须带这个位，
而不是拿解析结果反推。

## 3. 删除的验证流程（删 C++ / 删公共符号）

### 3.1 隔离 A/B：`git archive` 建 HEAD 树

```bash
SP=<scratchpad>
git archive <HEAD> | tar -x -C $SP/head_tree                  # 不含 .git 的纯源码树
cp pyqcd/einsum/cuda/einsum.cpython-*.so            $SP/head_tree/pyqcd/einsum/cuda/
cp pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so         $SP/head_tree/pyqcd/einsum/kupl/
cp <删除前|删除后>/libpyqcdeinsum_cuda.so           $SP/head_tree/pyqcd/einsum/cuda/
PYTHONPATH=$SP/head_tree mamba run -n <env> python -B -m pyqcd.testing
```

- **Python 固定为 HEAD**，只替换 `.so`；两组计数必须完全一致（本例 144/1/0 与 144/1/0）。
- 只跑"当前工作树 + 换 `.so`"**不构成证据**：并行改动会让失败集合变化，而你会
  分不清是 C++ 还是 Python 引起的。
- 双 cuBLAS 污染（conda 12.4 与 cupy 12.9 并存）会让 `LD_PRELOAD` 组产生假失败；
  数值探针须在 `import pyqcd.einsum` **之前**做一次小 GEMM 热身。

### 3.2 数值基线：在"能容纳的尺寸"上对照

删除融合核后，`[8,8,64,300,300]` 这类"靠融合核才装得下"的尺寸会 OOM——这是**容量**
回归，不是数值回归。正确做法：在同一方程上缩到可容纳尺寸（scale=64）逐条比
`max|Δ|/max|ref|`，并如实记录 OOM 的尺寸与算术。

### 3.3 删符号之前先 `grep` 全仓引用

`_kernel_force.AUTO_LABEL` 删除后，`_api.py` 三处引用立刻变成命中即崩的
`AttributeError`（`TUNE_OPTIMIZERS` 写出的每一行配置都会撞上）。
**任何"删符号"先 `grep` 全仓引用，含 `pyqcd/testing/**` 契约。**

```bash
grep -rn 'AUTO_LABEL' pyqcd/ --include=*.py | wc -l     # 目标 0
grep -rn 'AUTO_LABEL' pyqcd/testing/ --include=*.py     # 契约里的引用也要清
```

### 3.4 删死码必须给 grep 前后计数

```text
build_kernel            5 → 0（全部命中原在本文件）
DedicatedKernel         9 → 0
_try_family_apply 等     1 → 0（仅定义）
```

没有前后计数的"清理完成"无法复核，也无法在下次回归时定位是谁拿掉的。

## 4. 语义收敛：把"运行时猜"换成"实测结论"

### 4.1 同名不同义 = 该删的理由

`optimize="auto"` 在两个内核里是**不同算法**：

| 内核 | `auto` 的实际含义 |
|---|---|
| `generic`（`_plan_py.auto`） | 按算子数查表（1–4→optimal，5–6→branch-all，7–8→branch-2，9–14→branch-1，≥15→greedy），≤8 算子时再用 opt_einsum 的 DP 交叉校验 |
| `dedicated`（`_kernel_kupl`） | 在 `auto` 与 `greedy` 两条计划里按步表 FLOPs 取小，并列取 `auto`；仅在 CPU complex128 且 arena 超阈值时条件提升 branch-2 |

同一个参数名在两处是两套算法，"用户写了 auto，实际跑了什么"在报告里看不出来。
**判据**：一个运行时开关若无法在输出里被证实，就该由 `--tune` 的实测结论替代。

### 4.2 拒绝 `bool("false")`

字符串开关必须显式解析。本库实测：

```
bool("false")                    = True
_profile_cache.enabled("false")  = True     <-- 期望 False
_kernel_force.parse("false")     -> generic  （已显式解析，正确）
```

`_normalize_as_equivalence`（`_api.py:151`）是正确写法：显式映射 `true/1/force` 与
`false/0`，其余抛 `ValueError`。**凡是把外部字符串直接交给 `bool()` 的地方都是 bug**。

### 4.3 三态/四态参数，收不干净就写明

| 参数 | 现状 | 建议 |
|---|---|---|
| `kernel_force` | 单值 + 缺省 `generic`（无 `auto`） | 保持；docstring 必须与实现一致 |
| `as_equivalence` | `bool | None`，`None` = 跟随环境 | 保持；CLI 的第四态 `force` 只在该层有意义 |
| `specialize` | `bool | None` | `None` 跟随环境是合理三态 |
| `profile_dir` | `None` 同时表示"没给"和"沿用上次" | 非幂等，docstring 必须明写"进程级 sticky" |

### 4.4 拼接参数前先确认层级

`KernelUnavailableError` 的 `requested` 曾按"序列"设计，调用点照抄 `(requested,)`
多包了一层，于是 `", ".join(self.requested)` 拿到的是**元组的元组**：

```
TypeError: sequence item 0: expected str instance, tuple found     # _kernel_force.py:134
```

一条 `TypeError` 连带 5 项契约连锁失败（`test_kernel_force_runs_kupl_dedicated` 等）。
反向的坑同样成立：为"兼容"逼调用方自己包元组时，`tuple("dedicated")` 会静默变成
`('d','e','d','i','c','a','t','e','d')`。

**判据**：任何 `", ".join(...)` 之前先 `assert all(isinstance(x, str) for x in seq)`，
或把收口做成"字符串入参自动包成 1 元组"（现在 `KernelUnavailableError.__init__` 就是
`(requested,) if isinstance(requested, str) else tuple(requested)`）。

### 4.5 术语收敛：`auto` 只剩两种含义

`kernel_force` 的 `auto` 已删除后，`auto` 在库内只剩 `optimize="auto"`（收缩序）和
`specialize/cache/explain auto`（跟随环境）两类"缺省值"语义。写文档或改代码时点名
是哪一个；历史文本里"`kernel_force` 的 `auto`"一律视为**已删除**。

## 5. 报错文案也是接口：必须真的渲染

`--help` 走 `argparse.RawDescriptionHelpFormatter`，**不渲染 markdown**。
`**x**` 会原样打出 `**x**`。改文案后必须跑一次：

```bash
python -m pyqcd.einsum.test --help | grep -c '\*\*'
```

同一个坑在 shell 里：方程里带 `->out` 会被当成重定向写进文件
（`cpp/cuda/einsum/src/todbfn` 就是这样产生的垃圾文件）。**方程必须加引号。**

## 6. 基线数字只在同一环境里可比

本机实测：

| 环境 | 有 scipy | 能否跑 `python -B -m pyqcd.testing` |
|---|---|---|
| `pyqcd-test` | 否 | 否（import 期 `ModuleNotFoundError`） |
| `pyqcd-einsum-cu124` | 是（1.17.1） | 是 |

所以"任务给的基线 129/2/10"在本机不可复现，实测为 134/1/11。**正确做法**：以本机
实测为基线，做"改前/改后同环境"对比（本例 145/1/0），并把不可复现的事实写进报告。

## 7. 六条规格的验收证据法

验收一条规格时，证据至少要能回答"哪条命令、什么输出、为什么这就够了"：

| 规格类型 | 最小证据 |
|---|---|
| "只读不许写" | 操作前后 `md5sum` 表不变 + 目录条目数不变 |
| "同键多行要报错" | 抛出的异常类型 + 正文里**全部**候选行（含行号） |
| "缺文件要警告并继续" | stderr 原文（含可跑的 tune 命令）+ 返回值形状与数值 |
| "冲突要非零退出" | `rc` + 正文（不含 traceback） |
| "超时只输出不退出" | `rc=0` + `tune write: rows=0` |
| "文案要能渲染" | `--help` 实跑片段，而不是源码里的字符串 |

## 8. 迁移完整性：唯一定义 + 导出表无增减

把 `pack.cu` / `batched_gemm.cu` 并入 `dedicated_apply.cu` 时，三条必须同时成立：

```bash
# 1) 每个 provider 符号唯一定义（源码与产物各 1）
grep -rE 'extern "C"[^;]*int qcd_cuda_permute\(' cpp/cuda/einsum/src/*.cu | wc -l   # 1
nm --defined-only libpyqcdeinsum_cuda.so | awk '$3=="qcd_cuda_permute"' | wc -l     # 1
# 2) 无未解析符号
nm -u libpyqcdeinsum_cuda.so | grep -i qcd_cuda
# 3) 动态符号表无增减（10 个 ABI 入口，与删除前一致）
nm -D --defined-only libpyqcdeinsum_cuda.so | wc -l
```

`make-nv.sh` 的源列表是 glob（`src/*.cu`），删文件**不需要**改脚本；
`PROVIDER_SYMBOLS` 与导出 map 只在这些**名字**变化时才需要同步。

## 9. 配置表（画像）的三条铁律

1. **只有 tune 能写**，普通 `contract()` 只读。读侧不建目录、不落盘。
2. **命中行即指令**：命中后"未施加"必须有明确理由（`out=` / `specialize=False` /
   用户显式给了别的值），且提示里要打 `<file>:<行号>`。
   —— 这条曾被 `caller_force is not None` 判据整段关掉（§2.4）。
3. **`wall_ms` 是整段 `contract()`**，不存在第二列可与它混比的计时；
   选优只按它比较。旧 `init_ms`/`compute_ms` 拆分已删除，`grep` 应为 0。

## 10. 自查清单（提交前逐条打勾）

- [ ] 每个"死代码"判定都有**入口可达性**证据（含环境开关被打开的那一次）
- [ ] 每个被删符号都给了 `grep` 前后计数（含 `pyqcd/testing/**`）
- [ ] C++ 删除的证据是 `git archive` 隔离 A/B（Python 固定），不是单跑当前树
- [ ] 数值基线在"能容纳的尺寸"上逐条对照，OOM 尺寸单独记账
- [ ] 报错/帮助文案实跑过一次，无 markdown 残留、无非法取值建议
- [ ] 基线数字与本机环境一起记录；不可复现的基线明确写出
- [ ] 只有注释/文档被改时，`git diff --stat` 里没有可执行行
- [ ] 只改注释时，**反向撤销编辑后代码 AST 必须一致**（§11 的做法）
- [ ] 契约/CLI 的行为验证在 `PYQCD_EINSUM_PROFILE=0` 下也跑一次
      （仓库里的 `.cache` 命中行会制造与代码无关的 `ProfileConflict`）

## 11. 只改文档/注释时怎么证明"没改代码"

`git diff` 在多代理并行的工作树里**不可用**（别人的未提交改动会混进同一 hunk）。
可靠做法是**逐次编辑自证**：

1. 把每处编辑的 `(old, new)` 记下来；
2. 从工作树文本里把 `new` 反向替换回 `old`；
3. 两侧都 `ast.parse` → 剥掉 docstring → `ast.unparse`，要求字符串**完全相等**；
4. 另用 AST 校验"新插入的每一行都落在注释行或 docstring 跨度内"。

这样得到的结论是"我这次编辑没有触及任何可执行节点"，与工作树里还有谁在改无关。
本轮 10 处编辑全部通过（`failures: 0`）；细节见时间线 §13.1。

## 相关

- 运行时间线与关键输出：`logs/einsum-framework-convergence-20261118.txt`
- 内部契约（真源）：`pyqcd/einsum/AGENTS.md`（§9.1b–9.1f 为强制内核与配置表，§10 为本轮经验）
- 使用者视角：`docs/einsum_user_guide_20261116.md`
- 原始巡查报告：`logs/kf/{DCA,B2,A1,A2,SPEC6,CUDA_MIG,CL1}.report.md`

## 本轮落到的代码注释（读者会踩的位置）

| 位置 | 为什么在那里写 |
|---|---|
| `_kernel_force.py` `forced_labels` | `()` 有两个来源，取值分不开"用户给过没有" |
| `_api.py` `last_kernel` / `contract(kernel_force=…)` | `requested` 的 `None` 只可能来自"没给"；旧 docstring 写的是自动分派 + 逗号链 |
| `_api.py` `_cuda_fused_shape` | 该名单是 CUDA 执行器的准入证，改它等于改选路 |
| `_api.py` `_fused_disabled` | 融合核已删，它**只**剩缓存键语义，别顺手删 |
| `_profile_cache.py` `_switch_on` | 显式解析 vs `bool(text)` 的反转陷阱 |
| `_kernel_kupl.py` `try_dedicated` | 这个 `auto` 与 `_plan_py.auto` 是两套算法 |
| `_kernel_kupl.py` family 段 | 无调用方但**有意预留**，判死不能只看 grep |
| `paths.py` `_BUILTIN_ATTRS` | 这里的 `auto` 与 `kernel_force` 的 `auto` 同名不同义 |
| `test.py` `build_parser` | `--help` 不渲染 markdown，改文案必须实跑 |
