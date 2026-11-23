# PyQCD einsum 用户指南

面向**使用者**的任务导向手册：只回答"下一步敲哪条命令、输出怎么读、读到的结果意味着什么"。
内部实现与契约见 `pyqcd/einsum/AGENTS.md`，领域技能见 `skills/pyqcd-einsum/SKILL.md`。

| 项 | 值 |
|---|---|
| 适用范围 | `python -m pyqcd.einsum.test`、`python -m pyqcd.einsum.verify_specialized`、`pyqcd.einsum.contract` |
| 版本基线 | pyqcd 0.5.19（选项与行为按当前实现核对） |
| 性能口径 | **本文不给出任何性能数字**；一切以你在目标机器上的实测为准 |
| 已删除的历史名 | `profile.csv`（旧表名）、独立等价表、`--compare` 的 numpy 档、`dim=4` 等长探针与 `--optimize-topology-v2`；本文其余部分不再提 |

---

## 1 安装

强烈建议用 pip 安装**最新版**：

```bash
pip install -U pyqcd
```

**请逐行阅读安装过程的输出与提醒。** 原生内核（CPU 的 KUPL、GPU 的 CUDA）在安装时
现场构建，输出会说明是否建成、缺哪个编译工具链或驱动、跳过了哪些可选依赖。
忽略这些提示，很容易拿着一个只能走纯 Python 规划的包，却以为它在用原生加速。
装完可核对：

```bash
python -m pyqcd           # 版本、原生后端、可选依赖、子包
python -m pyqcd --json    # 机器可读
```

需要 Python 3.9+。源码编译步骤、CUDA 架构选择、线程与环境变量清单属于构建/运维细节，
见 `pyqcd/einsum/AGENTS.md`；本文不展开。

---

## 2 快速开始

最小可运行示例（三行）：

```python
import numpy as np, pyqcd.einsum as pe
a, b = np.ones((2, 3)), np.ones((3, 4))
c = pe.contract("ij,jk->ik", a, b)      # 等价于 a @ b
```

公开接口与 `opt_einsum` 对齐：`contract`、`contract_path`、`contract_expression`、
`shared_intermediates`、`PathInfo`、`paths`、`backends`、`helpers`、`parser`、`blas`。

命令行跑一条方程：

```bash
python -m pyqcd.einsum.test --equation "ij,jk->ik" --shape "[4,4],[4,4]->[4,4]"
# PASS ij,jk->ik  <本次实测>ms  rss+<N>MiB
```

- `PASS` / `FAIL` 一行是结论；那个毫秒数是**本次实测值**，会随机器与负载浮动，不是基准。
- 退出码：`0` 通过 / `1` 数值或形状不符 / `2` 参数或环境错误。
- `--verbose true` 把逐项明细打到 stderr；`--json <path>` 把完整结果写成 JSON。

`--shape` 可以省略：此时按 `--scene` 从语料 roles 反推尺寸（见 §4.1）。方程既不在语料里、
也没有能命中的等价写法时，省略 `--shape` 直接退出码 2，必须显式给出形状。

---

## 3 性能与 tune

> **正式使用前强烈建议先跑一次 tune：**
>
> ```bash
> python -m pyqcd.einsum.test --tune --equation "<你的方程>" --shape "<你的操作数形状>"
> ```
>
> **任何性能都须以实测为准：本文档不给出、也不背书任何性能数字、加速比或基准表。**

### 3.1 为什么先 tune

`--tune` 的产物是一张**配置表**（§5）：在固定的方程 / 形状 / dtype / 设备 / 后端上，
它遍历若干组 `kernel_force × as_equivalence × optimize`，把实测最快的一组写进表；
之后普通调用按表选路。没有表时普通调用只警告一次"需要先 tune"、打印该配置的 tune 命令，
然后**照常计算**（按未命中处理）——方程、dtype、形状与数值都不受影响。

### 3.2 遍历方式：有序清单，不是笛卡尔积

| 项 | 行为 |
|---|---|
| 固定不动的维度 | 方程、形状、dtype、设备、后端 |
| 遍历的三个轴 | `kernel_force × as_equivalence × optimize` |
| 分支 | 探测得出（不计时）：**B1** 原方程可特化 / **B2** 只有等价写法可特化 / **B3** 都无法特化 |
| 组数 | **B1、B2 各 29 组；B3 28 组**（冻结口径） |
| 顺序 | 共同前缀（`original` → `dedicated` → `generic`，每个内核内 `greedy → branch_1 → branch_2 → dp → branch_all → optimal → none`）→ 一条分支专属 `topology` → `backend` 尾部（同样 7 个 optimize） |
| 剪枝 | `min_wall_ms` 上界初值 = `--tune-timeout`；某组单次 `contract()` 超界即**终止该组**（标 `pruned`、绝不写表）并继续下一组；任何一组测到更小耗时都**立刻压低**上界，后续全部按新上界剪枝 |
| 每组的数 | 沿用 `--warmup` / `--rounds`；该组样本最小值就是写进表里的 `wall_ms` |
| 表读写 | 整轮在 tune session 里：表**既不读也不写**；结束后只写**胜出的那一组**，并删除同键其它所有行 |
| 失败处置 | 抛异常标 `error=`、超 `--tune-timeout` 标 `timeout`、误差超 `--atol`/`--rtol` 标 `MISMATCH`：**只打印、不退出**，不参与选优、不写表 |

### 3.3 输出怎么读

```text
tune comb : 1/28  kernel_force=original  as_equivalence=false  optimize=greedy  backend=numpy  wall_ms=<t1>  状态=ok  原因=-
tune comb : 2/28  kernel_force=original  as_equivalence=false  optimize=branch_1  backend=numpy  状态=pruned  原因=耗时大于min_wall_ms,此组被终止  wall_ms=<t2>  min_wall_ms=<t1>
tune prune: N 组因超过 min_wall_ms 被提前终止；最终上界 min_wall_ms=<t_best>，由 kernel_force=dedicated  as_equivalence=false  optimize=dp 设定
tune file : /tmp/xxx/pyqcd.einsum.profile.csv
tune wrote: kind=config  shape='<shape>'  kernel=dedicated  optimize=dp  wall_ms=<t_best>  note=-
tune best : kernel_force=dedicated  as_equivalence=false  optimize=dp  wall_ms=<t_best>
```

- `tune comb`：第 n / N 组，**全列逐行打，与 `--verbose` 无关**；`状态` 是
  `ok` / `pruned` / `error=` / `timeout` / `MISMATCH` 之一。
- `tune prune`：剪枝汇总，并点名把上界压到最低的那一组。
- `tune file` / `tune wrote` / `tune best`：表在哪、写了什么、最优是哪组。
- `--verbose` 额外给"最快 5 行 + 每个内核最快一行"的速览。
- 末尾固定给一张对齐的汇总表（**全部组**，列 `# / kernel_force / as_equivalence /
  optimize / wall_ms / 状态`），表尾是 `best:` 与 `saved:`。

### 3.4 tune 相关选项

| 选项 | 缺省 | 含义 |
|---|---|---|
| `--tune [all\|pass]` | 关 | `all`（或裸 `--tune`）跑完整遍历；`pass` 跑快速通过模式；`false`/空串关闭 |
| `--tune-timeout` | `15` | **每一次** `contract()` 调用的上限秒数；整组窗口为 `15 × (1 + warmup + rounds)` 秒。`0` = 不设上限。超时只跳过该组，绝不中止整轮 |
| `--tune-budget` | `1000` | 整轮遍历的总时长上限（秒）；用尽即跳过剩余组合并在报告里给出"已测 N/M"。`0` = 不限 |
| `--compare`（tune 下） | 关 | 不给就不跑参考基准；显式给出时主路径照旧跑一次参考，但不影响遍历维度 |

显式 `--kernel-force` / `--as-equivalence` / `--optimize` 只要不在遍历网格里，就会
**被忽略**，每个参数在 stderr 上给一行警告，遍历照常进行、退出码仍为 0。

### 3.5 tune 之后怎么用

```bash
python -m pyqcd.einsum.test --equation "<同一条>" --shape "<同一形状>"
```

命中时打印一行 profile hit（含表文件的绝对路径与数据行号），`route` / `kernel` 证据行照旧。
`--profile false` 只让本次不读表；`--explain true` 恢复详解；`--verbose true` 每次调用都打命中行。

### 3.6 `python -m pyqcd.einsum.test` 是什么

它是单方程验证 / 复现 / 对照的**唯一合法入口**：真的算一遍、按容差判对错、报告实际
走的路线与实测耗时。临时脚本或一次性 `python -c` 得到的结论不予采信。完整选项见 §4。

---

## 4 CLI 参考

下文所有取值均区分大小写，缺省值取自当前实现。

### 4.1 输入、形状与数值

| 选项 | 缺省 | 说明 |
|---|---|---|
| `--equation` | 必填 | Einstein 方程，必须带显式 `->` |
| `--shape` | — | `'[a..],[b..],...->[c..]'`；输出段可省略（按方程输出段推导）。省略整个选项时按 `--scene` 反推 |
| `--scene` | `distill` | 反推尺寸用的档：`distill` / `distill_curr` / `scene` / `full`（另有 `smoke` 冒烟档）。这些形状只作路径与正确性证据，见 §7 |
| `--device` | `cpu` | `cpu` / `gpu` / `cuda`；后两者等价，都是**设备名**，后端由 `--backend` 选 |
| `--dtype` | `complex128` | `complex128` / `complex64` / `float64` / `float32` |
| `--backend` | `auto` | 数组库：`auto` / `numpy` / `cupy` / `torch`。原生 KUPL/CUDA 特化由 `--kernel-force` 选择，不是后端 |
| `--atol` | `1e-12` | **相对**容差（按 `max\|expected\|` 归一），不是逐元素绝对阈值；单精度常需 `1e-6~1e-4` |
| `--rtol` | `0` | 附加的相对余量 |
| `--seed` | `0` | 操作数随机种子（各路径看到同一批数据） |

当 `--shape` 省略且方程是靠**等价转换**才在语料里找到时，`--explain true`
（或 `--verbose true`）会多打一行 `shape source: corpus(equivalence) via <语料方程>`，
说明形状来自哪条语料写法。

### 4.2 选路

| 选项 | 缺省 | 说明 |
|---|---|---|
| `--optimize` | 不写（内部按 `auto`） | 收缩序。对外叶子 **7 个**：`greedy` / `optimal` / `dp` / `branch_1` / `branch_2` / `branch_all` / `none`（`none` = 按方程原序，不加规划）。`auto` 是缺省选择器（按算子数让规划器兜底选择），不是可写的叶子：它不在 `--optimize` 的 `choices` 里，显式写它以退出码 2 结束 |
| `--specialize` | `auto` | `auto` 跟随 `PYQCD_EINSUM_SPECIALIZE`；`true` 强制允许特化（不保证命中，原生门照样会拒）；`false` 强制通用路径，是排查原生内核问题的第一手段 |
| `--kernel-force` | `generic` | **只接受单个取值**：`generic`（完全跳过专用内核） / `dedicated` / `topology`（按操作数后端解析成 kupl/cuda） / `original`（交给 `opt_einsum.contract`） / `backend`（直接调后端 einsum）。强制而跑不成抛 `KernelUnavailableError`、退出码 2，**绝不回落**；强制尝试会旁路负缓存，保证真试一次 |
| `--as-equivalence` | `auto` | `auto` / `true` / `false` / `force`。`auto` 只在表里有等价行时改写；`force` 忽略负标记、真的重查一次。改写后返回的数组始终是**原方程**的答案 |
| `--equivalence` | `auto` | 只影响**报告**的等价段：`auto` / `true` / `false`。不改 `contract()` 行为、不改数值 |
| `--explain` | `auto` | 首次详解三段 `plan` / `kernel` / `why`；`auto` 跟随 `--verbose`，`true` 强制打印 |

#### 4.2.1 `optimize` 的对外取值面

> **qcd-einsum 对外提供的 `optimize` 取值只有 7 个：`greedy / optimal / dp / branch_1 / branch_2 / branch_all / none`。**
> 不写 `optimize` 时走内置选择器 `auto`（按算子数查表，≤8 算子且 opt_einsum 可用时改用其 DP 结果）；`auto` 是**缺省值而不是可写取值**，CLI 与配置表都不接受它。
> `none` = 按方程原序（亦可用 `optimize=False` 表示）。`branch_1` / `branch_2` / `branch_all` = 分支定界的窄/全宽变体，三者都是对外取值。
> `branch-1` / `branch-2` / `branch-all`（下划线主名的连字符拼写）与 `dynamic-programming`（≡ `dp`）是**兼容别名**，与主名解析到同一个函数对象，不算独立名字。
> `auto-hq` / `eager` / `opportunistic` / `random-greedy` / `random-greedy-128` 已**从名字面删除**，`paths.get_path_fn` 一律抛 `KeyError`。
> 契约：`pyqcd/testing/_einsum_optimize_surface_contract.py`（本节逐字一致）与 `_einsum_optimize_leaves_contract.py`（运行时名字面）。

注册表里另外 4 个名字只是**拼写兼容**（与主名是同一个函数对象），逐条标注如下：

| 名字 | 现状标注 |
|---|---|
| `branch-1` | 兼容别名 ≡ `branch_1`（同一函数对象） |
| `branch-2` | 兼容别名 ≡ `branch_2`（同一函数对象） |
| `branch-all` | 兼容别名 ≡ `branch_all`（同一函数对象） |
| `dynamic-programming` | 兼容别名 ≡ `dp`（同一函数对象） |

### 4.3 参照与对照

| 选项 | 缺省 | 说明 |
|---|---|---|
| `--compare` | `none` | `opt_einsum`（`oe.contract`）/ `backend`（在另一个后端上互证）/ `none`（只跑不比） |
| `--compare-optimize` | `auto` | **只管参照**的收缩序：`auto` / `greedy` / `dp` / `optimal` / `branch-all` / `branch-2` / `branch-1`。名字不跨库重映射：`opt_einsum` 不认识的取值退出码 2；`backend` / `none` 下没有可设的优化器，只打一行警告、退出码仍为 0 |

想拿 numpy 做对照，用等价写法 `--backend numpy --compare backend`（主路径 numpy，
参照落在另一个可用后端上，两侧数值互证）。

### 4.4 计时、缓存、配置与输出

| 选项 | 缺省 | 说明 |
|---|---|---|
| `--rounds` | `3` | 计时轮数；这些样本的**最小值**写入报告与 `wall_ms` |
| `--warmup` | `1` | 丢弃的预热次数；`0` 仍合法，用于明确把冷启动计入耗时时 |
| `--cache` | `auto` | `auto` / `true` / `false`；显式 `true`/`false` 时 `--verbose true` 多一行 `cache :` 统计，`false` 既不读也不写 |
| `--profile` | `auto` | 配置表三态：`auto` 跟随 `PYQCD_EINSUM_PROFILE`；`false` 本次不读不写、连目录都不建 |
| `--profile-dir` | — | 钉住配置表目录；缺省 `PYQCD_EINSUM_PROFILE_DIR` 或用户缓存目录 `$XDG_CACHE_HOME/pyqcd.einsum`（未设时 `~/.cache/pyqcd.einsum`）。**是"钉住"不是"单次覆盖"**：省略它不会回退到环境变量，只会沿用它 |
| `--json` | — | **路径**不是开关：`--json /path/out.json`；写不进去退出码 2 |
| `--verbose` | `false` | `true` 把逐项明细打到 stderr |
| `--language` | `auto` | `auto` / `en` / `zh-cn`；只改文案，不改结果、不改 JSON 字段、不写环境变量 |

### 4.5 `verify_specialized` 的选项

| 选项 | 缺省 | 说明 |
|---|---|---|
| `equation`（位置参数） | 必填 | 要查询的方程 |
| `--json` | 关 | 机器可读结果，不受 `--language` 影响 |
| `--language` | `auto` | `auto` / `en` / `zh-cn` |
| `--equivalence` | `auto` | `auto` / `true` / `false`；`false` 时等价段整段消失 |
| `--kernel-force` | 不打印该行 | 只在给出时记录"要求了什么"；本模块不收缩，`ran=` / `attempted=` 写 `unknown` |
| `--explain` | `auto` | 缺省等于关闭；`true` 多三行计划（`plan` / `kernel` / `why`） |

---

## 5 配置表

### 5.1 文件与位置

表只有一张，缺省落在 `$XDG_CACHE_HOME/pyqcd.einsum/`（未设时为
`~/.cache/pyqcd.einsum/`）下，文件名：

```text
pyqcd.einsum.profile.csv
```

- **只有 `--tune` 能写**；普通 `contract()` **只读**：不追加、不更新、不重写文件。
- `--profile-dir` 或 `PYQCD_EINSUM_PROFILE_DIR` 可改目录；`PYQCD_EINSUM_PROFILE=0`
  或 `--profile false` 整进程/本次关闭：不读、不写、连目录都不建。
- 旧位置 `./.cache/pyqcd.einsum.profile.csv` 只作为只读回退来源。
- 不要提交任何用户缓存：里面有主机名、方程与形状。

### 5.2 硬规矩

| 规矩 | 行为 |
|---|---|
| 同键只允许 1 行 | 同一 `(shape, dtype, device, equation, backend)` 出现 2 行以上 → 报错退出，并把**全部候选行**打到 stderr（不猜、不选"最新"） |
| 未 tune 则表是空的 | 文件不存在只警告**一次**"需要先 tune"并给出该配置的 tune 命令，然后继续执行；不会替你建空表 |
| 命中不得与显式配置打架 | 命中的内核 / `optimize` / 等价结论与**显式写出**的 `kernel_force` / `optimize` / `as_equivalence` 不一致 → 报错退出（打印键、表值、请求值、记录位置）。`--compare` 不参与；`optimize` 的缺省档等于没写 |
| 命中只施加建议 | 最多替你填 `kernel_force=` 与 `optimize=`，或按等价行改写；方程、dtype、形状、结果都不动 |
| 行数 = tune 过的键数 | 不是调用次数：普通调用跑多少次表都不变，这是设计 |

### 5.3 `wall_ms` 是唯一计时列

`wall_ms` = **整个 `contract()` 调用的墙钟时间**，取 `--rounds` 个受测样本的
**最小值**：含等价转换（若有）、读表、未命中时的规划与匹配、内核执行；不含参照
实现的计时与收尾落盘。
表里**没有第二列计时**可以与它相加或混比。

两类行靠"哪几列有值"区分：

- **性能行**：`kernel` + `optimize` + `wall_ms` 有值；
- **等价行**：`operand_order` / `operand_permutations` / `output_permutation`
  （+ `target_equation` 与一句 `note`）有值，内核与计时留空。

### 5.4 常见排查

| 症状 | 怎么办 |
|---|---|
| 目录/文件根本没出现 | 检查是否 `PYQCD_EINSUM_PROFILE=0` 或 `--profile false`；普通调用本就不建目录 |
| 一直有一行"需要先 tune" | 该目录下没有表：照它给的命令 tune 一次，或 `--profile false` 静音地跑 |
| 报 `ProfileConflict` | 显式参数与表值冲突：去掉那个显式参数、改表、或 `--profile false` |
| 报"同键多行" | 手工编辑或拼表造成：tune 一次该键（写前会删同键所有行），或手工删到 1 行 |

---

## 6 特化路径附录

### 6.1 强烈推荐：先查再跑

想知道一条方程会不会命中特化路径，用：

```bash
python -m pyqcd.einsum.verify_specialized "cadb,menf,ac,em,bd,fn->"
```

它是**纯查询**：不生成操作数、不调用 `contract()`、不改全局状态、无副作用。
判定次序固定：形状特化 → 语料索引（8638 条）→ 等价映射。
退出码：`0` 命中 / `1` 未命中（实际会走通用分步路径）/ `2` 方程非法或参数错误。

### 6.2 三段式读法

**① HIT**——结果行是 `HIT`，并带语料出处：

```text
结果        : HIT
  特化路径  : topology:a1c49862  [kernel(topology)]
  声明内核  : kupl_topology, cuda_topology
  语料出处  : seq=1  T00001  1. 介子两点函数
  代表写法  : cadb,menf,ac,em,bd,fn->
  操作数角色: peram_u, peram_u, gamma_M, gamma_MH, VDV_0, VDV_1
```

"声明内核"只表示契约允许哪些入口，**不表示这次真的会下发**；实际路线看 `test` 的
`route` / `kernel` 行（§4）。

**② MISS，但有等价写法**——`MISS` + 等价段（输入侧与输出侧两行恒在，某侧无需调整写 `-`）：

```text
结果        : MISS
  等价写法  : cadb,menf,ac,em,bd,fn->
  变换类型  : operand-axes
  张量调整  : transpose(1, 0, 2, 3), -, -, -, -, -
  结果调整  : -（改写后输出轴序已一致）
```

**③ MISS 且无等价**——只给一句"既不在语料…也不存在等价写法"，
**没有** `等价写法 / 变换类型 / 张量调整` 任何一行。

### 6.3 `--json`

```bash
python -m pyqcd.einsum.verify_specialized "acdb,menf,ac,em,bd,fn->" --json
```

关键字段：`equation` / `parsable` / `matched` / `digest` / `reason_key` /
`reason_args` / `equivalence`。**`reason_key` / `reason_args` 是语言无关的稳定形式**，
脚本请读它们，不要去解析 `reason` 那句人读文案。

`equivalence` 子对象的字段：

| 字段 | 含义 |
|---|---|
| `equation` | 改写后的方程（可直接交给 `contract`） |
| `rewritten` | 是否真的改写；`false` 表示查过了但没有等价写法 |
| `operand_order` | 张量列表新次序，或 `null` |
| `operand_permutations` | 逐操作数 `transpose` 置换（按**改写后**次序），或 `null` |
| `output_permutation` | 结果回填用的 `transpose` 置换（**已是逆置换**，直接喂），或 `null` |
| `kind` | 变换类别：`identity` / `operand-order` / `operand-axes` / `output-axes` 及其 `+` 复合 |

### 6.4 等价写法怎么用：输入变了，输出必须跟着变

- `operand_order[i]` = 改写后第 i 个操作数取你原来的第几个；
- `operand_permutations[j]` = 该位置的新轴 j 取旧轴 `perm[j]`（numpy `transpose` 语义）；
- `output_permutation` = 已经存成逆置换，直接喂 `np.transpose`。

**漏掉输出回填常常不会报错**：语料里大量轴长相同，形状检查会静默通过。
判据只有一条——改写后 `transpose(output_permutation)` 的结果必须逐元素等于原式直算的结果：

```python
import numpy as np, pyqcd.einsum as pe
from pyqcd.einsum import verify_specialized

original = "cwdx,eufv,mknl,oapb,qgrh,istj,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->GMHew"
m = verify_specialized.query(original)["equivalence"]
ops = [...]                                     # 按你原来的操作数次序准备

order = m["operand_order"] or list(range(len(ops)))
perms = m["operand_permutations"] or [None] * len(ops)
adjusted = [ops[i] if p is None else np.ascontiguousarray(np.transpose(ops[i], p))
            for i, p in zip(order, perms)]
got = pe.contract(m["equation"], *adjusted)
if m["output_permutation"]:
    got = np.transpose(got, m["output_permutation"])
assert np.allclose(got, pe.contract(original, *ops), rtol=1e-12, atol=0)
```

两个实操要点：

- `np.transpose` 返回**视图**，dedicated 内核要求操作数 C 连续；要它走 dedicated，
  就先 `np.ascontiguousarray` 落一份副本（代价是一份同大小内存与一次拷贝）。
- 等价改写只改"方程怎么写"，不改任何执行路径，也不改调用方的数据。

### 6.5 它不回答什么

- **HIT ≠ 原生内核一定会跑。** 资源门、dtype 策略、连续性与库可用性都会让它回落通用路径；
  要看"这次真的走了哪条"，读 `test` 的 `route` / `kernel` 行。
- **不验证数值、不测性能。** 数值用 `test` 判容差，性能只能靠 `--tune` 与你自己实测。
- **不列"所有支持的特化路径"。** 语料 8638 条，逐条罗列既无用又必然过期，用查询代替。

---

## 7 已知限制

1. **`--scene` 的形状不是生产规模结论。** 省略 `--shape` 反推出的尺寸只用于路径与
   正确性证据；要生产规模的耗时，必须把真实形状显式写进 `--shape`。
2. **本文没有任何性能承诺。** 机器、负载、dtype、形状、后端任一不同都不可比；
   先 tune，再以你自己机器上的实测为准。
3. **特化是"允许"不是"保证"。** 原生资源超限、dtype 被策略排除、操作数不连续或库缺失时
   一律回落通用路径——数值不变，只是路线不同。
4. **等价映射只在语料形态内工作。** 不做代数化简（不会把 `ij,jk->ik` 合并成矩阵乘），
   含省略号的方程不支持；正确性由调用方按 §6.4 复算。
5. **GPU 路径依赖本机栈。** 驱动、算力与构建产物任何一个不对，内核就跑不起来；
   先用 `python -m pyqcd --gpu-probe` 自查，再谈性能。
