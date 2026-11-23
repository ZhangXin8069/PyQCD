# tune 卡死与超时：判据与流程（2026-11-19）

面向**下一次改 `--tune` 的超时 / 参照 / 准入那几段代码的人**。这里只写"怎么判、怎么做、
怎么证明"；时间线、原始命令与输出摘要见 `logs/einsum-tune-timeout-20261119.txt`。

| 项 | 值 |
|---|---|
| 范围 | `pyqcd/einsum/test.py`：单次调用执行路径、参照、准入探测、`--tune` 取值语义 |
| 不在范围 | 收缩数值本身、原生内核实现、`docs/{en,zh}/performance.md`（主 agent 另管） |
| 证据口径 | 只认本机实测 + 可复跑命令；未复现的第三方数字标 `[转述]` 或 `unverified` |
| 本机 | Linux / RTX 4060 Laptop / cupy 14.2.0 / torch 2.15.0+cu132 / numpy 2.5.4 / Python 3.12.4 |

---

## 1. 三条硬结论（先背下来）

1. **缺省路线是子进程硬上限，进程内路线才是 best-effort**。缺省每组在全新解释器中
   运行，父进程可 `killpg`，因此能打断原生 C/CUDA；只有
   `PYQCD_EINSUM_TUNE_INPROCESS=1` 才退回 `signal.setitimer` + `SIGALRM`，其处理器只在
   **回到解释器**时派发。
2. **`fork` 子进程隔离在这台机器上不可用**：父进程只要碰过 CUDA，fork 出来的子进程
   第一次 `cupy.asarray` 就是 `cudaErrorInitializationError`。现行实现用 `spawn`
   全新解释器，代价见 §6。
3. **"tune 卡死无输出"最常见的两个位置，都不是收缩内核**：一是 `_tune_reference` 那次
   完整 `np.einsum`（发生在任何超时保护之外），二是 `optimal` 的穷举规划（纯 Python，
   但可能几十秒）。先按 §3 的四步定位，再动代码。

---

## 2. 进程内路线的能力边界（2026-11-19 实测）

`_guarded_call(call, limit, ...)`，`limit = 0.3 s`，本机 2026-11-19：

| 受测调用 | 实际墙钟 | 判定 | 结论 |
|---|---|---|---|
| 纯 Python `while perf_counter()-t < 5.0: pass` | **0.30 s** | `_TuneTimeout` | 能打断，精度就是上限 |
| 单次 `5000³` 原生 BLAS `a @ b`（该机 **2.34 s**） | **2.29 s** | `_TuneTimeout` | 只能等它返回再判：**超限 1.99 s** |
| 用户实测：`--tune` 一条 10 算子方程，`STAT=Rl+` / `%CPU=100` / `GPU 0%` | **3 分钟以上** `[转述]` | — | 闹钟完全不生效 |
| 非主线程 / 无 `SIGALRM` 平台 | 跑完才判 | `timer=posthoc` | 退化路径，如实标注 |

判据：**按"能否回到解释器"分档**，不按"耗时长不长"分档。进程内路线只能写
"能打断 Python 侧长循环、打不断原生 C/CUDA"，并同时给可操作替代
（把上限取足够大 / 先跑 `--tune pass`）。含糊句（"可能打断原生内核"、"建议在新进程里跑
tune"）**禁止**出现——前者把能力说反又等于没说，后者把问题推给用户，而且本机根本做不到。
契约 `pyqcd/testing/_einsum_tune_timeout_doc_contract.py` 逐字钉住这三处口径。

---

## 3. 定位"卡死无输出"：四步，按顺序做

1. **看 `tune init :` 起止行**。它们是成对的（`branch probe start/done`、
   `kernel admission start/done`、`session start`、`first group start/done`），最后一条
   `start` 没有配对的 `done` 就是卡住的位置。`pass` 模式里前两对写 `skipped (pass mode)`。
2. **看进程状态，不要猜**。`ps -o pid,stat,pcpu,etime -p <pid>` +
   `nvidia-smi --query-gpu=utilization.gpu --format=csv`：`STAT=Rl+`、`%CPU=100`、
   `GPU 0%` 说明在原生 C 代码里转（CPU 侧规划/打包），不是 GPU 在算。
3. **分离参照**。参照调用在遍历**之前**、在任何 `_guarded_call` 之外。把输入元素总数
   算出来与 `TUNE_REFERENCE_MAX_ELEMENTS`（`2_000_000`）比：超了就该走"跳过 + 自洽参照"
   那条路（§5）。实测用户那条 10 算子方程 = **256,010,297** 个输入元素，是门限的 **128 倍**。
4. **看内核准入**。`dedicated` 的准入会真做一次计划（只规划不下发），`topology` 的准入
   是 `_specialize.match`。两者都**不执行收缩**，正常是毫秒级；若这一段卡住，问题在计划
   arena 或签名计算，不在计时机制。

---

## 4. 超时粒度：为什么是"组级窗口"

一组有 `1 + warmup + rounds` 次调用（缺省 `warmup=1`、`rounds=3` → **5 次**），组级
窗口缺省为 `15 × 5 = 75 s`。若按"每一次调用装一个上限"，最坏情况仍会把
`5 × limit` 全花光，而且与 `min_wall_ms` 剪枝 **不同口径**（剪枝看整组样本最小值）。

现行实现：

```python
group_limit = float(limit) * float(1 + warmup + rounds)   # 一个窗口覆盖整组
seconds = _timed(lambda: _guarded_call(
    lambda: _measure(call, None, deadline, min_wall),     # 单次上限已置 None
    group_limit, deadline), warmup, rounds)
```

由此产生两条**必须同时写进文档**的语义后果：

1. 窗口是**整组预算**，不是"每次调用都必须在上限内"。单次调用超过 `limit` 但整组没超
   预算时**不会被判 timeout**——这是刻意的（否则一个慢 warmup 会废掉整组）。
2. 文案必须同时说清缺省子进程硬上限和进程内 best-effort 边界，否则用户会把两条路线
   混成一条。

判据：**改粒度之前先问"这个上限与谁同口径"**。与 `min_wall_ms` 同口径 → 整组预算；
与"单次调用不得超时"同口径 → 那就要接受 `5 × limit` 的代价，并把它写清楚。

---

## 5. 参照：门 + 退化路径，缺一不可

`_tune_reference` 跑一次完整 `np.einsum(optimize="greedy")`，它在**任何超时保护之外**，
生产形状上可能几分钟。所以按**输入元素总数**设门：

| 条件 | 参照 | 数值校验 |
|---|---|---|
| `total <= 2_000_000` | 真 `np.einsum` | `rel` 对真值，`rel > atol+rtol` 判 `mismatch` |
| `total > 2_000_000` | `None` + 原因 | 退化：**首个成功组合**的结果当参照（自洽校验） |
| `--tune pass` | `None`（`skipped (pass mode)`） | 整段不做，表尾写"pass 模式：跳过数值校验" |

**关键判据：只跳过、不补参照 = MISMATCH 永远检不出。** `rel` 依赖参照；参照为 `None`
时所有组 `rel=None`、`mismatch` 恒假，而契约里"`rel` 超容差必须判 MISMATCH"就变成
永真通过。所以大形状必须落到"首个成功组合自洽"这条路上——它零成本，却能抓住"不同内核
算出不同数"这类真实分歧。改这段代码时，两种合法写法只有：

- 保留小形状真参照；或
- 用自洽参照（首个成功组）。

**不能**只把参照设成 `None` 就完事。

---

## 6. 硬超时：`fork` 不可行，`spawn` 的真实价格

本机复现（父进程先做一次 cupy 乘法，再 `fork`）：

```
fork child exitcode: 0
fork child result: ('error', 'CUDARuntimeError: cudaErrorInitializationError: initialization error')
```

即：**进程能起来、退出码还是 0，但 CUDA 初始化必失败**（用户复测时是 22 组全 `error`）。
`cupy`/`cuda` 的上下文绑在进程上，`fork` 只复制 Python 堆，不复制驱动状态。

若坚持硬超时，唯一可行方案是**每组一个全新解释器**（`subprocess` / `spawn`）：

| 动作 | 本机中位（n=5） |
|---|---|
| `python -c pass` | 18.2 ms |
| `python -c "import numpy"` | 70.1 ms |
| `python -c "import cupy"` | 243.6 ms |
| `import cupy` + 建 CUDA 上下文 + 一次 `matmul` | **497.6 ms**（min 425.4 / max 1174.5） |

也就是说现行 28/29 组要多付 **约 14–15 s**（GPU 上还会更多），而且操作数要落盘
（`np.save` 逐位无损）再按后端载入。`pyqcd/einsum/_tune_worker.py` 已是现行子进程
路线的执行端，`_tune_runner.py` 负责启动、暂存、强杀与启动开销报告；这些开销明确
**不计入 `wall_ms`**。

> `[转述]` 任务书里的"spawn 每组约 1.7 s"本轮**未复现**：本机实测约 0.50 s（含一次
> CUDA gemm）。引用他人数字时必须标来源与本机是否复现。

---

## 7. 语义顺序：`as_equivalence=true` 是"先改写，再强制内核"

`as_equivalence=true` 的语义不是"允许改写"，而是"改写成语料形态后跑特化内核"。若代码
顺序写成"先拿原方程跑特化内核、失败才考虑改写"，那么 `kernel_force=topology` +
`as_equivalence=true` **必定失败**——原方程根本不在语料里。

本机验证（`cbda,menf,ac,em,bd,fn->`，改写后 `cadb,menf,ac,em,bd,fn->` 命中语料）：

```
as_equivalence=False kernel_force=topology -> KernelUnavailableError（原方程不在语料）
as_equivalence=True  kernel_force=topology -> OK rel=1.54e-16
```

正确顺序在 `_api.py` 的 `elif forced:` 分支：**先** `_contract_via_equivalence`，**再**
`_run_forced_kernel`。动这段之前先读那里现有的注释。

附带一条命名陷阱：`tune_branch()` 里的 `equiv_ok`（以及报障报告里叫的
`equivalent_ok`）**只表示"语料签名命中"**，不表示"能执行"。CUDA 侧还要过两道门：
编译期步表覆盖（`cpp/cuda/einsum/src/topology_steps.h`，本机 **8638 条表项 / 3477 条
不同步序**）与运行时 `status 4`（`QCD_TOPOLOGY_UNSUPPORTED`，`apply_contract.cu` 里
**4 个**返回点）。"命中但不可执行"是**真实存在**的状态，所以准入探测不能只靠 `match`。

---

## 8. `--tune` 的三种形态（写文档时最容易漏）

| 输入 | 语义 |
|---|---|
| 留空（裸 `--tune`）/ `true` / `True` / `all` | **完整模式**：分支探测 → 有序清单（B1/B2 = 29 组、B3 = 28 组）→ `min_wall_ms` 剪枝 → 参照校验 |
| `pass` | **快速通过**：强制 B3 那条线（28 组、无 `topology`）、`as_equivalence` 全 `false`、`--compare` 全关、不跑参照、不判 MISMATCH，表尾注"pass 模式：跳过数值校验" |
| 不给 / `false` / 空串 | 不遍历，只读配置表 |

实测（`ab,bc,cd->ad`，CPU、`--profile false`）：`pass` 模式共 **9 行** `tune init :`
（其中 `branch probe` / `kernel admission` / `first group` **三对**起止成对，其余
`mode` / `shapes ok` / `session start` 各一行）、遍历 28 组、末行
`mode: pass 模式：跳过数值校验`。

判据：**"不给"≠"给 true"**，而 `_as_bool("pass")` 是 `False`——任何新入口都必须走
`parse_tune_mode()`（唯一解析点），不要各自把字符串翻成布尔。

---

## 9. 文档诚实：两张网

### 9.1 三处文案必须同口径

`--help` 的 `help.tune_timeout`、tune 启动时的 `tune note :`、模块 docstring 的
「超时与预算」段——三处都要区分"缺省子进程硬上限"与"进程内 best-effort、打不断原生
C/CUDA"，并给 "`--tune pass` / 取足够大" 的可操作建议。契约逐字校验（4 项，含
"整份源码里不得出现含糊句"）。

### 9.2 性能页必须自证口径

旧 `docs/en/performance.md`（`/tmp/perf_en.bak`）与本轮新页的本机计数：

| 项 | 旧页 | 新页 |
|---|---|---|
| 行数 | 577 | 125 |
| 表格行 | 104 | 5（`--tune` 形态表，全是文字） |
| 形如 `0.861x` 的比例 | **30** | **0** |
| 表格内数字单元格 | 36 | **0** |
| `seq=` 提及 | 5 | 0 |

旧页的 30 个比例全部基于**已废弃的拓扑步表口径**（`production` 异形轴 / `uniform`
`dim=4` 探针），与 `ROLE_SHAPES` 不可比。现行做法是整页删减为"先 tune + 任何性能以实测
为准 + 怎么读输出"，**不给加速比**。

> `[转述]` 任务书说旧页"约 92 处数字"：按"含数字的行"数是 196、按"数字 token"数是 588、
> 按"比例型"是 30。**同一个文件三种数法差 20 倍**，引用时务必带上数法。

判据：性能页上的每个数都必须能回答"在哪个形状口径、哪台机、哪个 dtype、谁量的"。
答不上来就删，不要改写成"仅供参考"。

---

## 10. 改动检查单（改 tune 超时相关代码前）

1. **别动能力承诺的顺序**：先想清楚"这一段能不能被打断"，再决定文案；文案三处一起改。
2. **契约先跑**：`python -B -m pyqcd.testing._einsum_tune_timeout_doc_contract`
   （4 项）——它是唯一会因"文案说反能力"而红的门。
3. **三种 `--tune` 形态都要实跑一遍**（小方程即可，`--profile false` 避免污染配置表）。
4. **参照路径两条都验**：小形状（真参照、`rel` 有数）与大形状（跳过、自洽参照、`rel`
   仍有数）。只验一条会把 §5 那个"恒假"陷阱放回去。
5. **不许新增含糊句**：`grep -c '尽力而为\|best-effort' pyqcd/einsum/test.py` 应 ≥ 2，
   且 `--help` 段与 `tune note :` 行各自含一次。
6. **frozen 数字别外推**：本页所有毫秒都是 RTX 4060 Laptop + 单次运行；换机重测，
   不要标成"通用开销"。
7. **写报告时分清"我量的"与"别人说的"**：本轮的三个 `[转述]` 项（3 分钟挂死、1.7 s
   spawn、92 处数字）都单独标了。

---

## 11. 已知残余（下次开工先读）

1. `help.tune_timeout` 仍写"每一次 `contract()` 调用的上限"，实现是**组级窗口**
   （§4 第 2 条）。改文案要同步 `tune.signal_note` 与 docstring，并跑契约。
2. `pyqcd/einsum/_tune_worker.py` 是孤岛（无调用方，docstring 引用不存在的函数）。
   清理或接它都需要另一个任务，且接它之前先解决 §6 的操作数传递与开销记账。
3. `resolve_compare_optimize()` 里仍有 `if compare == "numpy":` 死分支（`--compare numpy`
   已从 `COMPARE_MODES` 删除），同理 `translate` 里 `error.numpy_compare_optimize`
   词条无人引用。属可执行代码，本轮未动。
4. `topology` 的 `status 4` 只能在**真下发之后**才知道，所以"命中语料"不等于"准入通过"
   （§7 结尾）。若要让这类失败从 `error` 变 `skip`，唯一可靠办法是预探测时真跑一次。
5. 本轮只写文档与注释，未改任何可执行语句，也未提交 git。
