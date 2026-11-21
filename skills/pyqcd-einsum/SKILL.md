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

## 单条方程查询：生产包 vs 测试包（2026-11-12）

**两个入口职责不同，不要混用**：

| 入口 | 位置 | 做什么 | 随 pip 发布 |
|---|---|---|---|
| `python -m pyqcd.einsum.verify_specialized "<eq>"` | 生产包 | **纯查询**：命中与否、拓扑签名/摘要、声明内核、语料出处（seq/T 号/角色） | 是 |
| **`python -m pyqcd.einsum.test`** | **生产包** | **单方程验证 + 计时（唯一合法入口）** | 是 |
| `python -m pyqcd.testing.einsum.verify_specialized "<eq>"` | 测试包 | **数值验证**：实际跑 contract，比对参考、判是否真跑原生内核、计时 | 否（`MANIFEST.in` `prune pyqcd/testing`） |

## `optimize` 对外只有 7 个取值（2026-11-18 硬性约定）

> **qcd-einsum 对外提供的 `optimize` 取值只有 7 个：`greedy / optimal / dp / branch_1 / branch_2 / branch_all / none`。**
> 不写 `optimize` 时走内置选择器 `auto`（按算子数查表，≤8 算子且 opt_einsum 可用时改用其 DP 结果）；`auto` 是**缺省值而不是可写取值**，CLI 与配置表都不接受它。
> `none` = 按方程原序（亦可用 `optimize=False` 表示）。`branch_1` / `branch_2` / `branch_all` = 分支定界的窄/全宽变体，三者都是对外取值。
> `branch-1` / `branch-2` / `branch-all`（下划线主名的连字符拼写）与 `dynamic-programming`（≡ `dp`）是**兼容别名**，与主名解析到同一个函数对象，不算独立名字。
> `auto-hq` / `eager` / `opportunistic` / `random-greedy` / `random-greedy-128` 已**从名字面删除**，`paths.get_path_fn` 一律抛 `KeyError`。
> 契约：`pyqcd/testing/_einsum_optimize_surface_contract.py`（本节逐字一致）与 `_einsum_optimize_leaves_contract.py`（运行时名字面）。

写代码 / 写文档 / 写 tag 消息时只从这 7 个里挑。注册表（`paths.PATH_OPTIONS`，11 个名字 =
7 叶子 + 4 别名）里另外 4 个**只是拼写兼容**，逐条标注如下（不许沉默地当它们可用）：

| 名字 | 现状标注 |
|---|---|
| `branch-1` | 兼容别名 ≡ `branch_1`（同一函数对象） |
| `branch-2` | 兼容别名 ≡ `branch_2`（同一函数对象） |
| `branch-all` | 兼容别名 ≡ `branch_all`（同一函数对象） |
| `dynamic-programming` | 兼容别名 ≡ `dp`（同一函数对象） |

**内部实现对照**（排查时按这几行读代码，别按名字猜）：

| 名字 | 落点 | 证据 |
|---|---|---|
| `auto`（**不是可写叶子**） | `_plan_py.auto` | `_AUTO_CHOICES`：1–4→`optimal`、5–6→`branch_all`、7–8→`branch_2`、9–14→`branch_1`、≥15→`greedy`；随后 `_reference_dp_path`（`len(inputs) > 8` 放弃）用 `opt_einsum` DP 覆盖。只经 `paths._IMPLICIT_DEFAULT_NAMES` 承接缺省调用，`get_path_fn("auto")` 抛 `KeyError` |
| `none` | `paths._NO_PATH` 哨兵 | 注册表里的值就是该哨兵；`_api._normalize_optimize("none")` 另外把对 optimize 有效的内核归一成 `False`（"整条链一次 n 元 einsum"）。`kernel_force='topology'` 上原样保留（顺序由内核决定） |
| `greedy` / `optimal` / `dp` | `_plan_py.greedy` / `.optimal` / `.dynamic_programming` | `paths._BUILTIN_ATTRS`（`dynamic-programming` 是 `dp` 的别名） |
| `branch_1` / `branch_2` / `branch_all` | `_plan_py.branch_1` / `.branch_2` / `.branch_all` | 宽度由 `nbranch=1/2/None` 决定，三个都是对外叶子；`branch-1`/`branch-2`/`branch-all` 只是连字符拼写，解析到同一对象 |

**踩坑**：`branch_all` = `branch(nbranch=None)`，与 `branch_1`/`branch_2` 是**不同**的计划（窄/全宽），
三个都可写；`branch`（无后缀）与已删的 `eager`/`opportunistic`/`random-greedy*`/`auto-hq`
现在一律 `KeyError`。`--compare-optimize` 是参照侧的名字空间（numpy / opt_einsum），与本节无关。

契约：`pyqcd/testing/_einsum_optimize_surface_contract.py`（**4 项**：`get_path_fn` 解析、
`contract()` 七名字数值 rel ≤1e-12、集合**恰好**这 7 个且别名逐条标注、三份文档逐字一致）
与 `pyqcd/testing/_einsum_optimize_leaves_contract.py`（**8 项**：运行时名字面、
`PATH_OPTIONS_MAIN`、已删拼写必须抛错、`none` 的内核语义、`auto` 只是缺省值）。
改名字面时先改这两个契约再改实现。

## 单方程测试的唯一接口 `python -m pyqcd.einsum.test`（2026-11-15，**强制**）

> ⛔ **硬性规定**：`qcd-einsum` 的**单方程验证 / 复现 / 性能对比**只有这一个合法入口。
> **禁止**新增 `/tmp/*.py`、scratchpad 脚本、一次性 `python -c`、或
> `pyqcd/testing/` 下的临时脚本替代它。违反者的验证结论**不予采信**，不得写进报告、
> 文档或 tag 消息。

唯一例外是整库回归 `python -m pyqcd.testing`——它答"整库有没有退化"，本入口答
"这一条方程对不对、多快"。**除此之外没有第三个入口。**

强制理由（三次实测教训，不是形式主义）：

| 教训 | 实测 |
|---|---|
| warmup 口径不一 | 临时脚本普遍测单次，而首调含 plan 缓存 + 原生库绑定 + JIT；实测数字见 `pyqcd/einsum/AGENTS.md` §9.0 的 warmup 论证（本卡不重复） |
| 容差口径不一 | `np.allclose` 的逐元素绝对阈值误判近零元素；本入口统一相对容差（按 `max\|expected\|` 归一） |
| 参照不固定 | 有人比 `np.einsum`、有人比 `opt_einsum`、有人不比，结论无法互比 |

```bash
python -m pyqcd.einsum.test --help      # 全部选项与缺省值，以 --help 为准

python -m pyqcd.einsum.test --device gpu --dtype complex128 \
    --equation 'ambn,cgdh,ekfl,oipj,ce,mo,gi,bdf,np,hjl->ak' \
    --shape '[400,4,1,1],[4,4,9,1],[4,400,400,400],[4,4,9,9],[4,4],[4,4],[4,4],[1,9,400],[1,9],[1,9,400]->[400,400]' \
    --atol 1e-12 --backend cupy --verbose true --compare opt_einsum
```

| 选项 | 缺省 | 可选 |
|---|---|---|
| `--device` | `cpu` | cpu / gpu / cuda |
| `--dtype` | `complex128` | complex128 / complex64 / float64 / float32（四类平权） |
| `--backend` | `auto` | auto / numpy / cupy / torch / kupl / cuda |
| `--compare` | `numpy` | numpy / opt_einsum / **backend**（跨后端互证）/ none |
| `--compare-optimize` | `auto` | **只作用于参照**的优化器：auto / greedy / dp / optimal / branch-all / branch-2 / branch-1。auto = 沿用旧行为（numpy→`greedy`；opt_einsum→`greedy` 或跟随 `--optimize`）。**不跨库重映射**：numpy 只认 `greedy`/`optimal`（`dp`/`branch-*` 报用法错误，退出码 2），opt_einsum 按 `oe.paths` 注册表校验；`--compare backend`/`none` 下无效（仅提示） |
| `--optimize` | `auto` | 同 `contract(optimize=...)`，**只管主路径** |
| `--specialize` | `auto` | auto / true / false：auto 跟随 `PYQCD_EINSUM_SPECIALIZE`；true 强制特化；**false 强制通用路径**（排查原生内核问题的第一手段） |
| `--equivalence` | `auto` | auto / true / false：auto 跟随 `PYQCD_EINSUM_EQUIVALENCE`（缺省开启，`0` 关闭）。未命中特化但存在等价写法时，`--verbose true` 给出改写后的方程 + **输入侧张量调整** + **输出侧结果回填**（无则 `-`）；已命中不出该段，无等价写法明写「无」。`--json` 带 `equivalence` 字段。**只做报告**：不改 `contract()` 行为、耗时与数值 |
| `--atol` | `1e-12` | **相对**容差（按 `max\|expected\|` 归一） |
| `--rtol` | `0` | 附加相对余量 |
| `--rounds` | `5` | 计时轮数（中位） |
| `--warmup` | `1` | 预热丢弃次数，**不允许 0** |
| `--seed` | `0` | 操作数随机种子 |
| `--verbose` | `false` | 明细打到 stderr |
| `--language` | `auto` | 输出语言：`auto` / `en` / `zh-cn`。`auto` 只在 `PYQCD_EINSUM_LANG` 显式设了时跟随（`en*`/`zh*`），未设时为**中文**（缺省行为零变化，不跟随 locale）；只改文案，不改结果、不改 `--compare`、不写环境变量 |
| `--shape` | 省略 | 省略则按 `--scene`（缺省 distill）从语料 roles 反推 |
| `--json` | 无 | 完整结果落盘 |

**两条不可协商的口径**：

1. `--atol` 是**相对**容差，不是 `np.allclose` 的逐元素绝对阈值（后者误判近零元素）。
2. **warmup 不得为 0**：首调包含 plan 缓存 + 原生库绑定 + JIT，与热态不是一个量级。
   实测对比与倍数只在 `pyqcd/einsum/AGENTS.md` §9.0 的 warmup 论证里，**本卡不复述数字**。

**输出里的 `kernel` 字段最有用**：直接给出实际原生路线
（`kupl_dedicated` / `cuda_topology` / `kupl_topology` / `kupl_topology_v2`），
不必再翻 `LAST_INFO`。`kupl_topology_v2` 意味着**回落 v2 通用执行器**、不是 v3 步表——
它是 `topology` 的第二条执行路线，不是独立内核。用户视角（与
`docs/einsum_user_guide_20261116.md` §1.3、`pyqcd/einsum/AGENTS.md` §9.1b 同一口径）：

> **用户只需要选 `topology`。** `topology` 内部有两条执行路线：
> ① 命中编译期步表 → **v3**（per-拓扑 GEMM 分步，快）；
> ② 该拓扑/形状没有编译步表 → 回落通用执行器 → **v2**（仍为原生执行，无 Python 逐步骤调度）。
> 两条都是**同一个内核**，用户不可选、也不需要选；`LAST_INFO['path']` / `route` 会如实显示
> 实际走了哪条（`kupl_topology` = v3，`kupl_topology_v2` = v2 回落）。
> **`kupl_topology_v2` 不是独立内核，不出现在 `kernel_force` 里。**

实测基准（本条方程 10 操作数 / 2.56 亿输入元素 / 输出 `(400,400)`）：

| 设备 | backend | 路线 | 耗时 | rel |
|---|---|---|---:|---:|
| CPU | numpy | `kupl_dedicated` | **7205.2 ms** | 3.30e-15 |
| GPU (V100) | cupy | `cuda_topology`（9 步） | **20.6 ms** | 1.17e-15 |


生产包入口刻意不调用 `contract()`、不生成操作数、不触碰 `_specialize._announced`，
因此可在生产代码的启动自检里直接调用；库内 API 为
`pe.query_specialization(eq)`（`__init__` 的惰性包装，返回同一 dict）。

```bash
python -m pyqcd.einsum.verify_specialized "agbh,cmdn,eqfr,ospt,ikjl,ce,mo,gi,kq,bdf,np,hj,lrt->as"
python -m pyqcd.einsum.verify_specialized "<eq>" --language en     # 输出语言 auto/en/zh-cn
python -m pyqcd.einsum.verify_specialized "<eq>" --equivalence false  # 关掉等价段（原因回到旧版措辞）
python -m pyqcd.einsum.verify_specialized "<eq>" --json             # equivalence 字段随开关有无
```

期望（本机实跑）：`结果: HIT`、`特化路径: topology:8006c502 [kernel(topology)]`、
`语料出处: seq=1650  T00028  10. 重子四点函数`、退出码 `0`。

退出码口径：命中 `0`、语料之外 `1`、方程非法（含非字符串）`2`。

`--language {auto,en,zh-cn}`（缺省 `auto`）只改人读文案：`auto` 只在
`PYQCD_EINSUM_LANG` 显式设了时跟随（未设保持中文，**不跟随系统 locale**）；
判定、退出码、`--json` 的结构化字段都不受影响（`--json` 仍是 `query()` 的原样
序列化，`reason` 恒为历史中文文案，另有语言无关的 `reason_key`/`reason_args`）。

`--equivalence {auto,true,false}`（缺省 `auto`）只管等价段：`auto` 跟随
`PYQCD_EINSUM_EQUIVALENCE`（未设即开启），`true`/`false` 强制开关并**压过**环境变量。
开启时 MISS 多一段「等价写法 + `张量调整`（输入侧）+ `结果调整`（输出侧）」——两行
恒在，某侧无需调整写 `-`（输出侧为 `-（改写后输出轴序已一致）`）；关闭时整段消失、
MISS 原因回到旧版措辞。`--json` 的 `equivalence` 字段开启时总是写、关闭时缺省。
**不改判定与退出码**（同一方程 true/false 下退出码相同）。

### 等价映射开关：`PYQCD_EINSUM_EQUIVALENCE`（2026-11-16）

MISS 时本入口会查"换个写法能不能命中特化路径"（`pyqcd/einsum/_equivalence.py`），
覆盖**三族改写**：操作数位置 / 单操作数轴序 / 输出段轴序；三族可任意复合（如
`cadb,menf,db,em,ac,fn->` ↔ `cadb,menf,ac,em,bd,fn->`），`kind` 用 `+` 连接。
字母重命名不归本层，由 `canonical_signature` 负责。

| 环境变量 | 缺省 | 关闭取值 | 说明 |
|---|---|---|---|
| `PYQCD_EINSUM_EQUIVALENCE` | `"1"`（**开启**） | `0` / `false` / `off` / `no` | **只影响查询报告，不影响任何数值计算**：`contract()` 仍按调用方给的方程走 |

- **调用时**读环境变量（不在 import 时求值），同进程内改即时生效。
- 关闭后 `normalize_equation()` → `None`（零枚举）、`find_equivalents()` → `[]`、
  MISS 报告回到**没有本功能时**的原措辞且不含 `equivalence` 字段
  （契约逐字符比对 `fd26357e` 的实跑原文）。
- **默认路径零枚举**（2026-11-16 重写）：结构不变量分桶 + 桶内构造式同构，复杂度
  O(incidence + Σ Σ|term|²)；位置/轴序的「只取两两互换、位置个数只取 1 或 2」这类
  叠加限制**在默认路径上**已不存在，三族任意复合走同一条搜索（`probe=` 兼容路径仍
  保留 `changed∈{0,1,2}`）。`_MAX_SEARCH_NODES=4096` / `_MAX_LABEL_OPTIONS=512` 是
  **同构搜索**上限，触发只会漏判、不会误判；旧实现的 `_candidate_equations` /
  `_MAX_CANDIDATES` / `orders[:96]` 已删除，只有显式传 `probe=` 才退回有界改写枚举
  （`_MAX_PROBE_CANDIDATES=2048`），两个 CLI 都不传。
- **同桶多条同构写法怎么挑**：先按 `_alignment` 排序，再在前
  `_MAX_ZERO_COST_SCAN=64` 条里用轴序敏感指纹 `_Shape.axis_key` 预筛，取**第一条零
  操作数轴置换且零输出置换**的映射；找不到才退回"按对齐序取第一条成功者"。
  `operand_order` 不计代价（只换实参次序，张量仍 C 连续）。
- 映射给出 `operand_permutations` 时，它是 `transpose(perm)` 语义、返回**视图**而
  通常非 C 连续；若目标是原生特化路径，先落一份 `np.ascontiguousarray`（代价：一份
  额外内存 + 一次拷贝）。不做副本时落点按 dtype 二选一：**复数**落 `kupl_topology`
  （该层无操作数连续性门，在原生边界静默复制一次）；**实数**（float64/float32）
  直接落通用分步路径。
- 契约：`python -B -m pyqcd.testing._einsum_equivalence_contract`（**44 项**，含按
  `_label_sizes`+`dims("distill")` 取的形状对拍、4 个置换方向判别用例；该形状**不是**
  蒸馏生产形状，见根 `AGENTS.md` 硬性禁止 #3）。

**自动施加等价写法**：`--as-equivalence {auto,true,false}`（缺省 **`auto`**）；API 是
`contract(..., as_equivalence=)`。**`auto` 只按配置表的等价行改写**：原 equation 的键上
有 tune 写下的标记行（三个置换字段 + 一句备注）才施加；表缺失 / 缓存关闭 / 未命中 /
没有标记行都**不改写**。`true` 总是尝试（旧语义），`false` 从不改写；tune 内部用
`force` 刷新。施加时改写方程、重排操作数、结果转回原输出轴序——**返回的仍是原方程的
答案**（有独立形状守卫）。`PYQCD_EINSUM_EQUIVALENCE=0` 或 `specialize=False` 时即使
显式 `true` 也不施加；`out=`、额外 kwargs、逐元素方程、显式优化器路径同样不施加。

**缓存与命中统计**：`--cache {auto,true,false}`（缺省 `auto` = 跟随
`PYQCD_EINSUM_CACHE`，默认开启）；API 是 `contract(..., cache=)` 与
`contract_path(..., cache=)`。三处缓存：`specialize.match`（含未命中）、
`equivalence.normalize`（含 `None` 负结果）、`api.find_path`。
`pyqcd.einsum.cache_stats()` 看 `hits/misses/stores/evictions/hit_rate`，
`cache_clear()` 清空；`cache=False` 既不读也不写。报告行只在显式给 `--cache` 时出现：
`cache    : enabled=true  hits=4  misses=3  hit_rate=0.57`。

**`kernels` 字段是声明不是事实**：它来自 `_specialize._topology_entry` 的
`kernels=("kupl_topology","cuda_topology")`，而实际下发常是 `kupl_dedicated`
（分派顺序更靠前），且受 `out=`、dtype、`_probe_double`、原生库可用性影响。
判"真跑了什么"只能看运行时提示行或 `LAST_INFO['path']`，不要读这个字段。

**性能**：查询需要语料出处时首次建 `摘要 -> Task` 索引（8638 条，约 **0.13 s**），
之后 O(1)（热查询实测 **0.019 ms**）。早期实现逐次线性扫描是 12.9 ms/次，
不要退回那种写法。语料外方程不建索引，0 开销。

## 强制指定内核：怎么用 / 怎么排查（2026-11-16）

**它只改选路，不改数值。** 结果是哪条路算的永远等价，所以靠数值测试抓不到它的
缺陷——判断它有没有生效只能看 `last_kernel()` 或报告里的 `kernel :` 行。

### 取值（四种，全部是**强制**值——没有 `auto`）

| 取值 | 行为 | 失败时 |
|---|---|---|
| `generic`（缺省） | 完全跳过专用内核（= 不给该选项） | 不产生 warning（等价 `specialize=False`） |
| `dedicated` | 只准**当前后端的** dedicated（numpy → `kupl_dedicated`，cupy → `cuda_dedicated`） | 抛 `KernelUnavailableError`，**绝不**回落 |
| `topology` | 只准**当前后端的** topology（同上，`kupl_` / `cuda_`） | 同上 |
| `backend` | 直接调后端自己的 `einsum`，**不进**专用分派（与其它取值互斥） | 后端自己抛什么就是什么 |

`dedicated` / `topology` 是**逻辑名**（`_kernel_force.KERNEL_LABELS`），具体标签由操作数
后端在运行时解析——不要再写死 `kupl_dedicated` 这类名字（已不合法）。逗号可给多个、
按书写次序尝试（`topology,dedicated`）。写错取值不会静默忽略——`ValueError` 里列出
合法集合 `backend, dedicated, generic, topology`。

> **为什么没有 `auto`（2026-11-18 用户决策）**：`kernel_force` 是**强制**属性，不该有
> "未指定 → 自动分派"的隐藏档。自动选路交给 `--tune` 的**实测结论**：tune 在同一
> 方程/形状/dtype/设备/后端上遍历 `kernel_force × as_equivalence × optimize`
> （4 × 2 × 3 = 24 组），把胜出那一组写进配置表，普通调用按表施加。留 `auto` 会让
> "运行时猜的分派次序"与"表里的实测结论"两套规则互相竞争，而报告里看不出谁说了算。
> 写 `auto`（大写、或混在逗号列表里）直接报错，消息点名 **tune** 并列出四个合法值。
> `0/false/off/no` 这类"关掉"的旧拼写只映射到 `generic`，不是 `auto` 的替身。

**旧写法要迁移**（这几条现在直接报用法错误）：

| 旧 | 新 |
|---|---|
| `kupl_dedicated` / `cuda_dedicated` | `dedicated` |
| `kupl_topology` / `cuda_topology` | `topology` |

> **与 `docs/einsum_production_optimization_20261112.md` 不冲突（2026-11-18 复核）**：
> 那份文档通篇只用 `generic` / `dedicated` 表示**运行时路径**（`kupl_dedicated` 等），
> 没有把旧值当合法输入写过。口径：**输入侧**只认上表四个强制值（`_kernel_force.ALLOWED_LABELS`），
> **输出侧**（`LAST_INFO['path']` / `route :`）仍是 `kupl_*` / `cuda_*`。有疑义以代码为准，
> 内部契约见 `pyqcd/einsum/AGENTS.md` §9.1b。

`LAST_INFO['path']` / `kernel   :` 行里的 `kupl_dedicated`、`cuda_topology` 等**不受影响**：
那是运行时真实路径名，不是输入。`backend` 与其它取值互斥，且它**不**沿用调用方的
`optimize=`——`_backend_einsum` 自己按后端传 `optimize=True`，也不做形态匹配、规划与
等价改写。

### 两处入口与优先级

```bash
# 环境变量：整进程生效
PYQCD_EINSUM_KERNEL_FORCE=topology python -m pyqcd.einsum.test … --verbose true

# CLI：只作用于这次
python -m pyqcd.einsum.test … --kernel-force topology --verbose true

# 直接调后端自己的 einsum（不进专用分派；与其它取值互斥）
python -m pyqcd.einsum.test … --kernel-force backend --verbose true
```

```python
from pyqcd.einsum import contract
contract(eq, *ops, kernel_force="topology")

from pyqcd.einsum._api import last_kernel   # 注意：未在包级 __all__ 导出
last_kernel()   # {'requested': ('topology',), 'ran': 'kupl_topology', 'attempted': ('kupl_topology',), 'reasons': {}}
```

`requested` 是逻辑名，`ran` / `attempted` 是具体标签——两者不同源是刻意的：
前者回答"我要什么"，后者回答"哪条路真的跑了"。

**显式参数 > 环境变量 > 缺省 `generic`**，都在**调用时**解析（同进程改环境变量即时生效）。
`--kernel-force` 的 argparse 缺省已是 `generic`（不再是 `None`），所以 CLI 总以显式值
参战；`PYQCD_EINSUM_KERNEL_FORCE` 对 `contract(kernel_force=None)` 的 **API** 调用仍然生效。
没有"跟随环境变量就别给这个选项"这一说了——省略该选项就是 `generic`。

### `kernel :` 行怎么读

```
kernel : requested=<generic|label,...>  ran=<label|none>  attempted=<label,...>
```

与它上面那行 `kernel   :`（三空格，`LAST_INFO` 汇总的 dict）**不是一回事**：
`requested` 说"要求了什么"，`ran` 才是"真跑了哪个"。`ran`/`attempted` 取不到写
`unknown`（没观测到 ≠ 没有）。这一行由 `--kernel-force` 触发、只在 `--verbose true`
的明细里出现。

### 排查顺序（照抄即可）

1. **先看最近一次到底跑了什么**，别猜：

   ```python
   from pyqcd.einsum._api import last_kernel
   print(last_kernel())          # ran=None → 一个都没跑成；reasons 里有逐标签原因
   ```

2. **再强制一次**，把 warning 变成明确的报错（强制路径**旁路负缓存**，所以这是
   "真试一次"，不会被上一次的 `status=4` 挡住）：

   ```bash
   PYQCD_EINSUM_KERNEL_FORCE=dedicated python -m pyqcd.einsum.test --equation … --shape … --verbose true
   ```

   跑成了 → `kernel : requested=dedicated ran=kupl_dedicated`（逻辑名 / 具体标签）；
   跑不成 → 退出码 2 + 逐标签原因 + 恢复建议。

3. **仍怀疑是缓存导致"没试"**，显式关掉负缓存复核：

   ```bash
   PYQCD_EINSUM_KERNEL_FORCE=dedicated PYQCD_EINSUM_NEGATIVE_CACHE=0 \
       python -m pyqcd.einsum.test … --verbose true
   ```

   负缓存的症状很好认：`reasons` 里出现 `negative-cache hit:`。强制路径自带旁路，
   所以**强制模式下不该看到这个前缀**——看到就说明旁路没生效。

### 与既有开关的组合（三条例外会早失败）

| 组合 | 结果 |
|---|---|
| `PYQCD_EINSUM_KERNEL_KUPL=0` + 强制 `dedicated` / `topology` | `KernelUnavailableError`，原因 `kernel disabled (…)` |
| `--specialize false` / `PYQCD_EINSUM_SPECIALIZE=0` + 强制取值 | `ValueError`（`test.py` 侧表现为 FAIL + 退出码 1） |
| `out=` + 强制取值 | `ValueError` |
| `generic` + 上面任意一个 | **不冲突**（generic 本就不走专用路径） |

### 常见失败原因对照

| 原因串（`reasons` 里） | 含义 / 下一步 |
|---|---|
| `... （dedicated）` / `... （topology）` | 括号里是逻辑名，冒号前是它在本机后端解析出的具体标签 |
| `kernel disabled (PYQCD_EINSUM_KERNEL_* = 0, or library missing)` | 后端被环境变量关了，或原生库没构建 |
| `real dtype topology is declined by policy` | float64/float32 被 `_TOPOLOGY_DECLINED_DTYPES` 排除；`PYQCD_EINSUM_TOPOLOGY_REAL=1` 可做 A/B |
| `... plan is pack-bound on CPU (...)` | 计划被 pack/flops 判据拒；这是如实拒绝，不是 bug |
| `dtype ... native probe failed at stage=<阶段>: <真实异常>` | 该 dtype 的原生自检没过；按 `stage` 查（`native` = 归档缺入口 → 重建；`tolerance` = 真算错；`dtype/shape` = 结构不符） |
| `negative-cache hit: ...` | 上一次的拒绝被记住了。强制路径不该出现；自动模式下属正常 |

### 用户实报的那个 case

远端集群上 `qcd_cuda_topology_apply failed with status 4 (plan not executable as
version-2 steps)`。自查路径是**先确认它到底能不能跑**，而不是先怀疑构建：

```bash
PYQCD_EINSUM_KERNEL_FORCE=topology python -m pyqcd.einsum.test \
    --device gpu --backend cupy --equation … --shape … --verbose true
```

- 能跑 → `kernel : ran=cuda_topology`（cupy 上 `topology` 解析成它），说明之前只是
  自动分派选了别的入口；
- 不能跑 → 这条命令会给出**逐标签原因**（例如那个 dtype 的 probe 没过、或 arena
  超限、或顺序守卫拒绝），以及"重建原生库 / `--gpu-probe` / `PYQCD_CUDA_ARCHITECTURES`"
  三条恢复建议；
- 再往下才是 `python -m pyqcd --gpu-probe` 看设备与 kernel status，以及
  `PYQCD_CUDA_ARCHITECTURES` 是否与卡的算力匹配。

`status 4` 在 CUDA 侧同时覆盖"步表不认这条计划"与"顺序守卫拒绝了这条计划"，
**只看状态码分不出来**，必须靠强制模式给出的 reason 串。

### `verify_specialized --kernel-force` 的边界（别当成验证器）

那个入口**不做收缩**，所以：

- `--kernel-force` 只决定报告里**出不出** `kernel :` 行、以及 `requested=` 显示什么；
- `ran=` / `attempted=` 恒为 `unknown`——本进程没跑过任何内核，如实写"不知道"；
- 不给 `--kernel-force` 时该行**不出现**（缺省输出与改造前逐字节相同），而且
  `PYQCD_EINSUM_KERNEL_FORCE` 在那儿**完全不起作用**（它不执行，谈不上强制）。

真要验证内核能不能跑，只能用 `python -m pyqcd.einsum.test --kernel-force <value>`。

契约：`python -B -m pyqcd.testing._einsum_kernel_force_explicit_contract`（5 项，取值口径：
`auto` 报错并点名 tune、缺省 `generic`、CLI 缺省、环境变量 `auto` 报错、四标签次序）。
旧的 `_einsum_kernel_force_contract`（9 项）仍断言 `auto` 三态，待随 `_api` 接线重基线。

## 配置表：怎么用 / 怎么排查（2026-11-17 TV 重设计）

`pyqcd/einsum/_profile_cache.py` 维护**一张**配置表 **`pyqcd.einsum.profile.csv`**
（当前工作目录的 `./.cache/`，或 `profile_dir` / `PYQCD_EINSUM_PROFILE_DIR` 指定的目录）。
它把 tune 标定出的 `(shape, dtype, device, equation, backend)` → `(kernel, optimize,
wall_ms)` 落成一行 CSV；下一次同键调用用**一次顺序读**换掉重复的试跑与匹配。它**只改
选路建议、不改数值**：方程、dtype、形状与结果都不动，关掉它永远不会有数值差异。

- 旧名 `profile.csv` **已废弃**；`profile_equivalence.csv` **已删除**——不再创建、不再读取。
  等价结论只以主表里的"等价行"存在（三个置换字段 + 一句备注，没有性能配置）。
- **只有 `--tune` 能写**（`write_winner`）；普通 `contract()` 走 `read_winner`，**只读**。
  **为什么**：这张表是**标定产物**。普通调用也写的话，不同运行环境、不同负载、不同版本
  的观测会互相覆盖，表就不再是任何一台机器上的可信结论。

### 从零到用上缓存：三步

| 步 | 做什么 | 看到什么 |
|---|---|---|
| 1 | 直接跑一次（不 tune） | 警告"需要先 tune" + **该配置的 tune 命令**；然后**照常执行**（按未命中） |
| 2 | 复制那行命令加 `--tune` | 遍历、按 `wall_ms` 选优、写入选优行（同键其它行被删；等价胜出时另写一条等价行） |
| 3 | 再跑同样的调用 | 命中并施加：只打一行 `profile hit …:12 kernel=… optimize=… wall_ms=… (applied)` |

第 1 步打印的模板由 `_profile_cache.ProfileMissing.cli_hint` 按键生成，可直接复制：

```bash
python -m pyqcd.einsum.test --device <device> --dtype <dtype> --equation "<equation>" \
    --shape "<shapes>" --tune --compare opt_einsum
```

### 开关与目录

| 我想… | 敲什么 |
|---|---|
| 用缺省（开，CSV 落在**当前工作目录**的 `./.cache/`） | 什么都不做（`--profile auto` 等于不给） |
| 换缓存目录 | `PYQCD_EINSUM_PROFILE_DIR=/path/to/dir …` 或 `--profile-dir /path/to/dir` |
| 整进程关掉（不读不写不建目录） | `PYQCD_EINSUM_PROFILE=0 …` |
| 只管这一次关掉 | CLI `--profile false`；API `contract(…, profile=False)` |
| 钉住一个缓存目录 | CLI `--profile-dir /path`；API `contract(…, profile_dir="/path")`。这是**钉住**、不是"单次覆盖"：后续不传就沿用它 |

CLI 只有 `test.py` 有这两个选项：`--profile {auto,true,false}`（缺省 `auto` = 跟随
`PYQCD_EINSUM_PROFILE`）与 `--profile-dir DIR`，两个都原样交给
`contract(profile=…, profile_dir=…)`。`verify_specialized` **没有**它们——那个入口不做
收缩、也从不读画像。

**`profile_dir` 的"保持"语义**：省略它（`None` 与空串都算"未给"）**不会**回到
`PYQCD_EINSUM_PROFILE_DIR` 或 `./.cache`，而是保持已经钉住的目录——`_api` 只在
`profile_dir` 为真时调 `configure()`，否则调 `ensure_configured()`，后者"只补齐、不改写"
（`_api.py:1114-1115`）。所以第一次传 A、之后不传仍是 A；想换只能显式传另一个目录，
而换目录 = 换一张表（解析缓存按路径失效）。只有从未配置过的首次调用才解析
env/默认值。

```bash
# 缺省就开着，但**普通调用不建表**：第一次跑只会警告"需要先 tune"（见上面三步）
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill \
    --rounds 2 --warmup 1 --verbose true

# 只这一次关掉配置表（不读、不写、不建目录）
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill \
    --rounds 2 --warmup 1 --profile false

# 把画像放到仓库外，别弄脏工作树
PYQCD_EINSUM_PROFILE_DIR=/tmp/einsum-profile python -m pyqcd.einsum.test …
python -m pyqcd.einsum.test … --profile-dir /tmp/einsum-profile

# 整进程关掉（这条命令不会创建任何目录/文件）
PYQCD_EINSUM_PROFILE=0 python -m pyqcd.einsum.test …
```

### 看状态（只读快照）

```python
from pyqcd.einsum import _profile_cache as p

p.summary()
# {'dir': ..., 'file': ..., 'enabled': True, 'rows': 0, 'keys': 0, 'last_error': None}
p.profile_path()   # '<...>/.cache/pyqcd.einsum.profile.csv'（不建目录、不读文件）
p.last_error()     # 最近一次**读**失败的原因；None 表示没出过问题
```

`summary()` 的键（以 `_profile_cache.summary()` 实现为准）：

| 键 | 含义 |
|---|---|
| `dir` / `file` | 生效目录的绝对路径 / `pyqcd.einsum.profile.csv` 的完整路径 |
| `enabled` | 开关打开（`PYQCD_EINSUM_PROFILE` / `profile=`） |
| `rows` / `keys` | 有效数据行数 / 不同键数。每键一行 → 两者相等；不等就是同键多行 |
| `last_error` | 最近一次**读**失败原因（写失败如实外抛，不记在这里） |

`summary()` 是**只读**快照：不建目录、不写文件。模块**没有** `flush()` / `notice()`：
普通调用根本不写，tune 在 `tune_session()` 窗口**之外**写一次。

### 表长什么样

表头（列序 = `_profile_cache.PROFILE_COLUMNS`；主机名与时间戳按本机变化）：

```
shape,dtype,device,equation,backend,kernel,optimize,wall_ms,target_equation,operand_order,operand_permutations,output_permutation,note,host,updated_at,pyqcd_version
```

- **键 = 前 5 列**：`shape` / `dtype` / `device` / `equation` / `backend`；`host` 只记录、
  **不参与匹配**（同一张表可以复制到别的机器继续用）；
- `shape` 规范化成 `2x3|3x4`（`x` 连轴、`|` 连操作数），`equation` 去掉全部空白；
- **一个键只有 1 行**：行数 = tune 过的键数。普通调用既不追加也不更新，所以"跑了很多次
  但表没变"是正确行为；
- 两类行靠"哪几列有值"区分：
  - **性能行**：`kernel` + `optimize` + `wall_ms` 有值，等价四列留空；
  - **等价行**：`operand_order` / `operand_permutations` / `output_permutation`（+
    `target_equation` 与一句 `note`）有值，`kernel` / `optimize` / `wall_ms` 留空。
    读侧看到它就按它改写（`as_equivalence=auto` / `true`），不提内核；
- 没有 `calls` 列、没有 `schema_version`、没有每键多条：这些都随"每键多条 / 稀疏写"
  一起删除。

### 健壮性与上限

- **坏行**：列数不对 / 编码坏的行逐行跳过（只丢那一行）；
- **读盘**：单次封顶 `_MAX_LOAD_BYTES = 64 << 20`（64 MiB）；
- **字段截断**：`note` 按 200 字符截断（`_CLIP_LIMITS`）；
- **读失败**：只记 `last_error()` 并按"未命中"返回——一行坏表不该让调用方拿不到结果；
- **写失败**：**如实外抛**（唯一调用方是 tune，它打印"写配置表失败"后继续）；
- **fork**：POSIX 下 `os.register_at_fork` 在子进程重建锁，否则父进程持锁时 fork 会让
  子进程首次读表永久阻塞；
- **没有**待写队列、没有 `atexit` 兜底、没有稀疏写门、没有整表预算重写：普通调用不写，
  tune 每次只做一次"读整表 → 删同键 → 只写 1 行"的原子替换。

### 排查对照

| 症状 | 原因 / 下一步 |
|---|---|
| 目录/文件根本没出现 | `PYQCD_EINSUM_PROFILE=0`，或这次是 `profile=False`。注意：环境变量为 0 时**连 `profile=True` 也不 IO**；另外**普通调用不建表**——要表只能 `--tune` |
| 表里没我这次的方程 | 这个键没 tune 过（行数 = tune 覆盖的键数，不是调用次数）。按警告里的命令 tune 一次 |
| 表一直在，但跑多少次都不变 | **这就是设计**：普通调用只读，只有 `--tune` 会写 |
| 启动时有一行"需要先 tune"警告 | 该目录下没有表（或换过 `--profile-dir`）。照它给的命令 tune 一次，或用 `--profile false` 静音地跑 |
| 看不到 `[pyqcd.einsum] profile hit: …` 这行 | 它受库提示开关管：`PYQCD_EINSUM_NOTICE=0` 会静音；未命中只在 `PYQCD_EINSUM_SPECIALIZE_VERBOSE=1` 时才打一行 |
| 命中被记下但没施加 | 这次带 `out=` / `specialize=False`，或显式给了 `kernel_force` / `optimize`——显式要求永远压过表（冲突时**报错**，不是静默） |
| 报 `ProfileConflict` 后退出 | 命中行与你的**显式** `kernel_force` / `optimize` / `as_equivalence` 冲突（`optimize` 的 `auto` 等于没给；`kernel_force` 已无 auto 档，没给即 `generic`）；报错带键、表值、请求值与 `<file>:<行号>`。去掉那个参数、改表，或 `--profile false` 让本次不读表 |
| 报"同键多行"后退出 | 手工编辑或拼表造成重复键。读侧把**全部候选行**打出来；tune 一次那个键（写前删同键所有行），或手工删到 1 行 |
| 命中的建议跑不成 | 自动退回原分派一次（`profile=False` 重试），**不抛** `KernelUnavailableError`；`last_profile()["fallback"]` 为 `True` |
| 旧文件还在，为什么完全没用 | `profile.csv` 是旧名，`profile_compare.csv` / `profile_equivalence.csv` 已删除——新代码都不读。重新 tune |
| 想把配置表整个作废 | 删掉 `pyqcd.einsum.profile.csv`（或换 `PYQCD_EINSUM_PROFILE_DIR`） |

**`.cache` 不要提交**：默认目录 `./.cache`（`_profile_cache.DEFAULT_DIR`）里是主机名、
方程与形状；根 `.gitignore` 的 `.*` 规则已忽略它（`git check-ignore -v
.cache/pyqcd.einsum.profile.csv` 复核），`MANIFEST.in` 也不收录任何缓存目录。

想强制"这次完全按我的来，别读表"：`contract(..., profile=False, kernel_force="dedicated")`。
想确认进程里表的状态：`p.summary()`（只读快照；读表本身是纯字典查找）。

### 配置表的六条硬判据（2026-11-17 TV 重设计）

- **唯一写者是 `--tune`**：普通 `contract()` 只读；"跑多少次表都不变"是设计，不是 bug。
  未 tune 时读侧**警告一次**（"需要先 tune"）+ 打印**该配置的 tune 命令**，然后继续
  按未命中执行。
- **同键多行 = 硬错**：读侧不猜、不选最优，把**全部候选行**打到 stderr 并非零退出。
  tune 一次那个键（写前删同键所有行）是最省事的修法。
- **冲突 = 硬错**：只算**显式**的 `kernel_force` / `optimize` / `as_equivalence`；
  `--compare` 不参与，`optimize` 的 `auto` 等于没给（`kernel_force` 已无 `auto` 档，
  `None` 即 `generic`）。报错带键、表值、请求值
  与 `<file>:<行号>`；想绕开只能去掉那个参数或 `--profile false`。
- **`wall_ms` 是唯一计时列**：整段 `contract()` 的墙钟（含等价转换、缓存读写、
  未命中时的规划-匹配），**不含** compare 参考计时与收尾落盘；选优只比它，也没有第二列
  可以与它相加或混比。
- **`as_equivalence` 缺省 `auto`**：有 tune 写的等价标记行才改写；没有标记行、表缺失、
  缓存关闭、未命中都**不改写**。`true` / `false` 是显式强制，`force` 供 tune 刷新。
- **tune 期间缓存关闭**：`tune_session()` 让读一律未命中、写静默不写；`--tune-timeout`
  缺省 30 s，异常 / 超时 / MISMATCH **只打印不退出**；遍历
  `as_equivalence × optimize`（`{auto, none, compare 目标}`）共 10 组，写入选优行并
  **删同键其它所有行（含更快的历史行）**，防异环境污染。

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
- 额外 43 条分层 full-scene 路径：CPU 中位 `0.861x`、42/43 不慢于 generic；
  GPU 中位 `0.276x`、42/43 不慢于 generic。确认普遍加速，同时允许
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
| `[kernel: kupl_topology]` | 整条计划由 `qcd_kupl_topology_apply` 执行成功；同一内核有两条路线，是否走 **v3 编译期步表** 看 `LAST_INFO["path"]`：`kupl_topology` = 真 v3（`specialized=True`），`kupl_topology_v2` = 步表未命中、回落 v2 通用执行器（`specialized=False`） |
| `[kernel: cuda_topology]` | 同一入口的 CUDA 实现执行成功；CUDA 侧步表未命中直接返回 status=4，因此只要成功就是真命中 |
| `[kernel: kupl_nucleon_2pt]` / `[kernel: kupl_3pt]` 等 | 实际跑的是分派顺序更靠前的形态专用 kernel，不是拓扑入口 |
| `[plan-cache]` | 命中了形态但没有专用 kernel 成功执行：`out=` 非空、缺库/缺符号、操作数混合 dtype、计划超出 v2 单步 arity≤2 模型、或 kernel 内部报错；只复用计划，数值不变 |

**v3 与 v2 必须分开统计（2026-11-15 起强制，仅维护者归因需要）**：`topology_kernels.inc` **已于
2026-11-15 扩到全语料 8638 条**（`_topology_kernel_gen.MAX_KERNELS = 8638`，
基础表 2582 + 附录 6056）。在此之前的版本只含基础表，附录会静默走 v2 且仍报
`path=kupl_topology`，导致 benchmark 把两类实现混在一起取中位 —— 那段历史数字
（中位 4.89–8.18×）已作废。现在 `LAST_INFO` 带 `specialized`，`path` 区分
`kupl_topology`（真 v3）与 `kupl_topology_v2`（回落）；引用任何 topology 性能数字
前先按 route 分组（用户视角只有一个 `topology`，分组只是维护者归因）。

**步表以 gzip 入库**（`src/topology_kernels.inc.gz`，约 1.8 MB）：裸 `.inc` 是
13.8 MB，超过本库 10 MB 单文件限制，而 `topology_apply.cpp` 无条件 include 它。
`make.sh` 编译前自动解压；重新生成后用 `gzip -9 -c` 覆盖 `.gz`。

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
  **四类 dtype 完全平权、没有"主精度"**：每类都由同一把原生 2×2 实测自证
  （`_probe_dtype`，每进程每 dtype 一次、fail-closed 并缓存），见
  `TOPOLOGY_DTYPES` / `topology_dtypes()` / `topology_enabled_dtypes()`。
- 静态生成内核**四类 dtype 平权**（`complex128`/`complex64`/`float64`/`float32`），
  由 `TOPOLOGY_DTYPES` 声明、每类由原生 2×2 实测自证后才下发（探测容差分档：
  c128/f64 `1e-9`、c64/f32 `1e-4`）。
- 计划缓存已从 32 项 FIFO 改为 **真 LRU + 64 MiB 字节预算**
  （`_TOPOLOGY_PLAN_CACHE` 与 `_DEDICATED_PLAN_CACHE` 各自记账，实测 8 个场景
  记账偏差全为 0）；键是**调用方写的方程 + 各 shape + dtype**，整语料扫描仍会
  反复建计划，单遍扫描耗时不能当稳态。
- KUPL 拓扑入口走 **v3 步表**（真实逐轴长度 `label_sizes`，不要求等长超立方体）：
  生产 `distill` 档 `S1` 实测 153.04 ms（比 `opt_einsum:dp` 的 271.49 ms 快 1.8×）。
  **证明于 2582 条表时代**：`S1` 当时不在表内，该数字实为 `kupl_topology_v2`
  （v2 通用执行器）。扩表后附录方程已进表，重测应得 `kupl_topology`；引用旧数字
  时仍须标注它测的是 v2。
- **收缩序已改为按生产尺寸定型**（2026-11-15，`plan_steps(equation, sizes)` +
  `optimize` 固定 `auto`）：抽样 21 条语料，MAC 中位改善 222×、最差 42738×，
  更优/持平/更差 = 17/4/0。
- **复数已真正进入 v3**：`run_specialized` 的 dtype→`elem_kind` 覆盖四类，
  复数额外过 `cblas_api().complex_complete()` 闸门（缺 cgemm/zgemm 时 fail closed
  回落 v2，绝不降精度）。证据：`PYQCD_EINSUM_TOPOLOGY_TRACE=1` 下复数打印
  `topology-specialized: ... dtype=0/1`；契约
  `test_complex_dtypes_reach_the_v3_step_table` 钉住这一点。
- **v3 输入侧有 direct 直连**（`kDirectPlain`/`kDirectTrans` 的运行期等价物，
  含 `count_batch > 1` 与单侧直连）：原先只有输出侧恒等检测，输入侧无条件
  `gather_extents`，seq=3 为此多付 32×（49 ms vs dedicated 1.49 ms）。
- **topology 有 CPU pack-bound 守卫** `_topology_pack_bound_reason()`：按步表逐侧
  记账（只算真正 pack 的一侧、输出侧非恒等才计 unpack），pack ≥ 1 GiB 且
  FLOPs ≤ 1e9 即 fail closed 回落。仍被拒的例子 seq=646（绕过守卫直调原生 8.9 s
  vs 通用 0.39 s）。
- **真 v3 的性能口径**（只算 `path=kupl_topology`，13 条基础表 seq × 4 dtype，
  distill，热态 3 轮中位）：中位 2.71–3.03×、最差 4.29–4.72× vs dedicated；
  快于 dedicated 的格子 3/12（seq=9 达 0.045–0.195×，seq=1700/2000 为 0.26–0.79×）。
- **CUDA 侧收缩序守卫**（`topology_order_acceptable`）：表内序相对"最小并集乘加
  贪心参考序"劣化超 8× 即 fail closed。语料 2582 条中拒 1692 条；seq=510 守卫
  关闭时 5099.88 ms、开启后回落通用路径 13.7 ms。
- **callable / `PathOptimizer` 类型的 `optimize`**：计划缓存键需要一个稳定身份。
  2026-11-14 修掉了「两个不同 callable 共享同一 `plan_key`、第二个从未执行」的
  静默 bug（`_api._OptimizeIdentity` + `_plan_cache_optimize_key`）；契约见
  `_einsum_optimize_callable_contract`（8 项）。
- CUDA 的 `TopologyProgram` 是**单槽**缓存（按计划字节 `memcmp`），换拓扑即覆盖；
  CUDA 拓扑入口还要求 `stream == nullptr`。
- CUDA 描述符摘要必须用 blake2b-**128**（`_descriptor_words`）：`_specialize`
  的索引摘要只有 96 位，零填充后前 12 字节相同、后 4 字节不同 ⇒ key 恒查不到表，
  语料元数 7–16 的方程全部 status=4。契约
  `test_cuda_descriptor_reaches_the_step_table_for_high_arity` 钉住这条。

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

下表是 2026-11-12 对当前完整任务表的 full-shape 实跑，不涉及缩放维度或冒烟场景。
**唯一权威出处是 `docs/einsum_production_optimization_20261112.md:185-194`（§8）**，
本卡只留结论：`match()` 热轮转已是**亚微秒级/条**（首轮冷扫描为数百毫秒量级）、
dedicated 计划热扫描是**数十微秒级/条**，且整合语料后不再反复淘汰计划缓存。
具体数值、单位（`us` = µs）与测量环境都在该 §8 的表里，改数只改那一处。
复现时先 `source "$HOME/env.sh"`，再设 `PYQCD_EINSUM_NOTICE=0` 读取
`_wick_tasklist.TASKS`，记录耗时与 `_match_of_normalized.cache_info()`。

**性能数据门（2026-11-12，2026-11-14 依 `refer/sush/sush-einsum.py` 重写）**：
性能结论只接受生产尺寸上的实跑，且**必须写明口径**——蒸馏张量与"空间×本征模展平"
矩阵乘是两条互不替代的口径：

| 口径 | 场景档 | 尺度由谁决定 | 能支撑的结论 |
|---|---|---|---|
| `distill`（主） | `distill`(Nev=100)、`distill_curr`(Nev=400) | `Nev`(100/400)、`Mom`(1/3)、`Ns=4` | 蒸馏收缩的全部性能结论 |
| `spatial` | `scene`、`full` | `Nev*Nx`(3200²)、`Lt*Nx`(2048²) | 仅"空间×本征模展平后的矩阵乘" |

依据：参考脚本 `refer/sush/sush-einsum.py` 的 `Nev_src=100`(`:170`)、
`Nev_curr=400`(`:171`)、`Mom_src=1`(`:172`)、`Mom_curr=3/9`(`:173`，**可变量**，
生产里两者都在用)、`Ns=4`(`:200`)；
其张量是 peram `(4,4,nev[tq],nev[ta])`、`VVV (Mom,Nev,Nev,Nev)`、`VDV (Mom,Nev,Nev)`，
**没有空间轴**（`Nx`/`Lt`/`lattice` 全文匹配数为 0）。

`smoke`、`--scale`、缩小维度和其他代理尺寸只能用于正确性/路径验证，必须丢弃其中的
耗时与倍率，不能改写成生产性能。把 `scene`/`full` 的倍率当蒸馏结论（或反之）、
用 `_label_sizes` 从 `dims("distill")` 反推的形状当蒸馏生产形状（该函数表达不出
蒸馏轴），同样作废。生产规模装不下时，正确状态是 `unverified`，不是用较小场景替代。
闸门与跨口径判定：`python -m pyqcd.testing._production_perf_gate_contract`
（`classify_measurement()` + `claim_basis()`）。

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

### §4 精度策略：四类 dtype 完全平权（2026-11-14 重写）

`complex128` / `complex64` / `float64` / `float32` **同等对待**：不设主精度、不设
默认精度、不设“单精度恒可下发”的特权。每类的启用条件完全相同——原生 2×2 实测
（`_probe_dtype`，每进程每 dtype 一次）自证正确；没通过只回落**这一类**，不影响
其它三类，也绝不把数组换成另一类精度。

| 项 | 规则 |
|---|---|
| 契约 | `TOPOLOGY_DTYPES = (complex128, complex64, float64, float32)`，顺序与 `QCD_DTYPE_*` 同值，Python/C++ 直接互传 dtype id |
| 只读视图 | `topology_dtypes()` 返回契约（不探测）；`topology_enabled_dtypes()` 返回本进程实测通过子集 |
| 探测 | `_probe_dtype(dtype)`：2×2 `ij,jk->ik` 对比 numpy；**整段在 `try` 内**，任何异常都只返回 `False` |
| 缓存 | `_TOPOLOGY_DTYPE_SUPPORT[dtype]`，每进程每 dtype 一次；失败也缓存（不重探、不重抛） |
| 降级 | 只回落该 dtype 到通用路径；跨 dtype 转换一律禁止；混合 dtype 直接 `fail closed` |
| 拒绝原因 | 契约外 dtype → `outside the topology dtype contract`；契约内但探测没过 → `probe failed ... Rebuild the native library (<make 命令>)` |

**内核侧（2026-11-14 补齐）**：KUPL 的参数化步表执行器原先只实例化 `float`/`double`，
复数一律交回通用执行器——这是四类里唯一的不对称点，已消除：

| 改动 | 位置 |
|---|---|
| `ElementKind` 改为四值且与 `QCD_DTYPE_*` 同值（complex128=0 … float32=3） | `cpp/kupl/einsum/src/topology_gemm.h` |
| 交错 re/im 的 POD 元素 `ComplexElement<T>`（`float2`/`double2` 布局，与 PyQCU `LatticeComplex<T>` 同构），打包/转置仍是纯 memcpy | 同上 |
| CBLAS 增加 `cblas_cgemm`/`cblas_zgemm`（alpha/beta 为指针）；实数符号缺失依旧服务实数路径，复数符号缺失只让复数回落 | 同上 |
| `gemm_available(api, T*)` 四类各自判符号，取消“先看 sgemm+dgemm 齐不齐”的全局门 | 同上 |
| `run_specialized` 的 dtype 分派从 `if float/else if double` 改为四类直传 `dtype_id` | `cpp/kupl/einsum/src/topology_apply.cpp` |
| 未命中原因进 trace（`topology-step-miss: <reason>`），排错不必再猜 | 同上 |

**正确性门（2026-11-14，全语料 × 四 dtype × 两条路径）**：

```bash
python -m pyqcd.testing.einsum.verify_dtype_parity --backend kupl --dim 3
# complex128 dedicated hit=8638 bad=0 max_rel=2.9e-14 ；topology max_rel=2.0e-14
# complex64  dedicated bad=0 max_rel=1.6e-05；topology 2.0e-05
# float64    dedicated bad=0 max_rel=4.2e-13；topology 1.0e-13
# float32    dedicated bad=0 max_rel=5.0e-05；topology 8.6e-05
# PASS（判据：同 dtype 通用路径为参考；dtype/形状/有限性 + rel 容差）
```

**单精度残余散射（2026-11-14 实测，已知限制）**：`float32` + `topology` + 16 算子
链（seq=1973）相对 float64 真值的偏差为 **1.25e-4**，而同路径 `generic` 是 4.0e-5，
`dedicated` 是 **3.0e-5**——即步表路径在这一格比通用路径多约 3× 的舍入散射（仍是 1e-4 量级，不是
错数）：属于单精度链式求和顺序差异，不是 dtype 或内核错误。
判据因此写成"双精度 ≤1e-12、单精度不得劣于同 dtype 通用路径"；该格会如实报出来，
需要更严精度时应回落到 `dedicated`（或直接用 `specialize=False`）。
CUDA 同格相反：`dedicated32 vs float64 真值 = 5.3e-6 < generic32 的 3.9e-5`。

判据是"特化复现**同 dtype**通用路径"，不是跨精度比较：跨精度会把该 dtype 固有的
舍入当成内核损失。**这一步抓到过一个真 bug**——`unpack_step` 的"二维互换"快捷
路径把分块指数的两侧写反了（`left = dim^split` 应为 `dim^(rank-split)`），只有
`split == rank/2` 时二者才相等，所以 120 条抽样与 2582 条基础表契约都没抓到，
全语料对拍在 16 条 4 输出轴拓扑上给出 `rel≈1`（值对但被错误转置）。抽样覆盖
不能替代全语料对拍。

**性能（2026-11-14，KUPL，生产场景 `scene` = Nx=64/Nev=50/Lt=40/Mom=5，2–3 轮中位）**：

| seq（元数） | dtype | generic | dedicated | topology |
|---|---|---:|---:|---:|
| 129（14） | complex128 | **131.1 ms** | 1.45× | 1.21× |
| 129（14） | complex64 | **57.3 ms** | 2.15× | 2.12× |
| 129（14） | float64 | **45.3 ms** | 2.58× | 1.97× |
| 129（14） | float32 | **38.2 ms** | 1.97× | 2.33× |
| 1623（13） | complex128 | 7.63 ms | 1.95× | 1.18× |
| 1623（13） | complex64 | 3.87 ms | **0.65×** | 1.99× |
| 1623（13） | float64 | 3.20 ms | **0.72×** | 2.36× |
| 1623（13） | float32 | 2.15 ms | 1.06× | 11.80× |

读法（三条都重要）：

1. **损失与 dtype 无关，与方程有关**：seq=129 上四类 dtype 全部亏（1.2–2.6×），
   seq=1623 上复杂/双精度赚（0.65–0.72×）。路径选择必须按方程代价门，不能按
   dtype 一刀切，也不能因为 complex 命中率高就认为 complex 更快。
2. **pack 开销是固定字节成本，所以对最快的 dtype 相对最贵**：seq=129 的亏损比
   在 float32（1.97×/2.33×）大于 complex128（1.45×/1.21×），因为实数 GEMM
   更快、打包占比更高。float32 + topology 在 seq=1623 的 11.80× 就是这个效应的
   极端值——不是 dtype 缺陷，是该拓扑的步表在小 GEMM 太多时没有优势。
3. **本机生产数字是 provisional**：单机共享（18 核，其它作业在跑）、2–3 轮中位，
   seq=1623 complex128 在两次运行间从 0.30× 变到 1.95×。要下"生产性能结论"
   必须按性能数据门重测（隔离机器、≥10 次中位、同线程/同 dtype 对照）。

**CUDA 生产档（2026-11-14，V100-SXM2-32GB，scene 档，3 轮中位，隔离 mamba 环境）**：

| seq | complex128 | complex64 | float64 | float32 |
|---|---:|---:|---:|---:|
| 129 | 0.78× | 0.60× | 0.48× | 0.43× |
| 1623 | 0.20× | 0.24× | 0.24× | 0.19× |
| 1863 | 0.20× | 0.19× | 0.18× | 0.19× |
| 3461 | 0.67× | 0.57× | 0.54× | 0.51× |

16/16 单元格 `dedicated ≤ generic`（1.3–5.5×），四类同一量级；CUDA `topology` 在这四条
路径上全部 MISS（描述符+`topology_steps.h` 覆盖面窄），属已知限制而非 dtype 缺陷。
正确性同机复验：`verify_dtype_parity --backend cuda --dim 3 --limit 600` →
dedicated 2400/2400 命中、0 失配（c128 2.8e-15 / c64 8.2e-7 / f64 1.6e-14 / f32 8.3e-6）。

CUDA 复测环境（不改基础环境）：`mamba create -n pyqcd-einsum-cu124 python=3.12 numpy
scipy h5py opt_einsum cupy-core cuda-version=12.4 openblas` + `cuda-nvrtc/cuda-cudart
(含 -dev)/libcublas`；`CUDA_VISIBLE_DEVICES=0` 锁 V100，`mamba run -n pyqcd-einsum-cu124`
执行。基础环境里 pip 的 `nvidia-cuda-nvrtc 13.2` 会被 cupy dlopen 到（CUDA 13 已删
sm_70），这是"cupy 坏掉"的真因。

复跑命令：

```bash
python -m pyqcd.testing.einsum.bench_dtype_matrix --backend kupl --scene scene \
    --seqs 129,1623 --rounds 3 --mem-budget-gib 50        # 生产档（唯一有效口径）
python -m pyqcd.testing.einsum.bench_dtype_matrix --backend kupl --dims 2,4,8
    # 小尺寸只作路径/正确性证据，倍率一律不作为性能结论
```

**GPU 环境坑（2026-11-14 实测）**：cupy 14.2 会加载 pip 装进来的
`nvidia-cuda-nvrtc 13.2`，而 toolkit/driver 是 12.4；CUDA 13 已删掉 sm_70/sm_60，
于是任何 cupy kernel 都报 `nvrtc: error: invalid value for --gpu-architecture`，
GPU 契约测试全部失败（症状像“内核坏了”，其实是环境）。判据与修法：

```bash
python -c "from cupy_backends.cuda.libs import nvrtc; print(nvrtc.getVersion())"
# 输出 (13, 2) 而 nvcc 是 12.4 → 版本错配
LD_PRELOAD=$CONDA_PREFIX/lib/libnvrtc.so.12 python -c "import cupy; print(cupy.zeros(4).sum())"
# 或按 mamba 口径把 cupy 与 cuda-version 对齐：
#   mamba install -c conda-forge "cupy-core=13.*" "cuda-version=12.4"
```

### §4b 形状特化路径：语料之外的第三条链（2026-11-14 新增）

特化链现在有三条，职责不同，别混：

| 链 | 来源 | 适用 | 命中判据 | 失败行为 |
|---|---|---|---|---|
| 语料拓扑 + 原生步表 | `_topology_registry`（8638 条 wick） | 语料内方程 | 规范签名在索引里 | 回落计划缓存 / 通用 |
| `_dedicated_kernels` | 任意方程 | 固定归并序、逐对执行（未接线，独立引擎） | 调用方显式 `kernel_for(eq)` | 需自行处理 |
| **`_shape_paths`** | **手写算法** | **结构特殊、可一次算完的方程** | 规范签名精确匹配 | `try_execute` 返回 `None`，绝不抛异常 |

已注册：`x,ax,bx,cx->abc`（`shape:abc-sandwich`，内核标签 `shape_abc_sandwich`）

    out[a,b,c] = Σ_x w[x]·A[a,x]·B[b,x]·C[c,x]
    Q[a·n+b, x] = A[a,x]·B[b,x]         # n³ 次乘，与输出同阶
    W[c, x]     = C[c,x]·w[x]
    out[a,b,c]  = reshape(Q @ Wᵀ)       # M=n², K=n, N=n 的单个 GEMM

规划器拆三步（`x,ax->xa`、`bx,cx->bxc`、`xa,bxc->abc`），这里压成一次 BLAS
调用；中间体与输出同为 n³，不引入更高阶中继。

**实测（numpy/float64，中位 7 轮，`contract(specialize=False)` 为基线）**：

| n | 8 | 32 | 64 | 100 | 128 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| 形状特化/通用 | **0.19×** | 0.66× | 1.05×（单次离群） | 0.61× | 0.92× | **0.59×** |
| 相对 `np.einsum(dp)` | 0.26× | 0.70× | 0.93× | 0.84× | 0.99× | 0.86× |

数值偏差 ≤ 4e-16（与 numpy 同 BLAS 口径）。n=64 那一格是唯一边界样本（换轮次
在 0.73–1.05× 之间波动），不要据此宣称"全尺寸都快"。

**接线点**：`_api.contract` 在元素级委派之后、规划之前调用
`_shape_paths.try_execute`；只在 `out is None`、无 kwargs、`specialize is not False`
时尝试。**`out=` 与 `specialize=False` 一定绕过**（契约里有用例守着）。

**新增一条形状路径**：

1. 在 `_shape_paths.py` 里写一个 `run(operands) -> ndarray | None`，前置条件
   （秩、轴长一致、dtype 一致、C 连续、中间体预算）不满足就返回 `None`；
2. `_register("<代表方程>", "<name>", "<note>", ("<kernel_label>",), run)`——
   键是 `_specialize.canonical_signature`，所以标签改名自动命中；
3. 契约：`python -B -m pyqcd.testing._shape_paths_contract`（8 项：签名匹配、
   查询 HIT、四 dtype 数值与保真、真走内核、形状不符回落、`out=`/
   `specialize=False` 绕过、内存预算与混合 dtype 回落、内核抛异常被吞）；
4. 查询侧无需改动：`query_specialization(eq)` / `python -m pyqcd.einsum.verify_specialized`
   已先查 `_shape_paths` 再查语料，HIT 时给 `name/kind/kernels/note`，不伪造
   `corpus` 字段。

**抽样复核（更多特化路径，2026-11-14）**：`bench_specialized --sample 12 --seed 5
--scene scene`（cpu, double）12 条抽出 10 条可用，中位 **0.36×**（pyqcd 更快），
逐条 0.02×–1.00×，全部命中 `kupl_dedicated`；1 条 `REF-FAIL`（`opt_einsum(dp)` 要
149 GiB 中间体，是参考实现的病态）。复跑同一命令即可得到同 seed 同集合。

### §4c 方案最优性：推导结论与改进候选（2026-11-14，理论推导）

问题：当前"三条特化链 + 五道分派门"是不是最佳实现？结论是 **部分是**——分派顺序、
fail-open 契约、dim≤6 特化收益（0.19–0.53×）与"低元数大张量主动委派 BLAS"都站得住；
但有三处**可证的非最优**，都在内核外围，不是"特化"本身的上限。

**成本模型**（步 i = (B,M,N,K)，元素 s 字节）：FLOPs $\sum 2BMNK$；pack 字节
$2s\sum(BMK+BKN+BMN)$；中间体 $s\sum BMN$；BLAS 调用 $\sum B_i$（当前是串行循环，
不是 batched）。单步算术强度 $AI = MNK/[s(MK+KN+MN)]$，本机 F/B ≈ 2 flop/byte：
**AI≲4 时 pack 是主项**（dim=8 实测占 13–33%，中位 20.8%），**AI≳8 时只该优化 GEMM
效率与计划 FLOPs**（生产 16 算子 pack 约 4.57 GiB ≈ 占 10.3 s 的 3%，为推算值）。

**"一次单 GEMM vs 三步链"的判据**（推导，非实测）：三步链最后一步已是 $2n^4$，
FLOPs 相同，差异只在流量与派发次数：

$$T_{chain}-T_{single} \approx \frac{(k-1)2Is}{B_{eff}}-(k-1)(c_{blas}+c_{dispatch})$$

即 **单 GEMM 胜 ⟺ $n^3 \gtrsim (c_{blas}+c_{dispatch})B_{eff}/s$**，取 $c\approx20\,\mu s$
得 $n^*\approx30$–$55$（f64/DRAM≈43、f64/L3≈49、c128/DRAM≈34、f32/DRAM≈54）。收益上限
$1-3/5=0.4$，与 §4b 实测 0.59–0.61 一致；n=64 落在 $n^*$ 附近、实测在 0.73–1.05× 跳动，
**正好说明该判据的方向对、但不能用它定符号**。

**三处非最优 + 改进排序**（收益为推导估计，需按性能数据门复测）：

| # | 候选 | 预期收益 | 代价 | 风险 |
|---|---|---|---:|---|
| 1 | 步表 batch 折进 M（删 `for batch: cblas`，packed 布局 `lda=K, ldc=N` 本就允许零拷贝折平，`topology_gemm.h` 的 batch 循环） | 1.2–2×（生产最重步 64 次串行 zgemm → 一次 (409600,400,256)） | C++ 约 20 行 | 低（累加顺序不变） |
| 2 | 计划判据从"只看 FLOPs"推广到 (FLOPs, arena, pack) 三项门，四 dtype 各标定 | 回收 seq=129 类亏损 1.18–2.24× | 中（阈值标定表） | 中（易过拟合单机） |
| 3 | 派发瘦身（缓存 pointers、跳过重复校验、复用 match） | dim≤8 端到端 1.3–2×（派发 70 µs vs 内核 63 µs，dim=4 二者同量级） | 低 | 低 |
| 4 | `_TOPOLOGY_PLAN_CACHE` 32 → 16384 / 64 MiB（对齐 dedicated） | 冷 2.7 ms → 0.08 ms，语料热扫描进一步降 | 极低 | 无 |
| 5 | 消除 pack（直接 stride/batched） | **上界仅 1.05–1.26×**：95.3% 的 pack 是真多轴置换，用不上 TransA/B | 高（步表格式变更） | 中 |
| 6 | JIT loop nest（TACO/TVM/ExTensor 式） | 上界 = pack+中间体流量 ⇒ dim≤8 ≤1.26×、全场景 ≤1.03× | 很高 | 高 |
| 7 | cuBLAS batched / cuTENSOR | docs §9 已测无稳定收益；GPU 通用路径已交回 cuBLAS | — | 不建议重做 |
| 8 | TF32/混合精度 | GPU fp32 潜在 4–8× | 中 | **高**：破坏"同 dtype 复现"判据（float32 实测 5e-5 已在 1e-4 边缘），仓库零证据 |

**不迁移到生成式/编译器路线的理由**（对照结论）：③ 的独占收益只在"无 BLAS 结构的逐
元素形态"，而这一块已被门 3 整体委派给 backend（Hadamard 4000² 0.33×→1.00×）；④ 与
自包含 ABI、`nm -D` 白名单、四 dtype 平权与"单精度不得劣于通用"判据冲突。另有三条
结构空洞值得记录：**(a)** 无 loop-nest 生成器，加一条特化只能人工写 `run()`；
**(b)** 参数化维度只有 `dim`（`topology_gemm.h` 明写"轴长恒为 dim"），异形生产轴
（Nx=64/Nev=50/Lt=40）进不了步表——CUDA `topology` 在四条生产路径全 MISS 即此；
**(c)** 一元/重复指标步两条原生链都不支持（只有未接线的 `_dedicated_kernels` 支持）；
**(d)** 计划缓存键是**调用方写的方程**，无跨方程复用（原 32 项 FIFO 的问题记录见
`docs/handover_einsum_20261112.md:76-77`；容量与逐出策略已于 2026-11-14 改为真 LRU + 64 MiB，
见本节 `#4` 行，**不要再按 FIFO 引用**）。

**2026-11-14 攻坚结果（按测量否决/采纳）**：

| 候选 | 结论 | 证据 |
|---|---|---|
| #1 步表 batch 折进 M / `?gemm_batch` | **否决，已回退** | batch 维**不能**折进 M（packed A/B 都随 batch 变化，折平会混不同 batch 的 B）。改试 `cblas_{s,d,c,z}gemm_batch`：本机 A/B 中性到更慢（seq=500/dim=8 8.81 对 8.18 ms；seq=2584/dim=4 0.20 对 0.17 ms，9 轮中位），代码已全部回退，不留默认关闭的分支 |
| #4 拓扑计划缓存 32 项 FIFO → 真 LRU + 64 MiB | **采纳** | 全语料 8638 条：冷 1446 µs/条，**热 14.56 µs/条**（旧 32 项下等价全 miss），缓存 8638 项 / 14.5 MiB；`_kernel_kupl` 与 `_kernel_cuda` 两侧同口径 |
| #3 派发瘦身 | **未做**（收益低于噪声） | 端到端 `contract` 0.130 ms 对 `try_topology_apply` 0.118 ms ⇒ 派发约 **12.5 µs**，远小于 dim=4 内核 63 µs 的同量级估计；按"测量优先"不做无证据的瘦身 |

**修掉的两个真 bug（2026-11-14）**：

1. **`out=` dtype 不符静默 cast**（`_execute.py`）：`out[...] = new_view` 靠 numpy
   赋值语义做隐式转换，complex→float **丢虚部**（只发一条 ComplexWarning），
   float64→float32 完全静默。现在 dtype 不符直接 `ValueError`（`out dtype must
   match the contraction result: ...`），失败路径不写坏调用方缓冲区。契约：
   `_contract_edge_cases_contract.test_out_dtype_mismatch_fails_closed`。
2. **cuBLAS 双副本污染**（CUDA）：conda 的编译器 wrapper 无条件给
   `libpyqcdeinsum_cuda.so` 注入绝对 `RUNPATH`，进程里出现两份 `libcublas`
   （构建机 12.4 与宿主 cupy 的 12.9）。症状是**调过拓扑/步表入口之后**，cupy 的
   complex64 GEMM 恒报 `CUBLAS_STATUS_NOT_SUPPORTED`（c128/f32/f64 不受影响），
   极易误判成"内核算错"。修法：`cuda/_program._bind_host_cublas()` 在 dlopen
   归档**之前**先让宿主绑定自己的 libcublas，使 `libcublas.so.12` 这个 soname 已有
   映射。验证：`/proc/self/maps` 里 `libcublas.so` 只剩 1 份，topology 仍命中，
   c64 GEMM 正常。**注意不要用"清空 RUNPATH"来修**：实测清空后 dlopen 直接失败、
   topology 整体 miss；`make-nv.sh` 只保留 `PYQCD_EINSUM_STRIP_RPATH=1` 这个手动开关。

**四 dtype 全语料对真值对拍（2026-11-14，V100 + 隔离环境）**：

```bash
CUDA_VISIBLE_DEVICES=0 mamba run -n pyqcd-einsum-cu124 \
  python -m pyqcd.testing.einsum.verify_dtype_parity --backend cuda --dim 3
# complex128 dedicated hit=8638 bad=0 max_rel=3.3e-14 ; topology 4/8638 (2 条命中仍正确)
# complex64  dedicated hit=8638 bad=0 max_rel=2.8e-06
# float64    dedicated hit=8638 bad=0 max_rel=3.8e-13
# float32    dedicated hit=8638 bad=0 max_rel=3.4e-05
# PASS
```

CUDA 侧 `topology` 只覆盖 4/8638（描述符 + `topology_steps.h` 覆盖面窄），是已知限制，
**不是** dtype 问题；`dedicated` 四类全量命中且对真值全部在容差内。

**测试项必须覆盖四类 dtype（2026-11-14）**：`verify_native`（49 项：五类形态 × 四
dtype + 缓存值 + `out=` + `contract_expression` + 重复轴回落 + mixed-backends）、
`verify_plan_gate`（11 项：CPU/CUDA 资源门与负缓存 dtype 维度）、`verify_topology`
（四 dtype 各 30 条，偏差 1.7e-15 / 5.7e-7 / 1.8e-15 / 9.8e-6）都已参数化到四类。
扩覆盖时抓到两处**测试口径**缺陷，值得记住：

1. 相对误差不能逐元素除以 `|expected|`：sandwich 这类输出含近零元素，会把正确的
   float32 结果放大成 1.95e-4 假失败。统一按**输出最大幅值**归一。
2. **不要假设资源门对四类 dtype 单调**：门看 arena/pack **字节**，而规划器在不同
   dtype 上会选出不同收缩顺序 —— seq=6585 的 complex64 arena（9.41 GB）反而大于
   complex128（6.80 GB）。契约写成"每类各自标定 + 分歧格必须由已声明的三条门解释"。

**生产档 10 轮（性能数据门，2026-11-14，V100 scene）**：16/16 单元格
`dedicated ≤ generic`；按 seq 129 0.41–0.92×、1623 0.20–0.25×、1863 0.15–0.20×、
3461 0.61–0.80×；与 3 轮结果逐格差 ≤0.1x，四类同量级（无主精度）。

**证据缺口**：`refer/git-rep/opt_einsum/` 在本 checkout 是空目录，对手实现只能由仓内
fork 反推；Kubernetes/节点 F_eff（100–200 GFLOP/s）与"生产 pack 占 3%"均为推算；
无 PMU/cache-miss 数据；dtype 分别选路的收益只来自 seq=129/1623 两个样本。

### §4d 测试组：三个与内核无关的"结构契约"（2026-11-14 新增）

内核测试（步表/契约/全语料对拍）只能证明"数字对"，抓不到 Semantics / 分派 / 缓存
这三类"结果对但结构错"的缺陷。新增三组，全部覆盖四类 dtype：

| 组 | 模块 | 覆盖 | 项数 |
|---|---|---|---:|
| 语义 | `_einsum_semantics_contract` | 结合律、线性性/分配律、共轭与转置律、闭式真值（迹/置换/外积/单位阵）、语料随机方程、缓存不返回旧值、逐位确定性、负缓存可恢复、8 线程并发、0 长度轴与标量、非法方程报错 | 11 |
| 分派 | `_einsum_dispatch_contract` | 元素级门、形状特化优先于规划器、`out=`/`specialize=False` 绕行、batch 判据不按 ndim、原生计划门、混合 dtype 提升 | 7 |
| 缓存/内存 | `_einsum_cache_contract` | LRU 语义（命中刷新顺序）、字节预算独立生效、dedicated 缓存键含 dtype+optimizer、输入不可变、输出不别名、反复失败后块池存活、`out=` 缓冲 | 7 |

统一套件从 118 项 → **124 passed / 1 skipped / 0 failed**。

**写这三组时踩到两个"测试自身"的坑，都值得记住**：

1. **并发段不能 `redirect_stdout`**：8 个线程同时替换同一个 `sys.stdout` 并写同一个
   `StringIO`，会让进程**静默终止**（退出码 0、无 traceback，只有 `faulthandler`
   能看到停在首次并发调用内部）。表现极像"原生内核在并发下崩溃"，实际与内核无关。
   并发用例改为用 `PYQCD_EINSUM_NOTICE=0` 关提示，不做重定向。
2. **契约不能依赖跨组状态**：`_einsum_cache_contract.test_out_caller_buffer_survives_failed_call`
   早先假设"失败即回落且仍要写对 out"，单跑绿、套件里红（前一组留下的 patch/负缓存
   改变走向）。契约要验的是单次调用的确定性行为，不能把别的组的状态算进来。
   同类教训：`test_concurrent_*` 必须先做一次串行预热——首次 `contract` 的懒加载/
   ctypes 绑定/线程档初始化不是为并发设计的，验的是"稳态并发安全"而非"加载期可重入"。

**注册方式**：三组都必须在 `pyqcd/testing/conftest.py` 的显式 import 列表里加一行，
否则 `python -m pyqcd.testing` **不会收集**（该套件是显式注册制，不是 `test_*` 自动
发现）。加完用 `grep -c "def test_" pyqcd/testing/__init__.py` 与套件汇总行对照。

### §4e 各 dtype 的普遍加速效果（2026-11-14，scene 档，10 轮中位，含绝对量）

**先说结论**：加速效果**不随 dtype 变化，随方程变化**；CPU 与 GPU 结论相反。

#### KUPL（CPU，18 核共享机）—— **已去掉兜底，topology 列只表示真实步表覆盖**

| dtype | seq | 元数 | 出秩 | generic | dedicated | ded/gen | GFLOP/s | pack MiB | arena MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| complex128 | 129 | 14 | 4 | **124.08 ms** | 180.89 ms | 1.46× | 7.75 | 524.6 | 275.2 |
| complex64 | 129 | 14 | 4 | **61.80 ms** | 113.79 ms | 1.84× | 15.56 | 262.3 | 137.6 |
| float64 | 129 | 14 | 4 | **39.88 ms** | 106.50 ms | 2.67× | 24.11 | 262.3 | 137.6 |
| float32 | 129 | 14 | 4 | **18.02 ms** | 67.81 ms | 3.76× | 53.38 | 131.2 | 68.8 |
| complex128 | 1623 | 13 | 2 | 6.18 ms | **5.00 ms** | 0.81× | 17.62 | 29.2 | 19.3 |
| complex64 | 1623 | 13 | 2 | 3.71 ms | **3.01 ms** | 0.81× | 29.35 | 14.6 | 9.6 |
| float64 | 1623 | 13 | 2 | 3.10 ms | **2.38 ms** | 0.77× | 35.16 | 14.6 | 9.6 |
| float32 | 1623 | 13 | 2 | 1.90 ms | **1.31 ms** | 0.69× | 57.46 | 7.3 | 4.8 |
| complex128 | 1863 | 16 | 2 | 6.26 ms | **5.99 ms** | 0.96× | 17.50 | 29.3 | 20.5 |
| complex64 | 1863 | 16 | 2 | 3.81 ms | 4.02 ms | 1.06× | 28.75 | 14.6 | 10.3 |
| float64 | 1863 | 16 | 2 | 3.53 ms | **3.24 ms** | 0.92× | 31.07 | 14.6 | 10.3 |
| float32 | 1863 | 16 | 2 | 2.29 ms | **1.86 ms** | 0.81× | 47.78 | 7.3 | 5.1 |
| complex128 | 3461 | 16 | 4 | **122.36 ms** | 145.67 ms | 1.19× | 18.63 | 570.0 | 536.0 |
| complex64 | 3461 | 16 | 4 | **58.16 ms** | 83.32 ms | 1.43× | 39.19 | 285.0 | 268.0 |
| float64 | 3461 | 16 | 4 | **47.15 ms** | 68.06 ms | 1.44× | 48.34 | 285.0 | 268.0 |
| float32 | 3461 | 16 | 4 | **24.03 ms** | 39.74 ms | 1.65× | 94.84 | 142.5 | 134.0 |

`topology` 列在 scene 档**全部 MISS**：步表要求等长超立方体（`uniform_dimension`），
scene 是异形轴；去掉兜底后它会诚实地返回 status=4，而不是偷偷跑通用执行器冒充命中。

#### 一句话结论

**GPU 上四类 dtype 普遍受益（1.4–3.7×，单精度受益略大）；CPU 上四类普遍不受益
（0.72–1.67× 即略慢，仅轻路径赚）。加速取决于"对方付费的是启动开销还是算术"，
与 dtype 无关。** 因此 §4c 的改进 #2（计划判据按 dtype 各自标定 + 方程代价门）
比继续优化单一 dtype 更值得做。

### §4f KUPL 拓扑入口已去掉兜底（2026-11-14 破坏性变更）

**改了什么**：`cpp/kupl/einsum/src/topology_apply.cpp::apply_topology` 在
`run_specialized` 返回非 OK 时**不再**调用 `execute_parsed_contract` 兜底，而是直接
返回 `QCD_TOPOLOGY_UNSUPPORTED`（status 4）。

**为什么**：旧行为让"命中拓扑"与"走了通用执行器"在外部完全无法区分——
全语料统计会得到 8638/8638 的假命中，CPU 的 `topology` 与 `generic` 也就几乎同速
（0.98–1.02×），把"步表只覆盖等长超立方体"这个真实缺口彻底掩盖了。CUDA 侧的
descriptor 里根本没有步骤（只有 `input_labels`/`output_labels`/`sizes`/`key`），
查不到表只能返回 UNSUPPORTED，于是**同一列在两侧含义相反**。

**现在两侧同义**：查不到步表 = status 4，由 Python 侧回落 `dedicated` 或通用路径。
`dedicated` 不要求等长超立方体、能力严格更强，且分派顺序本来就更靠前，所以
**不丢任何可执行路径**。

**连带必须改的两处**（否则会死锁）：

1. **dtype 探测方程必须来自语料**：`_PROBE_EQUATION` 从 `ij,jk->ik`（语料外）改成
   `cadb,menf,ac,em,bd,fn->`（语料第一条，arity=6，`dim=2`）。旧方程靠兜底才"探测
   通过"；去兜底后它恒 status 4 → 探测判负 → dtype 永不启用 → 连语料方程也被门控
   拦住，形成死锁。
2. **探测容差按精度分档**：`_PROBE_TOLERANCE = {complex128: 1e-9, float64: 1e-9,
   complex64: 1e-4, float32: 1e-4}`。旧的统一 `1e-9` 会把**正确**的 complex64 判成
   不支持（实测 rel=3.6e-7，是单精度在 6 算子链上的正常舍入）。

**契约与用例的连带约束**：凡是验"真的下发成功"的用例，方程必须来自
**基础表前 2582 条**（`_topology_kernel_gen.MAX_KERNELS = 2582`，这是步表的真实
边界；附录 2583+ 一律 status 4）。`_topology_kernel_contract` 里的
`PROBE_EQUATION`/`probe_operands()` 与 `_einsum_semantics_contract` 的冷启动/负缓存
用例都已按此改写；`ij,jk->ik` 只能用来验"应当回落"。

**真实覆盖率（去兜底后，dim=2/3 等长超立方体）**：基础表抽样 44/44 命中、0 失配；
跨附录抽样 26/88（miss 全部来自 2583+ 的无步表条目）。scene 档（异形轴）全 MISS。
回归：**124 passed / 1 skipped / 0 failed**。

### §4g opt_einsum 对照测试组（2026-11-14 新增）

`_opt_einsum_parity_contract`（9 项），把 `pyqcd.einsum` 的"兼容替代 opt_einsum"
这句话拆成三层可验的断言：

| 层 | 验什么 | 抓什么缺陷 | 项数 |
|---|---|---|---|
| 数值 | 11 种形态（matmul/batch/chain/implicit/ellipsis/hadamard/trace/outer/inner/reduce/four-way）× 3 种 `optimize` × **四类 dtype** | 实现语义偏差 | 1 |
| 专用路径 | 语料拓扑（seq 2584/2588/4305）+ 形状特化 `x,ax,bx,cx->abc` 也对照 oracle | 专用内核自洽但与 oracle 不一致 | 1 |
| 路径 | `opt_cost`、`largest_intermediate`、`contraction_list` 步数逐项相等（greedy/optimal + 语料抽样 13 条） | **规划器退化——只改耗时、数值测不到** | 2 |
| 行为 | `contract_expression` 复用、`shared_intermediates` 命中、非法方程两边都报错 | API 兼容性破裂 | 3 |
| 性能 | 交替计时取中位，`ratio ≥ 8` 视为数量级退化（**不设"必须更快"**） | 路径退化成重规划/逐元素 | 1 |
| 溯源 | 打印 oracle 来源（仓库内 submodule vs 站点包） | 用错版本的基准做差分 | 1 |

**oracle 来源必须可见**：优先 `refer/git-rep/opt_einsum`（`_oracle.oracle()`），该
submodule 在本 checkout 为空，实际回落到站点包 **opt_einsum 3.4.0**，输出里会打印
`oracle=installed:3.4.0`。站点包与参考仓库版本不同会让差分失去意义，所以这一条
本身就是一个断言。

**写这组时踩到的测量坑（值得记住）**：第一版只给 pyqcd 套了 `_quiet()`（内部是
`redirect_stdout` + `patch.dict`），oracle 侧裸调。结果每调用约 **0.7 ms** 全是脚手架
开销，算成 pyqcd 的耗时后倍率虚高到 **13×**，触发退化门误报。修正为"计时循环外
一次性设 `PYQCD_EINSUM_NOTICE=0`，循环内两边都不套壳"后，真实比值是 0.93–4.22×
（小尺寸固定开销为主）：

```
matmul 2.20x | batch 4.22x | chain 1.06x | implicit 1.97x
ellipsis 3.52x | hadamard 0.94x | four-way 0.93x
```

**这些毫秒数不是性能结论**：它们是 dim≤4 的小尺寸，按性能数据门只能用于"没有
数量级退化"这一条判断；生产口径见 §4e（scene 档、≥10 轮中位）。

### §4h CPU 步表选路：allowlist 的真实口径与一个真 bug（2026-11-14）

**修掉一个真 bug**：`_dedicated_allowlist.allow()` / `seqs()` 一调用就
`NameError: name 'ALLOW' is not defined`。原因是 `ALLOW` 只在模块 `__getattr__`
里对**外部** `module.ALLOW` 惰性注入，而**模块内的全局查找不触发 `__getattr__`**。
白名单不参与数值计算，所以这个缺陷可以一直躺到有人真的用它选路。修法：内部统一走
`_resolved()`（`globals().get("ALLOW")` → 首次解析并缓存），`__getattr__` 也复用它。
回归：`test_allowlist_is_callable`（含 `seqs()` 自洽 + 不存在的 seq 必须 False）。

**allowlist 的口径边界（不要误用）**：它按 `dim ∈ {4,6,8}` 的**等长超立方体** +
`complex128` 标定（覆盖 8055/7806/4932 条）。生产 scene 是异形轴，此时 seq=129
（全亏 1.53–3.62×）与 seq=1623（赚 0.72–0.88×）在 allowlist 里**都是 True**，
即它对本场景**没有分辨力**。任何"用 allowlist 给生产选路"的改动都会失效。

**代价门也分辨不了"结构性不适用"**（实测标定，2026-11-14）：

| 证据 | 数值 | 结论 |
|---|---|---|
| seq=129 vs 1623 的 `pack/flops` | 0.572 vs 0.281（只差 2×） | 比值当不了判据 |
| 两者 arena | 275 MiB vs 19 MiB（差 14×，不到数量级） | 绝对量也当不了判据 |
| seq=129 沿 `scale 8→32` 的 ded/gen | **1.68 / 1.96 / 2.00 / 2.30 / 2.12×** | **全程都亏，没有规模拐点** |

所以「再加一个尺寸门槛」修不了它：seq=129/3461 这类 **4 输出轴 + arity ≥ 14** 的
拓扑对 CPU 步表是**结构性不适用**（pack 是纯字节搬运，而通用路径已是大 BLAS）。
候选判据是"输出秩 ≥ 4 且 arity ≥ 14"，但样本只有 2 亏 + 2 赚，**不足以据此改运行时
选路**——这一点也写成了断言（`test_losing_paths_share_the_wide_output_shape`），
避免有人拿 4 个样本就上生产门限。

契约：`python -B -m pyqcd.testing._dedicated_routing_contract`（5 项）。

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
| 计划缓存 | 键是**调用方写的方程 + 各 shape + dtype**（不是拓扑签名）；全语料扫描会反复建计划，单遍扫描耗时不能当稳态。**容量/逐出**已在 2026-11-14 改为真 LRU + 64 MiB（`_TOPOLOGY_CACHE_LIMIT = 16_384`、`_TOPOLOGY_CACHE_BUDGET = 64 << 20`，`_kernel_kupl.py:210-213`）；"32 项 FIFO"是 `docs/handover_einsum_20261112.md:76-77` 记录的历史状态，已失效 |
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
| 命中日志 | `[pyqcd.einsum] 专用路径: topology:xxxxxxxx [match: topology] [kernel: kupl_dedicated]`（`[match: …]` 是命中的形态族，`[kernel: …]` 才是实际执行的路线） | — |

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

### [T2] 收敛重构的经验（2026-11-18）

全文见 `docs/einsum_framework_convergence_20261118.md`，时间线见
`logs/einsum-framework-convergence-20261118.txt`。动手前先看四条：

1. **判死**：入口藏在 `PYQCD_*` 门后仍是活代码（`e4_fused`/`sandwich` 实测可达）；
   provider 符号（`qcd_cuda_permute` 等）在 Python 侧 grep 为 0 是常态，不判死。
2. **删 C++**：证据是 `git archive <HEAD>` 建树 + 只换 `.so` 的隔离 A/B（144/1/0 vs
   144/1/0）；单跑当前工作树会被并行 Python 改动带偏。
3. **删符号/删常量**：先 `grep` 全仓（含 `pyqcd/testing/**`），并给前后计数；
   `AUTO_LABEL` 那次的教训是每个带 `optimize` 列的命中行都会撞 `AttributeError`。
4. **语义陷阱**：`forced_labels(None) == ()` 要用显式布尔位区分"用户真的给过"；
   `bool("false") == True`；`optimize="auto"` 在 generic/dedicated 是两套算法；
   `kupl_topology`/`kupl_topology_v2` 是**路线**（判据为原生 5 道门，不是 seq 阈值）。

### [T3] 拓扑步表统一：口径与交付（2026-11-18）

判据与检查单：`docs/einsum_topology_unification_20261118.md`；时间线：
`logs/einsum-topology-unification-20261118.txt`。

1. **口径只有一个**：`ROLE_SHAPES`（peram `[4,4,100,100]`、gamma `[4,4]`、
   VDV `[100,100]`、VVV `[100,100,100,100]`）；`production`/`uniform` 已删除，
   旧表与旧性能数字**全部作废**，不要混用或"迁移"。
2. **重生成三条硬判据**：新表 `sha256` 必须与旧**不同**（相同=默认路径没换）；
   连跑两次 `sha256` 必须**相同**；`--verify` 必须 `missing=0 extra=0 mismatch=0`。
3. **两表不可互译**：`.inc` 是 cblas 字节码，`.h` 是 `int8` 活表位置对；只共享
   "语料枚举 + 路径求解"，格式层分别实现。
4. **大文件**：只发布 `.gz`（KUPL 8.1×、CUDA 4.4×），构建侧"本体缺失或 `.gz` 更新"
   才解压；KUPL/CUDA **两侧脚本都要有这段**，`.gz` 用 `mtime=0` 保证可复现。
5. **图纸是生成物**：`_wick_tasklist.py` 只能由 `wick_tasklist_gen` 重建，加条目写
   `_wick_extra.jsonl`；`seq` 连续 `1..N`（T 号在两张表里各起 `T00001`，720 个重复），
   整表改格式的验收是 `normalize(旧行)==新行` 逐行成立。
6. **提示语跟着功能走**：删 `kernel_force=auto` 后"先 dedicated 再 topology"就是假话；
   `[match: X]` 是命中形态、`[kernel: Y]` 才是实际执行（`LAST_INFO['path']`），X≠Y 正常。
7. **通用核对四项**：删常量先 `grep` 全仓（含 `pyqcd/testing/**`）；异常元组别多包一层
   （`KernelUnavailableError((requested,), …)` 会让 `", ".join()` 崩）；CLI 默认值用
   `None` 表示"没给"（填 `"generic"` 会被当成显式指定）；路径先 `expanduser()`
   再 `abspath()`，否则 `~/.cache` 变成字面目录、`--profile-dir` 静默失效。

### [T4] tune 的卡死 / 超时 / 参照（2026-11-19）

判据与流程：`docs/einsum_tune_timeout_20261119.md`；时间线：
`logs/einsum-tune-timeout-20261119.txt`；`AGENTS.md` 对应小节 §9.1h。

1. **`--tune-timeout` 是 best-effort，不是硬超时**：`SIGALRM` 只在字节码之间派发，
   原生 C/CUDA 调用不返回解释器就永不触发。本机实测：纯 Python 循环 0.30 s 准点抛；
   单次 **2.34 s** 的原生 BLAS、上限 0.3 s → **2.29 s 才抛（超限 1.99 s）**。文档只能
   写"能打断 Python 长循环、打不断原生"，并给 `--tune pass` / 加大上限两条替代。
2. **窗口是整组不是单次**：一组 `1+warmup+rounds`（缺省 **7**）次调用，装的是
   `limit × 7` 的组级预算，才与 `min_wall_ms` 剪枝同口径；按单次设限会白花 `7 × limit`。
3. **卡死先查两处非内核位置**：`_tune_reference` 的完整 `np.einsum`（在超时保护之外，
   按输入元素总数 `> 2_000_000` 跳过；生产方程实测 **2.56e8** 个元素）与 `optimal` 的
   纯 Python 穷举。定位靠 `tune init :` 起止行 + `STAT=Rl+/%CPU=100/GPU 0%`。
4. **硬超时的两条路都别顺手换**：`fork` 在本机必失败（父进程碰过 CUDA 后子进程
   `cudaErrorInitializationError`）；`spawn` 每组全新解释器实测 **497.6 ms**（含一次
   CUDA gemm，n=5，min 425.4 / max 1174.5）→ 22 组约 11 s 固定开销。
5. **语义顺序写死在 `_api.py`**：`as_equivalence=true` = "先改写成语料形态、再跑特化
   内核"。实测 `cbda,...->` 原方程 `kernel_force=topology` 必失败，同一条 `as_equivalence=true`
   → `rel=1.54e-16`。另：`equiv_ok` / `match` 命中只表示**签名命中**，CUDA 侧还有步表
   覆盖与 `status 4` 两道门，"命中但不可执行"真实存在。
6. **验收只用三条**：`python -B -m pyqcd.testing._einsum_tune_timeout_doc_contract`（4 项）、
   `--tune` / `--tune pass` / 不给 三种形态各跑一次小方程（`--profile false`）、
   小形状与大形状各验一次 `rel`（只验小形状会把"参照为 None ⇒ mismatch 恒假"放回去）。

## 2026-11-20 增补：original 内核 / 峰值内存 / 记录表 / 自动 tune

以下四条是同一批落地的事实，**细节证据**见 `logs/einsum-tune-20261119-facts.md`，
使用者视角见 `docs/einsum_tune_diagnostics_20261119.md`。

### `kernel_force="original"`

`_kernel_force.ALLOWED_LABELS` 现在是五个（`generic` / `dedicated` / `topology` /
`original` / `backend`）。`original` 把收缩交给 `opt_einsum.contract`，与 `backend`
同地位（都不进专用分派），tune 里排在 `backend` **之前**。

**名字必须转型**（实测结论，别凭记忆写）：

| 本库叶子 | opt_einsum 实参 |
|---|---|
| `greedy` / `optimal` / `dp` | 同名 |
| `branch_1` / `branch_2` | `branch-1` / `branch-2` |
| `branch_all` | `branch-all` |
| `none` | `False` |

转型单点是 `_kernel_force.opt_einsum_path_name()`；`_api._original_einsum()` 只调它，
不写第二份映射。`backend=` 由操作数类型推断（`_api._opt_einsum_backend_name`），
**不读命令行**——`oe.contract` 的 backend 缺省 `'auto'` 按数组类型选，只传 numpy
副本等于在另一台机器上做参照。

### 峰值内存逐组测量

`pyqcd/einsum/_memory.py` 提供 `reset_peak_memory()` / `peak_memory_mb()` /
`release_memory()` / `memory_probe_available()` / `format_peak_mb()`，四个都**绝不抛
异常**（失败进 `LAST_ERROR`）。约定：

* 进一组前 `reset_peak_memory()`，组末 `peak_memory_mb()`，然后 `release_memory()`；
* 子进程路线由 worker 做（显存属于那个进程），父进程只搬运 `peak_memory_mb`；
* 单位是 **MiB**（1024² 字节），`format_peak_mb()` 输出 `1234MB` / `-`；
* CPU 侧 `ru_maxrss`/`VmHWM` 是生命周期峰值、清不零，所以那个值是**下界**——docstring
  里如实写着，别把它当精确组内峰值。

### 记录表 `pyqcd.einsum.record.csv`

与 `profile.csv` 同目录、同列 + 一列 `peak_memory_mb`，**整表重写**（每次 tune 一份
快照）、**从不参与命中**：`_api` / 选路 / `read_winner` / 命中提示**一处都不读它**。
（"存在更省内存的配置"那条提示已在 2026-11-21 按用户要求删除，热路径上因此不再有
任何读记录表的 I/O；它现在是纯报告产物，只给报告与工具用。）

### 首次调用自动 tune

`pyqcd/einsum/_autotune.py`：配置表缺行时跑一整轮 tune 再重读表。三道边界——开关
`PYQCD_EINSUM_AUTOTUNE`（缺省开）、同键每进程只试一次、`_RUNNING` 互斥（防 tune
自己在内部调用回来）。**测试套件里必须关掉它**（`conftest.main()` 已设
`PYQCD_EINSUM_AUTOTUNE=0`），否则一次回归会被拖成几十分钟。

### 组数与剪枝口径（改了就要同步三处）

B1/B2 = **29** 组、B3 = **28** 组（`original` 各占 7 组）；`TUNE_MAX_COMBOS = 32`；
剪枝截止 = `2 × min_wall_ms`（`_tune_axes.MIN_WALL_FACTOR`），`原因=` 里同时给
`min_wall_ms` 与 `cutoff_ms`。同步点：`pyqcd/einsum/test.py` 的模块 docstring 与
`--help` 文案、`docs/{en,zh}/performance.md`、`pyqcd/testing/_einsum_tune_plan_contract.py`
的 `SPEC_SIZES`。改一处漏一处会让契约红在"文案过期"而不是真的行为变了。
