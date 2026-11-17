# qcd-einsum 用户操作手册 —— "我想做 X，该敲什么"

面向**使用者**的任务导向手册。API 参考不在这里（`--help` 已覆盖），
这里只回答"下一步敲哪条命令、输出怎么读、读到的结果意味着什么"。

| 项 | 值 |
|---|---|
| 适用范围 | `pyqcd/einsum` 的**用户可见入口**（`python -m pyqcd.einsum.test`、`python -m pyqcd.einsum.verify_specialized`、`pyqcd.einsum.contract`） |
| 版本基线 | `HEAD = aeea4faf9befac174c8e8f15d3c0138f2958851f`（2026-11-16，`pyqcd 0.4.3`），工作树干净 |
| 校验方式 | 文中**每条命令、每个输出块都在该 HEAD 上实跑过**（清单见 §10）。源码行号一律指该 HEAD |
| 本版重点 | `--equivalence` / `PYQCD_EINSUM_EQUIVALENCE` / `--language` 三条能力**已全部合入**；等价映射层已重写为"结构不变量分桶 + 桶内构造同构"，**输入侧与输出侧必须同时改写**这一条是本手册最重要的一节（§4）。另新增 §11：强制指定专用内核（`--kernel-force` / `PYQCD_EINSUM_KERNEL_FORCE`）——**只改选路、不改数值** |
| 不覆盖 | 内部契约（`pyqcd/einsum/AGENTS.md` §9）、领域知识（`skills/pyqcd-einsum/SKILL.md`）、CUDA 构建流程 |

---

## 0. 三条入口，先选对

| 我想… | 用哪条 | 一句话区别 |
|---|---|---|
| 知道这条方程会走哪条路 / 快不快 / 对不对 | `python -m pyqcd.einsum.test` | **会真的算**，出耗时与容差结论 |
| 只想知道它命不命中语料特化，不想跑数值 | `python -m pyqcd.einsum.verify_specialized` | **纯查询**，不构造张量、不调 `contract()`、无副作用 |
| 确认"换一种写法结果还一样吗"（等价映射） | 先看 §3，再照 §4 复算 | 映射只给建议，**正确性由你复算保证** |
| 只准跑某个内核 / 完全不用原生内核 | 看 §11 | **选路**开关，不改数值；跑不成会明确报错而不是 warning |
| 整库有没有退化 | `python -m pyqcd.testing` | 全量回归。与上面三条互补，**不可互相替代** |

`python -m pyqcd.einsum.test` 是 qcd-einsum 单方程验证/复现/性能对比的**唯一合法入口**
（硬性规定，2026-11-15 起）；临时脚本、一次性 `python -c` 的结论不予采信。
本手册只在 §4 做过**跨方程等价性对拍**——那不是"单方程对不对、多快"，没有对应的
CLI 入口，且它的结论由 §4.4 的独立验证器 `python -m pyqcd.testing.einsum.verify_equivalence`
另行背书。

```bash
# 环境（本机可用）
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pyqcd-einsum-cu124
cd /home/zhangxin/PyQCD
export PYTHONPATH=$PWD PYQCD_EINSUM_NOTICE=0
```

---

## 1. 我有一条方程，想知道它会走哪条路

### 1.1 命令

```bash
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" \
    --scene distill --verbose true
```

- `--scene distill`：省略 `--shape` 时按**语料 `roles`** 反推尺寸。
  方程不在语料里会直接报错退出 2（见 §1.5），此时必须显式给 `--shape`。
  这套尺寸**只能当路径证据**，不能当生产规模——理由见 §1.6。
- `--verbose true`：逐项明细打到 **stderr**；那行 `PASS …` 在 **stdout**。
  想一次看全就 `2>&1`。

### 1.2 实跑输出（原样）

> 耗时/内存是**当次实测值**，会随机器负载浮动；行与字段的结构是稳定的。
> 同一个方程两次跑出 `0.106ms` 与 `0.101ms` 都正常，别拿毫秒位比对历史结论。

stdout：

```
PASS cadb,menf,ac,em,bd,fn-> rel=1.54e-16 vs np.einsum(optimize='greedy')  0.106ms (ref 0.202ms@greedy, 0.53x)  rss+7MiB
```

stderr（`--verbose true`）：

```
equation : cadb,menf,ac,em,bd,fn->
device   : cpu  dtype=complex128  backend=auto
operands : 6 个，共 66 元素；输出 [] (1 元素)
shape    : 共 6 个操作数形状（按方程中从左到右的顺序）
           [4,4,1,1], [4,4,1,1], [4,4], [4,4], [1,1], [1,1]
specialize: auto（实际 True）
time     : 0.106 ms （warmup=1，rounds=5，中位）
ref opt  : 要 auto → 参照实际用 'greedy'
reference: np.einsum(optimize='greedy')
ref time : 0.202 ms  （pyqcd/参照 = 0.53×）
rel      : 1.540e-16  (atol=1e-12)
memory   : 输入张量 +2 MiB RSS
peak     : RSS +7 MiB（进程共 76 MiB）
pyqcd    : 0.4.3  /home/zhangxin/PyQCD/pyqcd/__init__.py
kernel   : {'kupl': {'path': 'kupl_dedicated', 'dtype': 'complex128'}}
route    : 特化 · 原生 qcd_kupl_dedicated_apply（path=kupl_dedicated）
```

### 1.3 怎么读 `route` 行 —— 这是"到底跑了什么"的权威答案

只有两种形态：

```
route    : 特化 · 原生 qcd_kupl_dedicated_apply（path=kupl_dedicated）
route    : 通用（无原生命中：Python 分步 / numpy.einsum / BLAS）
```

- 多个入口同时命中时用 ` + ` 连接（`test.py:1731-1742`，`detail.route_joiner`）。
- 原生符号映射见 `test.py:1724-1730`（`_route_of` 的 `native_symbols`）：

| `kernel_info[*]['path']` | route 行里显示的原生符号 |
|---|---|
| `kupl_dedicated` | `qcd_kupl_dedicated_apply` |
| `kupl_topology` | `qcd_kupl_topology_apply` |
| `kupl_topology_v2` | `qcd_kupl_topology_apply`（**看 `path=` 后缀区分**） |
| `cuda_dedicated` | `qcd_cuda_dedicated_apply` |
| `cuda_topology` | `qcd_cuda_topology_apply` |

**必须区分 `kupl_topology` 与 `kupl_topology_v2`**：两者符号相同，只有
`（path=…）` 后缀不同。前者表示真的走了编译期 v3 步表，后者是 v2 通用执行器在跑；
两者**数值都正确**，只有性能含义不同（`_kernel_kupl.try_topology_apply` 的
`LAST_INFO['path']` 契约）。

> 本机实测（CPU/KUPL）：`seq 3/4/6/7/8/9/10/2583/2584/3431` 的 `route` 全是
> `kupl_dedicated`；只有被 pack-bound 守卫逼回通用时才会出现 `通用（无原生命中…）`。
> `kupl_topology*` 这两条**在本机没能用 CLI 复现**（`try_dedicated` 在分派顺序里
> 更靠前，先把这些计划接走了），上表后三行取自源码，不是实跑。

`route` 行只读 `kernel_info` 里的 `path`；`kernel_info` 本身由 `test.py:1414-1425`
（`_kernel_info`）从 KUPL/CUDA 两个模块的 `LAST_INFO` 汇总，键为 `kupl`/`cuda`，
每项只保留实际存在的 `path / specialized / steps / dtype / copied`。

### 1.4 怎么读 `kernel` 行

```
kernel   : {'kupl': {'path': 'kupl_dedicated', 'dtype': 'complex128'}}
kernel   : 无原生命中（通用路径）
```

`kernel` 是嵌套字典（要读两层），`route` 是它的一句话翻译。
**`kernel` 为空 / 显示"无原生命中"才是"真的没走原生"**；
`specialize: auto（实际 True）` 只说明**开关开着**，不说明命中。

反例（实跑，`ij,jk->ik`）：

```
specialize: auto（实际 True）
kernel   : 无原生命中（通用路径）
route    : 通用（无原生命中：Python 分步 / numpy.einsum / BLAS）
```

`specialize_effective=True` + route 通用 = 开关开着但这条方程没有原生路径。
这两个字段答的是不同问题，别混读。

### 1.5 常见报错（都实跑过）

```bash
# 省略 --shape 但方程不在语料里                    → 退出码 2
python -m pyqcd.einsum.test --equation "ij,jk->ik"
# error: 省略 --shape 时方程必须在语料里（用于按 roles 反推生产尺寸），
#        但 'ij,jk->ik' 不在 distill 语料中。请显式给 --shape。
```

```bash
# --warmup 0 被拒                                  → 退出码 2
python -m pyqcd.einsum.test --equation "ij,jk->ik" --shape "[4,4],[4,4]->[4,4]" --warmup 0
# error: --warmup 不能为 0：首调包含 plan 缓存/原生库绑定/JIT，不丢弃会让耗时完全失真
```

### 1.6 ⚠️ `--scene` 的任何一档都**不是**生产尺寸

这是本手册最容易误用的一条，请连着 `pyqcd/einsum/AGENTS.md` §6.1 一起读：

| 档 | 本机实测形状（`cadb,menf,ac,em,bd,fn->` 的 6 个操作数） |
|---|---|
| `--scene distill` | `[4,4,1,1], [4,4,1,1], [4,4], [4,4], [1,1], [1,1]` |
| `--scene distill_curr` | `[4,4,9,1], [4,4,9,1], [4,4], [4,4], [1,9], [1,9]` |
| `--scene scene` | `[4,4,5,40], [4,4,5,40], [4,4], [4,4], [40,5], [40,5]` |
| `--scene full` | `[4,4,8,64], [4,4,8,64], [4,4], [4,4], [64,8], [64,8]` |

四档**全部**经过 `bench_specialized._label_sizes`。该函数的形状口径表达不出蒸馏
张量的真实轴：参考脚本 `refer/sush/sush-einsum.py` 的 peram 是
`(4, 4, nev[tq], nev[ta])`，同一张图里同时出现
`(4,4,100,100)`、`(4,4,100,400)`、`(4,4,400,100)`、`(4,4,400,400)`
四种异形轴长，而 `_label_sizes` **一个方程里只放得下一个 `Nev`**。
所以 `--scene distill_curr` 给出的 `[4,4,9,1]` 与真实 peram `(4,4,100,400)`
**不是一回事**——那个 `9` 是 `Mom_curr` 漏进 peram 轴的产物。

结论：

- 以上四档的形状只能当**路径证据**（走没走原生、命中哪条路），
  **不得**写成性能结论，也不要拿它们的倍率去标定任何门限；
- 真实生产形状只认 `pyqcd.einsum._bench_scene.distill_tensor_shapes()`
  （本机实跑给出 29 个张量，peram 四种异形轴长如上，`VDV` 为 `(1,100,100)` /
  `(3,400,400)`）；
- `test.py` **没有**"用 `distill_tensor_shapes()` 自动填形状"的开关。
  要测生产规模，必须把真实形状**显式写进 `--shape`**，例如
  `--shape "[4,4,100,400],[4,4,100,400],…"`（按你那条方程的操作数逐项对齐）。
- 小尺寸（`smoke`、`--scale`、等长 `dim` 扫描）同理由 AGENTS.md §6.1 判定为
  "只能用于路径与正确性证据"。

---

## 2. 我想确认它是否命中语料特化

### 2.1 命令

```bash
python -m pyqcd.einsum.verify_specialized "cadb,menf,ac,em,bd,fn->"
python -m pyqcd.einsum.verify_specialized "<eq>" --json                  # 结构化
python -m pyqcd.einsum.verify_specialized "<eq>" --language en           # 英文文案
python -m pyqcd.einsum.verify_specialized "<eq>" --equivalence false     # 关掉等价段
```

纯查询：不生成操作数、不调用 `contract()`、不改全局状态
（`verify_specialized.py` 模块 docstring）。判定顺序是
**形状特化（`_shape_paths`）→ 语料索引（`_specialize.match`，8638 条）→ 等价映射**。

三个选项都只影响**报告**，不改 `matched`、不改退出码：

| 选项 | 缺省 | 作用 |
|---|---|---|
| `--json` | 关 | 输出机器可读结果（不受 `--language` 影响） |
| `--language {auto,en,zh-cn}` | `auto` | 只翻人读文案。`auto` **只在 `PYQCD_EINSUM_LANG` 显式设了**时跟随，未设保持中文；**不跟随系统 locale** |
| `--equivalence {auto,true,false}` | `auto` | `auto` 跟随 `PYQCD_EINSUM_EQUIVALENCE`（未设即开启）；`true`/`false` 强制并**压过**该变量 |

### 2.2 三段式读法

**① HIT**——结果行是 `HIT`，且带语料出处：

```
方程        : cadb,menf,ac,em,bd,fn->
归一化      : cadb,menf,ac,em,bd,fn->
拓扑签名    : ((0, 1, 2, 3), (4, 5, 6, 7), (1, 0), (5, 4), (3, 2), (7, 6), ())
元数/槽位   : 6 个操作数 / 16 个指标槽（输出秩 0）
摘要        : a1c498625ef00836abc4196b
特化开关    : 开启（PYQCD_EINSUM_SPECIALIZE）
结果        : HIT
  特化路径  : topology:a1c49862  [kernel(topology)]
  声明内核  : kupl_topology, cuda_topology
  语料出处  : seq=1  T00001  1. 介子两点函数
  代表写法  : cadb,menf,ac,em,bd,fn->
  操作数角色: peram_u, peram_u, gamma_M, gamma_MH, VDV_0, VDV_1
  说明      : 声明内核只表示契约允许；实际是否下发取决于 out=、dtype、原生库可用性。
              数值验证见 python -m pyqcd.testing.einsum.verify_specialized
```

（上面这次是 `cadb,menf,ac,em,bd,fn->` 的实跑；完整 16 操作数的 HIT 见 §3.4 例 3。）

**"声明内核"要对照 `route` 读。** 同一个方程，`verify_specialized` 声明
`kupl_topology, cuda_topology`，而 §1.2 里 `test.py` 的 `route` 实际是
`qcd_kupl_dedicated_apply`——两者都对，只是回答的问题不同：
前者说"契约允许哪些入口"，后者说"这次真的走了哪个入口"。

**② MISS，但有等价写法**——`MISS` + `等价写法` 段（可映射），
**输入侧与输出侧两行恒在**（某侧无需调整写 `-`）：

```
结果        : MISS
  原因      : 直接写法不在语料里，但存在等价写法可命中特化路径（操作数 0 轴序改为 cadb（张量需 transpose(1, 0, 2, 3)））；需按给出的张量调整同步转换后再算，结果与直接写逐元素相同
  等价写法  : cadb,menf,ac,em,bd,fn->
  变换类型  : operand-axes
  变换说明  : 操作数 0 轴序改为 cadb（张量需 transpose(1, 0, 2, 3)）
  张量调整  : transpose(1, 0, 2, 3), -, -, -, -, -
  结果调整  : -（改写后输出轴序已一致）
  提示      : 等价改写只改「方程怎么写」，不改任何执行路径；按上面的张量调整同步转换后，
              用等价方程调用 contract() 即可走上特化路径，结果与直接写法逐元素相同。
```

（`张量调整` 是**输入侧**，`结果调整` 是**输出侧**，缺一不可；这两行的用法见 §4。）

当 `operand_order` 非空时会**多一行** `张量次序`：

```
  张量次序  : 按 [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] 重排操作数
```

**③ MISS 且无等价**——`MISS` + 一句"既不在语料…也不存在等价写法"：

```
结果        : MISS
  原因      : 既不在 wick_12cases 语料里，也没有形状特化路径，且不存在可落到已知形式的等价写法，走通用分步路径（无特化内核）
```

注意这时**没有** `等价写法 / 变换类型 / 张量调整` 任何一行——"没有"就是如实没有，
不是被截断。

### 2.3 `--json`（结构化）

```bash
python -m pyqcd.einsum.verify_specialized "acdb,menf,ac,em,bd,fn->" --json
```

```json
{
  "equation": "acdb,menf,ac,em,bd,fn->",
  "parsable": true,
  "matched": false,
  "digest": "91104304ce1d8e05ab572591",
  "reason_key": "reason.equivalent_rewrite",
  "reason_args": {
    "description": "操作数 0 轴序改为 cadb（张量需 transpose(1, 0, 2, 3)）"
  }
}
```

（上面只摘了关键字段；完整结果还有 `equation_normalized` / `signature` / `arity` /
`slots` / `output_rank` / `specialize_enabled` / `equivalence` / `reason`。）

**`reason_key` / `reason_args` 是语言无关的稳定形式**，`--json` 的消费者要本地化
就直接查 `verify_specialized.MESSAGES`（46 个 key × 中英两套，本机核对全部齐备），
**不要**去解析 `reason`（那是历史中文文案，逐字保留）。

`equivalence` 子对象的字段（§3.4 例 2 的实跑）：

| 字段 | 含义 |
|---|---|
| `equation` | 改写后的方程（可直接交给 `contract`） |
| `rewritten` | 是否真的改写（`false` 表示查过了但没有等价写法） |
| `operand_order` | 张量列表新次序，或 `null` |
| `operand_permutations` | 逐操作数 `transpose` 置换（按**改写后**次序），或 `null` |
| `output_permutation` | 结果回填用的 `transpose` 置换（**已是逆置换**，直接喂），或 `null` |
| `kind` | 变换类别：`identity` / `operand-order` / `operand-axes` / `output-axes` 及其 `+` 复合 |
| `description` | 中文说明（**始终中文**，见 §6.4） |

`equivalence` 字段的**出现规则**（两个入口同口径）：

- 开启时**总是写**，没有改写也写 `rewritten=false`——用来区分"查过了没有"与"根本没查"；
- `--equivalence false` 或 `PYQCD_EINSUM_EQUIVALENCE=0` 时**整个字段不出现**；
- 已 HIT 的方程**不查也不写**（`test.py` 与 `verify_specialized` 都是），
  本机实跑 `grep -c '^equiv'` 对 HIT 方程得 `0`。

### 2.4 退出码

| 码 | 含义 |
|---:|---|
| 0 | 命中（语料或形状特化） |
| 1 | 未命中——实际会走通用分步路径 |
| 2 | 方程无法解析，或参数非法 |

本机实跑：`cadb,…->` → 0；`ij,jk->ik` → 1；`a->b` → 2；
`--equivalence maybe` → 2（argparse 的 `choices` 拦下）。

```bash
python -m pyqcd.einsum.verify_specialized "a->b"   # 输出"方程无法解析（…）"，退出码 2
```

### 2.5 一个必须知道的陷阱

```bash
PYQCD_EINSUM_SPECIALIZE=0 python -m pyqcd.einsum.verify_specialized "cadb,menf,ac,em,bd,fn->"
# 特化开关    : 关闭（PYQCD_EINSUM_SPECIALIZE）
# 结果        : HIT          ← 仍然 HIT
```

因为它是**纯查询**：`HIT` 说的是"语料里有这条拓扑"，与实际会不会走原生无关。
要判断"这次会不会真的走"，请用 §1 的 `route` 行。

---

## 3. 我的方程没命中，但我觉得它"应该"命中

**先看清它为什么没命中，再决定要不要改写。**

### 3.1 为什么没命中：签名只归一化"字母改名"

`_specialize.canonical_signature`（`_specialize.py:164-186`）把指标按**全式首次出现
顺序**重新编号，于是 `ij,jk->ik`、`ij,jb->ib`、`ab,bc->ac` 签名相同。它的 docstring
把边界写得很清楚（`:171-179`）：

> 从头到尾扫描**全部操作数**（操作数顺序不变，每个操作数内部的指标顺序也不变），
> 再扫描输出项，把指标按"全式首次出现顺序"重新编号为 `0,1,2,…`

也就是说，签名**只**消掉：

- 字母改名（`i→a`）
- 空白差异（`normalize` 去空白）
- 省略 `->` 时的隐含输出推导

它**不**消掉：**操作数位置**、**单操作数内部轴序**、**输出段轴序**。
这三种写法数学上等价、签名却不同 → 一律 MISS。补上这一层的正是 §3.2。

> 一个直接的旁证（`_specialize.py:176-179` 的 docstring 原文）：
> `ij,kj->ik`、`ij,jk->ki` 与 `ij,jk->ik` 签名不同，因为"签名只认谁和谁共享、
> 共享的指标是否留在输出、以什么顺序留在输出"。

### 3.2 等价映射层怎么工作（`_equivalence.py`，**已重写**）

**旧实现**是"枚举候选方程 → 逐个 probe"，再靠上界（`orders[:96]`、
`_MAX_CANDIDATES`）防爆炸；上界本身造成覆盖缺口，因此**三族叠加解不出来**。
**现实现（本 HEAD）完全不枚举候选**，改成两次 O(incidence) 的判定：

1. **结构不变量分桶**（`_Shape`，`_equivalence.py:258-301`）：每个标签记
   `(出现重数, 是否在输出, 零共现伙伴数, 共现重数多重集)`，每个操作数记
   `(秩, 其标签签名多重集)`；桶键 = (操作数类多重集, 输出标签签名多重集)。
   这些量对**字母改名 / 操作数次序 / 操作数内轴序 / 输出轴序全都不变**，
   所以等价方程**必然同桶**（不漏判）。
2. **桶内构造式同构**（`_match`，`:336-453`）：对桶里每条语料条目构造
   `(操作数双射, 标签双射)`，用「秩 / 签名 / 重数 / 共现」四类约束回溯剪枝，
   命中即得**精确**映射；节点数超过 `_MAX_SEARCH_NODES`（`:107`，4096）就放弃
   （放弃只会漏判，不会误判）。
3. **字段由双射机械翻译**（`_build`，`:456-538`）：位置、轴序、输出轴序三段纯置换
   合成，翻完再做一次**符号回放自检**（`:495-497`、`:508-511`），不通过就丢掉该候选。

因为第 3 步的回放自检，本层**只会给出可验证的映射**：桶键撞了（两条不同构的方程
落进同一桶）也只会被拒绝，不会把错映射交出去。

覆盖的变换族（`kind` 字段的取值由 `_kind`，`:541-562` 给出）：

| 族 | 例子 | 字段 |
|---|---|---|
| `operand-order` 操作数位置 | `qgrh,istj,…` ↔ `isjt,qgrh,…` | `operand_order` |
| `operand-axes` 单操作数轴序 | `acdb,…` ↔ `cadb,…` | `operand_permutations` |
| `output-axes` 输出段轴序 | `…->GMHew` ↔ `…->MGHew` | `output_permutation` |
| 以上任意复合 | 例 2（三族叠加） | 三个字段同时非空，`kind` 用 `+` 连接 |
| 字母改名 | — | 由 `canonical_signature` 负责，本层不重复处理 |

**本层不做**代数化简（不会把 `ij,jk->ik` 合并成矩阵乘），也不做形状相关优化。
它也**不需要知道张量形状**——全部是符号层推导。

> 概念性的族例子（如 `ac,em->` ↔ `em,ac->`）**不在 8638 条语料里**，
> 用 CLI 查会得到"无等价写法"。这不是 bug：映射的靶子是**语料形态**，
> 语料外的形态没有可映射的目标。

### 3.3 ⚠️ 最容易出错的地方：输入变了，输出也必须变

**正确性必须由你自己保证。** 映射会逐条给出三种调整：

- `operand_order`：`operand_order[i]` = "改写后第 i 个操作数取你原来的第几个"；
- `operand_permutations[j]`：该操作数新轴 j 取旧轴 perm[j]（numpy `transpose` 语义）；
- `output_permutation`：**已经存成逆置换**，直接喂 `transpose` 即可。

最容易踩的坑：**只换了输入忘了换输出**。`_equivalence.py:156-162` 专门写了原因：

> 设原输出段 `out`、改写后 `new_out[j] == out[perm[j]]`，要还原成 `result_old` 需要
> `axes = perm⁻¹`。**对两两互换 perm 自逆、两者相同，所以测不出方向错误**
> ——必须用三循环这类置换验证。

本机实测三种错误的可见性（`cwdx,…,isjt,qgrh,…` 那 16 操作数方程，输出段轴长
强制两两不同 `2,3,4,5,6`）：

| 用例 | 正确置换 | 用逆置换 | 不回填 |
|---|---|---|---|
| 三循环 `HMGew`（映射给 `(2,0,1,3,4)`） | `relerr=4.35e-16` ✅ | 形状 `(4,2,3,5,6)` ≠ `(2,3,4,5,6)` → **形状抓到** | 形状 `(3,4,2,5,6)` ≠ → **形状抓到** |
| 两两互换 `GMHew`（映射给 `(1,0,2,3,4)`，自逆） | `relerr=4.35e-16` ✅ | `relerr=4.35e-16` ——**方向不可区分** | 形状 `(3,2,4,5,6)` ≠ → **形状抓到** |

读法：

- **自逆置换**（两两互换、双换）无论正转反转结果一样，所以"方向"这件事对它
  根本不成立；能出错的只有"漏转"。
- **非自逆置换**（三循环、五循环）方向错会被算错，但**只有当相关轴长不同时
  形状才会帮你抓到**。语料里大量轴长是 `4/4`、`1/1` 甚至全是 `1`，
  这时形状检查**完全失效、静默出错**。

所以判据只有一条：**改写后的结果 `transpose(output_permutation)` 必须逐元素等于
原式直接算的结果**。形状检查只是廉价的辅助，不是判据。§4 给出可直接粘贴的复算脚本。

### 3.4 三个真实例子

#### 例 1（单族变换，可用）

```bash
python -m pyqcd.einsum.verify_specialized "acdb,menf,ac,em,bd,fn->"
```

```
方程        : acdb,menf,ac,em,bd,fn->
归一化      : acdb,menf,ac,em,bd,fn->
拓扑签名    : ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1), (5, 4), (3, 2), (7, 6), ())
元数/槽位   : 6 个操作数 / 16 个指标槽（输出秩 0）
摘要        : 91104304ce1d8e05ab572591
特化开关    : 开启（PYQCD_EINSUM_SPECIALIZE）
结果        : MISS
  原因      : 直接写法不在语料里，但存在等价写法可命中特化路径（操作数 0 轴序改为 cadb（张量需 transpose(1, 0, 2, 3)））；需按给出的张量调整同步转换后再算，结果与直接写逐元素相同
  等价写法  : cadb,menf,ac,em,bd,fn->
  变换类型  : operand-axes
  变换说明  : 操作数 0 轴序改为 cadb（张量需 transpose(1, 0, 2, 3)）
  张量调整  : transpose(1, 0, 2, 3), -, -, -, -, -
  结果调整  : -（改写后输出轴序已一致）
  提示      : 等价改写只改「方程怎么写」，不改任何执行路径；按上面的张量调整同步转换后，
              用等价方程调用 contract() 即可走上特化路径，结果与直接写法逐元素相同。
```

退出码 **1**（MISS）。操作：把第 0 个张量 `transpose(1,0,2,3)`，
方程换成 `cadb,menf,ac,em,bd,fn->`。输出是标量（`->`），**没有** 输出轴要转。

#### 例 2（三族叠加）——**现在能解了**

上一版手册把它标为"已知未覆盖"，现实现能给出精确映射。

```bash
python -m pyqcd.einsum.verify_specialized \
  "cwdx,eufv,mknl,oapb,qgrh,istj,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->GMHew"
```

```
元数/槽位   : 16 个操作数 / 55 个指标槽（输出秩 5）
摘要        : facfbf94f8e3377ee0e65fc9
结果        : MISS
  原因      : 直接写法不在语料里，但存在等价写法可命中特化路径（操作数次序改为 [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] 且 操作数 4 轴序改为 isjt（张量需 transpose(0, 1, 3, 2)） 且 输出轴序改为 MGHew（结果需 transpose(1, 0, 2, 3, 4)））；需按给出的张量调整同步转换后再算，结果与直接写逐元素相同
  等价写法  : cwdx,eufv,mknl,oapb,isjt,qgrh,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHew
  变换类型  : operand-order+axes+output-axes
  变换说明  : 操作数次序改为 [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] 且 操作数 4 轴序改为 isjt（张量需 transpose(0, 1, 3, 2)） 且 输出轴序改为 MGHew（结果需 transpose(1, 0, 2, 3, 4)）
  张量调整  : -, -, -, -, transpose(0, 1, 3, 2), -, -, -, -, -, -, -, -, -, -, -
  张量次序  : 按 [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] 重排操作数
  结果调整  : 拿到结果后需 transpose(1, 0, 2, 3, 4)
  提示      : 等价改写只改「方程怎么写」，不改任何执行路径；按上面的张量调整同步转换后，
              用等价方程调用 contract() 即可走上特化路径，结果与直接写法逐元素相同。
```

（完整输出还有 `方程` / `归一化` / `拓扑签名` / `特化开关` 四行，这里只留与结论相关的行。）

三处差异正好是三族叠加：

| # | 差异 | 族 | 需要的调整 |
|---|---|---|---|
| ① | 操作数 4↔5 互换（`qgrh,istj` vs `isjt,qgrh`） | operand-order | `operand_order = (0,1,2,3,5,4,6,7,8,9,10,11,12,13,14,15)` |
| ② | 新位置 4 的 `qgrh→isjt` 之后轴序还原 | operand-axes | 该位置 `transpose(0,1,3,2)` |
| ③ | 输出 `GMHew→MGHew` | output-axes | 结果 `transpose(1,0,2,3,4)` |

**这张映射我自己复算过**（§4.3 的脚本，输出段轴长两两不同 `2,3,4,5,6`）：

```
mapped : cwdx,eufv,mknl,oapb,isjt,qgrh,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHew
kind   : operand-order+axes+output-axes
shape  : (2, 3, 4, 5, 6) vs (2, 3, 4, 5, 6)
relerr : 2.802e-16
```

**张量调整建议**（为什么用非对称轴长）：把输出段各轴设成互不相同的长度
（这里 `2,3,4,5,6`），其余标签统一给 `2`。这样输出轴序一旦搞反或漏转，形状就会
对不上（实测漏回填得到 `(3,2,4,5,6)`）。若沿用语料里那种 `4/4/1/1` 的轴长，
输出置换的方向错误会被**完全掩盖**。

#### 例 3（长方程，HIT）

```bash
python -m pyqcd.einsum.verify_specialized \
  "cwdx,eufv,mknl,oapb,isjt,qgrh,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHew"
```

```
元数/槽位   : 16 个操作数 / 55 个指标槽（输出秩 5）
摘要        : 7d2f55866d377d893c4513fc
结果        : HIT
  特化路径  : topology:7d2f5586  [kernel(topology)]
  声明内核  : kupl_topology, cuda_topology
  语料出处  : seq=8628  T00710  动量指标与流分量指标 / 情形 14：五夸克到三夸克四点函数
  代表写法  : cwdx,eufv,mknl,oapb,isjt,qgrh,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHew
  操作数角色: peram_u, peram_u, peram_u, peram_u, peram_u, peram_u, gamma_M, C * gamma_B, gamma_J1, gamma_J2, C * gamma_BH, VDV_0, VVV_1, VDV_2, VDV_3, VVV_4
```

退出码 **0**。`--json` 里同一结果：`"arity": 16`、`"slots": 55`、
`"output_rank": 5`、`"corpus": {"seq": 8628, "tid": "T00710", …}`。
（上面的代码块同样省略了 `方程` / `归一化` / `拓扑签名` / `特化开关` / `说明` 五行。）

### 3.5 为什么这层"只改写 + 给建议"，不替你执行

- `_equivalence` 是**纯接口层**：`normalize_equation` / `find_equivalents` / `explain`
  只返回方程与张量调整方案（`:802-892`）。
- 生产代码里**只有两个入口**碰它：`verify_specialized.py`（`query`，`:432-511`）与
  `test.py`（`_equivalence_of`，`:1268-1292`），两者都是**报告层**。
  `_api.py` / `_execute.py` **完全不经过**它（`git grep` 实证）。因此 `contract()`
  的行为、选路、数值与这条等价层存在与否无关。
- 想真正走上特化路径，必须**你在调用侧**按建议改方程 + 改张量 + **改输出**；
  库不会悄悄改写你的输入——那会让"我传进去的是什么、算出来的是什么的哪个轴"
  彻底失去可追溯性。

### 3.6 覆盖面边界（都是实跑确认的）

| 边界 | 事实 | 证据 |
|---|---|---|
| 多族叠加 | **已覆盖**。三族叠加（例 2）能给出精确映射 | §3.4 例 2 + §4.3 复算 `relerr=2.8e-16` |
| 操作数位置族 | **没有旧版的"只试前 96 个"上限**。`swap(10,11)`、`swap(14,15)` 现在都能映射 | 本机实跑：两者都返回 `operand-order` 映射 |
| 映射算法 | 桶内构造同构，**不枚举候选**；节点上限 `_MAX_SEARCH_NODES=4096`（`:107`）、标签配对上限 `_MAX_LABEL_OPTIONS=512`（`:111`）。触发上限只会漏判 | 源码 + 契约 `test_default_path_never_enumerates_candidate_equations` |
| 靶子范围 | 只映射到**8638 条语料形态**。语料外的形态（如 `ij,jk->ik`、`xy,yz->xz`、`ij->i`、单操作数 `ij->ji`、`ac,em->`）一律返回"无等价写法" | 本机逐个实跑 |
| 省略号 | `_split`（`:205-230`）**直接拒绝**含 `.` 的方程 → 无映射 | 实跑 `...ij,...jk->...ik` → 无等价写法 |
| 不做 | 代数化简、形状相关优化、跨方程重写 | 模块 docstring |
| 显式 `probe=` 路径 | 只有调用方**自己传 `probe`** 时才退回有界枚举（`_MAX_PROBE_CANDIDATES=2048`、`_PROBE_FULL_PERM_LIMIT=6`）。**两个 CLI 都不传** | `_probe_candidates`（`:679-731`） |
| 查询开销 | 首次调用要建语料索引：本机实测 **0.8–1.1 s**（两次独立运行）；之后**需改写**的查询 **0.32–2.54 ms**（arity 6→16 逐档，中位 0.59）；**已命中**（`kind=identity`）的查询 **0.007–0.010 ms**。开销随 arity 增长，**不要按 ~0.3 ms 估大方程** | 本机计时，8 档 arity |

### 3.7 照做之前先落一份副本：`transpose` 是视图，视图会绕过 dedicated

**判断规则**：映射给出 `operand_permutations` 时，**先 `np.ascontiguousarray`**，
否则 `try_dedicated` 的 `C_CONTIGUOUS` 门会把这条路径挡掉。

原因全在源码里，只有两处：

- `operand_permutations` 的语义是 `transpose(perm)`（`pyqcd/einsum/_equivalence.py:147-150`），
  而 `np.transpose` 返回的是**视图**——除恒等与两两交换外都不连续；
- `try_dedicated` 在**进规划之前**就先查这一点（`pyqcd/einsum/_kernel_kupl.py:1452-1453`）：

```python
if any(not operand.flags["C_CONTIGUOUS"] for operand in operands):
    return _fail("dedicated kernel requires C-contiguous operands")
```

于是按"直接 transpose"的直觉做法，即使映射成功，也会在路由链的第一站被拒，
落到下一站——`pyqcd/einsum/_api.py:552-555` 的次序是
`kupl_dedicated` → `kupl_topology` → 通用。这不是"映射失败"，而是
"映射成功后按最自然的用法走，走的路和规范写法不一样"。

下面三条命令（本机 CPU、`HEAD=02fdd54b`、`python` 3.12.4 / base 环境、numpy 2.5.4、
complex128）在语料 `seq=3` 的 `axes` 族变体上逐段复现。变体只改一处：把操作数
`egfh` 反转为 `hgfe`（全轴 >1、秩 4、无重复标签，符合 §3.6 的边界）；形状取自
生产口径 `_bench_scene.dims("distill")`，不用任何缩小档。

**(1) 先看映射给了什么**

```bash
cd /home/zhangxin/PyQCD
export PYTHONPATH=$PWD PYQCD_EINSUM_NOTICE=0
python - <<'PY'
import numpy as np
from pyqcd.einsum import _bench_scene, _equivalence, _wick_tasklist, bench_specialized

task = _wick_tasklist.by_seq()[3]                      # 语料 seq=3
dims = _bench_scene.dims("distill")                    # 唯一生产尺寸口径
rng = np.random.default_rng(0)
build = lambda *s: (rng.standard_normal(s)
                    + 1j * rng.standard_normal(s)).astype(np.complex128)
ops = bench_specialized.operands_for(task.equation, dims, build, task.roles)
variant = task.equation.replace("egfh", "hgfe")        # axes 族：反转该操作数轴序
vops = list(ops)
vops[2] = np.transpose(ops[2], (3, 2, 1, 0))

equ = _equivalence.normalize_equation(variant)
print("variant :", variant)
print("mapped  :", equ.equation, "| kind:", equ.kind)
print("perm    :", equ.operand_permutations)
print("output  :", equ.output_permutation)
print("op2     :", vops[2].shape, "C_CONTIGUOUS",
      vops[2].flags["C_CONTIGUOUS"])
PY
```

退出码 **0**：

```
variant : ambn,codp,hgfe,ce,mo,bdf,nph->ag
mapped  : ambn,codp,egfh,ce,mo,bdf,nph->ag | kind: operand-axes
perm    : (None, None, (3, 1, 2, 0), None, None, None, None)
output  : None
op2     : (100, 100, 100, 4) C_CONTIGUOUS False
```

最后一行是要点：映射要的那个张量此刻是**非连续视图**（`C_CONTIGUOUS False`）。

**(2) 路线 A：照文档直接 `transpose`（视图）**

```bash
cd /home/zhangxin/PyQCD
export PYTHONPATH=$PWD PYQCD_EINSUM_NOTICE=0
python - <<'PY'
import numpy as np
from pyqcd.einsum import (_bench_scene, _equivalence, _kernel_kupl,
                          _wick_tasklist, bench_specialized, contract)

task = _wick_tasklist.by_seq()[3]
dims = _bench_scene.dims("distill")
rng = np.random.default_rng(0)
build = lambda *s: (rng.standard_normal(s)
                    + 1j * rng.standard_normal(s)).astype(np.complex128)
ops = bench_specialized.operands_for(task.equation, dims, build, task.roles)
variant = task.equation.replace("egfh", "hgfe")
vops = list(ops)
vops[2] = np.transpose(ops[2], (3, 2, 1, 0))

equ = _equivalence.normalize_equation(variant)
adjusted = [t if p is None else np.transpose(t, p)      # 视图，未落副本
            for t, p in zip(vops, equ.operand_permutations)]
print("C_CONTIGUOUS:", [t.flags["C_CONTIGUOUS"] for t in adjusted])
got = contract(equ.equation, *adjusted)
if equ.output_permutation is not None:
    got = np.transpose(got, equ.output_permutation)
print("path    :", _kernel_kupl.LAST_INFO.get("path"))
print("failure :", repr(_kernel_kupl.LAST_FAILURE["reason"]))
ref = np.einsum(variant, *vops, optimize="greedy")
print("max|d|  :", float(np.max(np.abs(got - ref))),
      "| rel:", float(np.max(np.abs(got - ref)) / np.max(np.abs(ref))))
PY
```

退出码 **0**：

```
C_CONTIGUOUS: [True, True, False, True, True, True, True]
path    : kupl_topology
failure : 'dedicated kernel requires C-contiguous operands'
max|d|  : 5.820766091346741e-11 | rel: 6.746673840372554e-16
```

**(3) 路线 B：同一份置换，外面套一层 `np.ascontiguousarray`（副本）**

```bash
cd /home/zhangxin/PyQCD
export PYTHONPATH=$PWD PYQCD_EINSUM_NOTICE=0
python - <<'PY'
import numpy as np
from pyqcd.einsum import (_bench_scene, _equivalence, _kernel_kupl,
                          _wick_tasklist, bench_specialized, contract)

task = _wick_tasklist.by_seq()[3]
dims = _bench_scene.dims("distill")
rng = np.random.default_rng(0)
build = lambda *s: (rng.standard_normal(s)
                    + 1j * rng.standard_normal(s)).astype(np.complex128)
ops = bench_specialized.operands_for(task.equation, dims, build, task.roles)
variant = task.equation.replace("egfh", "hgfe")
vops = list(ops)
vops[2] = np.transpose(ops[2], (3, 2, 1, 0))

equ = _equivalence.normalize_equation(variant)
adjusted = [t if p is None else np.ascontiguousarray(np.transpose(t, p))
            for t, p in zip(vops, equ.operand_permutations)]   # 落一份连续副本
print("C_CONTIGUOUS:", [t.flags["C_CONTIGUOUS"] for t in adjusted])
got = contract(equ.equation, *adjusted)
if equ.output_permutation is not None:
    got = np.transpose(got, equ.output_permutation)
print("path    :", _kernel_kupl.LAST_INFO.get("path"))
print("failure :", repr(_kernel_kupl.LAST_FAILURE["reason"]))
ref = np.einsum(variant, *vops, optimize="greedy")
print("max|d|  :", float(np.max(np.abs(got - ref))),
      "| rel:", float(np.max(np.abs(got - ref)) / np.max(np.abs(ref))))
PY
```

退出码 **0**：

```
C_CONTIGUOUS: [True, True, True, True, True, True, True]
path    : kupl_dedicated
failure : ''
max|d|  : 5.820766091346741e-11 | rel: 6.746673840372554e-16
```

对照：

| 路线 | 传进去的张量 | `LAST_INFO['path']` | 数值（vs `np.einsum`，`optimize="greedy"`） |
|---|---|---|---|
| A 直接 `transpose` | 1 个非连续视图 | **`kupl_topology`** | `rel=6.747e-16` |
| B `ascontiguousarray` | 全部连续 | **`kupl_dedicated`** | `rel=6.747e-16` |

两条路线的**数值一致**（只差浮点舍入），差别只在**路线**；而路线一旦错了，
"用映射换特化收益"这件事就没发生。这个格子上两条路线的耗时曾差 75×
（`logs/einsum-equivalence-gain-20261116.txt` §4.2 的 seq 3 `egfh` 行：
`kupl_topology` 117.776 ms vs `kupl_dedicated` 1.570 ms）——**该数字来自那份日志，
本手册未复跑**，只作"差多少"的量级参考。

**代价**：副本要**额外一份同大小内存**与**一次拷贝时间**，所以
**只有当你确实要重路径收益时才值得**；小方程上拷贝本身可能就把收益吃掉，
不确定就先按 §5 的方法量一次再决定。

**边界**（与 §3.5 同一口径）：本层的映射只给"方程与张量调整方案"，**是否落副本
由调用方决定**——库不会替你复制，也不会替你改写输入。

---

## 4. 如何验证映射是否可信（正确性必读）

用户最关心的是"照着改完，结果还对吗"。这一节给出一套**可粘贴、可复现**的做法。

### 4.1 三个字段各自的用法

| 字段 | 怎么用 | 缺省（`null`）含义 |
|---|---|---|
| `operand_order` | `adjusted[i] = ops[operand_order[i]]`——**先按它重排张量列表** | 次序不变 |
| `operand_permutations[j]` | 对**重排后**位置 j 的张量做 `np.transpose(t, perm)` | 该路不转 |
| `output_permutation` | 对 `contract(等价方程, *adjusted)` 的结果做 `np.transpose(result, perm)` | 输出轴序已一致 |

三个字段都是**纯重排**，不含改名：`equivalence.equation` 用的是你原来的字母，
只是把结构写法换成语料的那种。所以照做即可，不需要翻译字母。

### 4.2 硬规则：输入变了，输出必须跟着变

一句话：**只改输入不改输出，拿到的是轴序错了的答案**；而它常常**不会报错**
（§3.3 的实测表已经量化了这一点）。

所以任何一次"用等价写法换取特化路径"的操作，都必须做完整的三步：

1. 按 `operand_order` 重排张量列表；
2. 按 `operand_permutations` 逐路 `transpose`；
3. 按 `output_permutation`（若非 `null`）对**结果** `transpose`。

漏掉第 3 步在轴长不同时会被形状检查抓到，在轴长相同（语料里到处都是）
时**静默出错**。

### 4.3 可粘贴运行的复算示例

```python
import numpy as np
import pyqcd.einsum as pe
from pyqcd.einsum import verify_specialized

original = ("cwdx,eufv,mknl,oapb,qgrh,istj,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx"
            "->GMHew")
m = verify_specialized.query(original)["equivalence"]

terms, out = original.split("->")
terms = terms.split(",")
sizes = {c: 2 + i for i, c in enumerate(out)}      # 输出段轴长两两不同
for t in terms:
    for c in t:
        sizes.setdefault(c, 2)                     # 其余统一 2

rng = np.random.default_rng(0)
ops = [rng.standard_normal(tuple(sizes[c] for c in t)).astype(np.complex128)
       for t in terms]

order = m["operand_order"] or list(range(len(ops)))
perms = m["operand_permutations"] or [None] * len(ops)
adjusted = [ops[order[i]] if perms[i] is None
            else np.transpose(ops[order[i]], perms[i]) for i in range(len(ops))]

got = pe.contract(m["equation"], *adjusted, backend="numpy")
if m["output_permutation"]:                        # 输入变了，输出必须跟着变
    got = np.transpose(got, m["output_permutation"])

expected = pe.contract(original, *ops, backend="numpy")
print("mapped :", m["equation"])
print("kind   :", m["kind"])
print("shape  :", got.shape, "vs", expected.shape)
print("relerr : %.3e" % (np.abs(got - expected).max() / np.abs(expected).max()))
```

实跑输出（本机，`HEAD=aeea4faf`）：

```
mapped : cwdx,eufv,mknl,oapb,isjt,qgrh,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHew
kind   : operand-order+axes+output-axes
shape  : (2, 3, 4, 5, 6) vs (2, 3, 4, 5, 6)
relerr : 2.802e-16
```

**反证**（把第 3 步去掉）：

```python
raw = pe.contract(m["equation"], *adjusted, backend="numpy")
print(raw.shape)      # (3, 2, 4, 5, 6) —— 少了回填，形状直接不对
```

这就是 §4.2 那条硬规则的现场演示。

> 这个脚本判的是**跨方程的等价性**，它既不是"单方程对不对"也不是"快不快"，
> 因此不受 §0 那条"单方程入口唯一"的约束；但它**不能**用来替代
> `python -m pyqcd.einsum.test` 做数值/性能取数。

### 4.4 独立验证器：`python -m pyqcd.testing.einsum.verify_equivalence`

不想自己写脚本、或者想验证**整层**而不是一条，用这个。它**不 import**
`_equivalence` 的内部函数，方程拆分、置换语义、形状推导、收缩、误差判据全部
自己重写，并有"基准自检"一层专门防止"验证器坏了却去指责被测代码"。

```bash
python -B -m pyqcd.testing.einsum.verify_equivalence
python -B -m pyqcd.testing.einsum.verify_equivalence --fuzz 80 --seed 7
python -B -m pyqcd.testing.einsum.verify_equivalence --quick
```

本机实跑（默认档，2.0 s）汇总：

```
单层 A: 10/10   两两组合 B: 7/7   三层 C: 2/2
反证 D: 15/15（漏层后均不等价；覆盖 order=5 / axes=5 / out=5）
方向 E: 穷举真值后非自逆 5/5 条正向正确且逆置换必错；自逆 2 条标注**方向不可区分**；无输出置换 1 条；未验证 0 条
模糊 F: 命中率 40/40（失败 0 条；另有 0 条仅结构层通过、数值参考超预算；映射落回**原语料形态**的 40 条；耗时 0.4 s）
边界 G: 18/18
有界性: 最坏 0.00 s（8 操作数 秩8 输出段）

合计: PASS 107 / FAIL 0 / 未验证 0（共 107 条）
结论: 全部通过
```

退出码 0 = 全过，1 = 有 FAIL；`未验证` 单列，**既不算通过也不算失败**。

七个维度分别答什么：

| 维度 | 答什么 |
|---|---|
| A 单层 | 每一族单独作用一次都对 |
| B 两两组合 | 两族叠加都对 |
| C 三层 | 三族叠加（就是 §3.4 例 2）对 |
| D 反证 | **故意漏掉任一层就不再等价**——证明每一层都不可省 |
| E 方向 | 穷举全部输出置换找唯一真值；非自逆置换还用逆置换复算，证明方向确实可区分 |
| F 模糊 | 从 8638 条语料里随机抽样、随机叠加 1–3 层，固定 seed 可复现 |
| G 边界 | 单操作数 / 秩 0 / 秩 1 / 对角标签 / 长度 1 轴 / 非法输入 / `explain` 与 `normalize` 一致 |

E 维度特别值得注意：它把输出段轴长强制设为**两两不同**，否则方向错误会被形状
对称掩盖——这正是 §3.3 表格里那两条 `自逆` 用例被显式标注"方向不可区分"的原因。

另外，**单层契约**（改 `_equivalence.py` 必须同步的那份）在：

```bash
python -B -m pyqcd.testing._einsum_equivalence_contract
# 本机实跑：41 passed, 0 skipped, 0 failed
```

### 4.5 常见的"看起来对、其实错"清单

| 症状 | 原因 | 怎么查 |
|---|---|---|
| 形状对、数值全错 | 只改了输入没改输出，且输出轴长恰好相同 | 用两两不同的输出轴长跑一次（§4.3 脚本默认这么做） |
| 形状不对 | 漏了 `output_permutation`，或把它当成"正向置换"自己求了逆 | 直接喂映射给的值，它**已经是** `transpose` 该用的那个 |
| 改完还是 MISS | 只做了一族、而映射要求多族 | 看 `kind`——`+` 连接的每一族都要做 |
| `operand_permutations` 长度对不上操作数个数 | 拿原始张量列表去应用它（它是按**改写后**次序排的） | 先 `operand_order`，再逐位置转 |
| `equivalence` 字段是 `rewritten: false` | 查过了，但真的没有等价写法 | 不是配置错，走通用路径是正常结果 |

---

## 5. 我想跑性能对照

> **先读 §1.6**：`--scene` 的四个档全部是 `_label_sizes` 口径，**不能当生产尺寸**。
> 本节只讲"怎么测"，不给任何可外推的性能数字。要生产规模的结论，必须先用
> `--shape` 显式传入 `_bench_scene.distill_tensor_shapes()` 的真实形状。

### 5.0 `optimize` 只有 6 个对外取值（先看这个）

> **qcd-einsum 对外提供的 `optimize` 取值只有 6 个：`auto / none / greedy / optimal / dp / branch`。**
> `auto` 为默认值（选择器：按算子数查表，≤8 算子且 opt_einsum 可用时改用其 DP 结果）。
> `none` = 按方程原序（亦可用 `optimize=False` 表示）。`branch` = 全宽分支定界（≡ 内部 `branch_all`）。
> 内部另有窄宽变体 `branch_1` / `branch_2`，**仅供 `auto` 使用，不属于对外取值**。
> `auto-hq`（≡`auto`）、`eager`/`opportunistic`/`random-greedy`/`random-greedy-128`（≡`greedy`）
> 为**退化别名**，不属于对外约定；`branch-1`/`branch-2`/`branch-all` 属**待清除的历史名字**。

写 `optimize=` 或 `--optimize` 时只从这 6 个里挑。注册表里另外还有 9 个名字，它们**不是**
对外取值；照抄旧文档/旧脚本时对号入座：

| 名字 | 现状标注 |
|---|---|
| `auto-hq` | 退化别名 ≡ `auto`（同一函数对象） |
| `dynamic-programming` | 退化别名 ≡ `dp`（同一函数对象） |
| `eager` | 退化别名 ≡ `greedy`（同一函数对象） |
| `opportunistic` | 退化别名 ≡ `greedy`（同一函数对象） |
| `random-greedy` | 退化别名 ≡ `greedy`（同一函数对象） |
| `random-greedy-128` | 退化别名 ≡ `greedy`（同一函数对象） |
| `branch-all` | 待清除（≡ `branch`，全宽；注册表里仍是独立名字） |
| `branch-1` | 待清除（窄宽变体 `nbranch=1`，仅供 `auto`） |
| `branch-2` | 待清除（窄宽变体 `nbranch=2`，仅供 `auto`） |

两个容易踩的点：

1. **`none` 不在注册表里**。`contract(optimize="none")` 与 `optimize=False` 是同一条路
   （整条链一次 n 元 einsum，`path = [(0,1,...,n-1)]`），由 `_api._normalize_optimize`
   归一，`paths.get_path_fn("none")` 本身会抛 `KeyError`——这不是 bug。
2. `--compare-optimize` 是**参照侧**的开关，名字空间属于 numpy / opt_einsum，
   与上面这 6 个无关（见 §5.2）。

判据与回归：`python -B -m pyqcd.testing._einsum_optimize_surface_contract`
（4 项：解析、数值 rel ≤1e-12、集合恰好这 6 个、三份文档逐字一致）。

### 5.1 `--compare` 四档

| 档 | 参照是什么 | 实跑看到的 `reference` |
|---|---|---|
| `numpy`（缺省） | `np.einsum(optimize=<ref opt>)` | `np.einsum(optimize='greedy')` |
| `opt_einsum` | `oe.contract(optimize=<ref opt>)` | `opt_einsum.contract(optimize='dp')` |
| `backend` | **另一个后端**跑同一批张量（跨后端互证） | `pyqcd.contract(backend='cupy', cupy, specialize=auto)` |
| `none` | 不比，只跑主路径 | `PASS` 行不带 `rel=` 与 `vs …` |

`--compare backend` 是唯一能同时验证"两条原生路径都给对答案"的模式
（主路径跑 `--device`，参照在另一个后端上跑）。它要求**另一个后端可导入**：
本机 cupy 可用，所以 `--device cpu` 时参照落在
`pyqcd.contract(backend='cupy', …)`，退出码 0；没有可用的另一后端时这条模式用不了，
请退回 `numpy`/`opt_einsum`。

### 5.2 ⚠️ `--compare-optimize` 的陷阱：两个库的名字不通用

`--compare-optimize` **只管参照**，主路径永远看 `--optimize`。

```bash
# 会失败：numpy 侧只有 greedy / optimal                → 退出码 2
python -m pyqcd.einsum.test --equation "ij,jk->ik" --shape "[4,4],[4,4]->[4,4]" \
    --compare numpy --compare-optimize dp
```
```
error: --compare numpy 不接受 --compare-optimize 'dp'：np.einsum(optimize=...) 实测只认 ('greedy', 'optimal') 两个名字，其余抛 KeyError('Path name %s not found')。
  提示: 想要 'dp' 请用 --compare opt_einsum（该名字在 opt_einsum 的路径注册表里），不要把它硬映射成一个语义不同的 numpy 名字
```

**这不是你写错了**，也不是"静默退回 greedy"——是刻意的用法错误，
因为跨库硬映射会得出语义不同的对照结论。

| `--compare` | 可用的 `--compare-optimize` | 不支持的会怎样 |
|---|---|---|
| `numpy` | `greedy`、`optimal`（`auto`→`greedy`） | 退出码 2 |
| `opt_einsum` | `auto`、`greedy`、`dp`、`optimal`、`branch-all`、`branch-2`、`branch-1` | 退出码 2 |
| `backend` / `none` | 没有可设的优化器 | 只打一行 warning，**不报错**，退出码仍是 0 |

```bash
# 正确做法：dp 配 opt_einsum
... --compare opt_einsum --compare-optimize dp
# ref opt  : 要 dp → 参照实际用 'dp'
# reference: opt_einsum.contract(optimize='dp')

# backend / none 下无效（仅提示）
... --compare none --compare-optimize dp
# ref opt  : 要 dp → 参照实际用 不适用
# warning: --compare-optimize 'dp' 对 --compare none 无效（该模式没有可设的优化器）
```

报告里同时有 `compare_optimize`（要什么）与 `compare_optimize_effective`（实际用了什么）。

### 5.3 `--specialize false`：看通用路径的代价（**本手册不给数字**）

```bash
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" \
    --scene distill --specialize false --verbose true
# specialize: false（实际 False）
# kernel   : 无原生命中（通用路径）
# route    : 通用（无原生命中：Python 分步 / numpy.einsum / BLAS）
```

用法是对的：`--specialize false` 是排查原生内核问题的第一手段——先拿到"完全不经
原生"的基线，再和 `auto` 比，差值就是"原生尝试 + 原生执行"的净代价。

**但本手册不给出这个差值的数字。** 上一版手册引用过一组
（`auto` 与 `false` 在同一 `--scene distill` 形状上 route 完全相同、耗时差 2.45×），
那组数据用的是 `_label_sizes` 口径的形状，按 `pyqcd/einsum/AGENTS.md` §6.1
**已被判定作废**，不得作为性能结论。因此这里标记为 **`unverified`**：

> 观察（待生产尺寸复核）：同一个方程在 `--specialize auto` 与 `--specialize false`
> 下可能给出**完全相同的 `route` 行**（都是通用），但耗时不同——说明"原生拒绝"
> 本身不是免费的（签名匹配、计划推演、各道守卫都要花时间）。**倍率 `unverified`**，
> 在拿到 `distill_tensor_shapes()` 形状上的实测之前不得外推。

要得到可信数字，把 `--shape` 换成真实形状后再测。

**注意 `--specialize true` 不是"一定会走原生"**：它是"强制允许特化"，
原生侧的门（pack-bound / dtype 策略 / arena）照样会拒，拒绝时 route 行如实写通用。
本机 seq=129 实跑：

```
specialize: true（实际 True）
kernel   : 无原生命中（通用路径）
route    : 通用（无原生命中：Python 分步 / numpy.einsum / BLAS）
PASS aibj,ckdl,eqfr,msnt,oupv,gwhx,ce,og,ik,su,bdf,nph,jlr,tvx->aqmw rel=7.17e-16 …
```

同时 stderr 给出拒绝理由（这条是**唯一**能拿到原生拒绝原文的地方）：

```
RuntimeWarning: pyqcd.einsum: specialised kernel did not run for topology:0177583f
(topology declined: pack-bound on CPU (3456077824 pack bytes, 216326912 FLOPs):
 the layout traffic dominates the arithmetic; the generic path handles it). …
```

（上面只引用耗时与倍率之外的字段：那是 `--scene distill` 形状，属性见 §1.6。）

### 5.4 为什么必须先丢 warmup（而且不允许设成 0）

首调包含 plan 缓存、原生库绑定（必要时含 JIT），与热态不是一个量级（实测差近百倍），
具体数字与论证只在 `pyqcd/einsum/AGENTS.md` §9.0 的 warmup 论证里——**本手册不复述、也没有复跑**，
它只用来说明"warmup 不可省"，不是性能结论。因此本入口：

- `--warmup` 缺省 1，**传 0 直接报错**（退出码 2，见 §1.5）；
- 计时只取 `--rounds` 次中位；
- 内存采样点（构造前 / 构造后 / 收尾）**都在计时循环之外**，不污染中位。

---

## 6. 我要把结果喂给脚本

### 6.1 退出码（`python -m pyqcd.einsum.test`）

| 码 | 含义 | 本机实跑 |
|---:|---|---|
| 0 | 通过（`ok=true`），或 `--compare none` 跑完 | `--scene distill` 缺省、`--compare backend`、JSON 可写 |
| 1 | 数值或形状不符 | `--dtype float32` 用缺省 `--atol 1e-12` |
| 2 | 参数或环境错误（含 `--json` 写不进去、`--compare-optimize` 不兼容） | `--warmup 0`、省略 `--shape` 且语料无此方程、`--json` 缺参数、`--json /nonexistent/x.json`、`--compare numpy --compare-optimize dp` |

```bash
# 1 的典型来源：单精度用缺省 --atol=1e-12
python -m pyqcd.einsum.test --scene distill --dtype float32 \
    --equation "cadb,egfh,minj,okpl,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek"
# FAIL cadb,egfh,minj,okpl,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek
#   rel=2.117e-07 > atol=1e-12
#   actual shape=[100, 100] expected=[100, 100]        → 退出码 1

# 同一命令加 --atol 1e-4
# PASS cadb,egfh,minj,okpl,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek rel=2.12e-07 \
#      vs np.einsum(optimize='greedy')  10.617ms (ref 12.386ms@greedy, 0.86x)  rss+39MiB  → 0
```

`--atol` 是**相对**容差（按 `max|expected|` 归一），不是逐元素绝对阈值。
`float32/complex64` 常需 `--atol 1e-6~1e-4`，这是精度口径不是 bug。

### 6.2 `--json` 是**路径**不是开关

```bash
python -m pyqcd.einsum.test … --json /path/out.json     # 打印 "wrote /path/out.json"，退出码 0
python -m pyqcd.einsum.test … --json                    # error: argument --json: expected one argument → 2
python -m pyqcd.einsum.test … --json /nonexistent/x.json # error: 无法把 --json 写到 … → 2
```

### 6.3 字段表（实跑抓取，`test.py --json`）

| 字段 | 类型 | 含义 |
|---|---|---|
| `ok` | bool | 容差 + 形状是否通过（**脚本只认这个**） |
| `rel` | float | 相对误差（按 `max\|expected\|` 归一） |
| `atol` / `rtol` / `tolerance` | float | 本次实际使用的容差与判定用的合成阈值（`atol + rtol`） |
| `max_abs_diff` | float | 最大绝对差（诊断用） |
| `out_shape` | list[int] | 期望输出形状 |
| `actual_shape` | list[int] | 实得输出形状 |
| `shapes` | list[list[int]] | 各操作数形状（按方程从左到右） |
| `kernel_info` | dict | `{backend: {path, specialized?, steps?, dtype?, copied?}}`，可能为空 `{}` |
| `specialize` | str | 请求值（`auto`/`true`/`false`） |
| `specialize_effective` | bool | 开关实际是否打开。**注意：true ≠ 命中原生** |
| **`equivalence`** | dict | `_equivalence.explain()` 原样：`equation` / `rewritten` / `operand_order` / `operand_permutations` / `output_permutation` / `description` / `kind`。**只在开启且方程未命中任何特化路径时出现**；关闭或已 HIT 时**缺该键** |
| `compare_optimize` | str | 参照优化器**请求值** |
| `compare_optimize_effective` | str \| null | 参照**实际使用**值；`null` = 该模式无优化器 |
| `reference` | str | 参照实现的可读描述 |
| `reference_seconds` | float | 参照耗时 |
| `seconds` | float | 主路径中位耗时 |
| `equation` / `device` / `dtype` / `backend` / `compare` / `optimize` | str | 回显配置 |
| `rounds` / `warmup` | int | 计时口径（`--seed` 是 CLI 选项但**不**进 JSON） |
| `n_operands` / `input_elements` / `output_elements` | int | 规模 |
| `install` | dict | `{version, package}`，用来确认"跑的是哪个 pyqcd" |
| `memory` | dict | `inputs_delta` / `peak`（RSS 与显存池） |
| `shape_source` | str | 形状来源说明（**这是唯一会被 `--language` 改写的人读字段**） |

`equivalence` 字段**在 HEAD 已存在**（上一版手册说"不存在"是当时的状态，已过期）。
它**不进计时区域**、不改 `contract()` 入参，纯报告（`test.py:1185-1189`）。

### 6.4 稳定性契约

- **`--json` 的字段集合是稳定契约**，`--language` 不改它。
  本机实测同一命令 `--language zh-cn` vs `en`：`sorted(keys)` 完全相同，
  **只有 `shape_source` 的值被翻译**（`形状由语料 roles 反推（scene=distill）` →
  `shapes inferred from corpus roles (scene=distill)`）；`kernel_info` 逐字相同。
- **注意例外**：`equivalence.description` 与 `verify_specialized` 的 `reason`
  **始终是中文**——它们来自 `_equivalence.explain()` 的中文字面量，`--language`
  不翻译结构化字段。要本地化请自己查 `MESSAGES`，或用
  `verify_specialized` 的 `reason_key` / `reason_args` 自行渲染。
- `--language` **不写任何环境变量**，不影响同进程后续调用。
- 若要用脚本判断"走的哪条路"，请读 `kernel_info[*]['path']`（缺省即无原生），
  **不要**去解析 `route` 那句人话（它随 `--language` 变）。

---

## 7. 环境变量总表

| 变量 | 缺省 | 作用范围 | 典型用法 | 性质 |
|---|---|---|---|---|
| `PYQCD_EINSUM_NOTICE` | **开**（除 `0/false/off/no` 外都算开） | `_specialize.announce`（命中特化时）与 `announce_generic`（未命中时）的**一次性提示**，每进程每形态一次 | `PYQCD_EINSUM_NOTICE=0` 静音 | **日志开关**。不改选路、不改数值 |
| `PYQCD_EINSUM_SPECIALIZE` | **开** | 进程级专用路径总开关（`_specialize.enabled()`）。`=0` 后**同进程所有** `contract()` 都走通用 | `PYQCD_EINSUM_SPECIALIZE=0` 做整进程对照 | **能力/选路开关（进程级）**。只想改**一次**调用请用 `--specialize false` |
| **`PYQCD_EINSUM_EQUIVALENCE`** | **开**（`"1"`；只有 `0/false/off/no` 关闭） | `_equivalence.enabled()`，**调用时**读环境变量。关闭后 `normalize_equation()` 返回 `None`、`find_equivalents()` 返回 `[]`，两个 CLI 的等价段整段消失、MISS 原因回到引入本功能前的原文 | `PYQCD_EINSUM_EQUIVALENCE=0` 关掉；`--equivalence true` 可**压过**它 | **纯报告开关**。等价映射层本身不参与 `contract()`，关闭不改任何数值 |
| `PYQCD_EINSUM_LANG` | **未设** | `_specialize.notice_language()`（库提示）与两个 CLI 的 `--language auto` | `PYQCD_EINSUM_LANG=en` / `zh` | **只改文案**。两条入口的 `auto` 语义不同（见 §7.1） |
| `PYQCD_EINSUM_TOPOLOGY_REAL` | **未设**（`float64`/`float32` 被策略排除在拓扑入口外） | `_kernel_kupl._topology_dtype_declined`：`=1`（`true/on/yes`）解除该策略 | `PYQCD_EINSUM_TOPOLOGY_REAL=1` 做 A/B | **绕过选路做对照实验**，**不是**能力开关 |
| `PYQCD_EINSUM_KERNEL_KUPL` | **开** | KUPL 原生入口（dedicated + topology）是否参与下发 | `=0` 完全不用 KUPL 原生库 | **能力开关（模块级）**。与 `--specialize false` 的区别：后者是"这次别用"，前者是"这个入口别用" |
| **`PYQCD_EINSUM_KERNEL_FORCE`** | **未设**（= `auto`） | 强制指定专用内核，取值 `auto` / `generic` / `dedicated` / `topology` / `backend`。**调用时**解析。显式 `--kernel-force` / `kernel_force=` 可**压过**它 | `=topology` 只准跑当前后端的拓扑入口；`=generic` 完全跳过专用内核；`=backend` 直接调后端 einsum | **选路开关**。不改数值；强制失败会抛 `KernelUnavailableError` 而不是 warning（见 §11） |
| `PYQCD_EINSUM_TOPOLOGY_TRACE` | **关** | 原生 C++ 侧的逐步诊断行（`topology-gemm-step …`、`topology-specialized: …`） | `=1` 排查 v3 步表是否真的在跑 | **诊断日志**。`trace_enabled()` 在原生侧**静态缓存**：同进程里改环境变量已经晚了，必须一开始就设 |
| `PYQCD_EINSUM_PROFILE` | **开**（`"1"`；`0/false/off/no` 关闭） | 配置表总开关（`_profile_cache`），是 `contract(profile=)` 与 CLI `--profile {auto,true,false}` 的缺省来源。**调用时**求值；关闭时**不读不写、连目录都不建** | `PYQCD_EINSUM_PROFILE=0` 整进程关掉 | **只影响选路建议**：普通调用**只读**（未 tune 则表为空），命中最多替你填 `kernel_force=` / `optimize=` 或按等价行改写；方程、dtype、形状、数值都不动（见 §13） |
| `PYQCD_EINSUM_PROFILE_DIR` | **未设**（用当前工作目录下的 `./.cache`） | 配置表目录；显式 `contract(profile_dir=)` 与 CLI `--profile-dir` 压过它。**只在首次解析时用得上**：进程一旦钉住目录，省略 `profile_dir` 不会回退到这里（见 §13.2） | `PYQCD_EINSUM_PROFILE_DIR=/tmp/einsum-profile` | **落盘位置**。只在 **tune 写入时**创建（普通调用不建目录）；不可写则降级为"不缓存"，不外抛 |

### 7.1 `--language auto` 不读 locale（反直觉点，必须记住）

两条入口的 `auto` **刻意不同**：

| 入口 | `auto` 的行为 | 为什么 |
|---|---|---|
| CLI（`test.py` / `verify_specialized.py`） | **只在 `PYQCD_EINSUM_LANG` 显式设了**时跟随它；未设 → **保持中文** | 跟随 locale 会让本机（`LANG=C.UTF-8`）不传选项的输出**从中文静默变英文**，破坏历史输出 |
| 库提示（`_specialize.notice_*`） | `PYQCD_EINSUM_LANG` 优先，**否则跟随 `LC_ALL`/`LC_MESSAGES`/`LANG`**；都没有 → 英文 | 英文环境的用户不该在日志里看到中文 |

实跑证据：

```bash
# locale 是 C.UTF-8，但 CLI 的明细仍是中文（auto 不读 locale）
LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m pyqcd.einsum.test \
    --equation "cadb,menf,ac,em,bd,fn->" --scene distill --verbose true
# shape    : 共 6 个操作数形状（按方程中从左到右的顺序）
# specialize: auto（实际 True）
# route    : 特化 · 原生 qcd_kupl_dedicated_apply（path=kupl_dedicated）

# 库提示跟随 locale → 中文（NOTICE 没关）
LANG=zh_CN.UTF-8 LC_ALL=zh_CN.UTF-8 python -m pyqcd.einsum.test \
    --equation "ij,jk->ik" --shape "[4,4],[4,4]->[4,4]"
# [pyqcd.einsum] 通用路径: 未命中特化形态 (ij,jk->ik)；执行的是通用分步收缩。…

# 同一个环境变量让 CLI 变英文（对照第一块的三行）
PYQCD_EINSUM_LANG=en python -m pyqcd.einsum.test … --verbose true
# shape    : 6 operand shape(s), left to right, in the equation's own order
# specialize: auto (effective True)
# route    : specialized - native qcd_kupl_dedicated_apply (path=kupl_dedicated)

# 命令行优先于环境变量，但**不改**库提示的语言
PYQCD_EINSUM_LANG=en python -m pyqcd.einsum.test … --language zh-cn
# [pyqcd.einsum] specialized path: topology:a1c49862 …   ← 库提示仍是英文
```

### 7.1b 一条提示只用一种语言（`[pyqcd.einsum]` 前缀不变）

库内提示不再出现"英文标签 + 中文解释"拼在同一行：**整句**按
`_specialize.notice_language()` 单语输出——显式 `--language`（走
`_specialize.set_language`，只改进程内状态、**不写环境变量**）> `PYQCD_EINSUM_LANG` >
locale（`LC_ALL`/`LC_MESSAGES`/`LANG`）。

```bash
# 英文：整句英文
PYQCD_EINSUM_PROFILE=0 PYQCD_EINSUM_LANG=en python -m pyqcd.einsum.test \
    --equation "cadb,menf,ac,em,bd,fn->" --scene distill --rounds 1 --warmup 1
# [pyqcd.einsum] specialized path: topology:a1c49862 [match: topology] [kernel: kupl_dedicated] corpus form (…)

# 中文：整句中文
PYQCD_EINSUM_PROFILE=0 PYQCD_EINSUM_LANG=zh python -m pyqcd.einsum.test \
    --equation "cadb,menf,ac,em,bd,fn->" --scene distill --rounds 1 --warmup 1
# [pyqcd.einsum] 专用路径: topology:a1c49862 [match: topology] [kernel: kupl_dedicated] 真实语料形态（…）
```

**脚本可解析的部分两种语言逐字相同**：`[pyqcd.einsum]` 前缀、`topology:<摘要>` 形态名、
`[match: <形态族>]` 与 `[kernel: <实际路线>]` / `[plan-cache]` 三个机器段，
以及 `route :` / `kernel :` 行的字段名与 `auto|generic|none|unknown` 这些取值都**不翻译**
（§6.4 稳定性契约）。跟着语言变的只有标签与说明文字。

`[match: …]` 与 `[kernel: …]` 必须分开读：语料形态默认先试展开式 `*_dedicated` 步表、
失败才试 `*_topology`（`_api._specialized_kernel_attempts` 的次序），所以
"`[match: topology]` + `[kernel: kupl_dedicated]`"是**正常**结果——命中的是拓扑形态，
执行的是 dedicated 内核。`[kernel: …]` 的取值来自执行侧 `LAST_INFO['path']`，
拓扑形态下因此可能是 `kupl_topology`（真走 v3 步表）或 `kupl_topology_v2`（v2 执行器完成）。

已知例外（实测）：`--verbose` 下"等价改写已跳过"那条（`_api._note_equivalence_skipped`）
目前仍是中文，不受 `PYQCD_EINSUM_LANG` 影响。

### 7.2 存在但本手册不展开的变量

以下变量在源码里可查（`git grep` 定位；本版逐名核对过全部存在），属于构建/线程/探测层：

| 变量 | 读取位置 |
|---|---|
| `PYQCD_EINSUM_SPECIALIZE_VERBOSE` | `_specialize.py:46` |
| `PYQCD_EINSUM_DISABLE_FUSED` | `_specialize.py:54` |
| `PYQCD_EINSUM_KERNEL_CUDA` | `_kernel_cuda.py:25` |
| `PYQCD_EINSUM_NATIVE` / `_LIB` / `_KUPL_LIB` / `_CUDA_LIB` / `_CBLAS_LIB` | `_native.py:24,39`、`kupl/einsum.py:82-83`、`_threads.py:53` |
| `PYQCD_EINSUM_THREADS` / `_OMP_THREADS` / `_BLAS_THREADS` | `_threads.py:50-52` |
| `PYQCD_EINSUM_POOL` / `_CONTEXT_CACHE_LIMIT` | `_execute.py:28`、`_api.py:55` |
| `PYQCD_EINSUM_SHAPE_PATH_MAX_BYTES` | `_shape_paths.py:45` |
| `PYQCD_EINSUM_NEGATIVE_CACHE` / `_DEDICATED_PLAN` / `_KUPL_PLAN` | `_kernel_kupl.py:58-59`、`_api.py:1278` |

---

## 8. 我的方程改了还是 MISS，怎么办（决策树）

```
verify_specialized 报 MISS
├─ 有「等价写法」行
│   ├─ 看 kind：`+` 连接的每一族都要做（位置 / 轴序 / 输出轴序）
│   ├─ 记住：输入变了，输出该转的一定要转（§3.3、§4.2）
│   └─ 改完按 §4.3 复算一次，relerr 应在 1e-15 量级
├─ 没有「等价写法」行
│   ├─ 方程确实不在 8638 条语料里 → 走通用路径是正常结果，不是配置错
│   ├─ 确认它与某条语料式等价（自己看拓扑签名）→ 现在的实现已覆盖
│   │   位置 / 轴序 / 输出轴序 / 复合四类；四类都试过还不行，就不再是
│   │   这一层能解决的范围（例如含省略号，或需要代数化简）
│   └─ 怀疑代数化简 → 本层不做，别等
├─ 等价段整段没出现
│   ├─ 是不是 `--equivalence false` 或 `PYQCD_EINSUM_EQUIVALENCE=0`？
│   ├─ 是不是方程已经 HIT（已命中的方程不查等价写法，这是刻意的）
│   └─ 都不是 → 看 `--json` 的 `equivalence` 是否存在，用 `rewritten` 区分
│       "查过没有" 与 "根本没查"
└─ 报 HIT，但 test.py 的 route 是通用
    ├─ 看 kernel 行：为空 → 原生门拒绝了（pack-bound / dtype 策略 / arena）
    ├─ 看 stderr 的 RuntimeWarning：拒绝原因写在里面
    └─ 用 --specialize false 量一次通用基线，再决定值不值得追原生（§5.3）
```

---

## 9. 能力边界（诚实版）

### 9.1 已支持（每条都有实跑证据）

| 能力 | 状态 | 本机证据 |
|---|---|---|
| `python -m pyqcd.einsum.test` | **已可用** | §1、§5、§6 |
| `--specialize {auto,true,false}` | **已可用** | §1.2、§5.3；`true` 不保证命中原生 |
| `--compare {numpy,opt_einsum,backend,none}` | **已可用** | 四档都实跑（§5.1） |
| `--compare-optimize` | **已可用** | 四档 + 两条错误路径（§5.2） |
| `--language {auto,en,zh-cn}`（**两个 CLI 都是**） | **已可用** | `test.py` 的 `MESSAGES` 97 key × 中英两套、`verify_specialized` 的 46 key × 中英两套，本机核对无缺；两入口的 `--help` 与输出都实跑过（§2.1、§7.1） |
| `reason_key` / `reason_args`（语言无关字段） | **已可用** | §2.3 的 `--json` 实跑 |
| **`--equivalence {auto,true,false}`（两个 CLI 都是）** | **已可用** | §2.1、§2.2、§5.3；`test.py:1820-1823`、`verify_specialized.py:696-699` |
| **`PYQCD_EINSUM_EQUIVALENCE`** | **已可用** | `_equivalence.py:99`；`=0` 与 `--equivalence false` 都实跑，行为一致（§2.3、§7） |
| **等价映射：单族（位置 / 轴序 / 输出轴序）** | **已可用** | §3.4 例 1；契约 41/41；验证器 A 10/10、B 7/7 |
| **等价映射：多族复合（含三族叠加）** | **已可用**（上一版标注"未覆盖"，**已过期**） | §3.4 例 2；验证器 C 2/2、D 15/15（漏任一层即不等价） |
| **操作数位置族无 96 上限** | **已可用**（上一版的"96 上限"已随重写消失） | 本机 `swap(10,11)` / `swap(14,15)` 均返回映射（§3.6） |
| 映射层的符号回放自检 | **已可用**（保证不误给映射） | `_equivalence.py:495-497`、`:508-511` |
| 独立验证器 `python -m pyqcd.testing.einsum.verify_equivalence` | **已可用** | §4.4 实跑 `PASS 107 / FAIL 0 / 未验证 0` |
| `kupl_topology` / `kupl_topology_v2` 的 route 展示 | 代码支持，**本机未复现** | 源码 `test.py:1724-1730`；§1.3 标注"取自源码，不是实跑" |
| `PYQCD_EINSUM_TOPOLOGY_REAL` 的可观测差异 | 变量存在，**本机测试用例上 route 无变化** | 本机 `TOPOLOGY_REAL=1` 与缺省都得到 `kupl_dedicated`（`try_dedicated` 先命中） |

### 9.2 仍未覆盖 / 做不到（如实保留）

| 边界 | 事实 | 证据 |
|---|---|---|
| 映射靶子只在语料内 | 只映射到 8638 条语料形态。`ij,jk->ik`、`xy,yz->xz`、`ij->i`、单操作数 `ij->ji`、`ac,em->` 都**无等价写法** | 本机逐个实跑 |
| 省略号 | 含 `.` 的方程 `_split` 直接拒绝 → 无映射 | 实跑 `...ij,...jk->...ik` |
| 代数化简 | 不做（不会把 `ij,jk->ik` 合并成 GEMM） | 模块 docstring |
| 搜索上界 | `_MAX_SEARCH_NODES=4096` 节点 / `_MAX_LABEL_OPTIONS=512` 配对。触发只漏判不误判；**本机没造出触发用例**，触发后的行为取自源码 | `_equivalence.py:107,111` |
| **性能数字** | **本手册不保留任何性能结论**。上一版引用的 `auto vs false`（2.45×）用的是 `_label_sizes` 口径形状，按 AGENTS.md §6.1 **作废**，标 `unverified`（§5.3） | AGENTS.md §6.1 |
| 生产规模实测 | 尚无。`test.py` 没有"用 `distill_tensor_shapes()` 填形状"的开关，必须手写 `--shape` | §1.6 |
| `kupl_topology*` 的 route 实跑 | 本机 CLI 复现不出来（`try_dedicated` 先命中） | §1.3 |
| `PYQCD_EINSUM_TOPOLOGY_REAL=1` 的选路变化 | 本机在已知用例上看不到差异 | §9.1 末两行 |

---

## 10. 本手册的实跑验证清单

环境：`pyqcd-einsum-cu124` / `PYTHONPATH=$PWD` / `PYQCD_EINSUM_NOTICE=0` /
`HEAD=aeea4faf` / 工作树干净；**未设置** `CUDA_VISIBLE_DEVICES`。

| # | 命令（要点） | 结果 |
|---|---|---|
| 1 | `python -m pyqcd.einsum.test --help` | `--equivalence {auto,true,false}` 在位；全部选项/缺省 |
| 2 | `python -m pyqcd.einsum.verify_specialized --help` | `--language` 与 `--equivalence` 都在位 |
| 3 | `… --equation "cadb,menf,ac,em,bd,fn->" --scene distill --verbose true` | `route=特化·qcd_kupl_dedicated_apply`，`rel=1.540e-16` |
| 4 | `… --equation "acdb,menf,ac,em,bd,fn->" --shape … --verbose true` | 多出 `equiv` 四行（方程 / 张量 / 结果 / 说明），`rel=1.763e-16` |
| 5 | 同上 `--equivalence false` / `PYQCD_EINSUM_EQUIVALENCE=0` | `equiv` 段消失，其余逐字不变 |
| 6 | `PYQCD_EINSUM_EQUIVALENCE=0 … --equivalence true` | `equiv` 段**回来**（命令行压过环境变量） |
| 7 | `test.py` HIT 方程 + `--verbose true` | `grep -c '^equiv'` = **0**（已命中不查等价） |
| 8 | `--language zh-cn` vs `en` 的 `--json` | 字段集合相同，只有 `shape_source` 变；`kernel_info` 逐字相同 |
| 9 | `verify_specialized` 例 1 / 例 2 / 例 3 | MISS+等价 / **MISS+三族映射（新）** / HIT（seq=8628） |
| 10 | 例 2 的 `--json` | `reason_key` + `equivalence{operand_order, operand_permutations, output_permutation, kind}` |
| 11 | `verify_specialized --json --equivalence false` | `equivalence` 键**不出现**，`reason_key=reason.no_specialized_path_disabled` |
| 12 | `verify_specialized --language en` | 全英文；`--json` 的 `reason` 仍中文 |
| 13 | `swap(10,11)` / `swap(14,15)` / 输出三循环 `HMGew` | 前二都返回 `operand-order` 映射（旧版说"不可"）；三者正常 |
| 14 | `_equivalence.normalize_equation` 探针：`ij,jk->ik`、`xy,yz->xz`、`ij->i`、`ij->ji`、`ac,em->`、`...ij,...jk->...ik` | 全部 `None`（如实"无等价写法"） |
| 15 | 跨方程复算（§4.3 脚本，输出轴长 2/3/4/5/6） | `relerr=2.802e-16`；漏回填形状 `(3,2,4,5,6)` ≠ `(2,3,4,5,6)` |
| 16 | 方向专测：三循环 / 自逆 | 三循环逆置换形状不符；自逆方向不可区分（§3.3） |
| 17 | `python -B -m pyqcd.testing._einsum_equivalence_contract` | `41 passed, 0 skipped, 0 failed` |
| 18 | `python -B -m pyqcd.testing.einsum.verify_equivalence` | `PASS 107 / FAIL 0 / 未验证 0`，2.0 s |
| 19 | 等价查询计时（arity 6/7/9/10/12/13/14/16 逐档） | 首次 0.8–1.1 s（建索引）；之后需改写的查询 0.32–2.54 ms（中位 0.59），identity 0.007–0.010 ms |
| 20 | `--compare opt_einsum --compare-optimize dp` | `ref opt 要 dp → 实际 'dp'` |
| 21 | `--compare numpy --compare-optimize dp` | 用法错误文案，退出码 2 |
| 22 | `--compare none --compare-optimize dp` | warning，退出码 0 |
| 23 | `--compare backend` | 参照 `pyqcd.contract(backend='cupy', cupy, specialize=auto)`，退出码 0 |
| 24 | `--dtype float32` 缺省 `--atol` / `--atol 1e-4` | 退出码 1（rel 2.117e-07）/ 0 |
| 25 | `--warmup 0`、省略 `--shape` 且语料无此方程 | 退出码 2 + 对应文案 |
| 26 | `--json` 无参数 / 不可写路径 / 可写路径 | 2 / 2 / 0（`wrote …`） |
| 27 | `PYQCD_EINSUM_NOTICE` 开/关 | 提示出现 / 消失 |
| 28 | `PYQCD_EINSUM_SPECIALIZE=0`（`test.py` 与 `verify_specialized`） | route 通用；`verify_specialized` 仍 `HIT` |
| 29 | `PYQCD_EINSUM_KERNEL_KUPL=0` | route 通用 + RuntimeWarning 点名该开关 |
| 30 | `PYQCD_EINSUM_TOPOLOGY_REAL=1` | route 仍是 `kupl_dedicated`（无变化） |
| 31 | `PYQCD_EINSUM_TOPOLOGY_TRACE=1`（seq=129、complex128） | 打出 `topology-gemm-step …` / `topology-specialized: … dtype=0 steps=5 sizes=8` |
| 32 | `LANG=C.UTF-8` CLI；`LANG=zh_CN` 库提示；`PYQCD_EINSUM_LANG=en`；`--language zh-cn` | 依次：CLI 仍中文 / 库提示中文 / CLI 英文 / CLI 中文而库提示仍英文 |
| 33 | `_bench_scene.distill_tensor_shapes("3pt")` | 29 个张量；peram 四种异形轴长 `(4,4,100,100)/(4,4,100,400)/(4,4,400,100)/(4,4,400,400)` |
| 34 | `--scene {distill,distill_curr,scene,full}` 的形状 | §1.6 的四行表；`distill_curr` 给 `[4,4,9,1]` 而**不是** `(4,4,100,400)` |
| 35 | `_specialize` 语料规模 | `len(TASKS) == 8638` |
| 36 | `--specialize true`（seq=129、complex128） | `specialize 实际 True` 但 `kernel` 为空、`route` 通用，`PASS`、退出码 0；stderr 给出 `pack-bound on CPU (3456077824 pack bytes, …)` 拒绝理由 |
| 37 | §3.7 的三条复算（`seq=3` 的 `axes` 变体 `egfh→hgfe`，`dims("distill")`） | 映射给 `perm=(None,None,(3,1,2,0),…)`；视图路线 `path=kupl_topology`（`failure='dedicated kernel requires C-contiguous operands'`）、副本路线 `path=kupl_dedicated`，两者 `rel=6.747e-16`，均退出码 0 |
| 38 | `--kernel-force {kupl_dedicated,kupl_topology,generic}`（`cadb,menf,ac,em,bd,fn->`） | 依次 `ran=kupl_dedicated` / `ran=kupl_topology`（`path=kupl_topology`，`steps=5`）/ `ran=none` + `route` 通用（§11.2） |
| 39 | numpy 操作数 + `--kernel-force cuda_topology`；`--kernel-force nope` | 前者退出码 2 + "未尝试（标签与本次操作数后端不匹配）"；后者退出码 2 + 合法集合（§11.1、§11.3） |
| 40 | float64 + `--kernel-force kupl_topology` | 退出码 2，原因 `real dtype topology is declined by policy`（§11.3） |
| 41 | `PYQCD_EINSUM_KERNEL_FORCE=kupl_dedicated`，再显式给 `--kernel-force generic` | 显式参数压过环境变量：`requested=generic`（§11.1） |
| 42 | `--specialize false` + `--kernel-force kupl_dedicated`；`out=` + 强制（API） | 均 `ValueError` 早失败；CLI 侧表现为 `FAIL` + 退出码 1（§11.5） |
| 43 | `PYQCD_EINSUM_KERNEL_KUPL=0` + 强制 `kupl_dedicated` | `KernelUnavailableError`，原因 `kernel disabled (…)`；退出码 2（§11.5） |
| 44 | 同一进程内连调两次（seq=129 方程） | 第 2 次 `reasons` 变 `negative-cache hit: …`；强制路径两次都报原始原因（§11.4） |
| 45 | `verify_specialized "<eq>" --kernel-force cuda_topology` | `强制内核    : requested=cuda_topology  ran=unknown  attempted=unknown`，结果仍 `HIT`、退出码 0（§11.7） |
| 46 | `python -m pyqcd --gpu-probe`（V100） | `kernel status: fallback`，与 §11.6 强制模式给出的失败原因一致 |
| 47 | `python -B -m pyqcd.testing._einsum_kernel_force_contract` | `9 passed, 0 skipped, 0 failed` |

**未复跑项（已在上文逐处标注）**：§5.3 的 `auto vs false` 倍率（数据作废，标
`unverified`）、§5.4 引用的 AGENTS.md `80×` warmup 数字、`kupl_topology*` 的 route 展示。

**取值口径提示**：上表 38–45 行是**取值收窄之前**的实跑记录，那时 `--kernel-force`
写的是具体标签（`kupl_dedicated` / `cuda_topology` …）。现在合法取值只有
`auto` / `generic` / `dedicated` / `topology` / `backend`，具体标签由操作数后端在运行时
解析；那几行保留其历史语境，**不要照抄命令**——现行写法见 §11。

---

## 11. 我要指定（或排除）某个专用内核

**这一节回答一个问题：这次到底准不准跑某个内核。** 它**不改任何数值**——走哪条路
结果都一样，改的只是"谁在算"。

### 11.1 三步操作

**第 1 步：先看这次实际跑了什么。** 别猜，读记录：

```bash
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill --verbose true
# kernel   : {'kupl': {'path': 'kupl_dedicated', 'dtype': 'complex128'}}
# route    : 特化 · 原生 qcd_kupl_dedicated_apply（path=kupl_dedicated）
```

（`kernel   :` 是**三空格**对齐的那个字段，值是 `LAST_INFO` 汇总的 dict。）

**第 2 步：想强制就加 `--kernel-force`**，取值有五种：

```bash
# 只准跑当前后端的 topology（numpy → kupl_topology）；它跑不成就报错，不回落
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill \
    --verbose true --kernel-force topology

# 只准跑当前后端的 dedicated（numpy → kupl_dedicated）
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill \
    --verbose true --kernel-force dedicated

# 完全跳过专用内核（不产生 warning），走通用路径
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill \
    --verbose true --kernel-force generic

# 直接调后端自己的 einsum，不进专用内核分派（与其它取值互斥）
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill \
    --verbose true --kernel-force backend

# 两个逻辑名都允许：按你写的次序试
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill \
    --verbose true --kernel-force topology,dedicated
```

合法取值只有五个：`auto` / `generic` / `dedicated` / `topology` / `backend`。
`dedicated` / `topology` 是**逻辑名**——具体标签由操作数后端决定（numpy → `kupl_*`，
cupy → `cuda_*`），不必也不能再写死 `kupl_dedicated` 这类名字。

旧写法要改（现在会报用法错误）：`kupl_dedicated` / `cuda_dedicated` → `dedicated`；
`kupl_topology` / `cuda_topology` → `topology`。**但 `kernel   :` 行与 `route :` 行里
出现的 `kupl_dedicated` 等不用改**——那是运行时真实路径名，不是输入。

> 与 `docs/einsum_production_optimization_20261112.md` 不冲突（2026-11-17 核对）：那份
> 文档只用 `generic` / `dedicated` 指代**运行时路径**，从未把旧值写成合法输入。
> 一句话口径：**输入侧**只认这五个取值，**输出侧**仍是 `kupl_*` / `cuda_*`；有疑义以代码为准
> （`_kernel_force.ALLOWED_LABELS`）。

写错不会被静默忽略：

```bash
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill --kernel-force nope
# error: --kernel-force 必须是 auto, backend, dedicated, generic, topology 之一
#        （多个用逗号分隔；backend 不可与其它标签混用），得到 'nope'   → 退出码 2
```

**第 3 步：读 `kernel :` 行确认。**

```
kernel : requested=topology  ran=kupl_topology  attempted=kupl_topology
```

| 字段 | 回答什么 |
|---|---|
| `requested` | 你**要求**了什么（逻辑名；`auto` 时反映那一刻环境变量的生效值） |
| `ran` | **真的**跑了哪个（具体标签）；`none` = 一个都没跑成（`generic` 下恒为 `none`） |
| `attempted` | 按次序真正**试过**哪些具体标签 |

这行只在 `--verbose true` 的明细里出现，且 `ran`/`attempted` 取不到时写 `unknown`
（"没观测到"，不是"没有"）。**它是机器可读的，不随 `--language` 翻译**；要写脚本
就解析它，别解析 `route` 那句人话。

### 11.2 一个完整例子（本机实跑）

同一条方程（语料 `seq=3`，`cadb,menf,ac,em,bd,fn->`），三种取值三条命令：

```bash
cd /home/zhangxin/PyQCD
export PYTHONPATH=$PWD PYQCD_EINSUM_NOTICE=0

# ① 缺省 auto：原次序（dedicated → topology）
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill --verbose true \
  2>&1 | grep -E "^kernel|^route"
```

```
kernel   : {'kupl': {'path': 'kupl_dedicated', 'dtype': 'complex128'}}
route    : 特化 · 原生 qcd_kupl_dedicated_apply（path=kupl_dedicated）
kernel : requested=auto  ran=kupl_dedicated  attempted=kupl_dedicated
```

```bash
# ② 强制 topology：绕过 dedicated，直接跑步表那条
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill --verbose true \
    --kernel-force topology 2>&1 | grep -E "^kernel|^route"
```

```
kernel   : {'kupl': {'path': 'kupl_topology', 'specialized': True, 'steps': 5, 'dtype': 'complex128', 'copied': ()}}
route    : 特化 · 原生 qcd_kupl_topology_apply（path=kupl_topology）
kernel : requested=topology  ran=kupl_topology  attempted=kupl_topology
```

```bash
# ③ generic：完全不走专用内核
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill --verbose true \
    --kernel-force generic 2>&1 | grep -E "^kernel|^route"
```

```
kernel   : 无原生命中（通用路径）
route    : 通用（无原生命中：Python 分步 / numpy.einsum / BLAS）
kernel : requested=generic  ran=none  attempted=none
```

三条命令的 `PASS`/`rel` 是同一量级（本机分别 `1.22e-16` / `1.22e-16` / 见 §5 口径），
差别只在**路线**。这就是"强制内核不改数值"的具体样子。

### 11.3 强制失败长什么样（两个真实例子）

**① 旧的具体标签已不是合法取值**——取值收窄成逻辑名（`dedicated` / `topology`），
`kupl_dedicated` / `cuda_topology` 这类名字现在直接是用法错误：

```bash
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill \
    --kernel-force cuda_topology     # 退出码 2
```

```
error: --kernel-force 必须是 auto, backend, dedicated, generic, topology 之一
       （多个用逗号分隔；backend 不可与其它标签混用），得到 'cuda_topology'
```

"这个标签是哪个后端"不再需要你操心：写 `topology` 就落到**当前后端**的
`*_topology`（numpy → `kupl_topology`，cupy → `cuda_topology`）。想跑 CUDA 内核仍要
让操作数真的在 GPU 上（`--device gpu --backend cupy`）。

**② 内核真的存在但跑不成**——`ran=none` 之外的另一种形态，原因会逐条列出：

```bash
python -m pyqcd.einsum.test --equation "ambn,codp,egfh,ce,mo,bdf,nph->ag" --scene distill \
    --dtype float64 --kernel-force topology    # 退出码 2
```

```
错误：强制内核 topology 没有跑成：equation 'ambn,codp,egfh,ce,mo,bdf,nph->ag'
实际尝试：kupl_topology（topology）
  - kupl_topology（topology）: real dtype topology is declined by policy: …
恢复自动分派：PYQCD_EINSUM_KERNEL_FORCE=auto；让失败不被缓存：PYQCD_EINSUM_NEGATIVE_CACHE=0；检查 GPU 栈：python -m pyqcd --gpu-probe
```

注意括号：`requested` 是你写的**逻辑名**，`attempted` / 逐条原因列的是它解析出的
**具体标签**——两者并列才知道"哪条路真的被试过"。

（实数档被策略排除在拓扑入口外，`PYQCD_EINSUM_TOPOLOGY_REAL=1` 可解开——见 §7。）

### 11.4 ⚠️ 负缓存的坑：为什么第二次"连试都不试"

`PYQCD_EINSUM_NEGATIVE_CACHE`（缺省开）会把"这条计划这条后端确定不行"记下来，
省掉重复尝试。副作用是**同一进程里第二次调用的原因会变成 `negative-cache hit:`**：

```bash
cd /home/zhangxin/PyQCD && export PYTHONPATH=$PWD PYQCD_EINSUM_NOTICE=0
python - <<'PY' 2>/dev/null
import numpy as np
from pyqcd.einsum import _bench_scene, _wick_tasklist, bench_specialized, contract
from pyqcd.einsum._api import last_kernel

task = _wick_tasklist.by_seq()[129]          # 这条方程两个原生入口都被 pack-bound 拒
dims, rng = _bench_scene.dims("distill"), np.random.default_rng(1)
build = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(complex)
ops = bench_specialized.operands_for(task.equation, dims, build, task.roles)

for i in (0, 1):
    contract(task.equation, *ops)
    rec = last_kernel()
    print(f"call {i}: ran={rec['ran']}")
    for label, why in rec["reasons"].items():
        print(f"         {label} -> {why.split('(')[0].strip()}")
PY
```

```
call 0: ran=None
         kupl_dedicated -> plan is pack-bound on CPU
         kupl_topology -> topology declined: pack-bound on CPU
call 1: ran=None
         kupl_dedicated -> negative-cache hit: plan is pack-bound on CPU
         kupl_topology -> negative-cache hit: topology declined: pack-bound on CPU
```

第二次那两个原因**不是新的失败，是上一次的复印件**。所以排查时：

- **强制路径自动旁路负缓存**（"强制即真试一次"），所以
  `--kernel-force <value>` 下不该看到 `negative-cache hit:`——看到就是旁路没生效；
- 想彻底排除缓存干扰，显式关掉它再复核一次：
  `PYQCD_EINSUM_NEGATIVE_CACHE=0 python -m pyqcd.einsum.test …`。

### 11.5 和既有开关一起用（三条例外会早失败）

强制**不是**和别的开关互斥，它们是不同的层——但下面三种组合会让内核根本到不了，
所以库选择**早失败**而不是静默绕过：

| 组合 | 结果 |
|---|---|
| `PYQCD_EINSUM_KERNEL_KUPL=0` + 强制 `dedicated` / `topology` | `KernelUnavailableError`：`kernel disabled (PYQCD_EINSUM_KERNEL_* = 0, or library missing)` |
| `--specialize false` / `PYQCD_EINSUM_SPECIALIZE=0` + 强制取值 | `ValueError`（`test.py` 里是 `FAIL` + 退出码 1） |
| `out=` + 强制取值 | `ValueError` |
| `--kernel-force generic` + 上面任意一个 | **不冲突**（generic 本就不走专用路径） |

```bash
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill \
    --specialize false --kernel-force dedicated     # 退出码 1
#  error: ValueError: specialize=False / PYQCD_EINSUM_SPECIALIZE=0 关掉了专用内核路径，
#         不能与强制指定内核同时使用；把 specialize 恢复为 None 或 True 再强制
```

判断口诀：`PYQCD_EINSUM_KERNEL_KUPL=0` 是"**关掉这个后端**"，`specialize=False` 是
"**整体跳过**"，`--kernel-force` 是"**只准跑这个**"，`--kernel-force backend` 是
"**别用专用内核，直接调后端**"。

### 11.6 我的内核在远端集群上 `status 4`，怎么自查

典型报错（CUDA 侧）：

```
qcd_cuda_topology_apply failed with status 4 (plan not executable as version-2 steps)
```

`status 4` 在 CUDA 侧同时覆盖"步表不认这条计划"和"顺序守卫拒绝了这条计划"，
**只看状态码分不出来**。自查顺序：

```bash
# ① 先确认它到底能不能跑（能跑就说明之前只是自动分派选了别的入口）
PYQCD_EINSUM_KERNEL_FORCE=topology python -m pyqcd.einsum.test \
    --device gpu --backend cupy --equation "cadb,menf,ac,em,bd,fn->" --scene distill --verbose true

# ② 不能跑时，上面那条命令会给出逐标签原因 + 三条恢复建议，按它走
#    ③ 再往下才是设备与构建面：
python -m pyqcd --gpu-probe
```

本机实跑 ②（V100，complex128）给出的是"该 dtype 的 2×2 自检没过 → 重建原生库"，
而不是含糊的 `status 4`：

```
错误：强制内核 topology 没有跑成：equation 'cadb,menf,ac,em,bd,fn->'
实际尝试：cuda_topology（topology）
  - cuda_topology（topology）: dtype complex128 is inside the topology dtype contract but its native probe failed (2x2 matmul disagreed with numpy); falling back to the generic path with no implicit downcast. Rebuild the native archive (bash cpp/cuda/einsum/make-nv.sh) to re-enable it
恢复自动分派：PYQCD_EINSUM_KERNEL_FORCE=auto；让失败不被缓存：PYQCD_EINSUM_NEGATIVE_CACHE=0；检查 GPU 栈：python -m pyqcd --gpu-probe
```

`python -m pyqcd --gpu-probe` 本机同一时刻的输出（节选）：

```
  device         : Tesla V100-SXM2-32GB (compute 7.0)
  kernel enabled : True
  kernel status  : fallback — topology entry did not run: dtype complex64 is inside the topology dtype contract but its native probe failed …
  verdict        : CUDA fast path is NOT active — results fall back
```

即**两者一致**：强制模式把"到底哪一环没过"说清楚了，probe 给的是同一件事的
设备视角。`PYQCD_CUDA_ARCHITECTURES` 是**构建期**变量（架构不匹配时内核根本编不
进来），与这里的运行期 probe 失败不是一回事。

### 11.7 `verify_specialized --kernel-force` **不能**验证内核能跑

那个入口是**纯查询、不做收缩**，所以：

```bash
python -m pyqcd.einsum.verify_specialized "cadb,menf,ac,em,bd,fn->" --kernel-force topology
# 强制内核    : requested=topology  ran=unknown  attempted=unknown
# 结果        : HIT
```

`ran=unknown` 是**如实**的"本进程没跑过任何内核"，不是"跑了但不知道"。规则：

- `--kernel-force` 在那儿**只**决定报告里出不出 `kernel` 这一行、以及
  `requested=` 显示什么；
- `PYQCD_EINSUM_KERNEL_FORCE` 在 `verify_specialized` 里**完全不起作用**（它不执行）；
- 不给 `--kernel-force` 时这行**不出现**，缺省输出与改造前逐字节相同；
- 真要验证"能跑"，只能用 §11.2 的 `python -m pyqcd.einsum.test --kernel-force`。

### 11.8 快速对照表

| 我想… | 敲什么 |
|---|---|
| 看这次跑了哪个内核 | `--verbose true`，读 `kernel :` 行的 `ran=` |
| 只准跑某一个 | `--kernel-force dedicated` / `topology`（两个都用就逗号分隔，按次序） |
| 完全不用专用内核 | `--kernel-force generic`（或 `--specialize false`） |
| 直接调后端自己的 einsum | `--kernel-force backend`（与其它取值互斥） |
| 整进程都生效 | `PYQCD_EINSUM_KERNEL_FORCE=<value>` |
| 排除缓存干扰 | 加 `PYQCD_EINSUM_NEGATIVE_CACHE=0` |
| 检查 GPU 栈 | `python -m pyqcd --gpu-probe` |
| 在代码里读结果 | `from pyqcd.einsum._api import last_kernel`（**未**在包级导出） |

### 11.9 强制 `generic` / `backend` 到底省了什么

`--kernel-force generic`（或 `contract(kernel_force="generic")` /
`PYQCD_EINSUM_KERNEL_FORCE=generic`）不是"少用一点内核"，而是**整段跳过选路**：

- **不做形态匹配**：一次 `_specialize.match` 都不调，语料索引也不查；
- **不做等价改写**：`as_equivalence` 那一步整段跳过（改写的唯一目的就是命中特化形态，
  而你已经明确要通用）；
- **不发那两条提示**：「专用路径」与「通用路径」都不打——已经要求通用，再提示一遍
  等于自说自话；
- 直接进通用收缩（Python 分步 / `numpy.einsum` / BLAS）。

`--kernel-force backend` 更极端：不进专用分派，直接调后端自己的 `einsum`，
本库的形态匹配、等价改写与收缩规划一概不参与（`route :` 行会写明 `backend einsum`）。

**这是省时间，不是省正确性。** 两条路算的是同一个方程，结果逐位可比（下面两条 `rel`
都在 1e-16 量级），所以做性能对照时它是最干净的"通用基线"。

```bash
EQ="cadb,menf,ac,em,bd,fn->"

# ① generic：不匹配、不改写、不提示
PYQCD_EINSUM_PROFILE=0 python -m pyqcd.einsum.test --equation "$EQ" --scene distill \
    --rounds 2 --warmup 1 --kernel-force generic --verbose true
# kernel   : no native hit (generic path)
# route    : generic (no native hit: Python steps / numpy.einsum / BLAS)
# kernel : requested=generic  ran=none  attempted=none
# PASS … rel=6.35e-16

# ② backend：直接后端 einsum
PYQCD_EINSUM_PROFILE=0 python -m pyqcd.einsum.test --equation "$EQ" --scene distill \
    --rounds 2 --warmup 1 --kernel-force backend --verbose true
# route    : backend einsum (--kernel-force backend: the specialised dispatch never ran)
# kernel : requested=backend  ran=backend  attempted=none
# PASS … rel=0.00e+00
```

读 `kernel :` 时记住 `ran=none attempted=none` 是**如实**的：一次内核都没试，
不是"试了没成功"。不给 `--verbose true` 时这些行都不出现，输出里连提示也没有。

两个容易混的点：

- **`generic` 与 `specialize=false` 不等价**：前者保留 `_LAST_KERNEL` 的如实记录
  （`requested=generic`、一次没试），后者只是把专用路径整体关掉；两者可以同时给，
  不冲突（`generic` 本来就不走专用路径）。
- **画像建议的 `generic` 不走这条路**：那是**计时结论**，不是"语义指令"——它不会顺手
  关掉等价改写（那条是独立开关），只把提示静音。区别见 §13.4 的"上次成功，这次却没被
  强制"一行。

## 12. 自动改写与缓存：让 MISS 的方程也能走特化、让重复查询不再重复算

`0.4.4` 起有两件事会**自动**发生（都可关）。它们都属于"底层优化"，但都有代价，
所以本节把"什么时候该关"写清楚。

### 12.1 自动施加等价写法（`--as-equivalence`，缺省 `auto`）

取值是 **`auto`（缺省）/ `true` / `false`**（`--tune` 内部另用 `force`，表示"忽略历史
结论、真查一遍"）。**`auto` 只在配置表里有等价标记行时才改写**：那行是 tune 写下的
"原 equation 走等价转换后的配置更佳"（三个置换字段 + 一句备注，没有性能配置）。
表缺失、缓存关闭、未命中、或该键没有标记行 → **一律不改写**（没有信息就不猜），
走普通路径。

`true` 总是尝试改写；`false` 从不改写。真正施加时是：重排操作数 → 按需转置并落成
连续副本 → 用等价方程收缩 → 把结果转回你要的输出轴序。**返回的数组永远是原方程的
答案**：形状与 dtype 都不变；轴序对不上时抛 `RuntimeError`，而不是把错结果交给你。

```bash
# 缺省 auto：表里有该方程的等价行才会改写（先按 §13.1 的三步 tune 一次）
python -m pyqcd.einsum.test --equation "menf,cadb,ac,em,bd,fn->" \
  --shape "[2,2,2,2],[2,2,2,2],[2,2],[2,2],[2,2],[2,2]->" \
  --rounds 2 --warmup 1 --verbose true
# route    : 特化 · 原生 qcd_kupl_dedicated_apply（path=kupl_dedicated）   ← tune 判定改写更快
# route    : 通用（无原生命中：Python 分步 / numpy.einsum / BLAS）         ← 没有标记行就不改写

# 想绕开表、强行验证改写这条路（旧语义）
python -m pyqcd.einsum.test … --as-equivalence true

# 从不改写
python -m pyqcd.einsum.test … --as-equivalence false
```

不施加的场合：`out=`、额外的 `**kwargs`、逐元素方程、cupy 的通用委派、
`--specialize false`、以及 `optimize` 传了 callable 或显式路径（它们绑定原操作数
次序）。注意：`PYQCD_EINSUM_EQUIVALENCE=0` 是**硬闸**，显式 `--as-equivalence true`
也压不过它。

**什么时候该显式关**：小形状、高频重复的批量收缩用 `--as-equivalence false`。缺省
`auto` 在没有等价行时本来就不改写，不需要为了省开销而关。

等价结论**没有独立的跨进程表**（`profile_equivalence.csv` 已删除，不再生成、不再读取，
2026-11-17 TV 重设计）：普通调用既不写也不读任何等价结论，跨调用只剩本进程内的查询
缓存。想跨进程复用，只能先 `--tune`——tune 会把"这条方程走哪条改写更快"写进主表里的
**等价行**（只有三个置换字段 + 一句备注，没有性能配置）。规则与表结构见 §13.3。

### 12.2 缓存与命中统计（`--cache`，缺省开启）

三处缓存：匹配器（`specialize.match`，含"没命中"这个结果）、等价查询
（`equivalence.normalize`，含 `None`）、规划器（`api.find_path`）。

```bash
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill \
  --rounds 2 --warmup 1 --verbose true --cache true
# cache    : enabled=true  hits=4  misses=3  hit_rate=0.57
```

- 不带 `--cache` 时**不会**出现这一行（旧输出逐字节不变）。
- `--cache false` 时该次调用**既不读也不写**，所以那行会如实写
  `enabled=false  hits=0  misses=0  hit_rate=none`。
- 在 Python 里：`pyqcd.einsum.cache_stats()` 看全部缓存，
  `contract(..., cache=False)` 单次关闭，`pyqcd.einsum.cache_clear()` 清空。

### 12.3 两者的典型组合

| 我想… | 敲什么 |
|---|---|
| 让 MISS 的方程也尽量命中特化 | 先 `--tune` 标定出等价行；之后缺省 `auto` 就按它改写（表里没有标记行时不改写） |
| 排查"到底是不是映射帮上忙" | 对照跑 `--as-equivalence true` / `false` 看 `route :` 行 |
| 小方程批量跑，怕被改写拖慢 | `--as-equivalence false`（缺省 `auto` 在没有等价行时本来就不改写） |
| 怀疑缓存串味 | `--cache false` 单次对照，或 `PYQCD_EINSUM_CACHE=0` |
| 看缓存有没有真的生效 | `--cache true` 读 `cache :` 行，或 `cache_stats()` |

---

## 13. 第二次跑同一个收缩，怎么复用 tune 标定的配置

配置表（`pyqcd/einsum/_profile_cache.py`）把 tune 量到的
`(形状, dtype, 设备, 方程, 后端)` → `(内核, optimize, wall_ms)` 记成一张 CSV
（文件名 **`pyqcd.einsum.profile.csv`**，落在**当前工作目录**的 `./.cache/`）。
**旧名 `profile.csv` 已废弃**；**`profile_equivalence.csv` 已删除**——不再创建、不再读取
（等价结论只以主表里的"等价行"存在，见 §13.3）。表是**只读 + 建议**：命中最多替你填
`kernel_force=`（逻辑名）与 `optimize=`，方程、dtype、形状、结果都不动——所以关掉它永远
不会有数值差异。

**为什么只有 tune 能写**：这张表是**标定产物**——它记的是"某台机器、某组形状上实测更快
的配置"。普通调用若也顺手写，不同运行环境、不同负载、不同版本的观测会互相覆盖，表就不再
是任何一台机器上的可信结论。

**五条硬规矩（2026-11-17 TV 重设计）**：

1. **只有 `--tune` 能写**：普通 `contract()` **只读**，既不追加、也不更新、更不重写文件；
2. **同键只允许 1 行**：同一 `(shape, dtype, device, equation, backend)` 出现 2 行以上 →
   **报错退出**并把**全部候选行**打到 stderr（不猜、不选最优）；
3. **未 tune 则表是空的**：文件不存在只**警告一次**"需要先 tune"+ **该配置的 tune 命令**，
   然后**继续执行**（按未命中处理）；它也不会顺手替你建一张空表；
4. **命中不得与显式配置打架**：命中的 `kernel` / `optimize` / 等价结论与**用户显式传入**
   的 `kernel_force=` / `optimize=` / `as_equivalence=` 不一致 → **报错退出**（打印键、表值、
   请求值与记录位置）。"显式"只算**非 `auto`** 的实参：`--compare` 不参与，显式写 `auto`
   等于没写；
5. **`wall_ms` 是唯一计时列**：含义见 §13.3c——**整个 `contract()` 的墙钟**，含等价转换、
   缓存读写与规划-匹配，不含 compare 参考计时与收尾落盘；表里没有第二列计时可以与它相加
   或混比。

### 13.1 从零到用上缓存：三步

缺省**只读**，所以拿到收益的路径只有三步：**第一次运行看警告 → 复制命令去 tune →
之后自动命中**。

```bash
# ① 第一次：没有表 → 一行警告 + 该配置的 tune 命令，然后照常算
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill \
    --rounds 2 --warmup 1
# [pyqcd.einsum] 警告：einsum 配置表缺失：…/pyqcd.einsum.profile.csv 不存在；需要先 tune：
#   python -m pyqcd.einsum.test --device <device> --dtype <dtype> --equation "cadb,menf,ac,em,bd,fn->" \
#     --shape "<本次的操作数形状>" --tune --compare opt_einsum
#   本次按未命中执行（方程、dtype、形状与数值都不受影响）。
# PASS …
# （警告由 `_api._warn_profile_missing` 打出，**每进程一次**；命令模板由
#   `_profile_cache.ProfileMissing.cli_hint` 按本次的键生成，可直接复制执行。）

# ② 照它给的命令 tune 一次：遍历 as_equivalence × optimize，按 wall_ms 挑最优，
#    写入选优行并删掉同键其它所有行
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill \
    --tune --compare opt_einsum

# ③ 之后再跑同形态：命中就按表选路
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill \
    --rounds 2 --warmup 1 --verbose true
```

跑过 tune 之后，`./.cache/` 里**只有一张表**：

```
./.cache/pyqcd.einsum.profile.csv     # 读（普通调用） + 写（只有 tune）
```

旧文件**都已作废**：`profile.csv` 是改名前的旧名；`profile_compare.csv` 与
`profile_equivalence.csv` **已删除**（不再创建、不再读取）。哪怕它们还躺在目录里，
新代码也完全看不见（等于"没 tune 过"）。

**命中并施加时只打一行**，而且钉到具体记录
（`<pyqcd.einsum.profile.csv 绝对路径>:<数据行号>`）：

```
[pyqcd.einsum] profile hit: <dir>/pyqcd.einsum.profile.csv:12 kernel=kupl_dedicated optimize=auto wall_ms=14.456 (applied)
[pyqcd.einsum] profile 命中：<dir>/pyqcd.einsum.profile.csv:12 kernel=kupl_dedicated optimize=auto wall_ms=14.456（已套用）
```

（格式见 `_api._profile_notice`；`wall_ms` 随本机变化，语言见 §7.1b。）

这一行的契约：

| 项 | 行为 |
|---|---|
| 条数 | 每个 `(方程, 形状, kernel, optimize)` **每进程一行**；`--verbose true` 时每次调用都打 |
| 行号 | 行号来自**文件里的数据行**（1-based）。表只有 tune 能写，所以"本进程刚记录的行"不存在 |
| 静音 | 同一形态**每进程一次**；`PYQCD_EINSUM_NOTICE=0` 整条关掉 |
| 抑制 | 这一行之后本次调用**不再**打解释性提示（`announce_generic` / `announce` / 等价说明） |
| 保留 | `route :` / `kernel   :` / `kernel :` 等**结果行照旧**（它们是 CLI 报告行，不是库提示） |
| 恢复详解 | `--explain true` / `--verbose true`（→ `_specialize.set_detail(True)`），或 `PYQCD_EINSUM_SPECIALIZE_VERBOSE=1` |

`--explain true` 只恢复**库内**解释性提示；CLI 自己在命中时仍只补一行画像状态
（`[pyqcd.einsum] profile hit → backend=… kernel=… (wall_ms=…)`），三段式
`plan` / `kernel` / `why` **不会**在命中时出现——命中本身就是解释。

`kernel=` 是**具体标签**（`kupl_dedicated`），回放时会翻成逻辑名 `dedicated` 交给
`kernel_force=`；所以你在 §11 里读到的"逻辑名 / 具体标签"两套说法在这里同时出现。

**命中行与你的显式参数冲突 = 硬报错**：表里记 `kernel=topology`、你却写
`--kernel-force generic`（或 `optimize=` / `--as-equivalence` 与表里不一致）时，程序
打印键、表值、你的值、记录位置（`<file>:<行号>`）后**以非零退出码结束**——不静默让
哪一方胜出。想绕开只有两种写法：去掉那个显式参数，或 `--profile false` 让本次不读表。

**同键多行 = 硬报错**：手工编辑或两份表拼接会造出这种情况。读侧把**全部候选行**
打到 stderr 再抛错，tune 一次就能把它收成一行（tune 写前先删同键所有行）。

### 13.2 换目录 / 关掉 / 钉住目录

```bash
# 把画像放到仓库外（推荐：别让它进工作树）
PYQCD_EINSUM_PROFILE_DIR=/tmp/einsum-profile python -m pyqcd.einsum.test …
python -m pyqcd.einsum.test … --profile-dir /tmp/einsum-profile

# 整进程关掉：不读、不写、连目录都不建
PYQCD_EINSUM_PROFILE=0 python -m pyqcd.einsum.test …

# 只这一次关掉（同样零副作用）
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill \
    --rounds 2 --warmup 1 --profile false
```

```python
from pyqcd.einsum import contract

contract(eq, *ops)                                   # 缺省：跟随 PYQCD_EINSUM_PROFILE
contract(eq, *ops, profile=False)                    # 只这次关掉，零副作用
contract(eq, *ops, profile_dir="/tmp/einsum-profile")  # 钉住目录（后续调用沿用它）
```

CLI 只有 `test.py` 有两个选项（`--help` 可查）：

```
--profile {auto,true,false}   缺省 auto = 跟随 PYQCD_EINSUM_PROFILE；
                              true/false 本次强制开/关（false 不读不写不建目录）
--profile-dir PROFILE_DIR     钉住本次进程的缓存目录（只有首次解析才按 env 或 ./.cache）
```

`verify_specialized` **没有**这两个选项——那个入口不做收缩、也从不读画像。
（`pyqcd.testing.einsum.*` 基准脚本里的 `--profile T1` 是**尺寸档**，与本层无关。）

#### `profile_dir` 是"钉住"，不是"单次覆盖"

这是最容易误解的一条：

- **省略 `profile_dir`（或传 `None` / 空串）不会回到 `PYQCD_EINSUM_PROFILE_DIR`，也不会
  回到 `./.cache`**——它保持进程里**已经钉住**的那个目录。`None` 与空串都按"未给"处理
  （`_resolve_dir_text`，`_profile_cache.py:959-969`）。
- 实现上 `_api` 每次 `contract()` 只做一件事：`profile_dir` 为真就
  `configure(profile_dir)`，否则 `ensure_configured()`；后者"只补齐、不改写"
  （`_api.py:1114-1115`、`_profile_cache.py:331-344`）。所以 `profile_dir="A"` 之后再调
  `contract(eq, *ops)`，用的仍是 A。
- 想换只能**显式传另一个目录**（或换一个进程，让它按 env / 默认值首次解析）。
  换目录 = 换一张表：解析缓存按路径失效，下一次读表读的是新目录的文件（没有待写行要
  冲，因为普通调用根本不写）。
- 只有**从未配置过**的第一次调用，才按 `profile_dir` > `PYQCD_EINSUM_PROFILE_DIR` >
  `./.cache` 解析一次。

### 13.3 表里有什么

表头（列序见 `_profile_cache.PROFILE_COLUMNS`；主机名与时间戳按本机变化）：

```
shape,dtype,device,equation,backend,kernel,optimize,wall_ms,target_equation,operand_order,operand_permutations,output_permutation,note,host,updated_at,pyqcd_version
```

- **键 = 前 5 列**：`shape` / `dtype` / `device` / `equation` / `backend`。`host` 只记录、
  **不参与匹配**（同一张表可复制到别的机器继续用）；
- `shape` 规范化成 `2x3|3x4`：`x` 连轴、`|` 连操作数，所以 `--shape "[2,3],[3,4]"`
  与 `2x3|3x4` 落在同一个键上；`equation` 去掉全部空白；
- **一个键只有 1 行**（硬约束）。行数 = tune 过的键数，不是调用次数：普通调用既不追加
  也不更新，所以"跑了很多次但表没变"是正确行为；
- 两类行靠"哪几列有值"区分：
  - **性能行**：`kernel` + `optimize` + `wall_ms` 有值，四个等价列留空；
  - **等价行**：`operand_order` / `operand_permutations` / `output_permutation`（+
    `target_equation` 与一句 `note`）有值，`kernel` / `optimize` / `wall_ms` 留空。
    读侧看到它就按它改写，不提内核；
- 没有 `calls` 列、没有 `schema_version`、没有每键多条：这些随"每键多条 / 稀疏写"一起
  删除（口径变更靠**换文件名**隔离——旧 `profile.csv` 根本不会被读）。

```bash
ls -l .cache/                                    # 只有 pyqcd.einsum.profile.csv
head -1 .cache/pyqcd.einsum.profile.csv          # 表头
column -s, -t < .cache/pyqcd.einsum.profile.csv  # 肉眼看每键那 1 行
```

### 13.3c `wall_ms` 怎么读，以及 `--tune` 到底写了什么

**一句话**：`wall_ms` = **整段 `pyqcd.einsum.contract(***)` 的墙钟时间**，含等价转换
（若有）、读表（若有）与未命中时的规划-匹配。这些成本你重跑一次就一定会付，所以它们
必须算在"这个配置有多快"里；挑配置只比这一列。

| 项 | 口径 |
|---|---|
| 起点 | `contract()` 的**第一行**（解析之前） |
| 终点 | 返回之前（结果已经算好） |
| **含** | 等价改写（重排/转置/连续副本/结果转回）、配置表读取、未命中时的规划与匹配、内核执行 |
| **不含** | compare 参考的计时、收尾落盘、调用方的循环、你打印输出的时间、CLI 的 warmup 与重复轮次编排 |
| 其他列 | **没有**第二列计时可以和它相加或混比。旧的拆分计时（`init_ms` = 规划/匹配、另一列 = 只包收缩执行）已删除；`init_ms` 若在某处还出现只作诊断，**不得参与选优** |

选优因此不再发生在读侧：读侧只做精确匹配（同键 ≤1 行，多行即报错）。**tune** 在遍历
里按 `wall_ms` 挑最优，然后只写那一条——见下。

**`--tune` 是唯一的写者，而且遍历期间既不读也不写**（`_profile_cache.tune_session()`）：

| 规格项 | 行为 |
|---|---|
| 锁定的维度 | 键 = `(shape, dtype, device, equation, backend)` 固定不动 |
| 遍历的维度 | `kernel_force ∈ {auto, dedicated, topology, generic, backend}` × `as_equivalence ∈ {true,false}` × `optimize ∈ {auto, greedy, optimal}`，共 **30 组** |
| 单次超时 | `--tune-timeout`，缺省 **30 s** |
| 失败处置 | 异常 / 超时 / 数值 MISMATCH **只打印、不退出**，继续遍历下一组 |
| 不读表 | 遍历期间 `read_winner` 一律当未命中：`auto` 那一组量的是**真实的自动分派**，不是"上次命中了什么"的回放 |
| 不写表 | 遍历期间表一个字都不变：`write_winner` **静默不写**（不排队推迟）。胜出配置在 `with tune_session():` 块**之外**写一次——所以同键旧行的删除与新行写入是一起完成的 |
| 选优 | 按 `wall_ms` 最小挑一组（其它诊断列若有，**不参与**） |
| 最优来自 `as_equivalence=false` | 以**原 equation** 写一行完整性能配置；删掉该键其它所有行（**含更快的历史行**，防异环境污染） |
| 最优来自 `as_equivalence=true` | ① 以 **`target_equation`** 写一行完整性能配置；② 另以**原 equation** 写一行**只含等价字段**（三个置换 + 备注）；两处都先删同键其它行 |

所以 tune 之后表里可能出现**同一个"方程族"的两行**：改写后那条（带内核与 `wall_ms`）
和原方程那条（只有置换字段）。读侧拿到后者就照它改写、不提内核；拿到前者就直接按它
选路。

**一条可复制命令**（全新临时目录，空表起步；`column` 只为好看）：

```bash
cd /home/zhangxin/PyQCD
export PYTHONPATH=$PWD PYQCD_EINSUM_NOTICE=0
PROF=$(mktemp -d)
python -m pyqcd.einsum.test --equation "ij,jk,kl->il" \
    --shape "[4,5],[5,6],[6,4]->[4,4]" --rounds 2 --warmup 1 --profile-dir $PROF   # 无表：警告 + 照常算
python -m pyqcd.einsum.test --equation "ij,jk,kl->il" \
    --shape "[4,5],[5,6],[6,4]->[4,4]" --rounds 2 --warmup 1 --profile-dir $PROF \
    --tune --compare opt_einsum                                                    # 写入选优行
python - "$PROF/pyqcd.einsum.profile.csv" <<'PY'
import csv, sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        print(f"{row['kernel'] or '(等价行)':12s} optimize={row['optimize'] or '-':8s}"
              f" wall={float(row['wall_ms'] or 0.0):8.3f} ms"
              f"  target={row['target_equation'] or '-'}")
PY
```

读法：**比的是不同配置在同一台机器上的 `wall_ms`**（tune 已经按它选过，这里只是复核），
不要拿它和别的文档里的数字横比——负载、dtype、形状、后端任何一项不同都不可比。
tune 只报**本次实测**的结果，不是基准；`--tune-budget` 之类的完整选项以 `--help` 为准。

### 13.4 排查：没生效 / 没文件 / 冲突 / 多行

```python
from pyqcd.einsum import _profile_cache as p
from pyqcd.einsum._api import last_profile

p.enabled()        # 开关打开且目录没被判死
p.profile_path()   # '<...>/.cache/pyqcd.einsum.profile.csv'（不建目录、不读文件）
p.last_error()     # 最近一次降级原因；None 表示没出过问题
p.read_winner({"shape": "4x5|5x6|6x4", "dtype": "complex128", "device": "cpu",
               "equation": "ij,jk,kl->il", "backend": "numpy"})
# 命中 → 1 行（带 line）；没有 → None；文件不存在 → ProfileMissing；同键多行 → DuplicateProfileEntry
last_profile()     # 最近一次 contract() 的观测：wall_ms / hit / fallback / note / key
```

`read_winner` 的**三种情形**就是排查的入口：

| 结局 | 说明 |
|---|---|
| 返回 1 行 | 正常命中。命中是否**施加**另看 `last_profile()["force"]` / `["optimize"]` / `["equivalence"]` |
| `None` | 表在、但这个键没 tune 过：按现有逻辑执行（**不会**自动补写一行） |
| `ProfileMissing` | 文件还没生成：先 tune。普通调用里它被降级成"每进程一次的警告 + 继续执行" |
| `DuplicateProfileEntry(rows)` | 同键多行：`rows` 是全部候选。删到只剩 1 行，或重新 tune 那个键 |

| 症状 | 原因 / 怎么办 |
|---|---|
| 目录/文件根本没出现 | `PYQCD_EINSUM_PROFILE=0`，或这次 `profile=False`。总开关为 0 时**连 `profile=True` 也不 IO**（不建目录） |
| 表一直在，但跑多少次都不变 | **这就是设计**：普通调用只读。要让新结论进表，只能 `--tune` |
| 表里只有几个键 | 只有 tune 过的键才有行；行数 = tune 次数覆盖的键数，不是调用次数 |
| 启动时有一行"需要先 tune"警告 | 该配置目录下没有表（或换过 `--profile-dir`）。照它给的命令跑一次 tune，或用 `--profile false` 静音地跑 |
| 报 `ProfileConflict` 后退出 | 命中行的 `kernel` / `optimize` / 等价结论与你的**显式** `--kernel-force` / `optimize=` / `--as-equivalence` 冲突。判据只算**非 `auto`** 的显式实参：`--compare` 不参与，显式写 `auto` 等于没写。去掉那个显式参数、改表、或 `--profile false` |
| 报"同键多行"后退出 | 手工编辑或拼表造成重复键：读侧不猜，直接把**全部候选行**打出来。tune 一次那个键（写前会删同键所有行），或手工删到 1 行 |
| 命中被记下但没施加 | 这次带 `out=` / `specialize=False`，或显式给了 `kernel_force`——显式要求永远压过表；冲突时前者不报错、后者报错 |
| 画像的建议跑不成 | 自动退回原分派一次（内部用 `profile=False` 重试），**不抛** `KernelUnavailableError`；`last_profile()["fallback"]` 为 `True` |
| 旧文件还在，为什么完全没用 | 文件名换了：`profile.csv` / `profile_compare.csv` / `profile_equivalence.csv` 新代码不读。重新 tune |
| 想把画像整个作废 | 删掉 `pyqcd.einsum.profile.csv`（或换 `PYQCD_EINSUM_PROFILE_DIR`） |

**别把 `.cache` 提交，它也不参与发布产物**：默认目录 `./.cache`
（`_profile_cache.DEFAULT_DIR`）里是主机名、方程与形状；根 `.gitignore` 的 `.*` 规则
已经忽略它（`.gitignore:73`，`git check-ignore -v .cache/pyqcd.einsum.profile.csv`
可复核），`MANIFEST.in` 也不收录任何缓存目录（只 `recursive-include pyqcd *.py …`），
所以它既不进 sdist/wheel 也不该进版本库。
