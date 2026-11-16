# qcd-einsum 用户操作手册 —— "我想做 X，该敲什么"

面向**使用者**的任务导向手册。API 参考不在这里（`--help` 已覆盖），
这里只回答"下一步敲哪条命令、输出怎么读、读到的结果意味着什么"。

| 项 | 值 |
|---|---|
| 适用范围 | `pyqcd/einsum` 的**用户可见入口**（`python -m pyqcd.einsum.test`、`python -m pyqcd.einsum.verify_specialized`、`pyqcd.einsum.contract`） |
| 版本基线 | `HEAD = 71971265a0e64167b32fa8703d6683deb9efd261`（2026-11-16，`pyqcd 0.4.3`） |
| 校验方式 | 文中**每条命令、每个输出块都实跑过**（清单见 §9）。源码行号一律指 HEAD 版本 |
| 并发说明 | 实跑发生在写作当天早些时候，当时工作树 = **HEAD + `verify_specialized.py` 的未提交改动**（新增 `--language` 与英文文案；其缺省中文输出与 HEAD 逐字节相同）。此后其它代理继续改动了 `test.py` / `AGENTS.md` / `SKILL.md` 并开始加入 `--equivalence`，**本手册未纳入这些后续改动**，冲突处一律以 HEAD 为准并在 §8 标注 |
| 不覆盖 | 内部契约（`pyqcd/einsum/AGENTS.md` §9）、领域知识（`skills/pyqcd-einsum/SKILL.md`）、CUDA 构建流程 |

---

## 0. 三条入口，先选对

| 我想… | 用哪条 | 一句话区别 |
|---|---|---|
| 知道这条方程会走哪条路 / 快不快 / 对不对 | `python -m pyqcd.einsum.test` | **会真的算**，出耗时与容差结论 |
| 只想知道它命不命中语料特化，不想跑数值 | `python -m pyqcd.einsum.verify_specialized` | **纯查询**，不构造张量、不调 `contract()`、无副作用 |
| 整库有没有退化 | `python -m pyqcd.testing` | 全量回归。与上面两条互补，**不可互相替代** |

`python -m pyqcd.einsum.test` 是 qcd-einsum 单方程验证/复现/性能对比的**唯一合法入口**
（硬性规定，2026-11-15 起）；临时脚本、一次性 `python -c` 的结论不予采信。
本手册未用它们做数值/性能取数（只额外用 JSON 解析把 `--json` 的字段名汇总成表，
数据本身仍由上面这条入口产出）。

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

- `--scene distill`：省略 `--shape` 时按**语料 `roles`** 反推生产尺寸。
  方程不在语料里会直接报错退出 2（见 §1.5），此时必须显式给 `--shape`。
- `--verbose true`：逐项明细打到 **stderr**；那行 `PASS …` 在 **stdout**。
  想一次看全就 `2>&1`。

### 1.2 实跑输出（原样）

> 耗时/内存是**当次实测值**，会随机器负载浮动；行与字段的结构是稳定的。
> 同一个方程两次跑出 `0.101ms` 与 `0.100ms` 都正常，别拿毫秒位比对历史结论。

stdout：

```
PASS cadb,menf,ac,em,bd,fn-> rel=1.54e-16 vs np.einsum(optimize='greedy')  0.101ms (ref 0.178ms@greedy, 0.57x)  rss+7MiB
```

stderr（`--verbose true`）：

```
equation : cadb,menf,ac,em,bd,fn->
device   : cpu  dtype=complex128  backend=auto
operands : 6 个，共 66 元素；输出 [] (1 元素)
shape    : 共 6 个操作数形状（按方程中从左到右的顺序）
           [4,4,1,1], [4,4,1,1], [4,4], [4,4], [1,1], [1,1]
specialize: auto（实际 True）
time     : 0.101 ms （warmup=1，rounds=5，中位）
ref opt  : 要 auto → 参照实际用 'greedy'
reference: np.einsum(optimize='greedy')
ref time : 0.178 ms  （pyqcd/参照 = 0.57×）
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

- 多个入口同时命中时用 ` + ` 连接（`test.py:1513`，`detail.route_joiner`）。
- 原生符号映射见 `test.py:1506-1512`：

| `kernel_info[*]['path']` | route 行里显示的原生符号 |
|---|---|
| `kupl_dedicated` | `qcd_kupl_dedicated_apply` |
| `kupl_topology` | `qcd_kupl_topology_apply` |
| `cuda_dedicated` | `qcd_cuda_dedicated_apply` |
| `cuda_topology` | `qcd_cuda_topology_apply` |
| `kupl_topology_v2` | 原样显示 `kupl_topology_v2`（映射表无此键，回退成 path 字符串） |

**必须区分 `kupl_topology` 与 `kupl_topology_v2`**：只有前者表示真的走了编译期 v3
步表，后者是 v2 通用执行器在跑；两者**数值都正确**，只有性能含义不同
（`_kernel_kupl.try_topology_apply` 的 `LAST_INFO['path']` 契约）。

> 本机实测（CPU/KUPL）：`seq 3/4/6/7/8/9/10/2583/2584/3431` 的 `route` 全是
> `kupl_dedicated`；只有被 pack-bound 守卫逼回通用时才会出现 `通用（无原生命中…）`。
> `kupl_topology*` 这两条**在本机没能用 CLI 复现**（`try_dedicated` 在分派顺序里
> 更靠前，先把这些计划接走了），上表后三行取自源码，不是实跑。

`route` 行只读 `kernel_info` 里的 `path`；`kernel_info` 本身由 `test.py:1229-1240`
从 KUPL/CUDA 两个模块的 `LAST_INFO` 汇总，键为 `kupl`/`cuda`，
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
# 省略 --shape 但方程不在语料里
python -m pyqcd.einsum.test --equation "ij,jk->ik"
# error: 省略 --shape 时方程必须在语料里（用于按 roles 反推生产尺寸），
#        但 'ij,jk->ik' 不在 distill 语料中。请显式给 --shape。   → 退出码 2
```

```bash
# --warmup 0 被拒
python -m pyqcd.einsum.test --equation "ij,jk->ik" --shape "[4,4],[4,4]->[4,4]" --warmup 0
# error: --warmup 不能为 0：首调包含 plan 缓存/原生库绑定/JIT，不丢弃会让耗时完全失真
#                                                              → 退出码 2
```

### 1.6 `--scene distill` 不等于"生产大张量"

`distill` 档由基准脚本 `refer/sush/sush-einsum.py` 的实际张量决定：`Mom_src=1`、
`Lt` 不是张量轴，因此很多轴长是 **1**。上面 `cadb,...` 的 6 个操作数总共只有
**66 个元素**——耗时是微秒级，只能当**路径证据**，不能当性能结论。
要看生产规模请用 `--shape` 显式给真实形状，或换 `--scene distill_curr`
（`Nev=400`、`Mom_curr=9`）。这一条与 `pyqcd/einsum/AGENTS.md` §6.1 的口径一致。

---

## 2. 我想确认它是否命中语料特化

### 2.1 命令

```bash
python -m pyqcd.einsum.verify_specialized "cadb,menf,ac,em,bd,fn->"
python -m pyqcd.einsum.verify_specialized "<eq>" --json     # 结构化
```

纯查询：不生成操作数、不调用 `contract()`、不改全局状态
（`verify_specialized.py` 模块 docstring）。判定顺序是
**形状特化（`_shape_paths`）→ 语料索引（`_specialize.match`，8638 条）→ 等价映射**。

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

**② MISS，但有等价写法**——`MISS` + `等价写法` 段（可映射）：

```
结果        : MISS
  原因      : 直接写法不在语料里，但存在等价写法可命中特化路径（…）
  等价写法  : cadb,menf,ac,em,bd,fn->
  变换类型  : operand-axes
  变换说明  : 操作数 0 轴序改为 cadb（张量需 transpose(1, 0, 2, 3)）
  张量调整  : transpose(1, 0, 2, 3), -, -, -, -, -
  提示      : 等价改写只改「方程怎么写」，不改任何执行路径；…
```

（`…` 处为排版省略，这段的**逐字全文**见 §3.4 例 1。）

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
  "equation_normalized": "acdb,menf,ac,em,bd,fn->",
  "parsable": true,
  "matched": false,
  "specialize_enabled": true,
  "signature": "((0, 1, 2, 3), (4, 5, 6, 7), (0, 1), (5, 4), (3, 2), (7, 6), ())",
  "arity": 6,
  "slots": 16,
  "digest": "91104304ce1d8e05ab572591",
  "output_rank": 0,
  "equivalent": {
    "equation": "cadb,menf,ac,em,bd,fn->",
    "rewritten": true,
    "operand_order": null,
    "operand_permutations": [[1, 0, 2, 3], null, null, null, null, null],
    "output_permutation": null,
    "description": "操作数 0 轴序改为 cadb（张量需 transpose(1, 0, 2, 3)）",
    "kind": "operand-axes"
  },
  "reason": "直接写法不在语料里，但存在等价写法可命中特化路径（…）"
}
```

（`reason` 的完整文本见 §3.4 例 1 的「原因」行，这里只为排版省略中间部分；
`operand_permutations` 实际是**多行缩进**的数组，等价于上面的紧凑写法。）

HIT 时 `equivalent` 不出现，改为 `name` / `kind` / `kernels` / `corpus`。
字段口径（含 `name` 与运行时提示 `topology:<摘要前 8 位>` 一致、`digest` 是完整
96 位摘要、`kernels` 只是**声明**）见该模块 docstring。

> 工作树版本还会多出 `reason_key` / `reason_args`（语言无关的稳定形式）。
> **HEAD 没有这两个键**，见 §8。

### 2.4 退出码

| 码 | 含义 |
|---:|---|
| 0 | 命中（语料或形状特化） |
| 1 | 未命中——实际会走通用分步路径 |
| 2 | 方程无法解析，或参数非法 |

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

这是本轮新增能力的核心场景。**先看清它为什么没命中，再决定要不要改写。**

### 3.1 为什么没命中：签名只归一化"字母改名"

`_specialize.canonical_signature`（`_specialize.py:164-186`）把指标按**全式首次出现
顺序**重新编号，于是 `ij,jk->ik`、`ij,jb->ib`、`ab,bc->ac` 签名相同。它的 docstring
把边界写得很清楚（`:171-172`）：

> 从头到尾扫描**全部操作数**（操作数顺序不变，每个操作数内部的指标顺序也不变），
> 再扫描输出项，把指标按"全式首次出现顺序"重新编号为 `0,1,2,…`

也就是说，签名**只**消掉：

- 字母改名（`i→a`）
- 空白差异（`normalize` 去空白）
- 省略 `->` 时的隐含输出推导

它**不**消掉：**操作数位置**、**单操作数内部轴序**、**输出段轴序**。
这三种写法数学上等价、签名却不同 → 一律 MISS。

> 一个直接的旁证（`_specialize.py:176-179` 的 docstring 原文）：
> `ij,kj->ik`、`ij,jk->ki` 与 `ij,jk->ik` 签名不同，因为"签名只认谁和谁共享、
> 共享的指标是否留在输出、以什么顺序留在输出"。

### 3.2 三族等价变换（`_equivalence.py:192-270`）

| 族 | 例子 | 你要动什么 | 代码位置 |
|---|---|---|---|
| ① `operand-order` 操作数位置 | `ac,em->` ↔ `em,ac->` | 按给出的 `operand_order` **重排张量列表** | `:192-211` |
| ② `operand-axes` 单操作数轴序 | `acdb,…` ↔ `cadb,…` | 对该操作数 `transpose(perm)` | `:213-224` |
| ③ `output-axes` 输出段轴序 | `…->MGHew` ↔ `…->GMHew` | 拿结果后 `result.transpose(output_permutation)` | `:226-270` |

三族都只做**符号层**推导，不需要知道张量形状；也**不做**代数化简（不会把
`ij,jk->ik` 合并成矩阵乘）。

### 3.3 ⚠️ 最容易出错的地方：输入变了，输出也必须变

**正确性必须由你自己保证。** 映射会逐条给出三种调整：

- `operand_order`：`operand_order[i]` = "改写后第 i 个操作数取你原来的第几个"；
- `operand_permutations[j]`：该操作数新轴 j 取旧轴 perm[j]（numpy `transpose` 语义）；
- `output_permutation`：**已经存成逆置换**，直接喂 `transpose` 即可。

最容易踩的坑：**只换了输入忘了换输出**。而且它常常**不会报错**——
`_equivalence.py:66-72` 专门写了原因：

> 设原输出段 `out`、改写后 `new_out[j] == out[perm[j]]`，要还原成 `result_old` 需要
> `axes = perm⁻¹`。**对两两互换 perm 自逆、两者相同，所以测不出方向错误**——
> 必须用三循环这类置换验证。

所以：

- 输出段是**两两互换**（`GMHew ↔ MGHew`）：形状本来就对称，用错方向/漏转都**静默出错**；
- 输出段是**三循环**（`MGHew → HMGew`，perm `(2,0,1,3,4)`，逆 `(1,2,0,3,4)`）：
  形状会从 `(2,3,4,5,6)` 变成 `(3,4,2,5,6)`，此时形状检查能帮你抓到。

判据只有一条：**改写后的结果 `transpose(output_permutation)` 必须逐元素等于原式直接算的结果。**
单族变换的正确性由契约
`pyqcd/testing/_einsum_equivalence_contract.py` 用 numpy 真值对拍钉住
（本机实跑 `12 passed, 0 skipped, 0 failed`）：

```bash
python -B -m pyqcd.testing._einsum_equivalence_contract
```

其中 `test_axis_permutation_mapping_is_numerically_equivalent` 就是例 1 的真实语料形状
对拍，做法可以照抄：

```python
mapped[0] = np.transpose(ops[0], result.operand_permutations[0])
via = pe.contract(result.equation, *mapped, backend="numpy", optimize="greedy")
# 与 pe.contract(原式, *ops, ...) 逐元素相同（≤1e-12 相对）
```

### 3.4 三个真实例子

#### 例 1（单变换，可用）

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
  提示      : 等价改写只改「方程怎么写」，不改任何执行路径；按上面的张量调整同步转换后，
              用等价方程调用 contract() 即可走上特化路径，结果与直接写逐元素相同。
```

退出码 **1**（MISS）。操作：把第 0 个张量 `transpose(1,0,2,3)`，
方程换成 `cadb,menf,ac,em,bd,fn->`。输出是标量（`->`），**没有** 输出轴要转。

#### 例 2（三变换叠加，**当前仍 MISS —— 已知未覆盖**）

```bash
python -m pyqcd.einsum.verify_specialized \
  "cwdx,eufv,mknl,oapb,qgrh,istj,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->GMHew"
```

```
元数/槽位   : 16 个操作数 / 55 个指标槽（输出秩 5）
摘要        : facfbf94f8e3377ee0e65fc9
结果        : MISS
  原因      : 既不在 wick_12cases 语料里，也没有形状特化路径，且不存在可落到已知形式的等价写法，走通用分步路径（无特化内核）
```

（完整输出还有 `方程` / `归一化` / `拓扑签名` / `特化开关` 四行，这里只留与结论相关的行，
其余逐字未改。）

相对语料式（例 3）有三处差异：

| # | 差异 | 族 | 需要的调整 |
|---|---|---|---|
| ① | 操作数 4↔5 互换（`qgrh,istj` vs `isjt,qgrh`） | operand-order | `operand_order = (0,1,2,3,5,4,6,7,8,9,10,11,12,13,14,15)` |
| ② | 原位置 4 的 `isjt→istj` | operand-axes | `transpose(0,1,3,2)` |
| ③ | 输出 `MGHew→GMHew` | output-axes | 结果 `transpose(1,0,2,3,4)` |

（上表 ①②③ 的置换值取自实跑输出：`②` 出现在「只修操作数轴序」那一族的候选里，
`③` 的逆置换 `(1,0,2,3,4)` 与 `_einsum_equivalence_contract` 的断言一致。）

**为什么枚举不出来**：`_candidate_equations` 每次只应用**一族**变换——
①、②、③ 各自是三条独立候选（`:198-211` 只改顺序、`:214-222` 只改一个操作数的轴、
`:232-268` 只改输出段），**没有任何一条候选是两族/三族叠加**。实跑证据：

| 起点（在例 3 基础上施加） | 结果 | 等价写法 |
|---|---|---|
| 只施加 ① | MISS | 无 |
| 只施加 ② | MISS | 无 |
| 只施加 ③ | MISS | 无 |
| ①② | MISS | 有（只需再补 ③） |
| ②③ | MISS | 有（只需再补 ①） |
| ①③ | MISS | 有（只需再补 ②） |
| **①②③ = 例 2** | **MISS** | **无** |

即：**离语料式还差一族时能映射，差两族/三族就不行。** 这是真实的能力边界，
不是用法错误。

> 三者叠加的**等价性**是用户在生产形状上实测的（相对误差 `7.9e-16`）。
> 本手册**没有**重跑这项验证：复现它需要同时施加三族变换后再对拍，
> 而仓库规定单方程验证只能走 `python -m pyqcd.einsum.test`，
> 那条入口只比较"同一条方程 vs 参照实现"，不能跨方程对拍。

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
  只返回方程与张量调整方案（`:274-333`）。
- 全仓库**只有** `verify_specialized.py:91-93` import 它；`_api.py` / `_execute.py`
  **完全不经过**它（`git grep` 实证）。因此 `contract()` 的行为、选路、数值
  与这条等价层存在与否无关。
- 想真正走上特化路径，必须**你在调用侧**按建议改方程 + 改张量；库不会悄悄改写你的
  输入——那会让"我传进去的是什么、算出来的是什么的哪个轴"彻底失去可追溯性。

### 3.6 覆盖面边界（都是实跑确认的）

| 边界 | 事实 | 证据 |
|---|---|---|
| 深度 | 每次只改**一族**，两族及以上叠加不覆盖 | §3.4 例 2 的 7 行探针表 |
| 操作数位置族上限 | 两两互换只试前 **96** 个（`:209` `orders[:96]`）。16 操作数时 `swap(0,1)`、`swap(4,5)` 可映射，`swap(10,11)`、`swap(14,15)` **不可** | 实跑四行探针（见下） |
| 单操作数轴序族 | 秩 ≤4 全排列；秩 >4 只试"整段逆序 / 首尾交换 / 循环移位"（`:42-44,143-160`） | 源码；秩 4 的 `(0,1,3,2)` 实跑可映射 |
| 输出段族 | 两两互换 + 整体轮转 + 子段三循环 +（秩≤4）全排列，上限 96（`:232-268`） | `test_output_axis_family_covers_pairwise_rotation_and_cycle` 实跑 PASS（`GMHew`/`MGHwe`/`HMGew`/`wMGHe` 都能映射回 `MGHew`） |
| 不做 | 代数化简、形状相关优化、跨方程重写 | 模块 docstring |

```
swap 0/1        → MISS，有等价写法（操作数次序 [1, 0, 2, 3, …, 15]）
swap 4/5        → MISS，有等价写法（操作数次序 [0, 1, 2, 3, 5, 4, …, 15]）
swap 10/11      → MISS，无等价写法
swap 14/15      → MISS，无等价写法
operand0 (0,1,3,2) → MISS，有等价写法（transpose(0, 1, 3, 2)）
```

（`…` 只是本手册的显示省略，工具实际打印完整的 16 元组。）

---

## 4. 我想跑性能对照

### 4.1 `--compare` 四档

| 档 | 参照是什么 | 实跑看到的 `reference` |
|---|---|---|
| `numpy`（缺省） | `np.einsum(optimize=<ref opt>)` | `np.einsum(optimize='greedy')` |
| `opt_einsum` | `oe.contract(optimize=<ref opt>)` | `opt_einsum.contract(optimize='dp')` |
| `backend` | **另一个后端**跑同一批张量（跨后端互证） | `pyqcd.contract(backend='cupy', cupy, specialize=auto)` |
| `none` | 不比，只跑主路径 | `PASS` 行不带 `rel=` 与 `vs …` |

`--compare backend` 是唯一能同时验证"两条原生路径都给对答案"的模式
（主路径跑 `--device`，参照在另一个后端上跑）。它要求**另一个后端可导入**：
本机 cupy 可用，所以 `--device cpu` 时参照落在
`pyqcd.contract(backend='cupy', …)`；没有可用的另一后端时这条模式用不了，
请退回 `numpy`/`opt_einsum`。

### 4.2 ⚠️ `--compare-optimize` 的陷阱：两个库的名字不通用

`--compare-optimize` **只管参照**，主路径永远看 `--optimize`。

```bash
# 会失败：numpy 侧只有 greedy / optimal
python -m pyqcd.einsum.test --equation "ij,jk->ik" --shape "[4,4],[4,4]->[4,4]" \
    --compare numpy --compare-optimize dp
```
```
error: --compare numpy 不接受 --compare-optimize 'dp'：np.einsum(optimize=...) 实测只认 ('greedy', 'optimal') 两个名字，其余抛 KeyError('Path name %s not found')。
  提示: 想要 'dp' 请用 --compare opt_einsum（该名字在 opt_einsum 的路径注册表里），不要把它硬映射成一个语义不同的 numpy 名字
```
退出码 **2**。**这不是你写错了**，也不是"静默退回 greedy"——是刻意的用法错误，
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
# warning: --compare-optimize 'dp' 对 --compare none 无效（该模式没有可设的优化器）
```

报告里同时有 `compare_optimize`（要什么）与 `compare_optimize_effective`（实际用了什么）。

### 4.3 `--specialize false`：看通用路径的真实代价

```bash
python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" \
    --scene distill --specialize false --verbose true
# specialize: false（实际 False）
# kernel   : 无原生命中（通用路径）
# route    : 通用（无原生命中：Python 分步 / numpy.einsum / BLAS）
```

实测一对（`aibj,ckdl,eqfr,msnt,oupv,gwhx,ce,og,ik,su,bdf,nph,jlr,tvx->aqmw`
= 语料 seq=129，`--scene distill`，complex128，`--rounds 1`，两次测量都重复到了同一量级）：

| 模式 | 耗时 | `route` | 说明 |
|---|---:|---|---|
| `--specialize auto` | 1635.455 ms | 通用 | 先尝试原生入口（签名匹配 / 计划推演 / 各道守卫），被拒后才回落 |
| `--specialize false` | 668.820 ms | 通用 | 完全不进原生入口 |

**两次的 `route` 一模一样，耗时差 2.45×。** 也就是说：
"原生拒绝"不是免费的——尝试这条路本身（签名匹配、计划推演、各道守卫）就要花钱，
本手册没有把它逐层归因到某一项。
所以 `--specialize false` 是排查原生内核问题的第一手段：先拿到"完全不经原生"的基线，
再和 `auto` 比，差值就是原生尝试 + 原生执行的净代价。

上面这条方程上原生路径**没赢**：`--specialize auto` 的 stderr 明确写了
`topology declined: pack-bound on CPU (3456077824 pack bytes, 216326912 FLOPs)`
（`pyqcd/einsum/AGENTS.md` §6.2「顺带核实的 bench 覆盖口径」一节也记了
"seq=129 被 pack-bound 守卫拒绝"）。但**别把它当通例**——
`AGENTS.md` §6.4 记录的附录 seq=2583 是 v3 步表快 1.7–2.9×。

**注意 `--specialize true` 不是"一定会走原生"**：它是"强制允许特化"，
原生侧的门（pack-bound / dtype 策略 / arena）照样会拒，拒绝时 route 行如实写通用
（本机 seq=129 实测：`--specialize true` 仍 PASS 且走通用）。

### 4.4 为什么必须先丢 warmup（而且不允许设成 0）

首调包含 plan 缓存、原生库绑定（必要时含 JIT），与热态不是一个量级。
`pyqcd/einsum/AGENTS.md` §9.0 记录过 `seq=2583` 首调 **45 ms** vs 热态 **0.56 ms**
（**80×**）。因此本入口：

- `--warmup` 缺省 1，**传 0 直接报错**（退出码 2，见 §1.5）；
- 计时只取 `--rounds` 次中位；
- 内存采样点（构造前 / 构造后 / 收尾）**都在计时循环之外**，不污染中位。

---

## 5. 我要把结果喂给脚本

### 5.1 退出码（`python -m pyqcd.einsum.test`）

| 码 | 含义 |
|---:|---|
| 0 | 通过（`ok=true`），或 `--compare none` 跑完 |
| 1 | 数值或形状不符 |
| 2 | 参数或环境错误（含 `--json` 写不进去、`--compare-optimize` 不兼容） |

实跑对照：

```bash
# 1 的典型来源：单精度用缺省 --atol=1e-12
python -m pyqcd.einsum.test --scene distill --dtype float32 \
    --equation "cadb,egfh,minj,okpl,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek"
# FAIL cadb,egfh,minj,okpl,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek
#   rel=2.117e-07 > atol=1e-12
#   actual shape=[100, 100] expected=[100, 100]        → 退出码 1

# 同一命令加 --atol 1e-4
# PASS cadb,egfh,minj,okpl,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek rel=2.12e-07 \
#      vs np.einsum(optimize='greedy')  6.932ms (ref 4.504ms@greedy, 1.54x)  rss+39MiB  → 0
```

`--atol` 是**相对**容差（按 `max|expected|` 归一），不是逐元素绝对阈值。
`float32/complex64` 常需 `--atol 1e-6~1e-4`，这是精度口径不是 bug。

### 5.2 `--json` 是**路径**不是开关

```bash
python -m pyqcd.einsum.test … --json /path/out.json     # 打印 "wrote /path/out.json"
python -m pyqcd.einsum.test … --json                    # error: argument --json: expected one argument → 2
python -m pyqcd.einsum.test … --json /nonexistent/x.json # error: 无法把 --json 写到 … → 2
```

### 5.3 字段表（实跑抓取，`test.py --json`）

| 字段 | 类型 | 含义 |
|---|---|---|
| `ok` | bool | 容差 + 形状是否通过（**脚本只认这个**） |
| `rel` | float | 相对误差（按 `max\|expected\|` 归一） |
| `atol` / `rtol` | float | 本次实际使用的容差 |
| `tolerance` | float | 判定时用的合成容差（与 `atol`/`rtol` 同源） |
| `max_abs_diff` | float | 最大绝对差（诊断用） |
| `out_shape` | list[int] | 期望输出形状 |
| `actual_shape` | list[int] | 实得输出形状 |
| `shapes` | list[list[int]] | 各操作数形状（按方程从左到右） |
| `kernel_info` | dict | `{backend: {path, specialized?, steps?, dtype?, copied?}}`，可能为空 `{}` |
| `specialize` | str | 请求值（`auto`/`true`/`false`） |
| `specialize_effective` | bool | 开关实际是否打开。**注意：true ≠ 命中原生** |
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

**`equivalence` 字段在 HEAD 的 `test.py --json` 里不存在**（`git show HEAD` 无该键）。
写作期间工作树正在加入它（配合 `--equivalence`），本手册**未验证**其形态。
在 HEAD 上，等价映射只出现在 `verify_specialized --json` 的 **`equivalent`**
（单数、子对象）里。

### 5.4 稳定性契约

- **`--json` 的字段集合是稳定契约**，`--language` 不改它。
  实测同一命令 `--language zh-cn` vs `en`：字段集合完全相同，
  **只有 `shape_source` 的值被翻译**；其余差异只有 `seconds`/`reference_seconds`/
  `memory` 的测量抖动。
- `--language` 也**不写任何环境变量**，不影响同进程后续调用。
- 若要用脚本判断"走的哪条路"，请读 `kernel_info[*]['path']`（缺省即无原生），
  **不要**去解析 `route` 那句人话（它随 `--language` 变）。

---

## 6. 环境变量总表

> 判定都是实跑：对应 §9 清单的第 18-23 行（NOTICE 开/关、locale 与 LANG、
> `PYQCD_EINSUM_SPECIALIZE=0`、`KERNEL_KUPL=0`、`TOPOLOGY_REAL=1`、`TOPOLOGY_TRACE=1`）。

| 变量 | 缺省 | 作用范围 | 典型用法 | 性质 |
|---|---|---|---|---|
| `PYQCD_EINSUM_NOTICE` | **开**（除 `0/false/off/no` 外都算开） | `_specialize.announce`（命中特化时）与 `announce_generic`（未命中时）的**一次性提示**，每进程每形态一次 | `PYQCD_EINSUM_NOTICE=0` 静音 | **日志开关**。不改选路、不改数值。`--verbose` 之上的额外噪音 |
| `PYQCD_EINSUM_SPECIALIZE` | **开** | 进程级专用路径总开关（`_specialize.enabled()`）。`=0` 后**同进程所有** `contract()` 都走通用 | `PYQCD_EINSUM_SPECIALIZE=0` 做整进程对照 | **能力/选路开关（进程级）**。只想改**一次**调用请用 `--specialize false`，不要用它的环境变量版 |
| `PYQCD_EINSUM_LANG` | **未设** | `_specialize.notice_language()`（库提示）与 CLI 的 `--language auto` | `PYQCD_EINSUM_LANG=en` / `zh` | **只改文案**。注意两条入口的 `auto` 语义不同（见下） |
| `PYQCD_EINSUM_EQUIVALENCE` | **HEAD 不存在** | 写作期间工作树开始加入（`EQUIVALENCE_ENV`，未提交） | — | **开发中/未合入**：HEAD 零命中。工作树的新实现本手册**未验证**，别照本文档假设它可用。注意：等价映射本身在 HEAD 已经是 MISS 时的**自动行为**（§2.2 ②），不依赖这个变量 |
| `PYQCD_EINSUM_TOPOLOGY_REAL` | **未设**（`float64`/`float32` 被策略排除在拓扑入口外） | `_kernel_kupl._topology_dtype_declined`：`=1`（`true/on/yes`）解除该策略 | `PYQCD_EINSUM_TOPOLOGY_REAL=1` 做 A/B | **绕过选路做对照实验**，**不是**能力开关：四类 dtype 契约上一直平权，探测也一直通过，只是分派层主动让 dedicated 优先 |
| `PYQCD_EINSUM_KERNEL_KUPL` | **开** | KUPL 原生入口（dedicated + topology）是否参与下发 | `=0` 完全不用 KUPL 原生库 | **能力开关（模块级）**。与 `--specialize false` 的区别：后者是"这次别用"，前者是"这个入口别用" |
| `PYQCD_EINSUM_TOPOLOGY_TRACE` | **关** | 原生 C++ 侧的逐步诊断行（`topology-gemm-step …`、`topology-specialized: …`） | `=1` 排查 v3 步表是否真的在跑 | **诊断日志**。`trace_enabled()` 在原生侧**静态缓存**：同进程里改环境变量已经晚了，必须一开始就设 |

### 6.1 `--language auto` 不读 locale（反直觉点，必须记住）

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
# ref opt  : 要 auto → 参照实际用 'greedy'
# route    : 特化 · 原生 qcd_kupl_dedicated_apply（path=kupl_dedicated）

# 库提示跟随 locale → 中文（注意 NOTICE 没关，且给的是显式 --shape）
LANG=zh_CN.UTF-8 LC_ALL=zh_CN.UTF-8 python -m pyqcd.einsum.test \
    --equation "ij,jk->ik" --shape "[4,4],[4,4]->[4,4]"
# [pyqcd.einsum] 通用路径: 未命中特化形态 (ij,jk->ik)；执行的是通用分步收缩。…

# 同一个环境变量让 CLI 变英文（对照上一块的四行）
PYQCD_EINSUM_LANG=en python -m pyqcd.einsum.test … --verbose true
# shape    : 6 operand shape(s), left to right, in the equation's own order
# specialize: auto (effective True)
# ref opt  : requested auto → reference actually uses 'greedy'
# route    : specialized - native qcd_kupl_dedicated_apply (path=kupl_dedicated)

# 命令行优先于环境变量，但**不改**库提示的语言（NOTICE 仍开着，所以能看到这行）
PYQCD_EINSUM_LANG=en python -m pyqcd.einsum.test … --language zh-cn
# [pyqcd.einsum] specialized path: topology:…   ← 库提示仍是英文（它只读环境变量）
```

### 6.2 存在但本手册不展开的变量

以下变量在源码里可查（`git grep` 定位），属于构建/线程/探测层，与本手册的任务无关：

| 变量 | 读取位置 |
|---|---|
| `PYQCD_EINSUM_SPECIALIZE_VERBOSE` | `_specialize.py:46`（每次命中都打印） |
| `PYQCD_EINSUM_DISABLE_FUSED` | `_specialize.py:54` |
| `PYQCD_EINSUM_KERNEL_CUDA` | `_kernel_cuda.py:25` |
| `PYQCD_EINSUM_NATIVE` / `_LIB` / `_KUPL_LIB` / `_CUDA_LIB` / `_CBLAS_LIB` | `_native.py:24,39`、`kupl/einsum.py:82-83,173` |
| `PYQCD_EINSUM_THREADS` / `_OMP_THREADS` / `_BLAS_THREADS` | `_threads.py:50-52` |
| `PYQCD_EINSUM_POOL` / `_CONTEXT_CACHE_LIMIT` | `_execute.py:28`、`_api.py:55` |
| `PYQCD_EINSUM_SHAPE_PATH_MAX_BYTES` | `_shape_paths.py:45` |
| `PYQCD_EINSUM_NEGATIVE_CACHE` / `_DEDICATED_PLAN` / `_KUPL_PLAN` | `_kernel_kupl.py:58-59`、`_api.py:1278` |

---

## 7. 我的方程改了还是 MISS，怎么办（决策树）

```
verify_specialized 报 MISS
├─ 有「等价写法」行
│   ├─ 只差一族 → 按「张量调整 / 张量次序 / 结果调整」改写调用侧，然后 contract() 走上特化
│   └─ 记得：输入变了，输出该转的也一定要转（§3.3）
├─ 没有「等价写法」行
│   ├─ 方程确实不在 8638 条语料里 → 走通用路径是正常结果，不是配置错
│   ├─ 你确认它与某条语料式等价（如例 2 三族叠加）→ **当前未覆盖**，
│   │   要么手工改写成语料式，要么接受通用路径
│   └─ 怀疑运算符数/输出轴序之外的东西（例如代数化简）→ 本层不做，别等
└─ 报 HIT，但 test.py 的 route 是通用
    ├─ 看 kernel 行：为空 → 原生门拒绝了（pack-bound / dtype 策略 / arena）
    ├─ 看 stderr 的 RuntimeWarning：拒绝原因写在里面
    └─ 用 --specialize false 量一次通用基线，再决定值不值得追原生
```

---

## 8. 能力边界与"开发中"清单（诚实版）

| 能力 | HEAD 状态 | 核验方式 | 本手册怎么写 |
|---|---|---|---|
| `python -m pyqcd.einsum.test` | **已可用** | 实跑 | §1、§4、§5 |
| `--specialize {auto,true,false}` | **已可用** | 实跑（§1.2、§4.3） | 三态语义已写；`true` 不保证命中原生 |
| `--compare-optimize` | **已可用** | 实跑四档 + 两条错误路径 | 陷阱已写（numpy 只有 greedy/optimal） |
| `--language {auto,en,zh-cn}`（`test.py`） | **已可用**（HEAD 就有） | 实跑（§6.1） | 反直觉点已写（auto 不读 locale） |
| `--language`（`verify_specialized`） | **开发中**（HEAD **没有**；工作树有） | `git show HEAD:pyqcd/einsum/verify_specialized.py` 只有 `--json` | §2 的输出按 HEAD 格式写；`reason_key`/`reason_args` 标为未合入 |
| `--equivalence`（CLI 选项） | **HEAD 没有**；写作期间工作树正在加入 | `git show HEAD:pyqcd/einsum/test.py` 的 `build_parser` 无该选项；`git grep HEAD` 零命中 | 明确写"HEAD 不存在、工作树开发中"；等价映射在 HEAD 已是自动行为 |
| `PYQCD_EINSUM_EQUIVALENCE` | **HEAD 没有**；工作树在加入 | 同上 | 明确写"HEAD 不存在、未验证" |
| 等价映射层（`_equivalence.py`） | **已可用**（自动） | 契约 12/12 PASS + 三个例子实跑 | §2.2 ②、§3 |
| 多族叠加映射（例 2 类） | **未覆盖** | 7 行探针（§3.4） | 如实写"已知未覆盖" |
| `kupl_topology` / `kupl_topology_v2` 的 route 展示 | 代码支持，**本机未复现** | 源码 `test.py:1506-1512` | §1.3 标注"取自源码，不是实跑" |
| `PYQCD_EINSUM_TOPOLOGY_REAL` 的可观测差异 | 变量存在，**本机测试用例上无差异** | §9 第 22 行 + 实跑 warning 文本点名该变量 | 如实写"`try_dedicated` 先命中，测不出差异" |

### 8.1 与 HEAD 的逐条对应

- 本文所有**选项名、缺省值、choices** 取自 `git show HEAD:pyqcd/einsum/test.py`
  的 `build_parser`（`:1565-1619`）与 `HEAD:pyqcd/einsum/verify_specialized.py:212-227`。
- 所有**环境变量名**取自 HEAD 的源码常量（`_specialize.py:46-54,312`、
  `_kernel_kupl.py:36,58-59,219`），并用 `git grep HEAD` 确认
  `PYQCD_EINSUM_EQUIVALENCE` **零命中**（工作树随后开始加入，见上表）。
- 等价层的**能力边界**取自 `HEAD:pyqcd/einsum/_equivalence.py` 的常量与分支
  （`:42-45`、`:143-160`、`:192-270`），并用实跑探针复核。
- 中文输出格式取自 `HEAD:pyqcd/einsum/verify_specialized.py:157-209` 的
  f-string 字面量；实跑用的是工作树（该文件的未提交改动只新增英文分支）。

---

## 9. 本手册的实跑验证清单

环境：`pyqcd-einsum-cu124` / `PYTHONPATH=$PWD` / `PYQCD_EINSUM_NOTICE=0` /
`HEAD=71971265`；**未设置** `CUDA_VISIBLE_DEVICES`。

| # | 命令（要点） | 结果 |
|---|---|---|
| 1 | `python -m pyqcd.einsum.test --help` | 全部选项/缺省/退出码/口径 |
| 2 | `… --equation "cadb,menf,ac,em,bd,fn->" --scene distill --verbose true` | `route=特化·qcd_kupl_dedicated_apply`，`rel=1.54e-16` |
| 3 | 同上 `--specialize false` | `route=通用`，`kernel=无原生命中（通用路径）` |
| 4 | `… --equation "ij,jk->ik" --shape "[4,4],[4,4]->[4,4]"` | `specialize 实际 True` 但 route 通用（关键反例） |
| 5 | `… --compare numpy --compare-optimize dp` | 用法错误文案 + 退出码 2 |
| 6 | `… --compare opt_einsum --compare-optimize dp` | `ref opt 要 dp → 实际 'dp'` |
| 7 | `… --compare none --compare-optimize dp` | warning，退出码 0 |
| 8 | `… --compare backend` | 参照 `pyqcd.contract(backend='cupy', …)`，退出码 0 |
| 9 | `… --dtype float32 --atol` 缺省 / `1e-4` | 退出码 1（rel 2.117e-07）/ 0 |
| 10 | `… --warmup 0` | 退出码 2 |
| 11 | `… --json <path>` / 无路径 / 不可写路径 | `wrote …` / 2 / 2 |
| 12 | `--json` 在 `--language zh-cn` vs `en` | 字段集合相同，仅 `shape_source` 变 |
| 13 | `verify_specialized` 例 1 / 例 2 / 例 3 | MISS+等价 / MISS 无解 / HIT（seq=8628） |
| 14 | 例 2 的 7 行族组合探针 | 单族与三族叠加 MISS，两族可映射 |
| 15 | `swap 0/1`、`swap 4/5`、`swap 10/11`、`swap 14/15`、`operand0 (0,1,3,2)` | 前二与末一可映射；中间两个不可（96 上限） |
| 16 | `verify_specialized "a->b"` | 退出码 2 |
| 17 | `PYQCD_EINSUM_SPECIALIZE=0 verify_specialized <HIT 式>` | `特化开关 关闭` 但 `结果 HIT` |
| 18 | `PYQCD_EINSUM_NOTICE` 开/关 | 提示出现/消失 |
| 19 | `LANG=zh_CN` 库提示；`LANG=C` CLI 输出 | 库提示中文；CLI 仍中文（auto 不读 locale） |
| 20 | `PYQCD_EINSUM_LANG=en`（+ `--language zh-cn`） | CLI 英文；CLI 中文而库提示仍英文 |
| 21 | `PYQCD_EINSUM_KERNEL_KUPL=0` | route 通用 |
| 22 | `PYQCD_EINSUM_TOPOLOGY_REAL=1`（`cadb,…`、seq=2583、seq=3431、float64） | route 无变化（`try_dedicated` 先命中） |
| 23 | `PYQCD_EINSUM_TOPOLOGY_TRACE=1`（seq=129、complex128） | 打出 `topology-gemm-step …` / `topology-specialized: … dtype=0 steps=5 sizes=8`，随后 warning 说明 `pack-bound` 拒绝 |
| 24 | `--specialize auto` vs `false`（seq=129、complex128、rounds=1） | 1635.455 ms vs 668.820 ms（重复测量 1629.937 / 653.719，量级一致） |
| 25 | `python -B -m pyqcd.testing._einsum_equivalence_contract` | `12 passed, 0 skipped, 0 failed` |

未复现项（已在上文逐处标注）：例 2 三变换叠加的 7.9e-16 数值对拍、`kupl_topology*`
的 route 展示、`PYQCD_EINSUM_TOPOLOGY_REAL=1` 带来的可观测选路变化。
