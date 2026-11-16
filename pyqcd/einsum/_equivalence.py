"""等价方程映射：把"同一个收缩换了个写法"映射回语料里的写法。

为什么需要这一层
----------------
``_specialize.canonical_signature`` 只做**字母重命名**的归一化（指标按全式首次
出现顺序重新编号），因此下面这些**数学上等价**的写法会被判成不同签名、一律 MISS：

1. **单操作数内部轴序**：``cadb,...`` 与 ``acdb,...``——后者只是第 0 个操作数把
   ``c``/``a`` 两个轴换了位置（调用方把对应张量 ``transpose`` 一下即可）。
2. **操作数位置**：``ac,em->`` 与 ``em,ac->``——两个输入交换次序。
3. **字母重命名**：已由 ``canonical_signature`` 覆盖，本层不重复处理。
4. **输出段轴序**：``...->GMHew`` 与 ``...->MGHew``——收缩完全相同，结果互为
   ``transpose``（调用方拿到结果后转回来即可）。
5. **以上任意复合**：位置、轴序、输出轴序**同时**变化。

这五类变换都**保持收缩结果逐元素不变**，本层据此给出"用哪种写法能命中、要动哪些
张量"。

算法：结构不变量分桶 + 构造式同构（**不枚举候选方程**）
--------------------------------------------------------
旧实现的做法是"枚举等价写法 → 逐个 probe"，再用一堆上界（``orders[:96]``、
``_MAX_CANDIDATES`` 等）防止爆炸；上界本身又造成覆盖缺口（16 操作数时
``swap(10,11)``/``swap(14,15)`` 无解，用户的三重复合直接 MISS）。本层改为两次
O(incidence) 的判定：

1. **结构不变量**（:class:`_Shape`）：每个标签记 ``(出现重数, 是否在输出,
   零共现伙伴数, 共现重数多重集)``，每个操作数记 ``(秩, 其标签签名多重集)``；
   桶键 = (操作数类多重集, 输出标签签名多重集)。这些量对**字母改名、操作数次序、
   操作数内轴序、输出轴序**全都不变，因此等价方程**必然同桶**（无漏判）。
   复杂度 O(incidence + Σ Σ|term|²)。
2. **构造式同构**（:func:`_match`）：在桶内对候选语料条目构造
   ``(操作数双射, 标签双射)``，用「秩/签名/重数/共现」四类约束做回溯剪枝，命中
   即得**精确**映射；节点数超过 :data:`_MAX_SEARCH_NODES` 就放弃（放弃只会漏判，
   不会误判）。真实方程高度刚性，实测每个候选几微秒。
3. **字段由双射机械翻译**（:func:`_build`）：位置、轴序、输出轴序三段纯置换合成，
   O(Σ|term|)；翻完再做一次**符号回放**自检（把调用方方程按三字段重排后必须与
   输出方程逐字相同），不通过就丢掉该候选。

由于第 3 步的回放自检，本层**只会给出可验证的映射**：桶键撞了（两条不同构的方程
落进同一桶）也只会被拒绝，不会把错映射交出去。

同桶多条同构写法怎么挑
----------------------
一个桶里常有多条**结构同构**的语料写法（语料本身就是同构重排的集合，最大桶 384
条），挑哪条只影响"调用方要动什么"。:func:`_select_mapping` 的口径是**零代价优先**：
先在前 :data:`_MAX_ZERO_COST_SCAN` 条里找"零操作数轴置换且零输出置换"的映射（只改
``operand_order`` 不算代价，那只是换个实参次序；每条先用 :attr:`_Shape.axis_key`
这道必要条件筛过），找不到才在同一排序下取第一条成功者。理由是转置返回视图、通常
非 C 连续，``_kernel_kupl.try_dedicated`` 的连续性门会直接拒掉它——同桶明明有零代价
写法，却因为 :func:`_alignment` 把"被重排位置上的字母巧合"当成"更接近"而挑了要转置
的那条，等于把查询推出特化路径。零代价档内仍按 :func:`_alignment` 排序，保留"改动
最小 / 最好读"的取向。

返回值口径
----------
``equivalence.equation`` 用的是**调用方的字母**、语料的**结构写法**：收缩结构与
语料条目完全相同，因此 ``_specialize.match`` 必命中（它只认拓扑签名，与字母无关），
而三个字段只表达纯粹的重排（不表达改名），可以直接照做。

开关
----
``PYQCD_EINSUM_EQUIVALENCE`` 默认 ``"1"``（开启）；设 ``0/false/off/no`` 关闭后
**产出映射的两个入口**变成空操作（:func:`normalize_equation` 返回 ``None``、
:func:`find_equivalents` 返回 ``[]``），``verify_specialized`` / ``test.py`` 的
等价段整段消失、MISS 原因回到**没有本功能时**的原措辞。开关在**调用时**读环境
变量（不在 import 时求值），同进程内改环境变量即时生效。
:func:`explain` **不读开关**：它是纯报告，且两个 CLI 的 ``--help`` 都写明
``--equivalence true`` 压过环境变量——若它也拦开关，显式 ``true`` 在环境变量为 0
时永远拿不到映射（``test.py`` 不做环境变量翻转）。关闭时它照实报告并附
``disabled=True``。

本层只影响查询报告，**不影响任何数值计算**：``contract()`` 仍按调用方给的方程走。

``probe`` 参数
--------------
``normalize_equation(..., probe=...)`` 是给调用方自定义"目标形态"的兼容入口，
语义仍是"找到一条通过 probe 的等价写法"。语料索引对这种自定义目标帮不上忙，
因此**只有显式传 probe 时**才退回有界改写枚举（:func:`_probe_candidates`，
上界 :data:`_MAX_PROBE_CANDIDATES`）。默认路径（``probe is None``）零枚举。

设计上的保守点
--------------
* 只在**能构造出精确同构**时给映射（轴序 / 位置 / 输出轴序），绝不做
  "看起来差不多"的猜测；``ij,jk->i`` 与 ``ij,jk->ik`` 这类真正不同的收缩不会被
  当成等价；
* 不做代数化简（如把 ``ij,jk->ik`` 合并成矩阵乘），那是改变计划的范畴；
* 映射完全由**符号层面**推导，不需要知道张量形状。
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, permutations, product
from typing import Any, Iterator, Sequence

__all__ = [
    "EQUIVALENCE_ENV",
    "Equivalence",
    "enabled",
    "normalize_equation",
    "find_equivalents",
    "explain",
]


#: 等价映射总开关：默认**开启**（``"1"``），``0/false/off/no`` 关闭。
#: 与 ``_specialize.NOTICE_ENV`` / ``DISABLE_ENV`` 同一风格；只影响查询报告，
#: 不影响任何数值计算。
EQUIVALENCE_ENV = "PYQCD_EINSUM_EQUIVALENCE"

#: 开关关闭时 :func:`explain` 附带的说明（机器可读别名见 ``disabled_reason``）。
DISABLED_REASON = "等价映射已关闭（PYQCD_EINSUM_EQUIVALENCE=0）"

# 单个候选的同构搜索节点上限。真实语料高度刚性（实测每候选几微秒），触发这个
# 上界意味着桶键撞上了一条结构相近但不同构的方程；此时放弃该候选即可——放弃
# 只会漏判，绝不会误判。
_MAX_SEARCH_NODES = 4096

# 单个操作数内"标签配对"的候选上限（秩 k 时有 k! 种）。语料全部操作数秩 ≤ 4
# （24 种），所以该上限只对语料外的高秩方程生效，且只影响是否继续尝试。
_MAX_LABEL_OPTIONS = 512

# 默认路径"零代价优先"这一趟最多看多少条同桶条目（见 :func:`_select_mapping`）。
# 依据：对语料 8638 条各做一次"交换前两个操作数"的 order 变体全量扫描，首个零代价
# 映射在对齐序里的名次**最大是 1**（6870 条落在第 0 位、1768 条落在第 1 位），
# 而最大桶有 384 条；取 64 既远大于实测的最坏名次（留出余量给语料将来长胖），又
# 只占最大桶的 1/6，第一趟不可能扫完整个桶（"未命中就退回旧行为"的代价因此有界）。
# 触发这个上界只会**降级到旧行为**（同一排序下取第一条成功者），不会误判：本层
# 每条映射都要过 :func:`_build` 的符号回放自检，"挑哪条"只影响报告的代价。
_MAX_ZERO_COST_SCAN = 64

# 显式 ``probe`` 兼容路径的枚举上界（默认路径不用）。
_MAX_PROBE_CANDIDATES = 2048
# 显式 ``probe`` 路径只为小方程保留"全排列"档；超过这个规模只试两两互换等
# 低阶改写，避免阶乘爆炸（该路径本身只服务调用方自定义的目标形态）。
_PROBE_FULL_PERM_LIMIT = 6

# 语料索引：``结构键摘要 -> [(seq, equation), …]``，第一次查询时建一次。
_INDEX: dict[int, list[tuple[int, str]]] | None = None
# 语料条目的结构描述按需展开后的缓存（只存真正比对过的）；上界防止长时间累积。
_SHAPES: dict[int, "_Shape"] = {}
_SHAPE_CACHE_LIMIT = 4096
# 已解析写法的缓存（桶内排序要逐条读语料写法）。
_PARSED: dict[str, tuple[list[str], str]] = {}


def enabled() -> bool:
    """等价映射是否启用（默认开启，``PYQCD_EINSUM_EQUIVALENCE=0`` 关闭）。

    在**调用时**读环境变量（不在 import 时求值），因此同进程内改环境变量即时生效。
    """
    return os.environ.get(EQUIVALENCE_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


@dataclass(frozen=True)
class Equivalence:
    """一条"等价写法"及其配套的张量调整方式。

    ``equation``
        改写后的方程（可直接交给 ``contract``）。它用**调用方的字母**、语料的
        结构写法，因此只差纯粹的重排，三个字段足以表达。
    ``operand_order``
        张量列表的新次序：``operand_order[i]`` 表示"改写后第 i 个操作数应取
        调用方原始张量的第几个"。``None`` 表示不需要重排。
    ``operand_permutations``
        每个操作数（**按改写后的次序**）自己的轴置换：``perm[j]`` 表示该操作数新轴 j 取旧轴 perm[j]（numpy ``transpose(perm)`` 语义）。``None`` 表示该操作数不需要转置。
        **注意 ``transpose`` 返回视图、通常不连续，而原生特化入口要求 C 连续（``_kernel_kupl.try_dedicated`` 第一道门），
        若目标是那条路径请先 ``np.ascontiguousarray``**，否则会被拒后落到 topology/通用。
    ``output_permutation``
        输出结果的轴置换，**已是逆置换**（``transpose`` 用的就是它）。非 ``None``
        时，调用方拿到改写后的结果需 ``result.transpose(output_permutation)``
        才是原方程的答案。

        为什么存逆置换：设原输出段 ``out``、改写后 ``new_out[j] == out[perm[j]]``。
        ``result_new`` 的轴 j 对应标签 ``new_out[j]``，要还原成 ``result_old``
        需要 ``new_out[axes[k]] == out[k]``，即 ``perm[axes[k]] == k``，故
        ``axes = perm⁻¹``。**对两两互换 perm 自逆、两者相同，所以测不出方向错误**
        ——必须用三循环这类置换验证（``MGHew -> HMGew`` 的 perm 是
        ``(2,0,1,3,4)``、逆是 ``(1,2,0,3,4)``，用错方向形状会从 ``(2,3,4,5,6)``
        变成 ``(3,4,2,5,6)``）。
    ``description``
        人类可读的说明，用于报告。
    ``kind``
        变换类别（``identity`` / ``operand-order`` / ``operand-axes`` /
        ``output-axes`` 及其用 ``+`` 连接复合形态），报告侧据此分类。
    """

    equation: str
    operand_order: tuple[int, ...] | None = None
    operand_permutations: tuple[tuple[int, ...] | None, ...] = ()
    output_permutation: tuple[int, ...] | None = None
    description: str = ""
    kind: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "equation": self.equation,
            "operand_order": (None if self.operand_order is None
                              else list(self.operand_order)),
            "operand_permutations": [
                None if perm is None else list(perm)
                for perm in self.operand_permutations] or None,
            "output_permutation": (None if self.output_permutation is None
                                   else list(self.output_permutation)),
            "description": self.description,
            "kind": self.kind,
        }


def _perm_inverse(perm: Sequence[int]) -> tuple[int, ...]:
    """逆置换：``perm[j]`` 是"新轴 j 取旧轴 perm[j]"，逆置换满足 ``inv[perm[j]] = j``。"""
    inverse = [0] * len(perm)
    for j, source in enumerate(perm):
        inverse[source] = j
    return tuple(inverse)


def _permuted(term: str, perm: Sequence[int]) -> str:
    """按 ``numpy.transpose`` 语义重排一个操作数的指标段（``new[j]=old[perm[j]]``）。"""
    return "".join(term[index] for index in perm)


def _split(equation: Any) -> tuple[list[str], str] | None:
    """拆成 (输入子脚本列表, 输出子脚本)，非法返回 ``None``。

    **每个 term 在入口统一去空白**：``"acdb, menf, ac"`` 若不去空白，第二个操作数
    会带上前导空格、被当成一个额外的轴位，产出的 perm 长度比张量秩多 1（旧实现的
    实测 bug，``operand-axes`` 族 2352/3543 条受影响）。这里按
    ``_specialize.normalize`` 的口径把全式空白一律去掉，然后校验
    "输出轴必须来自输入且不重复"（与 numpy 的拒绝口径一致）。
    """
    if not isinstance(equation, str):
        return None
    text = "".join(equation.split())
    if not text or text.count("->") != 1:
        return None
    lhs, rhs = text.split("->")
    terms = lhs.split(",")
    if any(not term for term in terms):
        return None
    if "." in text:
        return None                      # 省略号不在本层处理范围
    known: set[str] = set()
    for term in terms:
        known.update(term)
    if len(set(rhs)) != len(rhs) or not set(rhs) <= known:
        return None
    return terms, rhs


def _census(terms: Sequence[Sequence[str]]
            ) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    """一次遍历得到 (每个标签的出现重数, 共现计数)。

    共现 = 两个标签同时出现在某个操作数里（每个操作数内两两记一次，双向都记）。
    """
    degree: dict[str, int] = {}
    pairs: dict[tuple[str, str], int] = {}
    for term in terms:
        counts: dict[str, int] = {}
        for label in term:
            counts[label] = counts.get(label, 0) + 1
        for label, count in counts.items():
            degree[label] = degree.get(label, 0) + count
        unique = sorted(counts)
        for index in range(len(unique)):
            left = unique[index]
            for right in unique[index + 1:]:
                key = (left, right)
                pairs[key] = pairs.get(key, 0) + 1
                key = (right, left)
                pairs[key] = pairs.get(key, 0) + 1
    return degree, pairs


class _Shape:
    """一条方程的**字母无关结构描述**（桶键 + 同构搜索所需的全部局部信息）。

    ``sig``
        标签 → ``(出现重数, 是否在输出, 零共现伙伴数, 非零共现重数多重集)``。
    ``classes``
        操作数 → ``(秩, 其标签签名多重集)``。
    ``key``
        结构键 ``(操作数类多重集, 输出标签签名多重集)`` 的**摘要**。它对字母改名、
        操作数次序、操作数内轴序、输出轴序**全都不变**，所以"等价 ⟹ 同键"。
        存摘要而不是原键：原键含每个标签的完整签名（一条 16 操作数方程约 10 KB），
        8638 条全留会吃掉上百 MB；摘要只留一个整数，撞键由构造式同构兜住。
    ``pairs``
        共现计数（同构剪枝要用）。构建时已经算出来，直接留着——只有真正被保留的
        ``_Shape``（查询式 + 桶内比对过的条目）会占这份内存，索引本身只存摘要。
    ``axis_key``
        **轴序敏感**的结构指纹 ``(每个操作数按轴序排列的标签签名多重集, 输出段按轴序
        排列的标签签名)``。与 ``key`` 相反，它对操作数内轴序与输出轴序**敏感**：零代价
        映射（不转任何轴）要求目标的第 j 个轴对应调用方某操作数的第 j 个轴，而签名与
        字母名无关，所以两边的指纹必须逐位相等（操作数之间至多差一个位置置换、故比
        多重集；输出段没有位置可换、故逐位比）。这是**必要条件**，不是判据——真正是否
        零代价仍由 :func:`_build` 的结果说话；不相等就一定不可能零代价，用它先筛一遍
        可以免得为注定要转置的候选白跑同构搜索。
    """

    __slots__ = ("terms", "out", "sig", "classes", "key", "pairs", "axis_key")

    def __init__(self, terms: Sequence[Sequence[str]], out: Sequence[str]) -> None:
        self.terms = tuple(tuple(term) for term in terms)
        self.out = tuple(out)
        output = set(self.out)
        degree, pairs = _census(self.terms)
        partners: dict[str, list[int]] = {}
        for (left, _right), count in pairs.items():
            bucket = partners.get(left)
            if bucket is None:
                partners[left] = [count]
            else:
                bucket.append(count)
        total = len(degree)
        self.pairs = pairs
        self.sig = {
            label: (count,
                    1 if label in output else 0,
                    total - 1 - len(partners.get(label) or ()),
                    tuple(sorted(partners.get(label) or ())))
            for label, count in degree.items()}
        self.classes = tuple(
            (len(term), tuple(sorted(self.sig[label] for label in term)))
            for term in self.terms)
        self.key = hash((tuple(sorted(self.classes)),
                         tuple(sorted(self.sig[label] for label in self.out))))
        self.axis_key = (
            tuple(sorted(tuple(self.sig[label] for label in term)
                         for term in self.terms)),
            tuple(self.sig[label] for label in self.out))


def _search_order(query: _Shape, sizes: dict[Any, int]) -> list[int]:
    """同构搜索的操作数次序：从第 0 个操作数出发，此后优先与已处理的部分共享标签
    （每个新操作数的标签配对大多已被钉死，实测几乎零回溯），同类里优先下标小的
    ——这样"只差一处"的写法总会先被取到，报告的置换最小。"""
    count = len(query.terms)
    label_sets = [set(term) for term in query.terms]
    order: list[int] = [0]
    seen: set[int] = {0}
    covered: set[str] = set(label_sets[0])
    while len(order) < count:
        best_rank = None
        best_index = 0
        for index in range(count):
            if index in seen:
                continue
            shared = len(label_sets[index] & covered)
            rank = (shared > 0, shared,
                    -sizes.get(query.classes[index], 0), -index)
            if best_rank is None or rank > best_rank:
                best_rank, best_index = rank, index
        order.append(best_index)
        seen.add(best_index)
        covered |= label_sets[best_index]
    return order


def _output_matches(target: _Shape, query: _Shape, sigma: dict[str, str]) -> bool:
    """输出段的必要条件：``sigma`` 把调用方的输出标签映成目标输出标签的多重集。"""
    return (Counter(sigma.get(label) for label in query.out)
            == Counter(target.out))


def _match(target: _Shape, query: _Shape,
           budget: int = _MAX_SEARCH_NODES
           ) -> tuple[dict[int, int], dict[str, str]] | None:
    """在 ``query`` 与 ``target`` 之间构造一次**精确同构**，失败返回 ``None``。

    返回 ``(pi, sigma)``：``pi`` 是 query 第 j 个操作数 → target 操作数下标；
    ``sigma`` 是 query 标签 → target 标签。两者都是双射，且逐操作数的标签重数
    一致——因此"用 target 的方程、按返回的双射重排张量"必定逐元素等价。

    ``ij,jk->ik`` 与 ``ij,jk->i``（不同收缩）、``ii->i`` 与 ``ij->i``（对角 vs 一般）
    这类真正不同的方程在这里必然失败：标签重数与共现约束对不上。
    """
    if (len(target.terms) != len(query.terms)
            or len(target.out) != len(query.out)
            or len(target.sig) != len(query.sig)):
        return None
    if sorted(target.sig.values()) != sorted(query.sig.values()):
        return None                     # 标签签名多重集是必要条件（廉价否决）

    groups: dict[Any, list[int]] = defaultdict(list)
    for index, cls in enumerate(target.classes):
        groups[cls].append(index)
    sizes = {cls: len(indexes) for cls, indexes in groups.items()}
    order = _search_order(query, sizes)

    sigma: dict[str, str] = {}
    reverse: dict[str, str] = {}
    pi: dict[int, int] = {}
    used: set[int] = set()
    nodes = 0

    def candidates(query_index: int) -> list[int]:
        """同类候选按"与调用方该操作数的字母重合度"降序——只影响**取哪条**映射，
        不影响正确性：重合度高的一侧先试，得到的置换最小、报告最好读。"""
        available = [index for index in groups.get(query.classes[query_index], ())
                     if index not in used]
        if len(available) < 2:
            return available
        wanted = Counter(query.terms[query_index])
        available.sort(key=lambda index: (
            -sum(min(count, Counter(target.terms[index]).get(label, 0))
                 for label, count in wanted.items()),
            index))
        return available

    def consistent(query_label: str, target_label: str) -> bool:
        for other_query, other_target in sigma.items():
            if (query.pairs.get((query_label, other_query), 0)
                    != target.pairs.get((target_label, other_target), 0)):
                return False
        return True

    def options(query_index: int, target_index: int) -> tuple[tuple[str, str], ...]:
        query_counts = Counter(query.terms[query_index])
        target_counts = Counter(target.terms[target_index])
        for query_label, count in query_counts.items():
            if query_label in sigma:
                if target_counts.get(sigma[query_label], 0) != count:
                    return ()
        free_query = [l for l in sorted(query_counts) if l not in sigma]
        free_target = [l for l in sorted(target_counts) if l not in reverse]
        if len(free_query) != len(free_target):
            return ()
        found: list[tuple[tuple[str, str], ...]] = []
        for perm in permutations(free_target):
            pairs: list[tuple[str, str]] = []
            for query_label, target_label in zip(free_query, perm):
                if (query_counts[query_label] != target_counts[target_label]
                        or query.sig[query_label] != target.sig[target_label]):
                    break
                pairs.append((query_label, target_label))
            else:
                found.append(tuple(pairs))
            if len(found) >= _MAX_LABEL_OPTIONS:
                break
        if len(found) > 1:
            # 优先"同名对"最多的配对：轴置换与位置改写最小，读起来最省事。
            found.sort(key=lambda pairs: (
                -sum(1 for left, right in pairs if left == right), pairs))
        return tuple(found)

    def search(depth: int) -> tuple[dict[int, int], dict[str, str]] | None:
        nonlocal nodes
        nodes += 1
        if nodes > budget:
            return None
        if depth == len(order):
            return (dict(pi), dict(sigma)) if _output_matches(
                target, query, sigma) else None
        query_index = order[depth]
        for target_index in candidates(query_index):
            if target_index in used:
                continue
            for pairs in options(query_index, target_index):
                added: list[tuple[str, str]] = []
                ok = True
                for query_label, target_label in pairs:
                    if (query_label in sigma or target_label in reverse
                            or not consistent(query_label, target_label)):
                        ok = False
                        break
                    sigma[query_label] = target_label
                    reverse[target_label] = query_label
                    added.append((query_label, target_label))
                if ok:
                    used.add(target_index)
                    pi[query_index] = target_index
                    found = search(depth + 1)
                    if found is not None:
                        return found
                    del pi[query_index]
                    used.discard(target_index)
                for query_label, target_label in added:
                    sigma.pop(query_label, None)
                    reverse.pop(target_label, None)
        return None

    return search(0)


def _build(query_terms: Sequence[Sequence[str]], query_out: Sequence[str],
           target: _Shape, pi: dict[int, int], sigma: dict[str, str]
           ) -> Equivalence | None:
    """把 ``(pi, sigma)`` 翻译成三个置换字段，并做符号回放自检。

    回放自检是**唯一**的兜底：桶键可能把两条不同构的方程放进同一桶，``_match``
    的约束也可能被将来的改动破坏；这里把字段真的作用回调用方方程，逐字对上才放行。
    """
    reverse = {target_label: query_label
               for query_label, target_label in sigma.items()}
    spelling = ["".join(reverse[label] for label in term)
                for term in target.terms]
    spelling_out = "".join(reverse[label] for label in target.out)

    count = len(target.terms)
    inverse_order = [0] * count
    for query_index, target_index in pi.items():
        if not 0 <= target_index < count or not 0 <= query_index < len(query_terms):
            return None
        inverse_order[target_index] = query_index
    if sorted(inverse_order) != list(range(count)):
        return None

    perms: list[tuple[int, ...] | None] = []
    for index in range(count):
        source = query_terms[inverse_order[index]]
        consumed = [False] * len(source)
        perm: list[int] = []
        for label in target.terms[index]:
            query_label = reverse[label]
            for axis, actual in enumerate(source):
                if not consumed[axis] and actual == query_label:
                    consumed[axis] = True
                    perm.append(axis)
                    break
            else:
                return None
        identity = tuple(range(len(perm)))
        perm_tuple = tuple(perm)
        if _permuted(source, perm_tuple) != spelling[index]:
            return None                  # 自检：字段与方程不自洽
        perms.append(None if perm_tuple == identity else perm_tuple)

    out_perm: list[int] = []
    for label in query_out:
        target_label = sigma.get(label)
        if target_label is None or target_label not in target.out:
            return None
        out_perm.append(target.out.index(target_label))
    out_tuple: tuple[int, ...] | None = tuple(out_perm)
    if out_tuple == tuple(range(len(out_tuple))):
        out_tuple = None
    expected_output = (spelling_out if out_tuple is None
                       else _permuted(query_out, _perm_inverse(out_tuple)))
    if expected_output != spelling_out:
        return None                      # 自检：输出轴序方向必须自洽

    order: tuple[int, ...] | None = tuple(inverse_order)
    if order == tuple(range(count)):
        order = None
    moved_axes = any(perm is not None for perm in perms)
    if not moved_axes:
        perms = []

    parts: list[str] = []
    if order is not None:
        parts.append(f"操作数次序改为 {list(order)}")
    for index, perm in enumerate(perms):
        if perm is not None:
            parts.append(f"操作数 {index} 轴序改为 {spelling[index]}"
                         f"（张量需 transpose{perm}）")
    if out_tuple is not None:
        parts.append(f"输出轴序改为 {spelling_out}（结果需 transpose{out_tuple}）")
    description = " 且 ".join(parts) or "无需改写：本身已命中"

    return Equivalence(
        equation=",".join(spelling) + "->" + spelling_out,
        operand_order=order,
        operand_permutations=tuple(perms),
        output_permutation=out_tuple,
        description=description,
        kind=_kind(order, perms, out_tuple),
    )


def _kind(order: tuple[int, ...] | None,
          perms: Sequence[tuple[int, ...] | None],
          out_perm: tuple[int, ...] | None) -> str:
    """变换类别标签（报告侧的分类口径，与改造前的词表一致）。"""
    moved_order = order is not None
    moved_axes = any(perm is not None for perm in perms)
    moved_output = out_perm is not None
    if moved_order and moved_axes and moved_output:
        return "operand-order+axes+output-axes"
    if moved_order and moved_axes:
        return "operand-order+axes"
    if moved_order and moved_output:
        return "operand-order+output-axes"
    if moved_axes and moved_output:
        return "operand-axes+output-axes"
    if moved_order:
        return "operand-order"
    if moved_axes:
        return "operand-axes"
    if moved_output:
        return "output-axes"
    return "identity"


def _index() -> dict[int, list[tuple[int, str]]]:
    """``结构键摘要 -> [(seq, 语料方程), …]``；第一次查询时建一次并缓存。

    8638 条语料实测 0.75 s 量级（每条只算一次结构不变量，无枚举、无 probe）。
    条目里**不**保留 :class:`_Shape`：一条 16 操作数方程的签名表约 10 KB，全留会
    吃掉上百 MB；真正要比对的候选只在用到时展开（:func:`_shape_of`）。
    """
    global _INDEX
    if _INDEX is None:
        from . import _wick_tasklist as tasklist

        index: dict[int, list[tuple[int, str]]] = {}
        for task in tasklist.TASKS:
            parsed = _split(task.equation)
            if parsed is None:                       # 语料本身非法：跳过，不抛
                continue
            key = _Shape(*parsed).key
            index.setdefault(key, []).append((task.seq, task.equation))
        _INDEX = index
    return _INDEX


def _shape_of(seq: int, equation: str) -> _Shape:
    """语料条目的结构描述，按需展开并缓存（只缓存真正被比对过的条目）。"""
    shape = _SHAPES.get(seq)
    if shape is None:
        parsed = _parse_cached(equation)
        if parsed is None:                           # pragma: no cover - 语料已校验
            raise ValueError(f"corpus equation is not parsable: {equation!r}")
        shape = _Shape(*parsed)
        if len(_SHAPES) < _SHAPE_CACHE_LIMIT:
            _SHAPES[seq] = shape
    return shape


def _parse_cached(equation: str) -> tuple[list[str], str] | None:
    """``_split`` 的有界缓存：桶内排序要逐条读语料写法，重复解析不值得。"""
    parsed = _PARSED.get(equation)
    if parsed is None:
        parsed = _split(equation)
        if parsed is not None and len(_PARSED) < _SHAPE_CACHE_LIMIT:
            _PARSED[equation] = parsed
    return parsed


def _alignment(query_terms: Sequence[Sequence[str]], query_out: str,
               equation: str) -> tuple[int, int, int]:
    """桶内排序用的"写法接近度"：``(逐位置同集合, 逐位置逐字相同, 共有字母数)``。

    同一桶里常有多条**结构同构**的语料方程（语料本身就带同构重排），先试写法最
    接近的那条，报告的置换才最小（用户实际例子：命中 seq=8628 而不是同样同构、
    却要动四个位置的邻居）。这只影响**取哪条**映射，不影响正确性。
    """
    parsed = _parse_cached(equation)
    if parsed is None:                               # pragma: no cover - 语料已校验
        return (0, 0, 0)
    terms, out = parsed
    if len(terms) != len(query_terms):
        return (0, 0, 0)
    same_set = same_text = shared = len(set(query_out) & set(out))
    for left, right in zip(query_terms, terms):
        if set(left) == set(right):
            same_set += 1
            if left == right:
                same_text += 1
        shared += len(set(left) & set(right))
    return (same_set, same_text, shared)


def _position_permutations(count: int) -> list[tuple[int, ...]]:
    """操作数位置改写候选：小规模给全排列，大规模只给两两互换（O(n²)）。"""
    identity = tuple(range(count))
    if count <= _PROBE_FULL_PERM_LIMIT:
        return [perm for perm in permutations(range(count)) if perm != identity]
    out: list[tuple[int, ...]] = []
    for left, right in combinations(range(count), 2):
        perm = list(identity)
        perm[left], perm[right] = perm[right], perm[left]
        out.append(tuple(perm))
    return out


def _axis_permutations(rank: int) -> list[tuple[int, ...]]:
    """单个操作数的轴序候选：小秩给全排列，大秩只给逆序 / 首尾互换 / 轮转。"""
    identity = tuple(range(rank))
    if rank <= _PROBE_FULL_PERM_LIMIT:
        return [perm for perm in permutations(range(rank)) if perm != identity][
            :_MAX_LABEL_OPTIONS]
    out: list[tuple[int, ...]] = [tuple(reversed(identity))]
    swapped = list(identity)
    swapped[0], swapped[-1] = swapped[-1], swapped[0]
    out.append(tuple(swapped))
    for shift in range(1, rank):
        out.append(tuple(identity[shift:] + identity[:shift]))
    return out


def _output_permutations(rank: int) -> list[tuple[int, ...]]:
    """输出段轴序候选（``new_out[j] == out[perm[j]]``）。"""
    identity = tuple(range(rank))
    if rank <= _PROBE_FULL_PERM_LIMIT:
        return [perm for perm in permutations(range(rank)) if perm != identity][
            :_MAX_LABEL_OPTIONS]
    out: list[tuple[int, ...]] = []
    for shift in range(1, rank):
        out.append(tuple(identity[shift:] + identity[:shift]))
        out.append(tuple(identity[-shift:] + identity[:-shift]))
    for left, right in combinations(range(rank), 2):
        perm = list(identity)
        perm[left], perm[right] = perm[right], perm[left]
        out.append(tuple(perm))
    return out


def _probe_candidates(query_terms: Sequence[Sequence[str]],
                      query_out: str) -> Iterator[Equivalence]:
    """显式 ``probe`` 的兼容路径：按"改动量从小到大"产出等价写法。

    **默认路径不经过这里**：那条路由语料索引 + 构造式同构给出，零枚举。只有调用方
    自己给了 ``probe``（自定义目标形态，语料索引帮不上忙）时才退回这份有界枚举。
    产出次序：不动 → 只改位置 / 单操作数轴序 / 输出轴序 → 两族复合 → 三族复合；
    每个档位内先跑完全部位置次序，避免靠前的写法吃光预算。
    """
    count = len(query_terms)
    order_options: list[tuple[int, ...] | None] = [None] + _position_permutations(count)
    axis_options = [_axis_permutations(len(term)) for term in query_terms]
    out_options: list[tuple[int, ...] | None] = [None] + _output_permutations(
        len(query_out))
    emitted = 0

    for changed in (0, 1, 2):
        for positions in combinations(range(count), changed):
            for chosen in product(*[axis_options[position] for position in positions]):
                for order in order_options:
                    for out_perm in out_options:
                        if emitted >= _MAX_PROBE_CANDIDATES:
                            return
                        perms: list[tuple[int, ...] | None] = [None] * count
                        for position, perm in zip(positions, chosen):
                            perms[position] = perm
                        terms = [
                            query_terms[index] if order is None
                            else query_terms[order[index]]
                            for index in range(count)]
                        rewritten = [
                            term if perms[index] is None
                            else _permuted(term, perms[index])
                            for index, term in enumerate(terms)]
                        new_out = (query_out if out_perm is None
                                   else _permuted(query_out, out_perm))
                        if (order is None and out_perm is None
                                and all(perm is None for perm in perms)
                                and list(rewritten) == [str(t) for t in query_terms]):
                            continue
                        emitted += 1
                        yield Equivalence(
                            equation=",".join(rewritten) + "->" + new_out,
                            operand_order=order,
                            operand_permutations=tuple(perms) if any(perms) else (),
                            output_permutation=(None if out_perm is None
                                                else _perm_inverse(out_perm)),
                            description=_describe_rewrite(
                                order, rewritten, perms, new_out, out_perm),
                            kind=_kind(order, perms,
                                       None if out_perm is None
                                       else _perm_inverse(out_perm)),
                        )


def _describe_rewrite(order: tuple[int, ...] | None, rewritten: Sequence[str],
                      perms: Sequence[tuple[int, ...] | None], new_out: str,
                      out_perm: tuple[int, ...] | None) -> str:
    """显式 probe 路径的中文说明（与语料路径的措辞一致）。"""
    parts: list[str] = []
    if order is not None:
        parts.append(f"操作数次序改为 {list(order)}")
    for index, perm in enumerate(perms):
        if perm is not None:
            parts.append(f"操作数 {index} 轴序改为 {rewritten[index]}"
                         f"（张量需 transpose{perm}）")
    if out_perm is not None:
        parts.append(f"输出轴序改为 {new_out}"
                     f"（结果需 transpose{_perm_inverse(out_perm)}）")
    return " 且 ".join(parts) or "无需改写：本身已命中"


def _is_zero_cost(result: Equivalence) -> bool:
    """映射是否"零代价"：既不动任何操作数的内部轴序，也不动输出轴序。

    **只改 ``operand_order``（位置）不算代价**：调用方只是换个实参次序，张量本身
    不动、仍然 C 连续。而轴序 / 输出轴序要 ``transpose``，那返回的是**视图**、通常
    非 C 连续，``_kernel_kupl.try_dedicated`` 的第一道门就会把它拒掉——本来能走
    特化的查询反而被打回通用路径。
    """
    return (result.output_permutation is None
            and not any(perm is not None for perm in result.operand_permutations))


def _mapping_of(target: _Shape, query: _Shape, query_terms: Sequence[Sequence[str]],
                query_out: str) -> Equivalence | None:
    """一条桶内条目 → 一条通过自检的映射；不可映射返回 ``None``。

    与改造前 ``_normalize`` 的循环体逐句相同（``_match`` 失败或 ``_build`` 自检不过
    就跳过该条目），只是抽成函数给两趟扫描共用。
    """
    found = _match(target, query)
    if found is None:
        return None
    return _build(query_terms, query_out, target, *found)


def _first_mapping(entries: Sequence[tuple[int, str]],
                   query_terms: Sequence[Sequence[str]], query_out: str,
                   query: _Shape) -> Equivalence | None:
    """**旧行为**：按给定次序取第一条通过自检的映射（改造前的循环逐句保留）。"""
    for seq, text in entries:
        result = _mapping_of(_shape_of(seq, text), query, query_terms, query_out)
        if result is not None:
            return result
    return None


def _select_mapping(entries: Sequence[tuple[int, str]],
                    query_terms: Sequence[Sequence[str]], query_out: str,
                    query: _Shape) -> Equivalence | None:
    """在同桶条目里挑一条映射：**零代价优先**，给不出来就原样走旧行为。

    为什么需要偏好：同一桶里常有多条结构同构的语料写法（语料本身就是同构重排的
    集合，最大桶 384 条），而 :func:`_alignment` 的第三分量 ``shared`` 只数"逐位置
    的字面字母交集"，把**被重排位置上的字母巧合**也算成"更接近"。于是一个只换了
    两个操作数位置的查询可能挑到一条"还得再转置两个操作数"的写法（实例：
    ``codp,ambn,…`` 命中 seq=5 的 ``aobp,cmdn,…``，而同桶 seq=3 的
    ``ambn,codp,…`` 零置换）——转置是非连续视图，恰好被原生特化入口的连续性门
    挡住。:func:`_alignment` 因此只在**同一档内**起作用。

    这趟扫描是**纯附加**的：命中就多给一条零代价映射，没命中就一字不改地退回
    :func:`_first_mapping`，所以"找不到零代价"时结果与改动前逐字段相同。
    代价两层上界：候选条数封顶 :data:`_MAX_ZERO_COST_SCAN` 条，且每条先用
    :attr:`_Shape.axis_key` 这个**廉价必要条件**筛（不满足就绝不可能零代价），
    只有筛过的才跑 ``_match`` + ``_build``——绝不对整个桶做无界的同构搜索。
    """
    wanted = query.axis_key
    for index in range(min(len(entries), _MAX_ZERO_COST_SCAN)):
        seq, text = entries[index]
        target = _shape_of(seq, text)
        if target.axis_key != wanted:
            continue                     # 保守筛选：只会漏掉"更好的那条"，不会错选
        result = _mapping_of(target, query, query_terms, query_out)
        if result is not None and _is_zero_cost(result):
            return result
    return _first_mapping(entries, query_terms, query_out, query)


def _normalize(equation: str, probe: Any) -> Equivalence | None:
    """查表本体（**不含开关判定**）：语料索引 + 构造式同构，或显式 probe 的改写枚举。

    开关只在 :func:`normalize_equation` 判；:func:`explain` 是纯报告，不走开关
    ——``--equivalence true`` 这类显式要求必须能压过环境变量（两个 CLI 的
    ``--help`` 都是这么写的），而它们只经 ``explain`` 取报告。
    """
    parsed = _split(equation)
    if parsed is None:
        return None
    query_terms, query_out = parsed
    if probe is None:
        from . import _specialize

        if _specialize.match(equation) is not None:
            return Equivalence(equation=equation,
                               description="无需改写：本身已命中",
                               kind="identity")
        query = _Shape(query_terms, query_out)
        entries = _index().get(query.key, ())
        if len(entries) > 1:
            entries = sorted(entries, key=lambda item: (
                tuple(-value for value in _alignment(query_terms, query_out,
                                                     item[1])),
                item[0]))
        return _select_mapping(entries, query_terms, query_out, query)

    try:
        if probe(equation):
            return Equivalence(equation=equation,
                               description="无需改写：本身已命中",
                               kind="identity")
    except Exception:                              # noqa: BLE001 - 探测失败即放弃
        return None
    for candidate in _probe_candidates(query_terms, query_out):
        try:
            if probe(candidate.equation):
                return candidate
        except Exception:                          # noqa: BLE001 - 探测失败即跳过
            continue
    return None


def normalize_equation(equation: str,
                       probe: Any = None) -> Equivalence | None:
    """把方程改写成与某个"已知可用形式"完全相同的写法。

    默认（``probe is None``）走语料索引：先算结构不变量桶键，再在桶内用构造式
    同构确认，命中即返回映射；桶里没有可构造同构的条目就返回 ``None``。
    ``probe`` 是一个 ``callable(equation) -> bool``：给定候选方程，告诉本函数它是否
    已经是目标形式。显式给出时退回有界改写枚举（语料索引对自定义目标帮不上忙）。

    **本函数只改写方程与给出张量调整建议，绝不替调用方执行。**
    关闭（``PYQCD_EINSUM_EQUIVALENCE=0``）时直接返回 ``None``：零查询、零枚举。
    """
    if not enabled():
        return None
    return _normalize(equation, probe)


def _normal_form(terms: Sequence[Sequence[str]], out: str) -> str | None:
    """把方程写成"结构名次 + 首现编号"的规范样式（供人工阅读，不是等价判据）。

    与桶键不同，这里**需要**一个全序才能写出字符串；结构内的并列按字母名兜底，
    因此它只保证"同一条方程每次写出相同"，不保证"等价方程写出相同"——后者由
    :func:`normalize_equation` 的构造式同构负责。
    """
    labels = {label for term in terms for label in term}
    if len(labels) > 52:                             # 字母表不够用，放弃美化
        return None
    sig = _Shape(terms, out).sig
    rank = {label: index for index, label in enumerate(
        sorted(labels, key=lambda label: (sig[label], label)))}
    names = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rows = sorted("".join(names[rank[label]] for label in term) for term in terms)
    output = "".join(names[rank[label]]
                     for label in sorted(out, key=lambda label: rank[label]))
    return ",".join(rows) + "->" + output


def find_equivalents(equation: str, limit: int = 8) -> list[dict[str, Any]]:
    """列出该方程的等价写法（**不判是否命中**），供人工/工具挑选。

    新实现不再枚举候选：给出①**结构规范样式**（操作数按结构排序、操作数内轴序
    排序、标签按结构名次重编号）与②命中语料的那条写法（若存在）。条数是常数级，
    与元数无关；``limit`` 只截断返回。关闭（``PYQCD_EINSUM_EQUIVALENCE=0``）时
    返回 ``[]``。
    """
    if not enabled():
        return []
    parsed = _split(equation)
    if parsed is None:
        return []
    terms, out = parsed
    rows: list[dict[str, Any]] = []
    normal = _normal_form(terms, out)
    if normal is not None and normal != equation:
        rows.append(Equivalence(
            equation=normal,
            description="结构规范样式（操作数与轴序按结构排序、标签按名次重编号）",
            kind="normal-form",
        ).as_dict())
    matched = normalize_equation(equation)
    if matched is not None and matched.equation != equation:
        rows.append(matched.as_dict())
    return rows[:max(0, limit)]


def explain(equation: str, probe: Any = None) -> dict[str, Any]:
    """给 ``verify_specialized`` / ``test.py`` 用的完整说明：能否改写、改写成什么、
    要动哪些张量。

    **本入口不读开关**：它是纯报告，调用方（``--equivalence true`` 等显式要求）
    已经决定要查，真正的开关由 :func:`normalize_equation` / :func:`find_equivalents`
    与两个 CLI 的门负责（它们读同一个环境变量）。开关关闭时这里只是额外标一个
    ``disabled=True`` 与 :data:`DISABLED_REASON`，报告本身仍然如实。
    """
    normalized = _normalize(equation, probe)
    report: dict[str, Any] = {
        "equation": equation,
        "rewritten": normalized is not None
                     and normalized.kind != "identity",
    }
    if not enabled():
        report["disabled"] = True
        report["disabled_reason"] = DISABLED_REASON
    if normalized is None:
        report["reason"] = ("没有任何等价写法能落到已知形式上（本层只做单操作数轴序 / "
                            "操作数位置 / 输出段轴序的等价改写）")
        return report
    report.update(normalized.as_dict())
    if normalized.kind == "identity":
        report["reason"] = "本身已命中，无需改写"
    return report
