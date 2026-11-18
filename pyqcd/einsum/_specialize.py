"""方程形态专用优化路径（pattern specialization）。

格点 QCD 的收缩形式高度固定。对已知形态，本模块给出**专用执行路径**，
并在命中时打印提示，便于与通用路径做性能对照。

稳定冻结：自标签 ``stab31`` 起的匹配口径（:func:`canonical_signature` 的重编号规则、
:func:`match` 的单段式索引、摘要算法）属于 qcd-einsum 冻结稳定线（首个冻结版
0.4.1；0.4.2 仅修正文档措辞）。改动这些口径会让已在生产使用的方程换路径，
默认禁止；例外见 ``AGENTS.md`` §0。

开关优先级（高 → 低）：

1. 调用参数 ``contract(..., specialize=False)`` —— 单次调用关闭；
2. 环境变量 ``PYQCD_EINSUM_SPECIALIZE=0`` —— 进程级关闭；
3. 默认开启。

命中时输出（每个进程每个形态一次），整句只用**一种**语言：

    [pyqcd.einsum] 专用路径: topology:xxxxxxxx [match: topology] [kernel: kupl_dedicated]
    [pyqcd.einsum] specialized path: topology:xxxxxxxx [match: topology] [kernel: kupl_dedicated]

``[match: …]`` 是**命中的形态族**，``[kernel: …]`` 是**这次真正执行**的内核路线
（取自执行侧 ``LAST_INFO['path']``），没跑成内核时写作 ``[plan-cache]``。两段回答的是
两个不同的问题：命中的是拓扑**形态**，不代表走了拓扑**内核**，所以
"``[match: topology]`` + ``[kernel: kupl_dedicated]``"说的是"拓扑形态这次由 dedicated
内核执行"，不是标签取错。

``kernel_force`` **只接受单值强制**（``backend`` / ``dedicated`` / ``generic`` /
``topology``，缺省 ``generic``），**不存在"依次尝试多个内核"**：强制了哪个就只下发哪个，
跑不成就报错或回落。默认 ``generic`` 时一个专用内核都不试——语料命中只换来这一行形态
提示，执行仍是通用分步收缩。行尾说明因此按**本次执行语义**分三档拼（见
``_NOTE_ZH`` / ``_NOTE_EN`` 与 :func:`_execution_semantics`）：强制跑成的写"已按
kernel_force 强制走 <kernel>"，没强制的写"仅形态匹配，按 generic 执行"，强制后没跑成
而回落的写"已回落 generic"。

语言由 ``PYQCD_EINSUM_LANG`` / CLI 的 ``--language`` 决定（见
:func:`notice_language`）：``zh`` 用中文动词（``专用路径``）配中文说明，``en`` 全英文。
``[pyqcd.einsum]`` 前缀、形态名与 ``[match: …]`` / ``[kernel: …]`` / ``[plan-cache]``
三个机器段是**稳定部分**，两种语言逐字相同（所以它们不翻译）；
**绝不**把英文标签拼在中文解释前面（或反之）。

设置 ``PYQCD_EINSUM_SPECIALIZE_VERBOSE=1`` 可每次调用都输出。

匹配只有一段（:func:`match`）：先算 :func:`canonical_signature`，再按
``(元数, 槽位总数)`` 查当前 ``wick_12cases`` 的 8638 条拓扑语料。改名字母、
调整空白或改写为等价的省略输出仍会命中同一拓扑；语料之外的方程回落到通用计划。
历史上按字面 ``REGISTRY`` 匹配的旧入口已删除。

人工自检（不属于测试注册文件）：

    python -c "from pyqcd.einsum import _specialize; raise SystemExit(_specialize._selfcheck())"

也可用 ``python -m pyqcd.einsum._specialize``，但 runpy 会额外打印一条
``found in sys.modules`` 的 RuntimeWarning（``pyqcd.einsum`` 包已先行导入本模块），
不影响结论与退出码。
"""

from __future__ import annotations

import hashlib
import os
import string
import sys
import threading
from functools import lru_cache
from typing import Any

from ._cache import LRUCache

VERBOSE_ENV = "PYQCD_EINSUM_SPECIALIZE_VERBOSE"
# 首次命中提示的总开关：默认**开启**（用户要求"首次命中应有提醒"），
# 设 PYQCD_EINSUM_NOTICE=0 可完全关闭。与 VERBOSE_ENV 的区别：
#   NOTICE=0  → 一条都不打（含 verbose）
#   NOTICE=1（默认）→ 每个形态首次命中打一条
#   VERBOSE=1 → 每次命中都打（忽略"首次"去重）
NOTICE_ENV = "PYQCD_EINSUM_NOTICE"
DISABLE_ENV = "PYQCD_EINSUM_SPECIALIZE"
FUSED_DISABLE_ENV = "PYQCD_EINSUM_DISABLE_FUSED"

# 形态条目的值类型：字面条目只有字符串，拓扑条目另带 kernels/subscripts。
Entry = dict[str, Any]

# 已废弃：历史上这里有一张 5 条的字面形态表（gauge-propagator(E4)、gamma-sandwich(E5)、
# sush-2pt(S1)、sush-3pt-munu(S2)、sush-2pt-mu(S3)）。它们**不在** wick_12cases 语料里，
# 只是旧专用内核时代的遗留登记项。新版特化内核按拓扑覆盖全部 8638 条语料路径后，
# 这张表既不再有执行意义，也会让用户误以为"支持的特化路径只有 5 条"，故整体移除。
# 保留名字只为让旧的调用点报错清楚，见 _REMOVED_NAMES。
_REMOVED_NAMES = (
    "gauge-propagator(E4)",
    "gamma-sandwich(E5)",
    "sush-2pt(S1)",
    "sush-3pt-munu(S2)",
    "sush-2pt-mu(S3)",
)

# 指标字符集与广播位。签名里广播位固定记作 -1，不占用首现编号。
_LABELS = frozenset(string.ascii_letters)
_ELLIPSIS = "..."
_ELLIPSIS_ID = -1

# 形态名关键词 → 可服务该形态的专用 kernel 标签。执行侧（``_api._kernel_labels_for``）
# 以条目里的 ``kernels`` 为准，这里同时是字面条目与拓扑条目的唯一声明处。
# 旧的核子 2pt/3pt 专用 kernel 已整体删除，字面形态都只复用计划缓存或走拓扑入口。
_KERNEL_LABELS: tuple[tuple[str, tuple[str, ...]], ...] = ()

# The production corpus currently has 8638 unique equations. A smaller cache
# forces every full-corpus pass back through signature and index construction.
_CORPUS_CACHE_LIMIT = 16_384
_SIGNATURE_CACHE_LIMIT = _CORPUS_CACHE_LIMIT

# :func:`match` 的**结果**缓存：键是调用方给的原始方程字符串（含空白，未归一化），
# 值是返回的形态条目或 ``None``（负结果同样缓存——语料外的方程占调用的大头）。
# 容量与签名缓存同量级：键数上限就是调用方写过的不同写法数。
# 线程安全与统计由 :class:`._cache.LRUCache` 负责，本模块不加锁。
_MATCH_CACHE = LRUCache("specialize.match", maxsize=_CORPUS_CACHE_LIMIT)

# 缓存缺省值哨兵：与"命中且值恰为 None（负结果）"区分开。
_CACHE_MISS = object()


def normalize(subscripts: str) -> str:
    return "".join(subscripts.split())


def kernel_labels(name: str) -> tuple[str, ...]:
    """该形态名可用的专用 kernel 标签（按名称里的形态关键词判定）。"""
    for keyword, labels in _KERNEL_LABELS:
        if keyword in name:
            return labels
    return ()


def _encode_term(term: str, numbers: dict[str, int]) -> tuple[int, ...] | None:
    """把单个指标串编码为编号元组；非法字符返回 None。

    ``numbers`` 由调用方在整式范围内共享，因此编号顺序就是全局首次出现顺序；
    省略号编码成 :data:`_ELLIPSIS_ID`，单个 ``.`` 一律视为非法。
    """
    ids: list[int] = []
    index = 0
    while index < len(term):
        if term.startswith(_ELLIPSIS, index):
            ids.append(_ELLIPSIS_ID)
            index += len(_ELLIPSIS)
            continue
        label = term[index]
        if label not in _LABELS:
            return None
        if label not in numbers:
            numbers[label] = len(numbers)
        ids.append(numbers[label])
        index += 1
    return tuple(ids)


@lru_cache(maxsize=_SIGNATURE_CACHE_LIMIT)
def _signature_of_normalized(text: str) -> tuple | None:
    """已去空白的方程 → 拓扑签名（:func:`canonical_signature` 的实现）。"""
    if not text or text.count("->") > 1:
        return None
    lhs, separator, rhs = text.partition("->")
    terms = lhs.split(",")
    if any(not term for term in terms):
        return None

    numbers: dict[str, int] = {}
    operands: list[tuple[int, ...]] = []
    for term in terms:
        encoded = _encode_term(term, numbers)
        if encoded is None:
            return None
        operands.append(encoded)

    if separator:
        output_text = rhs
        for label in output_text:
            # 输出指标必须来自输入；否则该方程本身非法，交回旧行为（未命中）。
            if label in _LABELS and label not in numbers:
                return None
    else:
        # einsum 隐含输出：广播位在最前，其后是只出现一次的指标（按字母序）。
        counts: dict[str, int] = {}
        for label in lhs:
            if label in _LABELS:
                counts[label] = counts.get(label, 0) + 1
        output_text = (_ELLIPSIS if _ELLIPSIS in lhs else "") + "".join(
            sorted(label for label, count in counts.items() if count == 1))

    output = _encode_term(output_text, numbers)
    if output is None:
        return None
    if len(set(output)) != len(output):
        return None                       # 输出指标不可重复（numpy 同样拒绝）
    return tuple(operands) + (output,)


def canonical_signature(subscripts: Any) -> tuple | None:
    """方程的拓扑签名：与字母无关、与空白无关，保留操作数与指标顺序。

    签名定义（两步）：

    1. 先按 einsum 规则补全省略的输出项：有 ``->`` 时用其右侧字面写法；没有时
       输出 = ``...``（若任一操作数含省略号，放在最前）+ 只出现一次的指标按字母序。
    2. 然后从头到尾扫描**全部操作数**（操作数顺序不变，每个操作数内部的指标顺序
       也不变），再扫描输出项，把指标按"全式首次出现顺序"重新编号为 0,1,2,…；
       广播位 ``...`` 固定记作 ``-1``，不参与编号。

    返回值是 ``(操作数0, 操作数1, …, 输出)``：每项是该操作数的编号元组，最后一项是
    输出的编号元组。因此 ``ij,jk->ik``、``ij,jb->ib``、``ab,bc->ac`` 三者签名相同
    ``((0, 1), (1, 2), (0, 2))``；而 ``ij,kj->ik``（``((0,1),(2,1),(0,2))``）、
    ``ij,jk->ki``（``((0,1),(1,2),(2,0))``）与它们不同 —— 签名只认"谁和谁共享、
    共享的指标是否留在输出、以什么顺序留在输出"，不认字母，也不做操作数置换。

    无法解析的方程（类型不对、多重 ``->``、空操作数、非法字符、输出含输入之外或
    重复的指标）返回 ``None``，**不抛异常** —— 调用方据此保持旧的未命中行为。
    """
    if not isinstance(subscripts, str):
        return None
    return _signature_of_normalized(normalize(subscripts))


# 旧的字面形态表移除后这里没有可折叠的条目；语料形态统一走 _topology_hit。
# 保留该名字（空 dict）是为了兼容外部诊断代码的 `FAMILIES` 引用。
FAMILIES: dict[tuple, Entry] = {}

# 真实语料（refer/sush/wick_12cases.md 的 2582 条收缩）的拓扑摘要索引按需加载：
# 数据块 300 KB+，放进 import 路径会拖慢 `import pyqcd.einsum`。
_TOPOLOGY_DIGESTS: frozenset[str] | None = None
_TOPOLOGY_ENTRY_CACHE: dict[str, Entry] = {}


def _signature_digest(signature: tuple) -> str:
    """签名 → 128 位摘要；口径必须与 ``_topology_registry`` 的生成脚本一致。"""
    payload = "|".join(",".join(map(str, term)) for term in signature)
    return hashlib.blake2b(payload.encode(), digest_size=12).hexdigest()


def _topology_hit(subscripts: str, signature: tuple) -> str | None:
    """按 (元数, 槽位总数) 二级分叉查桶：只解压命中的那个桶。

    两级键都由方程本身算出，不需要预解析语料：
      元数   = 操作数个数（签名长度 - 1，末项是输出）；
      槽位数 = 各操作数 rank 之和 + 输出 rank。
    """
    from . import _topology_registry as registry
    arity = len(signature) - 1
    slots = sum(len(term) for term in signature)
    digest = _signature_digest(signature)
    return digest if registry.contains(digest, arity, slots) else None


def _topology_entry(digest: str) -> Entry:
    """语料拓扑的形态条目；代表写法第一次用到时才读。"""
    entry = _TOPOLOGY_ENTRY_CACHE.get(digest)
    if entry is None:
        from . import _topology_registry as registry
        entry = {
            "name": "topology:" + digest[:8],
            "kind": "kernel(topology)",
            # 这句只回答"命中了什么形态"，**不承诺**本次走哪条入口：实际路线一律以
            # `kernel` 行 / `LAST_INFO['path']` 为准。原先这里写死"dedicated 步表优先、
            # 拓扑步表兜底"，那是 `auto` 时代的分派次序；`auto` 删除后
            # `kernel_force` 只接受单值强制，一次调用不会连试两个内核，那句话既过时
            # 又会让读者误以为专用内核还在自动选路（2026-11-18 用户报告）。
            # 本次到底强制与否由 `announce` 按现场事实补上（见 `_NOTE_ZH`）。
            "note": "真实语料形态",
            # **英文同义说明**：``announce`` 在 ``en`` 下用它，绝不把中文解释拼到
            # 英文标签后面（那样一行里就有两种语言）。
            "note_en": "corpus form",
            # ``kernels`` 是这种形态**可能**由哪些执行条目服务，不是本次实际执行的那个：
            # 命中形态与执行路线是两个问题，``[match: X]`` + ``[kernel: Y]`` 里 X≠Y
            # 是正常结果（X 是形态族，Y 才是 ``LAST_INFO['path']``）。
            "kernels": ("kupl_topology", "cuda_topology"),
            "subscripts": "",
        }
        if len(_TOPOLOGY_ENTRY_CACHE) < _CORPUS_CACHE_LIMIT:
            _TOPOLOGY_ENTRY_CACHE[digest] = entry
    return entry


@lru_cache(maxsize=_CORPUS_CACHE_LIMIT)
def _match_of_normalized(text: str) -> Entry | None:
    """Cache the complete topology lookup for one normalized equation."""
    signature = _signature_of_normalized(text)
    if signature is None:
        return None
    digest = _topology_hit(text, signature)
    return _topology_entry(digest) if digest is not None else None


_announced: set[str] = set()
_lock = threading.Lock()


def enabled(specialize: bool | None = None) -> bool:
    """判定本次调用是否启用专用路径。"""
    if specialize is not None:
        return bool(specialize)
    return os.environ.get(DISABLE_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


def match(subscripts: str, *, use: bool | None = None) -> Entry | None:
    """返回命中的专用路径描述，未命中返回 None。

    唯一判据是**拓扑签名**：算出 :func:`canonical_signature` 后查语料索引
    （:mod:`._topology_registry`，覆盖 wick_12cases 的全部 8638 条路径）。
    因此 ``ij,jk->ik`` 与 ``ij,jb->ib`` 走同一条路，改字母、加空格、换输出宽度
    都不影响命中。未命中即语料之外的方程，调用方照旧走通用路径。

    结果缓存与命中统计
    ------------------
    本函数的结果（**含未命中的 ``None``**）进 ``_MATCH_CACHE``，键是入参原始字符串；
    命中统计读 ``_cache.stats()['caches']['specialize.match']`` 的
    ``hits/misses/stores/size``。``use`` 为 ``None`` 时跟随总开关
    ``PYQCD_EINSUM_CACHE``；显式 ``use=False`` 表示这一次**既不读也不写**缓存
    （返回值与缓存开启时相同）。非字符串入参直接返回 ``None``，不进缓存。
    """
    if not isinstance(subscripts, str):
        return None
    cached = _MATCH_CACHE.get(subscripts, _CACHE_MISS, use=use)
    if cached is not _CACHE_MISS:
        return cached
    entry = _match_of_normalized(normalize(subscripts))
    _MATCH_CACHE.put(subscripts, entry, use=use)
    return entry


def executed_path(kernel: str) -> str:
    """``kernel`` attempt 标签对应的**实际执行路线**；取不到就退回标签本身。

    attempt 标签只说明"试着下发哪个内核"，执行侧写的 ``LAST_INFO['path']`` 才是结论
    （``kupl_topology`` / ``cuda_topology`` / ``*_dedicated``）。KUPL 的拓扑入口**只有
    一条路线**：命中编译期步表就写 ``kupl_topology``，未命中直接 fail closed（status 4
    → 上层回落），既没有 ``kupl_topology_v2`` 这条回退路线，也没有路线查询符号可读
    ——不要再按 ``_kernel_kupl._TOPOLOGY_ROUTE_PATHS`` 去找（该符号已删除）。
    模块不在 ``sys.modules`` 里说明本次没有内核能留下记录（专用内核才 import 它），
    此时直接退回标签，绝不为了读一个提示去 import 原生模块。
    """
    if not kernel:
        return ""
    if kernel.startswith("cuda_"):
        module_name = f"{__package__}._kernel_cuda"
    elif kernel.startswith("kupl_"):
        module_name = f"{__package__}._kernel_kupl"
    else:
        return kernel                  # ``backend`` 之类没有原生 LAST_INFO
    module = sys.modules.get(module_name)
    path = getattr(module, "LAST_INFO", {}).get("path")
    return path if isinstance(path, str) and path else kernel


#: ``announce`` 行尾说明的**执行语义**分档：条目里只写"命中了什么形态"，本次到底有没有
#: 强制内核只有调用现场知道，所以说明文字在这里按档存模板。``{kernel}`` 填实际执行路线，
#: ``{requested}`` 填用户强制的逻辑名（``dedicated`` / ``topology``）。
_NOTE_ZH = {
    "forced": "（已按 kernel_force 强制走 {kernel}）",
    "generic": "（仅形态匹配；未强制内核，按 generic 执行）",
    "fallback": "（已按 kernel_force 强制走 {requested}，但没跑成，已回落 generic）",
}
_NOTE_EN = {
    "forced": " (kernel_force forced {kernel})",
    "generic": " (shape match only; no kernel forced, running generic)",
    "fallback": " (kernel_force forced {requested} but it did not run; "
                "fell back to generic)",
}


def _execution_semantics(kernel: str) -> tuple[str, str]:
    """本次提示的执行语义：``(档, 用户强制的逻辑名)``，档取 ``_NOTE_ZH`` 的键。

    只有 ``kernel`` 一个输入，所以结论全部来自**可观测事实**，不猜分派次序
    （``kernel_force`` 单值强制，每次调用至多下发一个内核）：

    - ``kernel`` 非空：这个专用内核真的执行了。``_api`` 里带非空标签调 :func:`announce`
      的只有"强制即真试一次"那条路，所以档是 ``forced``，逻辑名取标签后缀
      （``kupl_dedicated`` → ``dedicated``）。
    - ``kernel`` 为空：一个专用内核都没跑。``_LAST_KERNEL['requested']`` 只在调用方
      **显式要求**非 ``generic`` 时非空（缺省与显式 ``generic`` 都是空），它非空而
      ``ran`` 为空，说明强制的那条没跑成、已回落通用路径，档是 ``fallback``；
      其余情况只是形态命中，档是 ``generic``。
    """
    if kernel:
        return "forced", kernel.rpartition("_")[2]
    api = sys.modules.get(f"{__package__}._api")
    record = getattr(api, "_LAST_KERNEL", None) or {}
    requested = tuple(record.get("requested") or ())
    if requested and not record.get("ran"):
        return "fallback", ", ".join(requested)
    return "generic", ""


def _note_text(entry: Entry, key: str, table: dict[str, str], kernel: str) -> str:
    """条目文案 + 本次执行语义；条目没写文案时只打机器段（保持旧行为）。"""
    base = str(entry.get(key, ""))
    if not base:
        return ""
    semantics, requested = _execution_semantics(kernel)
    return base + table[semantics].format(
        kernel=executed_path(kernel), requested=requested)


def announce(entry: Entry, kernel: str = "") -> None:
    """命中时打印提示；默认每形态每个进程只打印一次。

    ``kernel`` 是本次真的跑成的 attempt 标签（空串表示一个专用内核都没跑），
    打印前经 :func:`executed_path` 换成执行侧的 ``LAST_INFO['path']``——
    attempt 标签是逻辑名（``dedicated`` / ``topology``），执行侧写的是真实路线名
    （``kupl_topology`` 等），两者不是一回事。

    提示把**命中形态**与**执行内核**分成两段：``[match: …]`` 是命中的形态族，
    ``[kernel: …]`` 是实际执行的路线（没跑内核时 ``[plan-cache]``）。两段不互相蕴含：
    命中拓扑形态却写 ``[kernel: kupl_dedicated]``，说的是"拓扑形态这次由 dedicated
    内核执行"，不是标签取错。条目里只写"命中了什么形态"，行尾说明由
    :func:`_execution_semantics` 按本次语义现场补上（强制跑成 / 未强制 / 强制失败回落）。

    整句只用**一种**语言（:func:`notice_language`）：``zh`` 用中文动词配 ``note``，
    ``en`` 用英文动词配 ``note_en``。``[pyqcd.einsum]`` 前缀、形态名
    （``topology:<摘要>``）与 ``[match: …]`` / ``[kernel: …]`` / ``[plan-cache]``
    三个机器段是**稳定部分**，两种语言逐字相同（``_einsum_notice_kernel_label_contract``
    逐字比对这三段）。原先英文标签后面直接拼中文 ``note``，一行里两种语言
    （2026-11-17 用户实测）。
    """
    name = entry.get("name", "unknown")
    # 下面两个 return 都是**抑制**：总开关关闭（``NOTICE_ENV``）/ 同一形态本进程已打过
    # （去重，免得一次运行把屏刷满）。``verbose`` 只解除去重、不解除总开关——两者的区别
    # 是"每次都打"与"一条都不打"（``verbose_enabled`` / ``notice_enabled`` 各管一头）。
    if not notice_enabled():
        return
    verbose = verbose_enabled()
    with _lock:
        if not verbose and name in _announced:
            return
        _announced.add(name)
    # 机器段先算好：两种语言逐字相同，所以只算一次、不跟着语言分支走。
    # ``[match: …]`` 取形态名前缀（``topology:<摘要>`` → ``topology``），它回答"命中的
    # 是什么形态"；``[kernel: …]`` 回答"实际执行的是谁"。两者缺一个就会把
    # "命中拓扑形态"读成"走了拓扑内核"。
    strategy = str(name).split(":", 1)[0]
    path = executed_path(kernel)
    executed = f"[kernel: {path}]" if path else "[plan-cache]"
    route = f"[match: {strategy}] {executed}"
    # 整句必须**单语**：动词与说明各自跟语言走（zh 用 ``note``，en 用 ``note_en``）。
    # ``[pyqcd.einsum]`` 前缀与上面的机器段两种语言逐字相同——脚本靠它们抓这一行
    # （``_api`` 的画像命中行用的是同一前缀），翻译掉就没人能解析。
    if notice_language() == "zh":
        note = _note_text(entry, "note", _NOTE_ZH, kernel)
        message = f"[pyqcd.einsum] 专用路径: {name} {route}"
    else:
        note = _note_text(entry, "note_en", _NOTE_EN, kernel)
        message = f"[pyqcd.einsum] specialized path: {name} {route}"
    print(f"{message} {note}" if note else message, flush=True)


LANGUAGE_ENV = "PYQCD_EINSUM_LANG"

#: 显式指定的提示语言（``"zh"`` / ``"en"``；``None`` = 按环境变量/locale 现算）。
#: ``--language``（CLI）经 :func:`set_language` 走这里——**不写环境变量**：改写
#: ``PYQCD_EINSUM_LANG`` 会污染同进程后续调用与调用方自己的子进程。
_LANGUAGE_OVERRIDE: str | None = None

#: 显式要求/关闭"完整详解"（``None`` = 按 :data:`VERBOSE_ENV` 现算）。
_DETAIL_OVERRIDE: bool | None = None


def set_language(language: str | None) -> str | None:
    """显式指定提示语言，返回设置前的取值（``None`` = 按 env/locale）。

    ``--language`` 是**本次输出**的选择，所以这里只改进程内状态，绝不写
    ``PYQCD_EINSUM_LANG``：写环境变量会让同进程后续调用（以及调用方自己的子进程）
    被悄悄改掉语言，而调用方无法预期。传入无法识别的取值（含 ``auto``/``None``）
    即回到"按环境变量"的口径。
    """
    global _LANGUAGE_OVERRIDE
    previous = _LANGUAGE_OVERRIDE
    text = "" if language is None else str(language).strip().lower()
    if text.startswith("zh"):
        _LANGUAGE_OVERRIDE = "zh"
    elif text.startswith("en"):
        _LANGUAGE_OVERRIDE = "en"
    else:
        _LANGUAGE_OVERRIDE = None
    return previous


def set_detail(enabled: bool | None) -> bool | None:
    """显式设定"完整详解"档，返回设置前的取值（``None`` = 按 ``VERBOSE_ENV``）。

    CLI 的 ``--explain true`` / ``--verbose true`` 用它恢复完整提示（包括画像命中时
    那唯一一行之外的解释）；不传或传 ``None`` 时回到环境变量口径，所以调用方不必
    猜"库现在的默认值是什么"。
    """
    global _DETAIL_OVERRIDE
    previous = _DETAIL_OVERRIDE
    _DETAIL_OVERRIDE = None if enabled is None else bool(enabled)
    return previous


def verbose_enabled() -> bool:
    """是否**每次**命中都打提示（``VERBOSE_ENV`` 或显式的 :func:`set_detail`）。"""
    if _DETAIL_OVERRIDE is not None:
        return _DETAIL_OVERRIDE
    return os.environ.get(VERBOSE_ENV, "").strip().lower() in (
        "1", "true", "on", "yes")


def notice_enabled() -> bool:
    """首次命中提示是否开启（默认开启，``PYQCD_EINSUM_NOTICE=0`` 关闭）。"""
    # 关掉提示等于删掉诊断。任何新增的抑制条件都必须同时配一条**反向断言**"该打的
    # 仍然打"：只验"它能沉默"，沉默本身就会把诊断能力一起砍掉（本轮就出现过"默认
    # 路径的通用提示不见了"的疑点，靠反向实测才排除，而不是靠读代码相信它）。
    return os.environ.get(NOTICE_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


def notice_language() -> str:
    """提示语言：显式 :func:`set_language` > ``PYQCD_EINSUM_LANG`` > locale。

    ``--language``（CLI）走 :func:`set_language`，因此**同一次运行的库内提示与报告
    正文是同一语言**；显式设成 ``en``/``zh`` 时一直用它。都没设时，环境 locale 是
    中文（``zh*``）就说中文，其余（含 ``C``/``C.UTF-8``/``en_US``）说英文——英文
    环境的用户不该在日志里看到中文。
    """
    if _LANGUAGE_OVERRIDE is not None:
        return _LANGUAGE_OVERRIDE
    explicit = os.environ.get(LANGUAGE_ENV, "").strip().lower()
    if explicit.startswith("zh"):
        return "zh"
    if explicit.startswith("en"):
        return "en"
    for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(name, "").strip().lower()
        if value:
            return "zh" if value.startswith("zh") else "en"
    return "en"


def announce_generic(subscripts: str = "", backend: str = "") -> None:
    """**通用内核**首次命中时的提醒（每个进程一次，可用 NOTICE_ENV 关闭）。

    与 :func:`announce` 的区别：那条讲"命中了哪个特化形态"，本条讲"这次没走特化，
    走的是通用分步收缩"，并提示可用的加速开关——用户据此能判断自己是否漏掉了
    专用路径。与 :func:`announce` 同口径：整句只用一种语言，``[pyqcd.einsum]``
    前缀与方程/命令名是两种语言共有的稳定部分。
    """
    if not notice_enabled():
        return
    verbose = verbose_enabled()
    key = "__generic__"
    with _lock:
        if not verbose and key in _announced:
            return
        _announced.add(key)
    where = f" backend={backend}" if backend else ""
    equation = f" ({subscripts})" if subscripts else ""
    # 与 :func:`announce` 同一条规矩：整句单语，``[pyqcd.einsum]`` 前缀与方程/命令名
    # 不翻译——提示和报告正文必须是同一种语言，否则一次运行里两套说法。
    if notice_language() == "zh":
        message = (
            f"[pyqcd.einsum] 通用路径: 未命中特化形态{equation}{where}；"
            f"执行的是通用分步收缩。若该方程属于蒸馏语料（wick_12cases），"
            f"可用 `python -m pyqcd.einsum._specialize` 查看已注册形态；"
            f"设 PYQCD_EINSUM_NOTICE=0 可关闭本提示。")
    else:
        message = (
            f"[pyqcd.einsum] generic path: no specialized shape matched"
            f"{equation}{where}; running the generic pairwise contraction. "
            f"If this equation comes from the distillation corpus "
            f"(wick_12cases), run `python -m pyqcd.einsum._specialize` to list "
            f"the registered shapes. Set PYQCD_EINSUM_NOTICE=0 to silence this "
            f"notice.")
    print(message, flush=True)


def describe(limit: int | None = None) -> list[tuple[str, str, str]]:
    """列出所有支持的特化路径：(name, kind, subscripts)。

    数据源是 wick_12cases 语料本身（8638 条），每条给出真实方程写法；
    名字用该路径的拓扑摘要前 8 位，与运行时提示里打印的完全一致。
    ``limit`` 只截断返回条数，不影响索引。
    """
    from . import _wick_tasklist as tasklist

    rows: list[tuple[str, str, str]] = []
    for task in tasklist.TASKS:
        signature = canonical_signature(task.equation)
        if signature is None:
            continue
        rows.append(("topology:" + _signature_digest(signature)[:8],
                     "kernel(topology)", task.equation))
        if limit is not None and len(rows) >= limit:
            break
    return rows


##### 人工自检入口（不属于测试注册文件）


def _selfcheck() -> int:
    """打印已支持的特化路径并跑等价性断言；返回进程退出码。"""
    failures: list[str] = []

    def check(label: str, condition: bool) -> None:
        print(f"  {'ok  ' if condition else 'FAIL'} {label}")
        if not condition:
            failures.append(label)

    from . import _wick_tasklist as tasklist

    rows = describe()
    print("支持的特化路径（来自 wick_12cases 语料，按拓扑签名匹配）：")
    print(f"  共 {len(rows)} 条；下面列出前 5 条：")
    for name, kind, subscripts in rows[:5]:
        print(f"  {name:22s} {kind:18s} {subscripts}")

    print("\n语料全量命中（每条都必须匹配上）：")
    missed = [task.seq for task in tasklist.TASKS if match(task.equation) is None]
    check(f"8638 条语料全部命中（未命中 {len(missed)} 条）", not missed)

    print("\n等价性（应命中同一签名/同一形态）：")
    pairs = [
        ("ij,jk->ik", "ij,jb->ib"),
        ("ij,jk->ik", "ab,bc->ac"),
        ("ij,jk->ik", "ij, jk -> ik"),
        ("ij,jk->ik", "ij,jk"),                       # 输出省略形式
        ("...ij,...jk->...ik", "...ij,...jk"),
        ("ii->i", "aa->a"),                           # 同操作数内重复指标（对角）
        ("ij,jk->ik", "ik,kj->ij"),                   # j↔k 改名，仍是同一收缩
    ]
    for left, right in pairs:
        check(f"{left} == {right}",
              canonical_signature(left) == canonical_signature(right))

    print("\n不等价（不得误命中）：")
    distinct = [
        ("ij,jk->ik", "ij,kj->ik"),                   # 第二个操作数指标换位
        ("ij,jk->ik", "ji,jk->ik"),                   # 第一个操作数指标换位
        ("ij,jk->ik", "ij,jk->ki"),                   # 输出顺序不同
        ("ij,jk->ik", "i,j->ij"),                     # 外积
        ("ii->i", "ij->i"),                           # 对角 vs 求和
    ]
    for left, right in distinct:
        check(f"{left} != {right}",
              canonical_signature(left) != canonical_signature(right))

    print("\n非法方程一律返回 None（不抛异常）：")
    for bad in ("i->->i", "i,,->i", "a..b->ab", "ij,jk->ikx", "ij->ii", "",
                "   ", None, 123, ["ij"]):
        check(f"canonical_signature({bad!r}) is None",
              canonical_signature(bad) is None)

    print("\n语料外的方程不误命中（走通用路径）：")
    for outside in ("ij,jk->ik", "ab,bc->ac", "ijk->i"):
        check(f"{outside} 不在语料 → None", match(outside) is None)

    print("\n语料方程的字母重命名写法按拓扑命中：")
    letters = string.ascii_letters

    def renamed(subscripts: str, shift: int = 7) -> str:
        table = str.maketrans({
            label: letters[(letters.index(label) + shift) % len(letters)]
            for label in set(subscripts) if label in _LABELS})
        return subscripts.translate(table)

    for task in tasklist.TASKS[:200]:
        original = match(task.equation)
        variant = renamed(task.equation)
        if variant == task.equation:
            continue
        check(f"seq={task.seq} 改名仍命中同一 name",
              (match(variant) or {}).get("name")
              == (original or {}).get("name"))

    print()
    if failures:
        print(f"FAILED: {len(failures)} 项")
        return 1
    print("OK: 全部自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
