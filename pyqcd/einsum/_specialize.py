"""方程形态专用优化路径（pattern specialization）。

格点 QCD 的收缩形式高度固定。对已知形态，本模块给出**专用执行路径**，
并在命中时打印提示，便于与通用路径做性能对照。

开关优先级（高 → 低）：

1. 调用参数 ``contract(..., specialize=False)`` —— 单次调用关闭；
2. 环境变量 ``PYQCD_EINSUM_SPECIALIZE=0`` —— 进程级关闭；
3. 默认开启。

命中时输出（每个进程每个形态一次）：

    [pyqcd.einsum] 专用路径: gauge-propagator(E4) [fused: cuda_e4_fused]

设置 ``PYQCD_EINSUM_SPECIALIZE_VERBOSE=1`` 可每次调用都输出。

匹配分两段（:func:`match`）：

1. **字面**：去空白后与 :data:`REGISTRY` 的键逐字符相同 —— 历史行为，一字不改；
2. **拓扑**：未命中时按 :func:`canonical_signature` 的签名查 :data:`FAMILIES`，
   于是把 ``ij,jk->ik`` 改写成 ``ij,jb->ib`` 或 ``ab,bc->ac`` 仍然命中同一形态。

两段都未命中时行为与以前完全一致（回落到通用的计划缓存路径）。

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
import threading
from functools import lru_cache
from typing import Any

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

# 形态键为规范化后的 subscripts（去空白）。
REGISTRY: dict[str, dict[str, str]] = {
    "zyxcTabn,zyxgTdef,Gae,Tzyxgc->TGdbfn": {
        "name": "gauge-propagator(E4)",
        "kind": "plan-cache",
        "note": "peram × peram × γ × 规范场；Python 层无专用 kernel（计划缓存），CUDA 执行器在体积门内会内部融合（z*y*x<=1）",
    },
    "ab,...bce,cd->...dae": {
        "name": "gamma-sandwich(E5)",
        "kind": "plan-cache",
        "note": "γ₅ 夹心；Python 层无专用 kernel（计划缓存），CUDA 执行器可能内部融合",
    },
    "eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN": {
        "name": "sush-2pt(S1)",
        "kind": "plan-cache",
        "note": "sush NN 2pt 生产方程（仅复用计划缓存）",
    },
    "gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG": {
        "name": "sush-3pt-munu(S2)",
        "kind": "plan-cache",
        "note": "sush 3pt 含 μν 插入（仅复用计划缓存）",
    },
    "manb,cedf,bd,Mfn,ac,MFem->MF": {
        "name": "sush-2pt-mu(S3)",
        "kind": "plan-cache",
        "note": "sush 2pt 含 μ 插入（仅复用计划缓存）",
    },
}

# 指标字符集与广播位。签名里广播位固定记作 -1，不占用首现编号。
_LABELS = frozenset(string.ascii_letters)
_ELLIPSIS = "..."
_ELLIPSIS_ID = -1

# 形态名关键词 → 可服务该形态的专用 kernel 标签。执行侧（``_api._kernel_labels_for``）
# 以条目里的 ``kernels`` 为准，这里同时是字面条目与拓扑条目的唯一声明处。
# 旧的核子 2pt/3pt 专用 kernel 已整体删除，字面形态都只复用计划缓存或走拓扑入口。
_KERNEL_LABELS: tuple[tuple[str, tuple[str, ...]], ...] = ()

# 「不可能命中」也是结果，缓存它才能保证 match() 的未命中开销与旧实现同级。
_SIGNATURE_CACHE_LIMIT = 512


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


def _build_families() -> dict[tuple, Entry]:
    """把 :data:`REGISTRY` 的字面写法折叠成拓扑等价类：签名 → 形态条目。"""
    families: dict[tuple, Entry] = {}
    for subscripts, entry in REGISTRY.items():
        signature = canonical_signature(subscripts)
        if signature is None or signature in families:
            continue
        families[signature] = {
            "name": entry["name"],
            "kind": entry["kind"],
            "note": entry["note"],
            "kernels": kernel_labels(str(entry["name"])),
            "subscripts": subscripts,
        }
    return families


# 拓扑等价类：改字母、加空白、换输出宽度都命中同一条目（只用于 REGISTRY 未命中的写法）。
FAMILIES: dict[tuple, Entry] = _build_families()

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


def _topology_index():
    """兼容旧接口：返回全部摘要的并集（仅诊断用，热路径不要调）。"""
    from . import _topology_registry as registry
    merged: set[str] = set()
    for key in registry._BUCKETS:
        raw = registry.bucket_raw(key)
        merged |= {raw[i:i + 16].hex() for i in range(0, len(raw), 16)}
    return frozenset(merged)


def _topology_entry(digest: str) -> Entry:
    """语料拓扑的形态条目；代表写法第一次用到时才读。"""
    entry = _TOPOLOGY_ENTRY_CACHE.get(digest)
    if entry is None:
        from . import _topology_registry as registry
        entry = {
            "name": "topology:" + digest[:8],
            "kind": "kernel(topology)",
            "note": "真实语料拓扑（整条计划交给原生拓扑入口；不支持时回落计划缓存）",
            "kernels": ("kupl_topology", "cuda_topology"),
            "subscripts": "",
        }
        if len(_TOPOLOGY_ENTRY_CACHE) < 4096:
            _TOPOLOGY_ENTRY_CACHE[digest] = entry
    return entry

_announced: set[str] = set()
_lock = threading.Lock()


def enabled(specialize: bool | None = None) -> bool:
    """判定本次调用是否启用专用路径。"""
    if specialize is not None:
        return bool(specialize)
    return os.environ.get(DISABLE_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


def match(subscripts: str) -> Entry | None:
    """返回命中的专用路径描述，未命中返回 None。

    两段式：先按字面查 :data:`REGISTRY`（与历史行为逐字符一致），未命中再按
    :func:`canonical_signature` 的拓扑签名查 :data:`FAMILIES`。两段都未命中就是
    旧实现的未命中 —— 调用方照旧回落到计划缓存路径。
    """
    if not isinstance(subscripts, str):
        return None
    text = normalize(subscripts)
    literal = REGISTRY.get(text)
    if literal is not None:
        return literal
    signature = _signature_of_normalized(text)
    if signature is None:
        return None
    entry = FAMILIES.get(signature)
    if entry is not None:
        return entry
    digest = _topology_hit(subscripts, signature)
    if digest is not None:
        return _topology_entry(digest)
    return None


def announce(entry: Entry, kernel: str = "") -> None:
    """命中时打印提示；默认每形态每个进程只打印一次。

    ``kernel`` 是**实际执行**的专用内核名（空串表示只命中计划缓存），
    提示里据实输出，避免把没有 kernel 的形态说成 fused。
    """
    name = entry.get("name", "unknown")
    if not notice_enabled():
        return
    verbose = os.environ.get(VERBOSE_ENV, "").strip().lower() in (
        "1", "true", "on", "yes")
    with _lock:
        if not verbose and name in _announced:
            return
        _announced.add(name)
    kind = f"kernel: {kernel}" if kernel else "plan-cache"
    note = entry.get("note", "")
    if notice_language() == "zh":
        print(f"[pyqcd.einsum] 专用路径: {name} [{kind}] {note}", flush=True)
    else:
        print(f"[pyqcd.einsum] specialized path: {name} [{kind}] {note}",
              flush=True)


LANGUAGE_ENV = "PYQCD_EINSUM_LANG"


def notice_enabled() -> bool:
    """首次命中提示是否开启（默认开启，``PYQCD_EINSUM_NOTICE=0`` 关闭）。"""
    return os.environ.get(NOTICE_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


def notice_language() -> str:
    """提示语言：``PYQCD_EINSUM_LANG`` 优先，否则跟随 ``LC_ALL``/``LANG``。

    显式设成 ``en``/``zh`` 时一直用它；未设时，环境 locale 是中文（``zh*``）就说
    中文，其余（含 ``C``/``C.UTF-8``/``en_US``）说英文——英文环境的用户不该在日志里
    看到中文。
    """
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
    专用路径。
    """
    if not notice_enabled():
        return
    verbose = os.environ.get(VERBOSE_ENV, "").strip().lower() in (
        "1", "true", "on", "yes")
    key = "__generic__"
    with _lock:
        if not verbose and key in _announced:
            return
        _announced.add(key)
    where = f" backend={backend}" if backend else ""
    equation = f" ({subscripts})" if subscripts else ""
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


def describe() -> list[tuple[str, str, str]]:
    """列出所有已注册形态：(name, kind, subscripts)。"""
    return [(item["name"], item["kind"], key) for key, item in REGISTRY.items()]


##### 人工自检入口（不属于测试注册文件）


def _selfcheck() -> int:
    """打印已注册签名并跑等价性断言；返回进程退出码。"""
    failures: list[str] = []

    def check(label: str, condition: bool) -> None:
        print(f"  {'ok  ' if condition else 'FAIL'} {label}")
        if not condition:
            failures.append(label)

    print("已注册字面形态与拓扑签名：")
    print(f"  {'name':22s} {'kind':28s} 签名")
    for key, entry in REGISTRY.items():
        print(f"  {entry['name']:22s} {entry['kind']:28s} {canonical_signature(key)}")
    print(f"字面条目 {len(REGISTRY)} 条，拓扑等价类 {len(FAMILIES)} 个")

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

    print(f"\n{len(REGISTRY)} 条字面写法仍按字面命中（回归不变）：")
    for key, entry in REGISTRY.items():
        hit = match(key)
        check(f"{entry['name']} 字面命中 name/kind/note 一致",
              hit is not None
              and (hit["name"], hit["kind"], hit["note"])
              == (entry["name"], entry["kind"], entry["note"]))

    print("\n同一形态的字母重命名写法按拓扑命中（name 一致）：")
    letters = string.ascii_letters

    def renamed(subscripts: str, shift: int = 7) -> str:
        table = str.maketrans({
            label: letters[(letters.index(label) + shift) % len(letters)]
            for label in set(subscripts) if label in _LABELS})
        return subscripts.translate(table)

    for key, entry in REGISTRY.items():
        variant = renamed(key)
        check(f"{entry['name']} 的改名写法命中同一 name",
              variant != key
              and (match(variant) or {}).get("name") == entry["name"])

    print("\n形态声明的 kernels 与执行侧解析一致（防漂移）：")
    from . import _api

    for entry in FAMILIES.values():
        check(f"{entry['name']} kernels={entry['kernels']}",
              _api._kernel_labels_for(entry) == tuple(entry["kernels"]))

    print()
    if failures:
        print(f"FAILED: {len(failures)} 项")
        return 1
    print("OK: 全部自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
