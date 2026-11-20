"""强制指定专用内核：解析、判定与"跑不成即抛"的异常。

为什么需要
----------
:func:`pyqcd.einsum.contract` 的专用 kernel 走哪条路必须**由调用方写明**：命中注册形态后
按请求的次序真试一次，试不成即抛 :class:`KernelUnavailableError`。这样调试时没有
"看不见的手"——``qcd_cuda_topology_apply failed with status 4`` 这类原因会直接进异常正文，
用户既能确认"我要的 kernel 到底跑没跑"，也能被明确告知跑不了，而不是只看到一条 warning
后悄悄回落到通用路径。

本模块只提供**取值解析与判定**这一层（零依赖、无全局状态），选路接线在
``pyqcd/einsum/_api.py``。

取值
----
===============  ==========================================================
取值               含义（全部是**强制**值，没有"不强制"档）
===============  ==========================================================
``generic``（缺省） 强制通用路径：完全跳过专用 kernel（等价 ``specialize=False``）
``dedicated``      只允许**当前后端的** dedicated（numpy → kupl，cupy → cuda）
``topology``       只允许**当前后端的** topology
``backend``        直接调用 ``backend.einsum``，不进专用内核分派
``original``       交给 ``opt_einsum.contract``，后端由操作数推断
===============  ==========================================================

**为什么不提供 ``auto``**：自动选路交给 ``--tune`` 的实测结论。tune 在同一形状/dtype/
设备上遍历 ``kernel_force × as_equivalence × optimize``，把**胜出那一组**写进配置表
（``_profile_cache``），之后普通调用按表施加——这是"实测过的最快路线"，不是运行时猜的
分派次序。``kernel_force`` 若还留一个 ``auto`` 档，同一个进程里就会出现两套互相竞争的
选路规则，且 "到底谁说了算" 在报告里看不出来。所以这里只接受强制值：写 ``auto``（或任何
拼错的名字）直接报错，正文里点名交给 tune 并列出合法集合。

``dedicated`` / ``topology`` 是**逻辑名**：具体标签由操作数后端决定
（``kupl_dedicated`` / ``cuda_dedicated`` / ``kupl_topology`` / ``cuda_topology``），
解析发生在 ``_api`` 的 attempt 过滤处，本模块只负责名字。

**只接受单个值**：逗号链式（"先试 A 再试 B"）已取消——同一个调用里出现两条互相
竞争的尝试次序，等于又在运行时猜"谁说了算"。写 ``"dedicated,generic"`` 一律抛
``ValueError``，消息里列出合法取值。``backend`` / ``generic`` 因此也不再需要"互斥"判定：
单值根本组不出混写。

优先级：显式参数 ``kernel_force=`` > 环境变量 ``PYQCD_EINSUM_KERNEL_FORCE`` > 缺省
``generic``。取值在**调用时**解析，同进程内改环境变量即时生效。
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "ALLOWED_LABELS",
    "BACKEND_LABEL",
    "DEFAULT_LABEL",
    "GENERIC_LABEL",
    "KERNEL_FORCE_ENV",
    "KERNEL_LABELS",
    "KernelUnavailableError",
    "ORIGINAL_LABEL",
    "OPT_EINSUM_PATHS",
    "forced_labels",
    "generic_only",
    "opt_einsum_path_name",
    "parse",
    "resolve",
]

#: 强制内核的环境变量名。
KERNEL_FORCE_ENV = "PYQCD_EINSUM_KERNEL_FORCE"

#: 强制通用路径：完全跳过专用 kernel。
GENERIC_LABEL = "generic"
#: 缺省取值：强制通用路径（不再是"未指定 → 自动分派"）。
DEFAULT_LABEL = GENERIC_LABEL
#: 直接调用 ``backend.einsum``，不进专用内核分派；与其它标签互斥。
BACKEND_LABEL = "backend"
#: 交给 ``opt_einsum.contract``：以 ``[shape,dtype,device,equation,backend]`` 为键
#: 调外部实现的 einsum，收缩序同样由 ``optimize`` 决定。与 ``backend`` 同一地位
#: （都不进专用内核分派），tune 里排在 ``backend`` **之前**。
ORIGINAL_LABEL = "original"

#: 可强制的**逻辑**标签：运行时按操作数后端解析成 ``*_dedicated`` / ``*_topology``。
KERNEL_LABELS = frozenset({
    "dedicated",
    "topology",
})

#: 全部合法取值（五个**具体**标签，没有 auto）：报错时按此集合列出。
ALLOWED_LABELS = frozenset(
    KERNEL_LABELS | {GENERIC_LABEL, BACKEND_LABEL, ORIGINAL_LABEL})

#: tune 的 ``optimize`` 叶子 → ``opt_einsum.contract_path(optimize=...)`` 的实参。
#:
#: **两侧拼写不同，实测得出**（``opt_einsum.paths._PATH_OPTIONS`` 里只有连字符）：
#: ``branch_1``/``branch_2``/``branch_all`` 必须传 ``branch-1``/``branch-2``/
#: ``branch-all``（传下划线是 ``KeyError: Path optimizer 'branch_1' not found``）；
#: ``none`` 必须传 ``False``（传字符串 ``'none'`` / ``'false'`` 同样 KeyError，
#: 只有 ``False`` 表示"不规划"，实测路径 ``[(0,1,2,3)]`` 与不优化一致）。
#: 这张表是**单向**的：本库对外一律用下划线叶子，转型只发生在下发给 opt_einsum 处。
OPT_EINSUM_PATHS: dict[str, Any] = {
    "greedy": "greedy",
    "branch_1": "branch-1",
    "branch_2": "branch-2",
    "dp": "dp",
    "branch_all": "branch-all",
    "optimal": "optimal",
    "none": False,
}


def opt_einsum_path_name(optimize: Any) -> Any:
    """本库的 ``optimize`` 叶子 → ``opt_einsum`` 的实参（``None`` = 不认识）。

    ``None``（未给）与 ``"auto"`` 都回到 ``"greedy"``：``original`` 是"交给 opt_einsum"，
    它没有"自动"这一档，缺省找一条确定的最短路即可——与主路径的 ``auto`` 语义不同，
    这里不假装两者同义。``False``/``True`` 按主路径的布尔口径归一后再查表。
    """
    if optimize is None or optimize is True:
        return "greedy"
    if optimize is False:
        return False
    text = str(optimize).strip().lower().replace("-", "_")
    if text in ("", "auto", "true"):
        return "greedy"
    if text in ("false", "off", "no", "0"):
        return False
    return OPT_EINSUM_PATHS.get(text)


#: "不要专用内核"的既有拼写：它们**不是** auto 的替身，只映射到 ``generic``。
#: 生态里 ``0/false/off/no`` 一向表示"关掉这个功能"，关掉专用内核就是走通用路径。
_OFF_ALIASES = frozenset({"0", "false", "off", "no"})


def _reject(label: str) -> ValueError:
    """非法取值 → ``ValueError``：点名交给 tune，并列出全部合法值。

    ``auto`` 走的就是这里（它已不是合法值）；拼错的名字同样得到这条消息——"只接受强制值"
    对两者是同一件事，没必要维护第二套文案。
    """
    allowed = ", ".join(sorted(ALLOWED_LABELS))
    return ValueError(
        f"内核标签 {label!r} 不是合法取值：kernel_force 只接受强制值，"
        f"自动分派已由 tune 取代（先用 --tune 实测出最快路线，再把结论写死）。"
        f"合法取值：{allowed}")


def _reject_multiple(label: str) -> ValueError:
    """写了多个值 → ``ValueError``：只接受单值，并列出合法值。

    逗号链式曾是"按书写次序依次尝试"，于是同一次调用里存在两条互相竞争的路线：
    报告里看不出谁说了算，也看不出"第二个标签是被拒绝的"还是"根本没轮上它"。
    取消它比解释它更省事。
    """
    allowed = ", ".join(sorted(ALLOWED_LABELS))
    return ValueError(
        f"内核标签 {label!r} 含有多个值：kernel_force 只接受单个值（逗号链式已取消，"
        f"一次只准写一个标签）。合法取值：{allowed}")


class KernelUnavailableError(RuntimeError):
    """强制指定了内核，但它没有跑成。

    强制即"真试一次"：标签不可用、被策略拒绝或原生库缺失时一律抛出，不做静默回落。
    携带足以定位问题的字段，避免用户再去翻 ``LAST_FAILURE`` 或负缓存内部状态。

    ``requested`` 是逻辑名（``dedicated`` / ``topology``），``attempted`` 与
    ``reasons`` 的键是**具体**标签（``kupl_dedicated`` …）：消息里两者并列，用户既能
    对上自己写的取值，也能直接看到哪个入口真的被下发过。
    """

    def __init__(self, requested: str | tuple[str, ...], subscripts: str,
                 reasons: dict[str, str], attempted: tuple[str, ...] = (),
                 resolved: dict[str, tuple[str, ...]] | None = None) -> None:
        # 单个逻辑名与"逻辑名序列"都收：逗号链式已取消，调用点手上只有一个名字，
        # 逼它自己包一层元组只会让 ``tuple("dedicated")`` 变成一串单字符。
        self.requested = (requested,) if isinstance(requested, str) else tuple(requested)
        self.subscripts = str(subscripts)
        self.reasons = dict(reasons)
        self.attempted = tuple(attempted)
        self.resolved = {name: tuple(labels)
                         for name, labels in (resolved or {}).items()}

        label = ", ".join(self.requested) or "(none)"
        lines = [f"强制内核 {label} 没有跑成：equation {self.subscripts!r}"]
        if self.attempted:
            lines.append("实际尝试：" + ", ".join(
                self._annotate(name) for name in self.attempted))
        for name in self.attempted:
            reason = self.reasons.get(name) or "（该内核未留下原因）"
            lines.append(f"  - {self._annotate(name)}: {reason}")
        missing = [name for name in self.requested if not self._was_attempted(name)]
        if missing:
            lines.append("未尝试：" + ", ".join(missing)
                         + "（标签与本次操作数后端不匹配）")
        lines.append(
            f"改走通用路径：kernel_force={GENERIC_LABEL}；"
            "让失败不被缓存：PYQCD_EINSUM_NEGATIVE_CACHE=0；"
            "检查 GPU 栈：python -m pyqcd --gpu-probe")
        super().__init__("\n".join(lines))

    def _annotate(self, concrete: str) -> str:
        """``kupl_dedicated`` → ``kupl_dedicated（dedicated）``：具体标签与逻辑名对照。"""
        for name, labels in self.resolved.items():
            if concrete in labels:
                return f"{concrete}（{name}）"
        return concrete

    def _was_attempted(self, name: str) -> bool:
        """该逻辑名是否真的下发过：看它解析出的具体标签有没有进 attempt 表。"""
        labels = self.resolved.get(name)
        if labels:
            return any(label in self.attempted for label in labels)
        return name in self.attempted


def parse(spec: object) -> tuple[str, ...]:
    """把一处取值解析成**单个**标签（长度 1 的元组）。

    ``None`` / 空串 → ``(GENERIC_LABEL,)``（缺省即强制通用路径）；
    ``0`` / ``false`` / ``off`` / ``no`` → 同样落到 ``generic``（"关掉专用内核"）；
    其余去空白、小写后必须是**一个**合法标签
    （``dedicated`` / ``topology`` / ``generic`` / ``backend``）。

    逗号链式已取消：``"dedicated,generic"`` 抛 ``ValueError`` 并说明"只接受单个值"。
    同一个调用里允许两条尝试次序，等于把"谁说了算"重新变成运行时猜测——那正是
    ``kernel_force`` 改成强制值的理由。任何不在 :data:`ALLOWED_LABELS` 里的名字
    （含已删除的 ``auto``）抛 ``ValueError``，消息点名交给 tune 并列出合法集合。
    """
    if spec is None:
        return (GENERIC_LABEL,)
    text = str(spec).strip()
    if not text:
        return (GENERIC_LABEL,)
    if "," in text:
        raise _reject_multiple(text)
    name = text.lower()
    if name in _OFF_ALIASES:
        name = GENERIC_LABEL
    if name not in ALLOWED_LABELS:
        raise _reject(text)
    return (name,)


def resolve(kernel_force: object = None) -> tuple[str, ...]:
    """解析生效取值：显式 ``kernel_force=`` 优先，其次环境变量，最后 ``generic``。

    返回值**永远非空**：未指定 → ``(GENERIC_LABEL,)``；环境变量写 ``auto`` 同样抛
    ``ValueError``（与显式参数一条消息）。
    """
    if kernel_force is not None:
        return parse(kernel_force)
    return parse(os.environ.get(KERNEL_FORCE_ENV))


def forced_labels(kernel_force: object = None) -> tuple[str, ...]:
    """要**尝试**的专用内核逻辑名；永不返回 ``None``。

    ``generic`` 不产生任何专用内核尝试，因此返回 ``()``（而不是一个"没强制"的哨兵）；
    其余取值原样返回，交给 ``_api`` 按操作数后端解析成具体标签。

    **``()`` 有两个来源**（缺省 ``generic`` 与用户显式写 ``generic``），取值本身分不开
    两者。调用方若要区分"用户真的给过"，必须在解析**之前**自己取显式布尔位
    （``_api.contract`` 的 ``caller_force_given``）；拿本函数的返回值反推，会把每一次
    默认调用都判成"用户点名了 generic"——画像的 ``force`` 槽与命中行号提示会整段失效。
    """
    labels = resolve(kernel_force)
    if labels == (GENERIC_LABEL,):
        return ()
    return labels


def generic_only(kernel_force: object = None) -> bool:
    """是否要求完全跳过专用 kernel（``resolve`` 恰为 ``generic``）。"""
    return resolve(kernel_force) == (GENERIC_LABEL,)
