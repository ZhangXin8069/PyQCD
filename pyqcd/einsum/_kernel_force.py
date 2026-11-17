"""强制指定专用内核：解析、判定与"跑不成即抛"的异常。

为什么需要
----------
:func:`pyqcd.einsum.contract` 的专用 kernel 是**拓扑匹配 + 依次尝试**：命中注册形态后
先试 ``*_dedicated``，再试 ``*_topology``，都不成就发一条 ``RuntimeWarning`` 并回落到
通用路径。数值契约上这是对的（结果始终正确），但调试时它是一只看不见的手：
``qcd_cuda_topology_apply failed with status 4`` 这类原因只在 warning 里出现一次，
用户既无法确认"我要的 kernel 到底跑没跑"，也无法强制它跑或被明确告知跑不了。

本模块只提供**取值解析与判定**这一层（零依赖、无全局状态），选路接线在
``pyqcd/einsum/_api.py``。

取值
----
=================  ==========================================================
取值               含义
=================  ==========================================================
``auto``（缺省）    保持原分派次序（dedicated → topology）；失败仍只 warning + 回落
``generic``        完全跳过专用 kernel（等价 ``specialize=False``），不产生 warning
``dedicated``      只允许**当前后端的** dedicated（numpy → kupl，cupy → cuda）
``topology``       只允许**当前后端的** topology
``backend``        直接调用 ``backend.einsum``，不进专用内核分派
=================  ==========================================================

``dedicated`` / ``topology`` 是**逻辑名**：具体标签由操作数后端决定
（``kupl_dedicated`` / ``cuda_dedicated`` / ``kupl_topology`` / ``cuda_topology``），
解析发生在 ``_api`` 的 attempt 过滤处，本模块只负责名字。可以用逗号给多个（按书写
次序即尝试次序）；``backend`` 与其它标签**互斥**（混写抛 ``ValueError``）。

优先级：显式参数 ``kernel_force=`` > 环境变量 ``PYQCD_EINSUM_KERNEL_FORCE`` > 未设置
（等价 ``auto``）。取值在**调用时**解析，同进程内改环境变量即时生效。
"""

from __future__ import annotations

import os

__all__ = [
    "ALLOWED_LABELS",
    "AUTO_LABEL",
    "BACKEND_LABEL",
    "GENERIC_LABEL",
    "KERNEL_FORCE_ENV",
    "KERNEL_LABELS",
    "KernelUnavailableError",
    "forced_labels",
    "generic_only",
    "parse",
    "resolve",
]

#: 强制内核的环境变量名。
KERNEL_FORCE_ENV = "PYQCD_EINSUM_KERNEL_FORCE"

#: 自动分派（等价于不设本功能）。
AUTO_LABEL = "auto"
#: 完全跳过专用 kernel。
GENERIC_LABEL = "generic"
#: 直接调用 ``backend.einsum``，不进专用内核分派；与其它标签互斥。
BACKEND_LABEL = "backend"

#: 可强制的**逻辑**标签：运行时按操作数后端解析成 ``*_dedicated`` / ``*_topology``。
KERNEL_LABELS = frozenset({
    "dedicated",
    "topology",
})

#: 全部合法取值（含不产生强制的 ``auto`` / ``generic``）：报错时按此集合列出。
ALLOWED_LABELS = frozenset(KERNEL_LABELS | {AUTO_LABEL, BACKEND_LABEL, GENERIC_LABEL})


class KernelUnavailableError(RuntimeError):
    """强制指定了内核，但它没有跑成。

    只在**强制**模式下抛出（``auto`` 模式仍是 warning + 回落，保持既有数值契约）。
    携带足以定位问题的字段，避免用户再去翻 ``LAST_FAILURE`` 或负缓存内部状态。

    ``requested`` 是逻辑名（``dedicated`` / ``topology``），``attempted`` 与
    ``reasons`` 的键是**具体**标签（``kupl_dedicated`` …）：消息里两者并列，用户既能
    对上自己写的取值，也能直接看到哪个入口真的被下发过。
    """

    def __init__(self, requested: tuple[str, ...], subscripts: str,
                 reasons: dict[str, str], attempted: tuple[str, ...] = (),
                 resolved: dict[str, tuple[str, ...]] | None = None) -> None:
        self.requested = tuple(requested)
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
            f"恢复自动分派：{KERNEL_FORCE_ENV}=auto；"
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
    """把一处取值解析成有序标签元组。

    ``None`` / 空串 / ``auto`` / ``0`` / ``false`` / ``off`` / ``no`` → ``()``（不强制）；
    ``generic`` → ``(GENERIC_LABEL,)``；其余按逗号切分、去空白、保序去重，返回**逻辑名**
    （``dedicated`` / ``topology`` / ``backend``）。未知标签抛 ``ValueError`` 并列出合法
    集合——静默忽略拼写错误会让用户以为强制生效了。
    """
    if spec is None:
        return ()
    text = str(spec).strip()
    if not text:
        return ()
    lowered = text.lower()
    # 与生态里其它开关一致：``0/false/off/no`` 表示"不用本功能"（等价 ``auto``）。
    # 早先这里把它当未知标签抛用法错误，与"关一个功能不该报错"自相矛盾。
    if lowered in ("0", "false", "off", "no") or lowered == AUTO_LABEL:
        return ()
    if lowered == GENERIC_LABEL:
        return (GENERIC_LABEL,)
    labels: list[str] = []
    for part in text.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name == AUTO_LABEL:
            continue
        if name not in ALLOWED_LABELS:
            allowed = ", ".join(sorted(ALLOWED_LABELS))
            raise ValueError(
                f"未知的内核标签 {part.strip()!r}；合法取值：{allowed}")
        if name not in labels:
            labels.append(name)
    # ``backend`` 是"别用专用内核"，与"只准跑某个专用内核"没有折中解释。先查它：
    # 只要列表里有 ``backend``，报的就是这条更具体的冲突（``backend,generic`` 亦然）。
    if BACKEND_LABEL in labels and len(labels) > 1:
        raise ValueError(
            f"{BACKEND_LABEL} 不能与其它内核标签同时给出：{labels}")
    if GENERIC_LABEL in labels and len(labels) > 1:
        raise ValueError(
            f"{GENERIC_LABEL} 不能与具体内核标签同时给出：{labels}")
    return tuple(labels)


def resolve(kernel_force: object = None) -> tuple[str, ...]:
    """解析生效取值：显式 ``kernel_force=`` 优先，其次环境变量，最后 ``()``。"""
    if kernel_force is not None:
        return parse(kernel_force)
    return parse(os.environ.get(KERNEL_FORCE_ENV))


def forced_labels(kernel_force: object = None) -> tuple[str, ...] | None:
    """需要强制的逻辑标签；``None`` 表示按原次序自动分派。"""
    labels = resolve(kernel_force)
    if not labels:
        return None
    if labels == (GENERIC_LABEL,):
        return ()
    return labels


def generic_only(kernel_force: object = None) -> bool:
    """是否要求完全跳过专用 kernel。"""
    return resolve(kernel_force) == (GENERIC_LABEL,)
