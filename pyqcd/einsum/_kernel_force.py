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
``<label>``        只允许该 label；跑不成就抛 :class:`KernelUnavailableError`
=================  ==========================================================

合法的 ``<label>`` 与 ``_api._specialized_kernel_attempts`` 给出的标签一致：
``cuda_dedicated`` / ``cuda_topology`` / ``kupl_dedicated`` / ``kupl_topology``。
可以用逗号给多个（按书写次序即尝试次序）。

优先级：显式参数 ``kernel_force=`` > 环境变量 ``PYQCD_EINSUM_KERNEL_FORCE`` > 未设置
（等价 ``auto``）。取值在**调用时**解析，同进程内改环境变量即时生效。
"""

from __future__ import annotations

import os

__all__ = [
    "AUTO_LABEL",
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

#: 可强制的专用内核标签，与 ``_api._specialized_kernel_attempts`` 的返回一致。
KERNEL_LABELS = frozenset({
    "cuda_dedicated",
    "cuda_topology",
    "kupl_dedicated",
    "kupl_topology",
})


class KernelUnavailableError(RuntimeError):
    """强制指定了内核，但它没有跑成。

    只在**强制**模式下抛出（``auto`` 模式仍是 warning + 回落，保持既有数值契约）。
    携带足以定位问题的字段，避免用户再去翻 ``LAST_FAILURE`` 或负缓存内部状态。
    """

    def __init__(self, requested: tuple[str, ...], subscripts: str,
                 reasons: dict[str, str], attempted: tuple[str, ...] = ()) -> None:
        self.requested = tuple(requested)
        self.subscripts = str(subscripts)
        self.reasons = dict(reasons)
        self.attempted = tuple(attempted)

        label = ", ".join(self.requested) or "(none)"
        lines = [f"强制内核 {label} 没有跑成：equation {self.subscripts!r}"]
        if self.attempted:
            lines.append("实际尝试：" + ", ".join(self.attempted))
        for name in self.attempted:
            reason = self.reasons.get(name) or "（该内核未留下原因）"
            lines.append(f"  - {name}: {reason}")
        missing = [name for name in self.requested if name not in self.attempted]
        if missing:
            lines.append("未尝试：" + ", ".join(missing)
                         + "（标签与本次操作数后端不匹配）")
        lines.append(
            f"恢复自动分派：{KERNEL_FORCE_ENV}=auto；"
            "让失败不被缓存：PYQCD_EINSUM_NEGATIVE_CACHE=0；"
            "检查 GPU 栈：python -m pyqcd --gpu-probe")
        super().__init__("\n".join(lines))


def parse(spec: object) -> tuple[str, ...]:
    """把一处取值解析成有序标签元组。

    ``None`` / 空串 / ``auto`` / ``0`` / ``false`` / ``off`` / ``no`` → ``()``（不强制）；
    ``generic`` → ``(GENERIC_LABEL,)``；其余按逗号切分、去空白、保序去重。未知标签抛
    ``ValueError`` 并列出合法集合——静默忽略拼写错误会让用户以为强制生效了。
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
        if name != GENERIC_LABEL and name not in KERNEL_LABELS:
            allowed = ", ".join(sorted(KERNEL_LABELS | {GENERIC_LABEL, AUTO_LABEL}))
            raise ValueError(
                f"未知的内核标签 {part.strip()!r}；合法取值：{allowed}")
        if name not in labels:
            labels.append(name)
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
    """需要强制的标签；``None`` 表示按原次序自动分派。"""
    labels = resolve(kernel_force)
    if not labels:
        return None
    if labels == (GENERIC_LABEL,):
        return ()
    return labels


def generic_only(kernel_force: object = None) -> bool:
    """是否要求完全跳过专用 kernel。"""
    return resolve(kernel_force) == (GENERIC_LABEL,)
