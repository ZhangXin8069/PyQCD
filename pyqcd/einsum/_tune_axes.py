"""``--tune`` 的**有序组合清单**与 ``min_wall_ms`` 剪枝记账（纯数据，无计时、无后端）。

为什么不是笛卡尔积
------------------
``kernel_force × as_equivalence × optimize`` 的笛卡尔积里绝大多数格子对某一条具体方程
根本不存在（``topology`` 在内核不可用、``branch_*`` 在方程本身不在语料里时连计划都
建不出来），而且笛卡尔积的**顺序**会把最慢的档排在前面——一次遍历有一半时间花在
永远不会有结果的格子上。这里改成一条**按实测价值排序的清单**：

1. 共同前缀（``as_equivalence=false``，``dedicated`` → ``generic``，每个内核内按
   ``greedy/branch_1/branch_2/dp/branch_all/optimal/none``）：把最容易跑、最可能快的
   格子放在最前，让 :class:`MinWall` 的上界尽早被压低，后面的格子一超界就直接终止。
2. 分支专属的一条（只有 ``as_equivalence=false`` 时原方程能特化 → 试 ``topology``；
   原方程不能特化但等价写法能 → 那条必须带 ``as_equivalence=true``；两者都不行 → 不试）；
3. 尾部 ``backend``（后端自带的 einsum，永远可跑，作为兜底基准）。

组数是**冻结口径**：``B1``/``B2`` = 22，``B3`` = 21。顺序逐项固定，改顺序就是改规范
——``--tune`` 的剪枝结论与 ``best`` 都建立在"先跑哪些格子"上。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "TUNE_BRANCHES", "TUNE_BRANCH_RAW", "TUNE_BRANCH_EQUIVALENT",
    "TUNE_BRANCH_NONE", "TUNE_BRANCH_LABELS", "TUNE_KERNEL_FORCE",
    "TUNE_AS_EQUIVALENCE", "TUNE_OPTIMIZERS", "TUNE_PREFIX", "TUNE_TAIL",
    "TUNE_BRANCH_EXTRA", "TUNE_PLAN_SIZES", "tune_plan", "tune_plan_size",
    "plan_without", "trimmed_plan_size", "TERMINATION_MESSAGE", "prune_reason",
    "MinWall", "Rate", "PREDICT_MIN_MS", "PREDICT_MARGIN",
]

#: 三个分支的稳定名（**机器可读**，不翻译）：原方程能特化 / 只有等价写法能特化 /
#: 两者都不能特化。
TUNE_BRANCH_RAW = "B1"
TUNE_BRANCH_EQUIVALENT = "B2"
TUNE_BRANCH_NONE = "B3"
TUNE_BRANCHES = (TUNE_BRANCH_RAW, TUNE_BRANCH_EQUIVALENT, TUNE_BRANCH_NONE)
#: 分支的人话说明（只进报告，不参与判定）。
TUNE_BRANCH_LABELS = {
    TUNE_BRANCH_RAW: "原方程可特化",
    TUNE_BRANCH_EQUIVALENT: "等价写法可特化",
    TUNE_BRANCH_NONE: "都无法特化",
}

#: 三个轴各自出现过的取值；``kernel_force`` 的顺序**就是**清单顺序，``optimize`` 的
#: 顺序是「先跑最可能命中的、``none`` 放最后」的冻结次序。
TUNE_KERNEL_FORCE = ("dedicated", "generic", "topology", "backend")
TUNE_AS_EQUIVALENCE = ("false", "true")
TUNE_OPTIMIZERS = ("greedy", "branch_1", "branch_2", "dp", "branch_all",
                   "optimal", "none")

#: 清单项是 ``(kernel_force, as_equivalence, optimize)``；``as_equivalence`` 用配置表
#: 的字符串口径（``"true"`` / ``"false"``），``optimize`` 用清单标签（``none`` 在调用
#: ``contract()`` 时换成 ``False``，写表时仍写 ``none``）。
TUNE_PREFIX: tuple[tuple[str, str, str], ...] = tuple(
    (kernel, "false", optimize)
    for kernel in ("dedicated", "generic")
    for optimize in TUNE_OPTIMIZERS)

#: 分支专属的那一条：``B1`` 试 ``topology`` 但**不**改写方程；``B2`` 必须让等价改写
#: 真正生效（``as_equivalence=true``）才有意义；``B3`` 没有可试的 topology 组。
TUNE_BRANCH_EXTRA: dict[str, tuple[tuple[str, str, str], ...]] = {
    TUNE_BRANCH_RAW: (("topology", "false", "none"),),
    TUNE_BRANCH_EQUIVALENT: (("topology", "true", "none"),),
    TUNE_BRANCH_NONE: (),
}

TUNE_TAIL: tuple[tuple[str, str, str], ...] = tuple(
    ("backend", "false", optimize) for optimize in TUNE_OPTIMIZERS)

#: 每个分支的组数（冻结口径：B1/B2 = 22，B3 = 21）。
TUNE_PLAN_SIZES = {
    branch: len(TUNE_PREFIX) + len(TUNE_BRANCH_EXTRA[branch]) + len(TUNE_TAIL)
    for branch in TUNE_BRANCHES}


def tune_plan(branch: str) -> list[tuple[str, str, str]]:
    """该分支的**有序**组合清单（共同前缀 → 分支专属 → 尾部）。

    未知分支名抛 ``ValueError``（不是静默返回空清单：空清单会让 ``--tune`` 报
    "没有任何可用组合"，把一次拼写错误说成物理结论）。
    """
    key = str(branch).strip()
    if key not in TUNE_BRANCH_EXTRA:
        raise ValueError(
            f"unknown tune branch {branch!r} (have {', '.join(TUNE_BRANCHES)})")
    return list(TUNE_PREFIX) + list(TUNE_BRANCH_EXTRA[key]) + list(TUNE_TAIL)


def tune_plan_size(branch: str) -> int:
    """:func:`tune_plan` 的组数（不做分配，供报告/契约直接问）。"""
    return TUNE_PLAN_SIZES.get(str(branch).strip(), 0)


def plan_without(combos: Sequence[tuple[str, str, str]],
                 unavailable: Iterable[str] | None = None
                 ) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    """去掉**该形状上本质不可用**的内核所对应的组，返回 ``(保留, 每个内核被去掉的组数)``。

    为什么要在**遍历前**裁剪而不是逐组报错：``dedicated`` 的计划 arena 超出内核自己的
    预算、``topology`` 的方程不可特化，这类"该内核对这个形状就是没有路"对每一组都恒
    成立。逐组跑一遍只能得到同一句错误的 7 份副本，把真正的异常淹掉（2026-11-18 用户
    实测：22 组里 8 组是这种噪音）。裁剪后剩下的组数才是报告里该出现的 ``N``。

    纯函数：不探测、不计时、不写表。``unavailable`` 只按 ``kernel_force``（三元组的
    第一项）匹配，未知内核名不猜。
    """
    blocked = {str(name) for name in (unavailable or ())}
    kept: list[tuple[str, str, str]] = []
    dropped: dict[str, int] = {}
    for combo in combos:
        if combo[0] in blocked:
            dropped[combo[0]] = dropped.get(combo[0], 0) + 1
        else:
            kept.append(combo)
    return kept, dropped


def trimmed_plan_size(branch: str, unavailable: Iterable[str] | None = None) -> int:
    """:func:`plan_without` 之后的组数（裁剪后**实际**会被遍历的组数）。"""
    kept, _dropped = plan_without(tune_plan(branch), unavailable)
    return len(kept)


#: 剪枝时的**固定原文**（用户口径，逐字不可改写；``--language`` 也不翻译它——它是
#: 用户给的规则原文，报告里要能 grep 到同一串字）。
TERMINATION_MESSAGE = "耗时大于min_wall_ms,此组被终止"


def prune_reason(wall_ms: float, min_wall_ms: float | None) -> str:
    """被剪枝那一组的 ``原因=``：固定原文 + 两个数值（``wall_ms`` / ``min_wall_ms``）。"""
    bound = "-" if min_wall_ms is None else f"{float(min_wall_ms):.6f}"
    return (f"{TERMINATION_MESSAGE}  wall_ms={float(wall_ms):.6f}  "
            f"min_wall_ms={bound}")


@dataclass
class MinWall:
    """``min_wall_ms`` 的当前上界与"把它压低的那一组"。

    ``ms`` 是**单次** ``contract()`` 调用的毫秒上界：初值来自 ``--tune-timeout``，
    之后每测到更小的一次调用就立刻改成那个值（后续所有组都用新上界剪枝）。
    ``config`` 是压低它那一组的 ``(kernel_force, as_equivalence, optimize)``。

    ``ms is None`` 有两个来源，语义相同——**不设上界、一组都不剪**：
    ``--tune-timeout 0`` 是用户明确关掉了上限。此时 :meth:`observe` 也不许把上界
    从 ``None`` 拉起来：那会把"不限"悄悄变成"按第一个实测值剪枝"。
    """

    ms: float | None
    config: tuple[str, str, str] | None = None
    source: str = "--tune-timeout"

    def exceeds(self, wall_ms: float) -> bool:
        """这一次调用是否超过当前上界（``ms is None`` = 不设上界，永不超界）。"""
        return self.ms is not None and float(wall_ms) > float(self.ms)

    def observe(self, wall_ms: float, config: Iterable[str]) -> bool:
        """记录一次实测：更小就立刻更新上界与对应配置，返回是否更新。

        不设上界时（``ms is None``）**只记配置不降上界**，返回值仍是 ``False``：
        "谁最快"由 :func:`tune_best` 从报表里选，本对象只负责剪枝上界。
        """
        if self.ms is None:
            return False
        value = float(wall_ms)
        if value >= float(self.ms):
            return False
        self.ms = value
        self.config = tuple(str(item) for item in config)       # type: ignore[assignment]
        return True

    def config_text(self) -> str:
        """对应配置的一行文本（报告/表尾用）；还没有任何实测时给 ``-``。"""
        if self.config is None:
            return "-"
        kernel, as_equiv, optimize = self.config
        return (f"kernel_force={kernel}  as_equivalence={as_equiv}  "
                f"optimize={optimize}")


#: 预判剪枝的两个门槛。它与 :data:`TERMINATION_MESSAGE` 那条**实测**规则是两件事：
#: 实测算出这一组超过了当前 ``min_wall_ms``，原文逐字是"耗时大于min_wall_ms,此组被终止"；
#: 而这里是**只凭规划器的 FLOPs** 就判定"它连跑都不值得跑"，原因里必须写明
#: ``plan-only estimate, not measured``——两者混为一谈就等于编造观测。
#:
#: 为什么需要它：``optimize=none`` 这类组的单次调用要算 ``7.6e17`` 次浮点运算
#: （2026-11-19 用户那条 10 算子生产方程实测），硬超时也只能等到上限才强杀
#: （缺省 ``30s × 7 = 210s``）。而规划器给这条路径的 FLOPs 只要 0.07 ms，
#: 于是"先问要跑多少运算，再决定跑不跑"能把 210 秒压到毫秒。
#:
#: 两个门槛同时满足才会剪，方向都是**保守**的：
#: * ``PREDICT_MIN_MS``：预测耗时不到 1 秒的组一概照跑（预测量级只在小样本上不可靠）；
#: * ``PREDICT_MARGIN``：预测要慢到当前上界的 4 倍才剪——速率来自**别的配置**的实测，
#:   跨配置外推本来就有量级误差，留 4 倍余量后"漏剪"远多于"误剪"。
PREDICT_MIN_MS = 1000.0
PREDICT_MARGIN = 4.0


@dataclass
class Rate:
    """已观测到的最高吞吐（FLOP/s）与"这一组要不要跑"的预判。

    速率取**最大**而不是最小：预测耗时 = ``flops / rate``，用最大速率得到的预测耗时
    最小，于是剪枝最少——"少剪"比"错剪"安全（错剪会让真正的 ``best`` 永远不出现，
    而报告里看不出来）。

    只由**成功且未超时**的组更新（``observe``），与 ``min_wall_ms`` 的口径一致：
    坏组不该定义"要多快才算值得跑"。
    """

    flops_per_s: float | None = None
    source: str = "-"

    def observe(self, flops: float | None, seconds: float | None,
                config: Iterable[str] | None = None) -> bool:
        """记一次实测吞吐，返回是否把已有速率抬高（决定"预测"值变得更宽松）。"""
        try:
            value = float(flops) / float(seconds)          # type: ignore[arg-type]
        except (TypeError, ValueError, ZeroDivisionError):
            return False
        if not value or value != value or value <= 0.0:     # NaN / 0 / 负数
            return False
        if self.flops_per_s is not None and value <= self.flops_per_s:
            return False
        self.flops_per_s = value
        if config is not None:
            kernel, as_equiv, optimize = (str(item) for item in config)
            self.source = (f"kernel_force={kernel}  as_equivalence={as_equiv}  "
                           f"optimize={optimize}")
        return True

    def predict_ms(self, flops: float | None) -> float | None:
        """按已观测吞吐预测这一组要多少毫秒；没有速率或没有 FLOPs 时给 ``None``。"""
        if self.flops_per_s is None or not flops:
            return None
        return float(flops) / float(self.flops_per_s) * 1000.0

    def should_skip(self, flops: float | None,
                    min_wall_ms: float | None) -> tuple[bool, str]:
        """预判：这一组是否**明显**比当前上界还慢，返回 ``(剪不剪, 原因)``。

        ``min_wall_ms is None``（不设上界）时永不剪：没有参照就没有"明显更慢"。
        原因串是**机器可读**的稳定形态，并明确写 ``plan-only estimate, not measured``。
        """
        predicted = self.predict_ms(flops)
        if predicted is None or min_wall_ms is None:
            return False, ""
        threshold = float(min_wall_ms) * PREDICT_MARGIN
        if predicted < max(PREDICT_MIN_MS, threshold):
            return False, ""
        reason = (f"predicted_flops={float(flops):.6e}  "
                  f"rate={float(self.flops_per_s):.6e}flop/s  "
                  f"predicted_ms={predicted:.6f}  "
                  f"threshold_ms={threshold:.6f}  "
                  f"(plan-only estimate, not measured)")
        return True, reason
