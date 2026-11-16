"""等价方程归一化：把"同一个收缩换了个写法"映射回语料里的规范形式。

为什么需要这一层
----------------
``_specialize.canonical_signature`` 只做**字母重命名**的归一化（指标按全式首次
出现顺序重新编号），因此下面三种**数学上等价**的写法会被判成不同签名、一律 MISS：

1. **单操作数内部轴序**：``cadb,...`` 与 ``acdb,...``——后者只是前者的第 0 个
   操作数把 ``c``/``a`` 两个轴换了位置（调用方把对应张量 ``transpose`` 一下即可）。
2. **操作数位置**：``ac,em->`` 与 ``em,ac->``——两个输入交换次序。
3. **字母重命名**：已由 ``canonical_signature`` 覆盖，本层不重复处理。

这三类变换都**保持收缩结果逐元素不变**（2026-11-16 实测：``cadb,menf,ac,em,bd,fn->``
配 16 槽随机张量，直接算 / 映射后算 / ``np.einsum`` 真值三者逐位相同
``-166.6946020118``）。

本层只做**改写与报告**，不改任何执行路径：``contract()`` 仍按调用方给的方程走。
若要把映射用于真实计算，调用方必须按返回的 ``operand_permutations`` /
``operand_order`` 同步调整张量——本模块会明确给出这些调整，不替调用方猜。

设计上的保守点
--------------
* 只在**能证明等价**时给映射（纯轴序/位置/改名），绝不做"看起来差不多"的猜测；
* 不做代数化简（如把 ``ij,jk->ik`` 合并成矩阵乘），那是改变计划的范畴；
* 映射完全由**符号层面**推导，不需要知道张量形状。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "Equivalence",
    "normalize_equation",
    "find_equivalents",
    "explain",
]


# 单个操作数内部轴序的全排列只对短子脚本可行；超过这个秩就只试"整段逆序"与
# "循环移位"，避免阶乘爆炸（rank 8 就会 40320 种）。
_MAX_FULL_PERM_RANK = 4
_MAX_CANDIDATES = 20000


@dataclass(frozen=True)
class Equivalence:
    """一条"等价写法"及其配套的张量调整方式。

    ``equation``
        改写后的方程（可直接交给 ``contract``）。
    ``operand_order``
        张量列表的新次序：``operand_order[i]`` 表示"改写后第 i 个操作数应取
        调用方原始张量的第几个"。``None`` 表示不需要重排。
    ``operand_permutations``
        每个操作数（**按改写后的次序**）自己的轴置换：``perm[j]`` 表示该操作数
        新轴 j 取旧轴 perm[j]（numpy ``transpose(perm)`` 语义）。``None`` 表示
        该操作数不需要转置。
    ``output_permutation``
        输出结果的轴置换。非 ``None`` 时，调用方拿到改写后的结果，还需
        ``result.transpose(output_permutation)`` 才是原方程的答案。
    ``description``
        人类可读的说明，用于报告。
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


def _split(equation: str) -> tuple[list[str], str] | None:
    """拆成 (输入子脚本列表, 输出子脚本)，失败返回 ``None``。"""
    if not isinstance(equation, str) or "->" not in equation:
        return None
    lhs, rhs = equation.split("->", 1)
    if "->" in rhs:
        return None
    terms = lhs.split(",")
    if not terms or any(not term for term in terms):
        return None
    return terms, rhs


def _minimal_axis_perm(old: str, new: str) -> tuple[int, ...] | None:
    """若 ``new`` 只是 ``old`` 的轴重排，返回该置换（``new[j] == old[perm[j]]``）。

    否则（字母集合不同）返回 ``None``——不做任何猜测。
    """
    if len(old) != len(new) or sorted(old) != sorted(new):
        return None
    positions: dict[str, list[int]] = {}
    for index, ch in enumerate(old):
        positions.setdefault(ch, []).append(index)
    used: dict[str, int] = {}
    perm: list[int] = []
    for ch in new:
        slot = used.get(ch, 0)
        if slot >= len(positions[ch]):
            return None
        perm.append(positions[ch][slot])
        used[ch] = slot + 1
    if list(perm) == list(range(len(old))):
        return None                      # 恒等置换，无意义
    return tuple(perm)


def _operand_permutation_candidates(term: str) -> list[tuple[int, ...]]:
    """单个操作数可能的轴重排（保守枚举，避免阶乘爆炸）。"""
    rank = len(term)
    candidates: list[tuple[int, ...]] = []
    indices = list(range(rank))
    if rank <= _MAX_FULL_PERM_RANK:
        for perm in itertools.permutations(indices):
            if list(perm) != indices:
                candidates.append(perm)
    else:
        # 秩较大时只试三种最有物理意义的：整段逆序、首尾交换、循环移位。
        candidates.append(tuple(reversed(indices)))
        swapped = indices[:]
        swapped[0], swapped[-1] = swapped[-1], swapped[0]
        candidates.append(tuple(swapped))
        for shift in range(1, rank):
            candidates.append(tuple(indices[shift:] + indices[:shift]))
    return candidates[:64]


def _candidate_equations(equation: str) -> list[Equivalence]:
    """枚举所有候选等价写法（不判是否在语料里，那由调用方查）。"""
    parsed = _split(equation)
    if parsed is None:
        return []
    terms, out = parsed
    results: list[Equivalence] = []
    seen: set[str] = set()

    def add(new_terms: list[str],
            order: tuple[int, ...] | None,
            perms: list[tuple[int, ...] | None],
            description: str, kind: str,
            output_permutation: tuple[int, ...] | None = None,
            new_out: str | None = None) -> None:
        out_text = out if new_out is None else new_out
        candidate = ",".join(new_terms) + "->" + out_text
        if candidate == equation or candidate in seen:
            return
        seen.add(candidate)
        results.append(Equivalence(
            equation=candidate,
            operand_order=order,
            operand_permutations=tuple(perms) if any(perms) else (),
            output_permutation=output_permutation,
            description=description,
            kind=kind,
        ))

    # ---- ① 操作数位置排列（只换顺序，张量按同一次序传即可）----
    if len(terms) > 1:
        orders: list[tuple[int, ...]] = []
        # **两两互换**优先：这是最常见也最小的变换（"我只把第 i 和第 j 个换了"），
        # 任何操作数个数下都是 O(n²) 且必须覆盖。原先"操作数多于 6 时只试相邻交换"
        # 的做法漏掉了 (1,0,2,3,4,5) 这种最自然的情形——实测 6 操作数方程
        # `menf,cadb,...` 要映射回 `cadb,menf,...` 正是它，却不在候选里。
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                order = list(range(len(terms)))
                order[i], order[j] = order[j], order[i]
                orders.append(tuple(order))
        # 操作数较少时再补全排列（秩 <= 4 才有意义，否则组合爆炸）
        if len(terms) <= 4:
            for perm in itertools.permutations(range(len(terms))):
                if perm not in orders and list(perm) != list(range(len(terms))):
                    orders.append(perm)
        for order in orders[:96]:
            add([terms[i] for i in order], order, [None] * len(terms),
                f"交换操作数位置为 {list(order)}", "operand-order")

    # ---- ② 单个操作数内部轴序 ----
    for which in range(len(terms)):
        for perm in _operand_permutation_candidates(terms[which]):
            new_terms = list(terms)
            new_terms[which] = "".join(terms[which][i] for i in perm)
            perms: list[tuple[int, ...] | None] = [None] * len(terms)
            perms[which] = perm
            add(new_terms, None, perms,
                f"操作数 {which} 轴序改为 {new_terms[which]}（张量需 transpose{perm}）",
                "operand-axes")
        if len(results) > _MAX_CANDIDATES:
            break

    # ---- ③ 输出段轴序（改写后需对**结果**做 transpose）----
    #
    # ``...->MGHew`` 与 ``...->GMHew`` 只差输出轴的前两位，收缩本身完全相同，
    # 结果互为 transpose（2026-11-16 实测：``ij,jk->ik`` 与 ``ij,jk->ki`` 在相同
    # 输入上满足 ``r2 == r1.T``）。这一族原先漏掉了，导致 `->GMHew` 这类"只换了
    # 输出顺序"的写法被判"不存在等价写法"。
    if len(out) > 1:
        out_perms: list[tuple[int, ...]] = []
        indices = list(range(len(out)))
        # 两两互换（最常见：只调换两个输出轴）
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                perm = list(indices)
                perm[i], perm[j] = perm[j], perm[i]
                out_perms.append(tuple(perm))
        # 循环移位（整体轮转）
        for shift in range(1, len(out)):
            out_perms.append(tuple(indices[shift:] + indices[:shift]))
            out_perms.append(tuple(indices[-shift:] + indices[:-shift]))
        # **子段三循环**：任意 [i, j, k] 三个位置的轮转（如 MGHew -> HMGew 是
        # 前三个字母 `(2,0,1,3,4)`）。整体轮转覆盖不到它，而这是用户实际遇到的形态。
        # 两个方向都列：(i<-j, j<-k, k<-i) 与 (i<-k, k<-j, j<-i)。
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                for k in range(j + 1, len(out)):
                    for a, b, c in (((j, k, i)), ((k, i, j))):
                        perm = list(indices)
                        perm[i], perm[j], perm[k] = a, b, c
                        entry = tuple(perm)
                        if entry not in out_perms:
                            out_perms.append(entry)
        # 短输出段才做全排列（秩 <= 4 时 24 种，秩 5 就 120 种、秩 6 是 720 种）
        if len(out) <= _MAX_FULL_PERM_RANK:
            for perm in itertools.permutations(indices):
                if perm not in out_perms and list(perm) != indices:
                    out_perms.append(perm)
        for perm in out_perms[:96]:
            new_out = "".join(out[i] for i in perm)
            add(list(terms), None, [None] * len(terms),
                f"输出轴序改为 {new_out}（结果需 transpose{perm}）",
                "output-axes", output_permutation=perm, new_out=new_out)
            if len(results) > _MAX_CANDIDATES:
                break
    return results


def normalize_equation(equation: str,
                       probe: Any = None) -> Equivalence | None:
    """把方程改写成与某个"已知可用形式"完全相同的写法。

    ``probe`` 是一个 ``callable(equation) -> bool``：给定一个候选方程，告诉本函数
    它是否**已经是**目标形式（例如"该方程在语料里能命中"）。缺省用
    ``_specialize.match`` 判非空。

    返回第一条通过 ``probe`` 的等价写法；都不通过则返回 ``None``。**本函数只改写
    方程与给出张量调整建议，绝不替调用方执行**。
    """
    if probe is None:
        from . import _specialize

        def probe(candidate: str) -> bool:        # noqa: PLW0108 - 显式绑定
            return _specialize.match(candidate) is not None

    parsed = _split(equation)
    if parsed is None:
        return None
    # 方程本身已经命中就不必改写。
    if probe(equation):
        return Equivalence(equation=equation, description="无需改写：本身已命中",
                           kind="identity")
    for candidate in _candidate_equations(equation):
        try:
            hit = probe(candidate.equation)
        except Exception:                          # noqa: BLE001 - 探测失败即跳过
            continue
        if hit:
            return candidate
    return None


def find_equivalents(equation: str, limit: int = 8) -> list[dict[str, Any]]:
    """列出该方程的全部候选等价写法（**不判是否命中**），供人工/工具挑选。

    这是纯枚举：调用方拿到列表后可自己决定拿哪一条去查语料。``limit`` 限制返回
    条数，避免高秩方程刷屏。
    """
    candidates = _candidate_equations(equation)
    return [item.as_dict() for item in candidates[:limit]]


def explain(equation: str, probe: Any = None) -> dict[str, Any]:
    """给 ``verify_specialized`` 用的完整说明：能否改写、改写成什么、要动哪些张量。"""
    normalized = normalize_equation(equation, probe=probe)
    report: dict[str, Any] = {
        "equation": equation,
        "rewritten": normalized is not None
                     and normalized.kind != "identity",
    }
    if normalized is None:
        report["reason"] = ("没有任何等价写法能落到已知形式上（本层只做轴序 / "
                            "操作数位置 / 字母重命名的等价改写）")
        return report
    report.update(normalized.as_dict())
    if normalized.kind == "identity":
        report["reason"] = "本身已命中，无需改写"
    return report
