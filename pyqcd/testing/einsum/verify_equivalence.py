"""``pyqcd.einsum._equivalence`` 的独立穷尽验证器（交叉检验，不是背书工具）。

被测契约
--------
``normalize_equation(eq)`` 给出"同一收缩的另一种写法"改回语料形式的映射：

* ``operand_order``：改写后第 i 个操作数取调用方原张量的第几个；
* ``operand_permutations``：按**新**次序、每个操作数自己的轴置换，
  ``perm[j]`` = 新轴 j 取旧轴 ``perm[j]``（numpy ``transpose(perm)`` 语义）；
* ``output_permutation``：结果要做的 ``transpose``，**已是逆置换**。

正确用法（本文件按它独立复算）::

    adjusted = [transpose(ops[order[i]], perms[i]) for i in range(n)]
    result   = transpose(contract(mapping.equation, *adjusted), output_permutation)
    # 必须逐元素等于 contract(变体方程, ops)

判据（三层，缺一不可）
----------------------
1. **结构层**（纯字符串，零浮点噪声）：映射字段必须精确解释它给出的方程——
   ``operand_order`` 是排列；新位置 i 的操作数文本 == 原操作数 ``order[i]`` 按
   ``perms[i]`` 重排；新输出文本 == 原输出按 ``invert(output_permutation)`` 重排。
2. **数值层**：按上面那段用法重算，与变体方程的直接结果比相对误差。
3. **基准层**：本文件按自己记录的变换历史推出一份**独立真值映射**，先证明它自己
   数值正确，再拿它去对照被测映射。若真值映射自己就不对，说明是验证器的错，
   不能算被测对象的缺陷——这一层专门防止"验证器坏了却去指责被测代码"。

结构层抓字段语义错误；数值层抓"字段自洽但整体不等价"与输出置换方向错误
（后者只有用**非对称**轴长才暴露）；基准层保证结论不依赖于被测实现。

方向维（E）的真值另有来源
-------------------------
``output_permutation`` 的方向不能拿被测实现自己的字段当参照，也不能只看结构
（自逆置换 ``perm == inverse`` 与全等轴长会让方向错误完全隐形）。E 的做法是
**穷举**：把映射方程的收缩结果按每一个输出置换 ``transpose`` 过去，看哪个与变体
方程的直接结果逐元素相等；输出轴长两两不同时这个置换唯一。再用"逆置换必须不等价"
证明该用例真的能区分方向；自逆置换显式标注**方向不可区分**，不计入方向验证。

为什么不 import 内部函数
------------------------
``_split`` / ``_perm_inverse`` / ``_permuted`` / ``_candidate_equations`` 一旦拿来用，
验证就退化成"自己验自己"。本文件只用两个公开入口：``normalize_equation`` /
``explain``（被测对象）与 ``_specialize.match``（判断是否命中语料）；方程拆分、
置换语义与合成、形状推导、张量生成、收缩、误差判据全部在这里重新实现。

跑法
----
::

    python -B -m pyqcd.testing.einsum.verify_equivalence
    python -B -m pyqcd.testing.einsum.verify_equivalence --fuzz 80 --seed 7
    python -B -m pyqcd.testing.einsum.verify_equivalence --quick

退出码 0 = 全过，1 = 有 FAIL。``未验证``（数值参考超出预算等）不计为 FAIL，也不会
被算成通过，汇总里单列。

七个维度：A 单层 / B 两两组合 / C 三层叠合 / D 单层反证 / E 置换方向 / F 随机模糊 /
G 边界与鲁棒性；另有 S 基准自检（验证器自证）与 T 有界性。
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
import time
from collections import Counter
from math import prod

import numpy as np

# 被测对象的公开入口 + 语料命中判定。除此之外不 import 任何内部实现。
from pyqcd.einsum import _equivalence as equivalence
from pyqcd.einsum import _specialize as specialize

# 真实轴长来源（只读；G 维度的"长度 1 轴"用它在语料真实形态上验证）。
from pyqcd.einsum._bench_scene import dims as scene_dims

# 相对误差判据：同一算式换求和次序的浮点噪声在 1e-14 量级，1e-9 留足余量；
# 真正漏掉一层或用错置换方向的误差是 O(1)，1e-6 是"确实不等价"的下限。
TOL_PASS = 1e-9
TOL_WRONG = 1e-6

# 单个中间张量的元素数上限（float64 ≈ 32 MB）。超了记"未验证"，不硬算。
BUDGET = 4_000_000

# 有界性判据：单次 normalize 的耗时上限（秒）。
TIME_LIMIT = 30.0

# 造形状用的长度池：互不相同，能把"轴序错"从数值上暴露出来。
_PALETTE = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)


class MappingError(Exception):
    """映射字段自相矛盾或无法作用（结构层判据失败）。"""


class BudgetExceeded(Exception):
    """数值参考的中间张量超出预算，该用例只能记"未验证"。"""


# ────────────────────────────────────────────────────────────────────────────
# 自实现的方程工具（不依赖被测模块）
# ────────────────────────────────────────────────────────────────────────────


def parse(equation):
    """``"a,b->c"`` → ``(["a", "b"], "c")``；非法返回 ``None``。"""
    if not isinstance(equation, str) or "->" not in equation:
        return None
    lhs, rhs = equation.split("->", 1)
    if "->" in rhs:
        return None
    terms = lhs.split(",")
    if not terms or any(not term for term in terms):
        return None
    return terms, rhs


def invert(perm):
    """逆置换：``inverse[perm[j]] = j``。"""
    inverse = [0] * len(perm)
    for j, source in enumerate(perm):
        inverse[source] = j
    return tuple(inverse)


def is_perm(perm, size):
    return len(perm) == size and sorted(perm) == list(range(size))


def permute(text, perm):
    """按 numpy ``transpose`` 语义重排指标段：新轴 j 取旧轴 ``perm[j]``。"""
    return "".join(text[index] for index in perm)


def structure_key(equation):
    """字母无关的结构键：指标按全式首次出现顺序编号，保留操作数次序与轴序。

    这是本文件自己实现的"形态相等"判定，用来回答"映射结果是不是期望的那个形态"，
    不借用被测模块的签名函数。输出含输入之外的指标、或输出指标重复时返回 ``None``
    （numpy 也拒绝这两种）。
    """
    parsed = parse(equation)
    if parsed is None:
        return None
    terms, out = parsed
    numbers = {}
    encoded = []
    for term in terms:
        row = []
        for char in term:
            if char not in numbers:
                numbers[char] = len(numbers)
            row.append(numbers[char])
        encoded.append(tuple(row))
    out_row = []
    for char in out:
        if char not in numbers:
            return None
        out_row.append(numbers[char])
    if len(set(out_row)) != len(out_row):
        return None
    return tuple(encoded) + (tuple(out_row),)


class Tracker:
    """按层施加变换并**自己记录**完整变换历史，用于推出独立真值映射。

    记录的是 base → variant 的正向变换：
    ``variant_terms[i] == permute(base_terms[order[i]], perms[i])``，
    ``variant_out == permute(base_out, outperm)``。
    """

    def __init__(self, base):
        parsed = parse(base)
        if parsed is None:
            raise MappingError(f"无法解析: {base!r}")
        self.base = base
        self.terms, self.out = parsed
        self.order = list(range(len(self.terms)))
        self.perms = {}
        self.outperm = tuple(range(len(self.out)))
        self.steps = []

    # ---- 正向：base -> variant ----
    def swap(self, i, j):
        self.order[i], self.order[j] = self.order[j], self.order[i]
        self.steps.append(("swap", i, j))
        return self

    def axes(self, position, perm):
        source = self.order[position]
        current = self.perms.get(source, tuple(range(len(self.terms[source]))))
        if not is_perm(perm, len(current)):
            raise MappingError(f"轴序 {perm} 不是秩 {len(current)} 的排列")
        self.perms[source] = tuple(current[index] for index in perm)
        self.steps.append(("axes", position, tuple(perm)))
        return self

    def out_axes(self, perm):
        if not is_perm(perm, len(self.out)):
            raise MappingError(f"输出轴序 {perm} 不是秩 {len(self.out)} 的排列")
        self.outperm = tuple(self.outperm[index] for index in perm)
        self.steps.append(("out", tuple(perm)))
        return self

    # ---- 读回 ----
    def terms_now(self):
        result = []
        for position, source in enumerate(self.order):
            perm = self.perms.get(source)
            text = self.terms[source]
            result.append(text if perm is None else permute(text, perm))
        return result

    def equation(self):
        return ",".join(self.terms_now()) + "->" + permute(self.out, self.outperm)

    def truth_mapping(self):
        """variant → base 的**独立真值** ``(order, perms, outperm)``。

        variant 位置 j 的张量是 ``transpose(base[order[j]], perms[j])``，因此 base
        位置 k 要取 variant 位置 ``inverse_order[k]`` 并施加 ``inverse(perms[j])``；
        结果为 ``transpose(variant_result, outperm)``（推导见模块 docstring）。
        """
        inverse_order = invert(tuple(self.order))
        perms = []
        for k in range(len(self.order)):
            j = inverse_order[k]
            perm = self.perms.get(self.order[j])
            perms.append(None if perm is None else invert(perm))
        outperm = None if list(self.outperm) == list(range(len(self.out))) \
            else tuple(self.outperm)
        return tuple(inverse_order), tuple(perms), outperm

    def layers_used(self):
        used = []
        if list(self.order) != list(range(len(self.order))):
            used.append("order")
        if any(perm is not None for perm in self.perms.values()):
            used.append("axes")
        if list(self.outperm) != list(range(len(self.out))):
            used.append("out")
        return used


class TruthMapping:
    """与 ``Equivalence`` 同形的纯字段容器（真值映射，不来自被测模块）。"""

    __slots__ = ("equation", "operand_order", "operand_permutations",
                 "output_permutation", "description", "kind")

    def __init__(self, equation, order, perms, outperm, kind="truth"):
        self.equation = equation
        self.operand_order = order
        self.operand_permutations = perms
        self.output_permutation = outperm
        self.description = "本验证器推出的真值映射"
        self.kind = kind

    def as_dict(self):
        return {"equation": self.equation,
                "operand_order": None if self.operand_order is None
                else list(self.operand_order),
                "operand_permutations": [None if p is None else list(p)
                                         for p in self.operand_permutations] or None,
                "output_permutation": None if self.output_permutation is None
                else list(self.output_permutation),
                "description": self.description, "kind": self.kind}


def describe_steps(steps):
    words = []
    for step in steps:
        if step[0] == "swap":
            words.append(f"换位{step[1]}<->{step[2]}")
        elif step[0] == "axes":
            words.append(f"位{step[1]}轴序{step[2]}")
        else:
            words.append(f"输出轴序{step[1]}")
    return " + ".join(words) or "（无变换）"


def build_variant(base, steps):
    """按步骤表构造变体，返回 ``(变体方程, Tracker)``。"""
    tracker = Tracker(base)
    for step in steps:
        if step[0] == "swap":
            tracker.swap(step[1], step[2])
        elif step[0] == "axes":
            tracker.axes(step[1], step[2])
        else:
            tracker.out_axes(step[1])
    return tracker.equation(), tracker


# ────────────────────────────────────────────────────────────────────────────
# 自实现的映射语义（把三个字段翻译成可执行的张量调整）
# ────────────────────────────────────────────────────────────────────────────


def resolve(equation, mapping):
    """把映射字段解析成 ``(order, perms, outperm)``；非法字段抛 :class:`MappingError`。

    ``perms`` 长度恒等于操作数个数（缺失补 ``None``），下标是**新**位置。
    """
    parsed = parse(equation)
    if parsed is None:
        raise MappingError(f"变体方程无法解析: {equation!r}")
    terms, out = parsed
    count = len(terms)

    order = mapping.operand_order
    order = tuple(range(count)) if order is None else tuple(order)
    if not is_perm(order, count):
        raise MappingError(f"operand_order={order} 不是 0..{count - 1} 的排列")

    raw = mapping.operand_permutations or ()
    if not raw:
        perms = (None,) * count
    else:
        if len(raw) != count:
            raise MappingError(
                f"operand_permutations 长度 {len(raw)} != 操作数个数 {count}")
        perms = tuple(None if item is None else tuple(item) for item in raw)

    for position, perm in enumerate(perms):
        rank = len(terms[order[position]])
        if perm is not None and not is_perm(perm, rank):
            raise MappingError(
                f"新位置 {position} 的 perm={perm} 不是秩 {rank} 的排列")

    outperm = mapping.output_permutation
    if outperm is not None:
        outperm = tuple(outperm)
        if not is_perm(outperm, len(out)):
            raise MappingError(
                f"output_permutation={outperm} 不是秩 {len(out)} 的排列")
    return order, perms, outperm


def structural_report(equation, mapping):
    """结构层判据：返回问题描述，空串表示字段与映射方程完全一致。"""
    parsed = parse(equation)
    if parsed is None:
        return f"变体方程无法解析: {equation!r}"
    terms, out = parsed
    mapped = parse(mapping.equation)
    if mapped is None:
        return f"映射方程无法解析: {mapping.equation!r}"
    new_terms, new_out = mapped
    if len(new_terms) != len(terms):
        return f"操作数个数 {len(new_terms)} != {len(terms)}"

    order, perms, outperm = resolve(equation, mapping)   # 字段非法即抛

    for position, source in enumerate(order):
        perm = perms[position]
        expect = terms[source] if perm is None else permute(terms[source], perm)
        if new_terms[position] != expect:
            hint = "" if perm is None else f"（transpose{perm}）"
            return (f"新位置 {position} 写成 {new_terms[position]!r}，"
                    f"按字段应为 {terms[source]!r}{hint} → {expect!r}")

    if outperm is None:
        if new_out != out:
            return (f"输出段 {new_out!r} 与变体 {out!r} 不同，"
                    f"但 output_permutation 为 None")
    else:
        expect_out = permute(out, invert(outperm))
        if new_out != expect_out:
            return (f"输出段 {new_out!r} 与 invert({outperm}) 的预期 "
                    f"{expect_out!r} 不符")
    return ""


# ────────────────────────────────────────────────────────────────────────────
# 自实现的形状推导、张量生成与数值收缩
# ────────────────────────────────────────────────────────────────────────────


def make_sizes(equation, scheme="auto", override=None):
    """标签 → 轴长。

    ``scheme="auto"``：标签数不超过长度池时**全部互不相同**（轴向错误在数值上可见）；
    否则只让**输出段**互不相同、其余统一 2（控制成本）。两条路都保证输出段各轴
    长度 pairwise 不同——否则输出置换的方向错误会被形状对称掩盖。
    """
    parsed = parse(equation)
    if parsed is None:
        raise MappingError(f"方程无法解析: {equation!r}")
    terms, out = parsed
    labels = []
    for term in terms:
        for char in term:
            if char not in labels:
                labels.append(char)
    if override is not None:
        sizes = dict(override)
        for char in labels:
            sizes.setdefault(char, 2)
        return sizes
    if scheme == "auto" and 0 < len(labels) <= len(_PALETTE):
        return {char: _PALETTE[index] for index, char in enumerate(labels)}
    sizes = {}
    for index, char in enumerate(out):
        sizes[char] = 2 + index
    for char in labels:
        sizes.setdefault(char, 2)
    return sizes


def generate(equation, sizes, seed=0):
    """按标签长度生成张量；同一标签在所有操作数里同长。"""
    parsed = parse(equation)
    terms, _ = parsed
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(tuple(sizes[char] for char in term))
            for term in terms]


def contract(equation, operands, sizes, budget=BUDGET):
    """自实现的收缩：按"结果张量最小"贪心两两收缩，带中间量预算。

    不用 ``numpy.einsum(optimize="greedy")`` 的整式路径：实测 16 操作数、全等轴长时
    它会把中间量撑到 GB 级（本机 68 s 未完成），而同一方程换一对轴长只要 1.5 ms——
    耗时对 numpy 的路径启发式过于敏感，不适合当固定判据的参考实现。
    """
    parsed = parse(equation)
    if parsed is None:
        raise MappingError(f"方程无法解析: {equation!r}")
    terms, out = parsed
    if len(terms) != len(operands):
        raise MappingError(f"{len(terms)} 个操作数但给了 {len(operands)} 个张量")
    if len(terms) == 1 or any(len(set(term)) != len(term) for term in terms):
        # 单操作数 / 含对角标签：交给 numpy 一次算完（这两种情形规模本来就小）。
        total = sum(int(np.size(item)) for item in operands)
        if total > budget:
            raise BudgetExceeded(f"操作数合计 {total} 元素超出预算 {budget}")
        return np.einsum(equation, *operands, optimize=False)

    work = [(list(term), array) for term, array in zip(terms, operands)]
    out_set = set(out)
    while len(work) > 1:
        later = Counter()
        for labels, _ in work:
            later.update(labels)
        best = None
        for i in range(len(work)):
            for j in range(i + 1, len(work)):
                shared = Counter(work[i][0])
                shared.update(work[j][0])
                union = list(dict.fromkeys(work[i][0] + work[j][0]))
                keep = [lab for lab in union
                        if lab in out_set or later[lab] > shared[lab]]
                size = prod(sizes[lab] for lab in keep)
                if best is None or size < best[0]:
                    best = (size, i, j, keep)
        size, i, j, keep = best
        if size > budget:
            raise BudgetExceeded(f"中间张量 {size} 元素超出预算 {budget}")
        step = (",".join(["".join(work[i][0]), "".join(work[j][0])])
                + "->" + "".join(keep))
        merged = np.einsum(step, work[i][1], work[j][1], optimize=False)
        expect = tuple(sizes[lab] for lab in keep)
        if merged.shape != expect:            # 自检：形状推导必须与 numpy 一致
            raise MappingError(
                f"内部收缩形状 {merged.shape} != 推导的 {expect}（步 {step}）")
        work[i] = (keep, merged)
        work.pop(j)
    labels, array = work[0]
    return np.einsum("".join(labels) + "->" + "".join(out), array, optimize=False)


def adjust(equation, mapping, operands, drop=None):
    """按映射调整张量；``drop`` 用于"故意漏掉某一层"的反证。

    返回 ``(adjusted, outperm)``，``adjusted`` 可直接喂给 ``mapping.equation``。
    """
    order, perms, outperm = resolve(equation, mapping)
    if drop == "order":
        order = tuple(range(len(order)))
    elif drop == "axes":
        perms = (None,) * len(perms)
    elif drop == "out":
        outperm = None
    elif drop is not None:                                # pragma: no cover
        raise MappingError(f"未知的漏层 {drop!r}")
    adjusted = []
    for position, source in enumerate(order):
        tensor = operands[source]
        perm = perms[position]
        adjusted.append(tensor if perm is None else np.transpose(tensor, perm))
    return adjusted, outperm


def compare(expected, got):
    """``(是否在容差内, 相对误差, 说明)``；形状不符即判不等价。"""
    if tuple(np.shape(expected)) != tuple(np.shape(got)):
        return False, float("inf"), f"形状 {np.shape(got)} != {np.shape(expected)}"
    if np.size(expected) == 0:
        return True, 0.0, ""
    diff = float(np.max(np.abs(np.asarray(got) - np.asarray(expected))))
    scale = max(float(np.max(np.abs(expected))), 1e-300)
    relative = diff / scale
    return relative <= TOL_PASS, relative, ""


def evaluate(variant, mapping, sizes=None, seed=0):
    """按契约重算并返回 ``(相对误差, 说明)``；形状不符时误差为 ``inf``。"""
    table = sizes if sizes is not None else make_sizes(variant)
    operands = generate(variant, table, seed)
    expected = contract(variant, operands, table)
    adjusted, outperm = adjust(variant, mapping, operands)
    canonical = contract(mapping.equation, adjusted, table)
    got = canonical if outperm is None else np.transpose(canonical, outperm)
    return compare(expected, got)


# ────────────────────────────────────────────────────────────────────────────
# 记录与打印
# ────────────────────────────────────────────────────────────────────────────


class Recorder:
    """逐条记录并即时打印，长跑时也能看到进度。"""

    def __init__(self):
        self.rows = []

    def add(self, dimension, name, status, error=None, note=""):
        self.rows.append({"dimension": dimension, "name": name,
                          "status": status, "error": error, "note": note})
        text = "误差=—" if error is None else f"误差={error:.1e}"
        tail = f"  {note}" if note else ""
        print(f"[{dimension}] {name}  {status}  {text}{tail}", flush=True)

    def counts(self, dimension):
        rows = [row for row in self.rows if row["dimension"] == dimension]
        passed = sum(1 for row in rows if row["status"] == "PASS")
        return passed, len(rows)

    def failures(self):
        return [row for row in self.rows if row["status"] == "FAIL"]

    def unverified(self):
        return [row for row in self.rows if row["status"] == "未验证"]


def record(rec, dimension, name, ok, error=None, note=""):
    rec.add(dimension, name, "PASS" if ok else "FAIL", error, note)


# ────────────────────────────────────────────────────────────────────────────
# 单条用例的完整判据
# ────────────────────────────────────────────────────────────────────────────


def check_mapping(rec, dimension, name, variant, mapping, sizes=None, seed=0):
    """结构层 + 数值层；返回相对误差（失败时 ``None``）。"""
    try:
        problem = structural_report(variant, mapping)
    except MappingError as exc:
        rec.add(dimension, name, "FAIL", None, f"结构层：{exc}")
        return None
    if problem:
        rec.add(dimension, name, "FAIL", None, f"结构层：{problem}")
        return None
    try:
        ok, relative, why = evaluate(variant, mapping, sizes=sizes, seed=seed)
    except BudgetExceeded as exc:
        rec.add(dimension, name, "未验证", None, f"结构层通过；数值参考不可用：{exc}")
        return None
    except MappingError as exc:
        rec.add(dimension, name, "FAIL", None, f"数值层：{exc}")
        return None
    note = f"kind={mapping.kind}" + (f"  {why}" if why else "")
    rec.add(dimension, name, "PASS" if ok else "FAIL", relative, note)
    return relative if ok else None


def check_case(rec, dimension, name, variant, tracker, sizes=None, seed=0,
               probe=None, expect_key=None):
    """一条完整用例：被测映射 + 真值映射双跑，任一层失败即 FAIL。"""
    started = time.perf_counter()
    try:
        mapping = equivalence.normalize_equation(variant, probe=probe)
    except Exception as exc:                              # noqa: BLE001
        rec.add(dimension, name, "FAIL", None,
                f"normalize_equation 抛 {type(exc).__name__}: {exc}")
        return None
    elapsed = (time.perf_counter() - started) * 1e3

    if mapping is None:
        rec.add(dimension, name, "FAIL", None,
                f"normalize 返回 None（耗时 {elapsed:.0f} ms）")
        return None

    if expect_key is not None:
        if structure_key(mapping.equation) != expect_key:
            rec.add(dimension, name, "FAIL", None,
                    f"映射方程 {mapping.equation!r} 不是期望形态")
            return None
    else:
        try:
            hit = specialize.match(mapping.equation)
        except Exception as exc:                          # noqa: BLE001
            rec.add(dimension, name, "FAIL", None,
                    f"match 抛 {type(exc).__name__}: {exc}")
            return None
        if hit is None:
            rec.add(dimension, name, "FAIL", None,
                    f"映射结果 {mapping.equation!r} 不命中语料")
            return None

    table = sizes if sizes is not None else make_sizes(variant)
    relative = check_mapping(rec, dimension, f"{name}", variant, mapping,
                             sizes=table, seed=seed)
    if relative is None:
        return None

    # 基准层：真值映射也必须数值正确，否则是验证器自己的语义错了。
    truth = self_truth(tracker, variant)
    truth_ok, truth_error, truth_why = evaluate(variant, truth, sizes=table,
                                                seed=seed)
    if not truth_ok:
        rec.add(dimension, f"{name} 基准自检", "FAIL", truth_error,
                f"真值映射自己不等价——验证器语义有误，本条结论作废：{truth_why}")
        return None

    if structure_key(mapping.equation) != structure_key(truth.equation):
        rec.add(dimension, f"{name} 与真值形态", "PASS", relative,
                "映射方程与真值方程的形态不同但数值均正确"
                f"（映射 {mapping.equation[:24]}… / 真值 {truth.equation[:24]}…）")
    return mapping


def self_truth(tracker, variant):
    """由变换历史推出的真值映射（强制结构自洽）。

    ``TruthMapping.equation`` 与该层的语义必须是**目标（base）方程**：调用方按
    ``operand_order`` / ``operand_permutations`` 调整后的张量要喂给它，结果再
    ``transpose(output_permutation)`` 才回到变体方程的答案。这里曾误填
    ``tracker.equation()``（= 变体方程本身），把"目标"写成了"输入"，于是结构层
    必然报"新位置 0 写成 X，按字段应为 Y"——那是本验证器的缺陷，不是被测对象的。
    """
    if tracker.equation() != variant:
        raise MappingError(
            "变换历史与待验方程不一致（验证器缺陷）："
            f"历史重建 {tracker.equation()!r} != 待验 {variant!r}")
    order, perms, outperm = tracker.truth_mapping()
    truth = TruthMapping(tracker.base, order, perms, outperm)
    problem = structural_report(variant, truth)
    if problem:
        raise MappingError(f"真值映射结构不自洽（验证器缺陷）：{problem}")
    return truth


# ────────────────────────────────────────────────────────────────────────────
# 语料样本（只取方程文本；形状由本文件自造）
# ────────────────────────────────────────────────────────────────────────────

SEQ1 = "cadb,menf,ac,em,bd,fn->"
SEQ2585 = "ambn,codp,egfh,ce,mo,Mbdf,Nnph->MNag"
SEQ4307 = "ambn,cgdh,eifj,okpl,ce,Gmo,gi,Nbdf,Nnp,Mhjl->MNGak"
USER16 = ("cwdx,eufv,mknl,oapb,isjt,qgrh,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx"
          "->MGHew")
# 任务给定的"三层叠合"实例：
#   位置 4<->5 + 新位置 5 的 isjt->istj perm(0,1,3,2) + 输出 MGHew->GMHew
USER16_LAYERED = ("cwdx,eufv,mknl,oapb,qgrh,istj,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,"
                  "Mlr,tvx->GMHew")
USER16_STEPS = [("swap", 4, 5), ("axes", 5, (0, 1, 3, 2)), ("out", (1, 0, 2, 3, 4))]


# ────────────────────────────────────────────────────────────────────────────
# A. 三层变换的单层验证
# ────────────────────────────────────────────────────────────────────────────


def run_single_layer(rec, seed=0):
    """每族单独作用一次。返回成功样例备查（D 只收多层样例，E 已自带独立用例）。"""
    cases = [
        ("A 位置 seq1 换 0<->1", SEQ1, [("swap", 0, 1)]),
        ("A 位置 seq2585 换 0<->1", SEQ2585, [("swap", 0, 1)]),
        ("A 位置 user16 换 4<->5", USER16, [("swap", 4, 5)]),
        ("A 轴序 秩2 seq1 op2 ac->ca", SEQ1, [("axes", 2, (1, 0))]),
        ("A 轴序 秩4 seq2585 op0 ambn->nbma", SEQ2585,
         [("axes", 0, (3, 2, 1, 0))]),
        ("A 轴序 秩4 三循环 seq4307 op1 cgdh->gdch", SEQ4307,
         [("axes", 1, (1, 2, 0, 3))]),
        ("A 输出 互换 GMHew", USER16, [("out", (1, 0, 2, 3, 4))]),
        ("A 输出 三循环 HMGew", USER16, [("out", (2, 0, 1, 3, 4))]),
        ("A 输出 轮转 wMGHe", USER16, [("out", (4, 0, 1, 2, 3))]),
    ]
    samples = []
    for index, (name, base, steps) in enumerate(cases):
        variant, tracker = build_variant(base, steps)
        sizes = make_sizes(variant)
        mapping = check_case(rec, "A", name, variant, tracker, sizes=sizes,
                             seed=seed + index)
        if mapping is not None:
            samples.append({"name": name, "variant": variant, "mapping": mapping,
                            "sizes": sizes, "tracker": tracker})
    return samples


# ────────────────────────────────────────────────────────────────────────────
# B. 两两组合（1+2 / 1+3 / 2+3 各两条）
# ────────────────────────────────────────────────────────────────────────────


def run_pairs(rec, seed=0):
    cases = [
        ("B 1+2 seq1 换位0<->1 且 op0 反转", SEQ1,
         [("swap", 0, 1), ("axes", 0, (3, 2, 1, 0))]),
        ("B 1+2 seq2585 换位0<->2 且 op0 反转", SEQ2585,
         [("swap", 0, 2), ("axes", 0, (3, 2, 1, 0))]),
        ("B 1+3 seq4307 换位0<->1 且 输出互换", SEQ4307,
         [("swap", 0, 1), ("out", (1, 0, 2, 3, 4))]),
        ("B 1+3 seq2585 换位1<->2 且 输出三循环", SEQ2585,
         [("swap", 1, 2), ("out", (2, 0, 1, 3))]),
        ("B 2+3 seq4307 op0 反转 且 输出三循环", SEQ4307,
         [("axes", 0, (3, 2, 1, 0)), ("out", (2, 0, 1, 3, 4))]),
        ("B 2+3 seq2585 op1 反转 且 输出互换", SEQ2585,
         [("axes", 1, (3, 2, 1, 0)), ("out", (1, 0, 2, 3))]),
    ]
    samples = []
    for index, (name, base, steps) in enumerate(cases):
        variant, tracker = build_variant(base, steps)
        sizes = make_sizes(variant)
        mapping = check_case(rec, "B", name, variant, tracker, sizes=sizes,
                             seed=seed + index)
        if mapping is not None:
            samples.append({"name": name, "variant": variant, "mapping": mapping,
                            "sizes": sizes, "tracker": tracker})
    return samples


# ────────────────────────────────────────────────────────────────────────────
# C. 三层叠合（任务给定实例）
# ────────────────────────────────────────────────────────────────────────────


def run_stack(rec, seed=0):
    variant, tracker = build_variant(USER16, USER16_STEPS)
    exact = variant == USER16_LAYERED
    rec.add("C", "C 实例逐字重建", "PASS" if exact else "FAIL", None,
            "本文件按三层步骤重建出的方程与任务文本"
            + ("一致" if exact else f"不一致：{variant}"))
    if not exact:
        return []
    sizes = make_sizes(USER16_LAYERED)
    mapping = check_case(rec, "C", "C 三层叠合 user16", USER16_LAYERED, tracker,
                         sizes=sizes, seed=seed)
    if mapping is None:
        return []
    return [{"name": "C 三层叠合 user16", "variant": USER16_LAYERED,
             "mapping": mapping, "sizes": sizes, "tracker": tracker}]


# ────────────────────────────────────────────────────────────────────────────
# D. 单层反证：故意漏掉一层，断言不再等价
# ────────────────────────────────────────────────────────────────────────────


def run_drop_layer(rec, samples, seed=0):
    """对每个多层样例，用**真值映射**逐层漏掉一层，断言不再等价。

    用真值而不是被测映射：即使被测实现返回 None，漏层反证依然成立，
    三个层也一定能各覆盖一次（真值映射按构造必然同时用到它记录过的层）。
    """
    covered = Counter()
    tested = 0
    for sample in samples:
        variant = sample["variant"]
        tracker = sample["tracker"]
        layers = tracker.layers_used()
        if len(layers) < 2:
            continue
        truth = self_truth(tracker, variant)
        for layer in layers:
            tested += 1
            covered[layer] += 1
            name = f"D 漏层[{layer}] {sample['name']}"
            try:
                ok, relative, why = evaluate(
                    variant, truth, sizes=sample["sizes"], seed=seed)
            except (BudgetExceeded, MappingError) as exc:
                rec.add("D", name, "未验证", None, f"数值参考不可用：{exc}")
                continue
            if not ok:
                rec.add("D", name, "未验证", None,
                        f"真值映射本身就不等价，漏层反证无意义：{why or relative}")
                continue
            table = sample["sizes"]
            operands = generate(variant, table, seed)
            adjusted, outperm = adjust(variant, truth, operands, drop=layer)
            try:
                expected = contract(variant, operands, table)
                canonical = contract(truth.equation, adjusted, table)
            except BudgetExceeded as exc:
                rec.add("D", name, "未验证", None, f"{type(exc).__name__}: {exc}")
                continue
            except (ValueError, MappingError) as exc:
                # 漏层后张量的轴序与目标方程标签不再对应：要么 numpy 拒绝广播，
                # 要么本文件的形状自检发现实测形状与推导不符。**同一份 contract
                # 在紧邻上方对未漏层的真值映射刚刚算通过**，所以这里的失败只可能
                # 来自被丢掉的这一层——它就是"该层不可省"的证据。
                rec.add("D", name, "PASS", float("inf"),
                        f"漏掉 {layer} 层后不再适用（{type(exc).__name__}）：{exc}")
                continue
            got = canonical if outperm is None else np.transpose(canonical, outperm)
            same, relative_drop, why_drop = compare(expected, got)
            if same or (relative_drop <= TOL_WRONG and not why_drop):
                rec.add("D", name, "FAIL", relative_drop,
                        f"漏掉 {layer} 层后仍然等价（误差={relative_drop:.1e}）"
                        "——该层似乎可省")
            else:
                detail = why_drop or f"误差放大到 {relative_drop:.1e}"
                rec.add("D", name, "PASS", relative_drop,
                        f"漏掉 {layer} 层后不等价：{detail}")
    for layer in ("order", "axes", "out"):
        if covered[layer] == 0:
            rec.add("D", f"D 漏层[{layer}] 覆盖", "未验证", None,
                    "本轮没有任何多层样例用到这一层，无法做该层的漏层反证")
    return tested, covered


# ────────────────────────────────────────────────────────────────────────────
# E. 置换方向专测（真值独立穷举；非自逆必须唯一正确，自逆标注不可区分）
# ────────────────────────────────────────────────────────────────────────────

#: 方向专测用例：``(名称, base 语料方程, 步骤表)``。
#: base 全是语料方程，变体含输出段轴序（方向专测的靶点），因此
#: ``normalize_equation`` 必须给出映射；给不出就是覆盖缺口，如实记 FAIL。
#: 覆盖：无置换 / 自逆互换 / 三循环**及其逆** / 五循环 / 四循环 / 自逆双换 /
#: 复合（换位+操作数轴序+输出三循环：映射方程的输入本身也不同，穷举数值真值
#: 不再与结构层同义反复）。
DIRECTION_CASES = (
    ("恒等（无输出置换）", USER16, []),
    ("自逆互换 GMHew", USER16, [("out", (1, 0, 2, 3, 4))]),
    ("三循环 HMGew", USER16, [("out", (2, 0, 1, 3, 4))]),
    ("三循环逆 GHMew", USER16, [("out", (1, 2, 0, 3, 4))]),
    ("五循环 wMGHe", USER16, [("out", (4, 0, 1, 2, 3))]),
    ("四循环 NgMa", SEQ2585, [("out", (1, 3, 0, 2))]),
    ("自逆双换 NMga", SEQ2585, [("out", (1, 0, 3, 2))]),
    ("复合 换位4<->5 + op5轴序 + 输出三循环", USER16,
     [("swap", 4, 5), ("axes", 5, (0, 1, 3, 2)), ("out", (2, 0, 1, 3, 4))]),
)


def brute_force_output_permutation(variant, mapping, table, seed):
    """穷举输出段的**全部**置换，返回 ``(命中的置换列表, 目标张量, 映射结果张量)``。

    真值完全由 numpy 决定：``transpose(canonical, p)`` 与 ``contract(variant)``
    逐元素相等的那些 ``p``。不读被测模块的任何内部量，也不经过 ``Tracker``。
    输出轴长 pairwise 不同时形状已经排掉大部分候选，剩余候选由数值比对钉死。
    """
    operands = generate(variant, table, seed)
    target = contract(variant, operands, table)
    adjusted, _ = adjust(variant, mapping, operands)
    canonical = contract(mapping.equation, adjusted, table)
    rank = np.ndim(canonical)
    matches = [perm for perm in itertools.permutations(range(rank))
               if compare(target, np.transpose(canonical, perm))[0]]
    return matches, target, canonical


def run_direction(rec, seed=0):
    """``output_permutation`` 方向专测：真值来自穷举，不来自被测实现。

    每条用例做四件事：①穷举出"哪个输出置换能把映射结果还原成变体结果"，它必须
    唯一且等于映射给的置换；②结构上把变体输出段按真值置换的逆重排，必须逐字等于
    映射方程的输出段；③非自逆置换还要断言用**逆置换**重算确实不等价（证明这条
    用例真的能区分方向）；④自逆置换显式标注"方向不可区分"，不计入方向验证。
    """
    stats = {"non_self": 0, "non_self_ok": 0, "self_inverse": 0,
             "identity": 0, "unverified": 0}
    for name, base, steps in DIRECTION_CASES:
        variant, _tracker = build_variant(base, steps)
        out_text = parse(variant)[1]
        rank = len(out_text)
        table = make_sizes(variant, scheme="output")   # 输出段长度 pairwise 不同
        lengths = tuple(table[char] for char in out_text)
        tag = f"E 方向 {name}"

        mapping = equivalence.normalize_equation(variant)
        if mapping is None:
            rec.add("E", tag, "FAIL", None,
                    "normalize 返回 None：变体没有映射，方向无从验证（覆盖缺口）")
            continue
        try:
            problem = structural_report(variant, mapping)
        except MappingError as exc:
            problem = str(exc)
        if problem:
            rec.add("E", tag, "FAIL", None, f"结构层：{problem}")
            continue
        if len(set(lengths)) != len(lengths):
            rec.add("E", tag, "FAIL", None,
                    f"输出轴长 {lengths} 有重复：方向不可区分，这条用例无效")
            continue

        try:
            matches, target, canonical = brute_force_output_permutation(
                variant, mapping, table, seed)
        except BudgetExceeded as exc:
            stats["unverified"] += 1
            rec.add("E", tag, "未验证", None, f"数值参考超预算：{exc}")
            continue
        except (ValueError, MappingError) as exc:
            rec.add("E", tag, "FAIL", None,
                    f"按映射重算失败（{type(exc).__name__}）：{exc}")
            continue

        identity = tuple(range(rank))
        given = (identity if mapping.output_permutation is None
                 else tuple(mapping.output_permutation))
        if len(matches) != 1:
            rec.add("E", tag, "FAIL", None,
                    f"穷举 {rank}! 个输出置换命中 {len(matches)} 个：{matches}"
                    "（应唯一：0 个=数值不等价，多个=轴长没把方向区分开）")
            continue
        truth = matches[0]
        ok, relative, why = compare(target, np.transpose(canonical, given))
        if truth != given:
            rec.add("E", tag, "FAIL", relative,
                    f"独立穷举的真值置换 {truth} != 映射给的 {given}；"
                    f"按 {given} 重算{'相等' if ok else '不等'}"
                    f"{'' if ok else f'（{why}）'}")
            continue

        mapped_out = parse(mapping.equation)[1]
        expect_out = permute(out_text, invert(given))
        if mapped_out != expect_out:
            rec.add("E", tag, "FAIL", relative,
                    f"映射方程输出段 {mapped_out!r} != 按真值置换推的 "
                    f"{expect_out!r}")
            continue

        inverse = invert(given)
        if given == identity:
            stats["identity"] += 1
            rec.add("E", tag, "PASS", relative,
                    f"输出轴长 {lengths}；"
                    + ("映射无输出置换" if mapping.output_permutation is None
                       else f"映射给的置换就是恒等 {given}")
                    + "，穷举也只认同位——三者一致")
            continue
        if given == inverse:
            stats["self_inverse"] += 1
            rec.add("E", tag, "PASS", relative,
                    f"输出轴长 {lengths}；置换 {given} 自逆（== 逆 {inverse}）："
                    "正向唯一正确，但**方向不可区分**，不据此判方向正确")
            continue

        stats["non_self"] += 1
        stats["non_self_ok"] += 1
        rec.add("E", tag, "PASS", relative,
                f"输出轴长 {lengths}；穷举唯一真值 {truth} == 映射给的置换；"
                f"逆置换 {inverse} 另测")

        back_ok, back_relative, back_why = compare(
            target, np.transpose(canonical, inverse))
        if back_ok:
            rec.add("E", f"{tag} 反向（逆置换）", "FAIL", back_relative,
                    f"用逆置换 {inverse} 转也相等——这条用例没区分出方向")
        elif back_relative > TOL_WRONG and not back_why:
            rec.add("E", f"{tag} 反向（逆置换）", "PASS", back_relative,
                    f"用逆置换 {inverse} 确实不等价：误差放大到 "
                    f"{back_relative:.1e}")
        else:
            rec.add("E", f"{tag} 反向（逆置换）", "PASS", back_relative,
                    f"用逆置换 {inverse} 确实不等价："
                    f"{back_why or f'误差 {back_relative:.1e}'}")
    return stats


# ────────────────────────────────────────────────────────────────────────────
# F. 随机模糊测试（固定 seed，可复现）
# ────────────────────────────────────────────────────────────────────────────


def random_steps(rng, equation, layers):
    """随机抽 ``layers`` 层互不相同的变换，返回 ``[(step_label, step), …]``。

    ``swap`` 会挪动位置，因此 ``axes`` 的秩必须按**当前**占该位置的操作数取：
    旧写法用 ``terms[position]``（base 上的秩），一旦先抽到 ``swap`` 就会给秩 3 的
    操作数配一个秩 2 的排列，生成器自己抛 "轴序不是秩 k 的排列"（验证器缺陷）。
    """
    parsed = parse(equation)
    terms, out = parsed
    order = list(range(len(terms)))
    labels = ["order", "axes", "out"]
    rng.shuffle(labels)
    steps = []
    for label in labels[:layers]:
        if label == "order":
            if len(terms) < 2:
                continue
            i, j = rng.sample(range(len(terms)), 2)
            step = ("swap", i, j)
            order[i], order[j] = order[j], order[i]
        elif label == "axes":
            positions = [k for k in range(len(terms))
                         if len(terms[order[k]]) >= 2]
            if not positions:
                continue
            position = rng.choice(positions)
            rank = len(terms[order[position]])
            perm = list(range(rank))
            rng.shuffle(perm)
            if perm == sorted(perm):
                perm = perm[1:] + perm[:1]               # 退化成轮转，保证非恒等
            step = ("axes", position, tuple(perm))
        else:
            if len(out) < 2:
                continue
            if len(out) <= 4:     # 秩 <= 4 时被测层枚举全排列，随便抽
                perm = list(range(len(out)))
                rng.shuffle(perm)
                if perm == sorted(perm):
                    i, j = rng.sample(range(len(out)), 2)
                    perm[i], perm[j] = perm[j], perm[i]
            else:                 # 秩 > 4 只枚举互换 / 三循环 / 轮转，不越界
                if len(out) >= 3:
                    i, j, k = rng.sample(range(len(out)), 3)
                    perm = list(range(len(out)))
                    perm[i], perm[j], perm[k] = j, k, i
                else:
                    i, j = rng.sample(range(len(out)), 2)
                    perm = list(range(len(out)))
                    perm[i], perm[j] = perm[j], perm[i]
            if perm == sorted(perm):
                continue
            step = ("out", tuple(perm))
        steps.append(step)
    return steps


def run_fuzz(rec, count=40, seed=20261116):
    from pyqcd.einsum import _wick_tasklist as tasklist

    tasks = list(tasklist.TASKS)
    rng = random.Random(seed)
    picked = rng.sample(tasks, min(count, len(tasks)))
    passed = unverified = 0
    exact_back = 0
    failures = []
    started = time.perf_counter()
    for task in picked:
        layers = rng.choice([1, 2, 3])
        steps = random_steps(rng, task.equation, layers)
        if not steps:
            continue
        try:
            tracker = Tracker(task.equation)
            for step in steps:
                if step[0] == "swap":
                    tracker.swap(step[1], step[2])
                elif step[0] == "axes":
                    tracker.axes(step[1], step[2])
                else:
                    tracker.out_axes(step[1])
            variant = tracker.equation()
        except MappingError as exc:                      # 生成器自检
            failures.append((f"seq={task.seq}", task.equation, "", str(exc)))
            rec.add("F", f"F seq={task.seq}", "FAIL", None, f"生成变体失败：{exc}")
            continue
        label = f"seq={task.seq} 层{len(steps)}"
        try:
            mapping = equivalence.normalize_equation(variant)
        except Exception as exc:                          # noqa: BLE001
            failures.append((label, variant, describe_steps(steps),
                             f"normalize 抛 {type(exc).__name__}: {exc}"))
            rec.add("F", f"F {label}", "FAIL", None, "normalize 抛异常")
            continue
        if mapping is None:
            failures.append((label, variant, describe_steps(steps), "返回 None"))
            rec.add("F", f"F {label}", "FAIL", None,
                    f"返回 None（{describe_steps(steps)}）")
            continue
        try:
            problem = structural_report(variant, mapping)
        except MappingError as exc:
            problem = str(exc)
        if problem:
            failures.append((label, variant, describe_steps(steps),
                             f"结构层：{problem}"))
            rec.add("F", f"F {label}", "FAIL", None, f"结构层：{problem}")
            continue
        if specialize.match(mapping.equation) is None:
            failures.append((label, variant, describe_steps(steps),
                             "映射结果不命中语料"))
            rec.add("F", f"F {label}", "FAIL", None, "映射结果不命中语料")
            continue
        if structure_key(mapping.equation) == structure_key(task.equation):
            exact_back += 1
        table = make_sizes(variant)
        try:
            ok, relative, why = evaluate(variant, mapping, sizes=table,
                                         seed=task.seq)
        except BudgetExceeded as exc:
            unverified += 1
            rec.add("F", f"F {label}", "未验证", None,
                    f"结构层通过；数值参考超预算：{exc}")
            continue
        except MappingError as exc:
            failures.append((label, variant, describe_steps(steps),
                             f"数值层：{exc}"))
            rec.add("F", f"F {label}", "FAIL", None, f"数值层：{exc}")
            continue
        if ok:
            passed += 1
            rec.add("F", f"F {label}", "PASS", relative, describe_steps(steps))
        else:
            detail = why or f"误差 {relative:.1e}"
            failures.append((label, variant, describe_steps(steps), detail))
            rec.add("F", f"F {label}", "FAIL", relative,
                    f"{describe_steps(steps)}；{detail}")

    if failures:
        print("\n--- F 失败样例明细 ---", flush=True)
        for label, variant, steps_text, actual in failures:
            print(f"  {label}  变换={steps_text or '（生成器）'}", flush=True)
            print(f"    变体: {variant}", flush=True)
            print(f"    实际: {actual}", flush=True)
    return {"total": len(picked), "passed": passed, "failures": failures,
            "unverified": unverified, "exact_back": exact_back,
            "seconds": time.perf_counter() - started}


# ────────────────────────────────────────────────────────────────────────────
# G. 边界与鲁棒性
# ────────────────────────────────────────────────────────────────────────────

FIXED_TARGETS = {
    "ij->ij": "ij->ij",
    "kj,ji->": "kj,ji->",
    "ji->i": "ji->i",
    "ii->i": "ii->i",
    "abcd->abcd": "abcd->abcd",
}


def target_of(name):
    """FIXED_TARGETS 里目标形态的结构键（语料之外的边界形态用）。"""
    return structure_key(FIXED_TARGETS[name])


def probe_for(name):
    key = target_of(name)
    return lambda candidate: structure_key(candidate) == key


def _compressed_real_sizes(equation, roles, scene="distill"):
    """按语料真实轴长（``_label_sizes``）等比压缩到 1/3/4/…

    保留真实的 ``1``（动量轴在 ``distill`` 档就是 1），因此能直接检验
    "长度 1 的轴转置后还等价吗"。
    """
    from pyqcd.einsum.bench_specialized import _label_sizes

    terms, _ = parse(equation)
    real = _label_sizes(terms, roles, scene_dims(scene))
    table = {}
    step = 3
    for value in sorted(set(real.values())):
        if value == 1:
            table[value] = 1
        else:
            table[value] = step
            step += 1
    return {label: table[value] for label, value in real.items()}


def run_edges(rec, seed=0):
    """单操作数 / 秩 0 输出 / 秩 1 / 对角标签 / 长度 1 轴 / 非法输入 / explain 一致。"""
    from pyqcd.einsum import _wick_tasklist as tasklist

    # 1. 单操作数 ij->ji：目标是恒等形态 ij->ij（语料不含这种形态）。
    variant, tracker = build_variant("ij->ji", [("axes", 0, (1, 0))])
    check_case(rec, "G", "G 单操作数 ij->ji 回到 ij->ij", variant, tracker,
               sizes={"i": 3, "j": 2}, seed=seed,
               probe=probe_for("ij->ij"), expect_key=target_of("ij->ij"))

    # 2. 输出秩 0：ij,jk-> 与 kj,ji-> 差三层改写（换位 + 两个操作数反转）。
    steps = [("swap", 0, 1), ("axes", 0, (1, 0)), ("axes", 1, (1, 0))]
    variant, tracker = build_variant("ij,jk->", steps)
    check_case(rec, "G", f"G 输出秩0 {variant} 回到 ij,jk->", variant, tracker,
               sizes={"i": 2, "j": 3, "k": 4}, seed=seed,
               probe=probe_for("kj,ji->"), expect_key=target_of("kj,ji->"))

    # 3. 秩 1 输出：ij->i 换位后反转。
    variant, tracker = build_variant("ij->i", [("swap", 0, 0),
                                              ("axes", 0, (1, 0))])
    check_case(rec, "G", "G 秩1输出 ji->i 回到 ij->i", variant, tracker,
               sizes={"i": 2, "j": 3}, seed=seed,
               probe=probe_for("ji->i"), expect_key=target_of("ji->i"))

    # 4. 对角标签 ii->i：本身即目标形态，映射应为 identity 且数值无误。
    variant, tracker = build_variant("ii->i", [])
    check_case(rec, "G", "G 对角 ii->i（identity）", variant, tracker,
               sizes={"i": 3}, seed=seed,
               probe=probe_for("ii->i"), expect_key=target_of("ii->i"))

    # 5. 对角反例：ij->i 不得被"等价"到 ii->i（那是把求和换成取对角）。
    diagonal = target_of("ii->i")
    started = time.perf_counter()
    try:
        bogus = equivalence.normalize_equation(
            "ij->i", probe=lambda candidate: structure_key(candidate) == diagonal)
    except Exception as exc:                              # noqa: BLE001
        rec.add("G", "G 对角反例 ij->i 不得落到 ii->i", "FAIL", None,
                f"抛 {type(exc).__name__}: {exc}")
    else:
        if bogus is None:
            rec.add("G", "G 对角反例 ij->i 不得落到 ii->i", "PASS", None,
                    f"正确返回 None（{(time.perf_counter() - started) * 1e3:.0f} ms）")
        else:
            rec.add("G", "G 对角反例 ij->i 不得落到 ii->i", "FAIL", None,
                    f"错误地给出 {bogus.equation!r}——对角与求和被当成等价")

    # 6. 长度 1 的轴（语料里大量 (4,4,1,1)）。
    for tag, sizes in (("4,4,1,1", {"a": 4, "b": 4, "c": 1, "d": 1}),
                       ("4,1,2,3", {"a": 4, "b": 1, "c": 2, "d": 3}),
                       ("1,2,1,3", {"a": 1, "b": 2, "c": 1, "d": 3})):
        variant, tracker = build_variant("abcd->dcba", [("out", (3, 2, 1, 0))])
        check_case(rec, "G", f"G 长度1轴 abcd->dcba ({tag})", variant, tracker,
                   sizes=sizes, seed=seed,
                   probe=probe_for("abcd->abcd"), expect_key=target_of("abcd->abcd"))

    # 7. 语料真实轴长（distill 档压缩）：动量轴的真实长度就是 1。
    for seq in (2585, 4307):
        task = tasklist.by_seq()[seq]
        out_rank = len(parse(task.equation)[1])
        steps = [("axes", 0, tuple(reversed(range(len(parse(task.equation)[0][0]))))),
                 ("out", tuple(range(1, out_rank)) + (0,))]
        variant, tracker = build_variant(task.equation, steps)
        try:
            sizes = _compressed_real_sizes(variant, task.roles)
        except Exception as exc:                          # noqa: BLE001
            rec.add("G", f"G 真实轴长 seq={seq}", "未验证", None,
                    f"语料角色轴长推导失败：{type(exc).__name__}: {exc}")
            continue
        check_case(rec, "G", f"G 真实轴长 seq={seq} 反转+输出轮转", variant,
                   tracker, sizes=sizes, seed=seed)

    # 8. 非法输入：不崩、不抛、不误给映射。
    bad_inputs = ["", "   ", "ij,jk", "i->->i", "i,,->i", None, 123, ["ij"]]
    problems = []
    for bad in bad_inputs:
        try:
            got = equivalence.normalize_equation(bad)
            explained = equivalence.explain(bad)
        except Exception as exc:                          # noqa: BLE001
            problems.append(f"{bad!r} → {type(exc).__name__}: {exc}")
            continue
        if got is not None:
            problems.append(f"{bad!r} → 返回映射 {got.equation!r}")
        if not isinstance(explained, dict):
            problems.append(f"{bad!r} → explain 返回 {type(explained).__name__}")
    record(rec, "G", "G 非法输入不崩", not problems, None,
           "；".join(problems[:2]) if problems
           else f"{len(bad_inputs)} 个非法输入均返回 None 且 explain 正常")

    # 9. explain 与 normalize_equation 必须说同一件事（两个公开入口的一致性）。
    for name, equation in (("G explain 一致 单层", "cadb,menf,ca,em,bd,fn->"),
                           ("G explain 一致 三层", USER16_LAYERED)):
        try:
            mapping = equivalence.normalize_equation(equation)
            report = equivalence.explain(equation)
        except Exception as exc:                          # noqa: BLE001
            rec.add("G", name, "FAIL", None, f"抛 {type(exc).__name__}: {exc}")
            continue
        if mapping is None:
            if report.get("rewritten") is False and report.get("reason"):
                rec.add("G", name, "PASS", None, "两者都表示未改写")
            else:
                rec.add("G", name, "FAIL", None,
                        f"normalize 返回 None，但 explain 说 rewritten="
                        f"{report.get('rewritten')!r}")
            continue
        trouble = []
        if report.get("equation") != mapping.equation:
            trouble.append(f"equation {report.get('equation')!r} != "
                           f"{mapping.equation!r}")
        if report.get("kind") != mapping.kind:
            trouble.append(f"kind {report.get('kind')!r} != {mapping.kind!r}")
        expected_perm = (None if mapping.output_permutation is None
                         else list(mapping.output_permutation))
        if report.get("output_permutation") != expected_perm:
            trouble.append("output_permutation 不一致")
        record(rec, "G", name, not trouble, None,
               "；".join(trouble) if trouble else f"kind={mapping.kind}")


# ────────────────────────────────────────────────────────────────────────────
# T. 有界性：16 操作数 / 秩 8+ 输出段
# ────────────────────────────────────────────────────────────────────────────

BOUNDED_CASES = (("16 操作数 user16", USER16),
                 ("8 操作数 秩8 输出段", "ai,bi,ci,di,ei,fi,gi,hi->abcdefgh"))


def run_boundedness(rec, seed=0):
    timings = []
    for name, equation in BOUNDED_CASES:
        started = time.perf_counter()
        try:
            mapping = equivalence.normalize_equation(equation)
        except Exception as exc:                          # noqa: BLE001
            rec.add("T", f"T 有界性 {name}", "FAIL", None,
                    f"抛 {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - started
        timings.append((name, elapsed))
        if mapping is None:
            note = "未命中，走完整枚举"
        elif mapping.kind == "identity":
            note = "本身已命中（identity，未走枚举）"
        else:
            note = f"kind={mapping.kind}"
        record(rec, "T", f"T 有界性 {name} {elapsed:.2f} s",
               elapsed <= TIME_LIMIT, None,
               note + ("" if elapsed <= TIME_LIMIT else f"；超过 {TIME_LIMIT:.0f} s 上限"))
    return timings


# ────────────────────────────────────────────────────────────────────────────
# 汇总
# ────────────────────────────────────────────────────────────────────────────


def print_summary(rec, drops, fuzz, direction, timings):
    print("\n=== 汇总 ===", flush=True)
    for dimension, label in (("A", "单层 A"), ("B", "两两组合 B"),
                             ("C", "三层 C")):
        passed, total = rec.counts(dimension)
        print(f"{label}: {passed}/{total}", end="   ", flush=True)
    print("", flush=True)
    passed, total = rec.counts("D")
    covered = drops[1] if drops else Counter()
    print(f"反证 D: {passed}/{total}（漏层后均不等价；覆盖 "
          f"order={covered['order']} / axes={covered['axes']} / out={covered['out']}）",
          flush=True)
    if direction is not None:
        print(f"方向 E: 穷举真值后非自逆 {direction['non_self_ok']}/"
              f"{direction['non_self']} 条正向正确且逆置换必错；"
              f"自逆 {direction['self_inverse']} 条标注**方向不可区分**；"
              f"无输出置换 {direction['identity']} 条；"
              f"未验证 {direction['unverified']} 条", flush=True)
    if fuzz is not None:
        failed = len(fuzz["failures"])
        print(f"模糊 F: 命中率 {fuzz['passed']}/{fuzz['total']}"
              f"（失败 {failed} 条{'，明细见上' if failed else ''}；"
              f"另有 {fuzz['unverified']} 条仅结构层通过、数值参考超预算；"
              f"映射落回**原语料形态**的 {fuzz['exact_back']} 条；"
              f"耗时 {fuzz['seconds']:.1f} s）", flush=True)
    passed, total = rec.counts("G")
    print(f"边界 G: {passed}/{total}", flush=True)
    if timings:
        worst = max(timings, key=lambda item: item[1])
        print(f"有界性: 最坏 {worst[1]:.2f} s（{worst[0]}）", flush=True)

    failures = rec.failures()
    unverified = rec.unverified()
    print(f"\n合计: PASS {sum(1 for r in rec.rows if r['status'] == 'PASS')} / "
          f"FAIL {len(failures)} / 未验证 {len(unverified)}"
          f"（共 {len(rec.rows)} 条）", flush=True)
    if failures:
        print("\nFAIL 清单:", flush=True)
        for row in failures:
            print(f"  [{row['dimension']}] {row['name']}: {row['note']}", flush=True)
    if unverified:
        print("\n未验证清单（不计为通过）:", flush=True)
        for row in unverified:
            print(f"  [{row['dimension']}] {row['name']}: {row['note']}", flush=True)
    print("\n结论: " + ("全部通过" if not failures else f"{len(failures)} 条 FAIL"),
          flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="独立穷尽验证 pyqcd.einsum._equivalence 的等价映射层")
    parser.add_argument("--fuzz", type=int, default=40,
                        help="模糊测试抽样条数（默认 40）")
    parser.add_argument("--seed", type=int, default=20261116,
                        help="随机种子（默认 20261116）")
    parser.add_argument("--quick", action="store_true", help="快速档：模糊抽样减半")
    args = parser.parse_args(argv)

    seed = args.seed
    print(f"被测模块文件: {getattr(equivalence, '__file__', '?')}", flush=True)

    rec = Recorder()
    run_single_layer(rec, seed=seed)
    pairs = run_pairs(rec, seed=seed + 100)
    stack = run_stack(rec, seed=seed + 200)
    drop_samples = stack + pairs
    drops = run_drop_layer(rec, drop_samples, seed=seed + 300)
    direction = run_direction(rec, seed=seed + 400)
    fuzz = run_fuzz(rec, count=max(4, args.fuzz // 2 if args.quick else args.fuzz),
                    seed=seed)
    run_edges(rec, seed=seed + 500)
    timings = run_boundedness(rec, seed=seed + 600)

    print_summary(rec, drops, fuzz, direction, timings)
    return 0 if not rec.failures() else 1


if __name__ == "__main__":
    sys.exit(main())
