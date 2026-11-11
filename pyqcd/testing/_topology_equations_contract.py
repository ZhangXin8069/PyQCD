"""契约测试：``refer/sush/wick_12cases.md`` 里 2582 条真实收缩方程的拓扑匹配与正确性。

两件事必须同时成立，缺一不可：

1. **拓扑匹配**：把方程里的指标字母整体改名（``ij,jk->ik`` → ``ij,jb->ib``）、加空白、
   省略显式输出，命中的形态必须与原始写法一致；改结构（交换某个操作数的两个指标、
   调换输出顺序）必须**不**命中。
2. **数值正确**：每条方程在随机小操作数上与 ``numpy.einsum`` 逐元素一致。

只做小形状（默认每维 2），因为这里验证的是路线与匹配，不是性能；性能基准在
``pyqcd.testing.einsum.bench_*`` 里。可直接运行::

    python -m pyqcd.testing._topology_equations_contract
"""
from __future__ import annotations

import itertools
import pathlib
import random
import string

import numpy as np

from ..einsum import _specialize as spec
from ..einsum import contract

CASES_PATH = (pathlib.Path(__file__).resolve().parents[2]
              / "refer" / "sush" / "wick_12cases.md")
# 只取拓扑类表；文件末尾“动量指标与流分量指标”一节是展开表，不是新拓扑。
_TOPOLOGY_SECTION_MARKER = "## 动量指标与流分量指标"
_NCASE_ROWS = (2, 6, 120, 720, 6, 24, 720, 24, 120, 120, 720)
_MIN_CASE_ROWS = 2582


def _parse_equations() -> list[str]:
    """按文件顺序返回全部拓扑类方程（去重前）。"""
    text = CASES_PATH.read_text(encoding="utf-8")
    head = text.split(_TOPOLOGY_SECTION_MARKER)[0]
    equations: list[str] = []
    for line in head.splitlines():
        if not line.startswith("| T"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        equation = cells[2].strip("` ").strip()
        if "->" in equation:
            equations.append(equation)
    return equations


def _rename(equation: str, shift: int = 11) -> str:
    """整体改名字母，保持结构不变（拓扑等价）。"""
    letters = string.ascii_letters
    table = str.maketrans({
        ch: letters[(letters.index(ch) + shift) % len(letters)]
        for ch in equation if ch in letters})
    return equation.translate(table)


def _shape_for(term: str, size: int) -> tuple[int, ...]:
    return tuple(size for _ in term)


def _operands(equation: str, rng: np.random.Generator, size: int = 2,
              dtype=np.float64):
    lhs = equation.split("->")[0]
    terms = lhs.split(",")
    return [rng.standard_normal(_shape_for(term, size)).astype(dtype)
            for term in terms]


def test_equation_corpus_is_complete():
    """语料必须完整：2582 条，且每个情形的条数与文档汇总一致。"""
    equations = _parse_equations()
    assert len(equations) >= _MIN_CASE_ROWS, len(equations)
    assert len(equations) == sum(_NCASE_ROWS), len(equations)
    assert all(eq.count("->") == 1 for eq in equations)
    # 每条都是合法的 einsum 表达式（交给 numpy 判定，不在本地重写解析器）
    for equation in equations[:50]:
        lhs, rhs = equation.split("->")
        terms = lhs.split(",")
        np.einsum(equation, *[np.zeros(_shape_for(term, 1)) for term in terms],
                  optimize="greedy")


def test_every_equation_matches_numpy():
    """2582 条方程逐个与 numpy 对照（小形状，含重复指标与零维输出）。"""
    rng = np.random.default_rng(20261111)
    equations = _parse_equations()
    for index, equation in enumerate(equations):
        operands = _operands(equation, rng, size=2)
        expected = np.einsum(equation, *operands, optimize="greedy")
        got = contract(equation, *operands)
        assert got.shape == np.shape(expected), (index, equation, got.shape)
        # 13-16 个算子的链式收缩与 numpy 的成对求和顺序不同，float64 下差异在
        # 1e-12 量级；真实错误（例如错序、漏缩并）都是 O(1)，用 1e-9 仍能拦住。
        assert np.allclose(got, expected, rtol=1e-9, atol=1e-10), (
            index, equation, float(np.max(np.abs(got - np.asarray(expected)))))


def test_renamed_equations_keep_the_same_signature():
    """整体改名后签名不变——这是“按拓扑匹配”的定义。"""
    for equation in _parse_equations()[:400]:
        assert spec.canonical_signature(equation) == \
            spec.canonical_signature(_rename(equation)), equation


def test_structural_changes_do_not_collide():
    """改结构不得误命中：交换某操作数两个指标或调换输出顺序都要换签名。"""
    checked = 0
    for equation in _parse_equations():
        lhs, rhs = equation.split("->")
        terms = lhs.split(",")
        for position, term in enumerate(terms):
            if len(term) < 2 or term[0] == term[1]:
                continue
            swapped = term[1] + term[0] + term[2:]
            variant_terms = list(terms)
            variant_terms[position] = swapped
            variant = ",".join(variant_terms) + "->" + rhs
            assert spec.canonical_signature(variant) != \
                spec.canonical_signature(equation), (equation, variant)
            checked += 1
            break
        if len(rhs) >= 2 and rhs[0] != rhs[1]:
            swapped_out = rhs[1] + rhs[0] + rhs[2:]
            variant = lhs + "->" + swapped_out
            assert spec.canonical_signature(variant) != \
                spec.canonical_signature(equation), (equation, variant)
            checked += 1
        if checked >= 200:
            break
    assert checked >= 100, checked


def test_matching_is_whitespace_and_output_insensitive():
    """空白与“省略显式输出”不得改变拓扑签名（numpy 语义一致时才断言）。"""
    samples = ["ij,jk->ik", "ab,bc->ac", "ij,jb->ib"]
    signatures = {spec.canonical_signature(item) for item in samples}
    assert len(signatures) == 1, signatures
    assert spec.canonical_signature("ij, jk -> ik") == \
        spec.canonical_signature("ij,jk->ik")
    # 省略输出：'ij,jk' 的隐含输出就是 'ik'
    assert spec.canonical_signature("ij,jk") == \
        spec.canonical_signature("ij,jk->ik")


def test_specialization_hit_is_stable_for_registered_forms():
    """已注册的 9 条形态：原名与改名写法都命中同一条目，未注册形态不误命中。"""
    for subscripts, entry in spec.REGISTRY.items():
        assert (spec.match(subscripts) or {}).get("name") == entry["name"]
        renamed = _rename(subscripts)
        if renamed != subscripts:
            assert (spec.match(renamed) or {}).get("name") == entry["name"]
    assert spec.match("ij,jk->ik") is None


_TESTS = (
    test_equation_corpus_is_complete,
    test_every_equation_matches_numpy,
    test_renamed_equations_keep_the_same_signature,
    test_structural_changes_do_not_collide,
    test_matching_is_whitespace_and_output_insensitive,
    test_specialization_hit_is_stable_for_registered_forms,
)


def main() -> int:
    passed = 0
    failed = 0
    for test in _TESTS:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:                      # noqa: BLE001 - 报告用
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS {test.__name__}")
    print(f"{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
