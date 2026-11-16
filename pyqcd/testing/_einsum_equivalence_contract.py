"""等价映射接口层的契约：`pyqcd.einsum._equivalence` + verify_specialized 的集成。

本层是**纯接口层**：只把方程改写成等价形式并给出张量调整建议，不改任何执行路径。
契约因此只断言三件事：
1. 三族变换的**数学等价性**（用 numpy 真值对拍，不是断言实现细节）；
2. 报告的**保守性**（非法/无解输入不崩、不猜、已命中的不改写）；
3. `verify_specialized` 的输出里这些字段真的出现。
"""

from __future__ import annotations

import io
import contextlib
import json

import numpy as np

import pyqcd.einsum as pe
from pyqcd.einsum import _equivalence, _specialize, verify_specialized


# ------------------------------------------------------------------ 1. 等价性

def test_axis_permutation_mapping_is_numerically_equivalent():
    """单操作数轴序置换：改方程 + 同步 transpose 张量 → 结果逐元素不变。

    这是用户实际报的场景：``cadb,menf,ac,em,bd,fn->``（语料命中）与
    ``acdb,...``（MISS）只差第 0 个操作数里 ``c``/``a`` 两轴的位置。
    """
    target = "acdb,menf,ac,em,bd,fn->"
    result = _equivalence.normalize_equation(target)
    assert result is not None, "接口层没找到等价写法"
    assert result.equation == "cadb,menf,ac,em,bd,fn->", result.equation
    assert result.operand_permutations[0] == (1, 0, 2, 3), result.operand_permutations

    # 用真实语料形状对拍
    from pyqcd.einsum.bench_specialized import _label_sizes
    from pyqcd.einsum._bench_scene import dims
    from pyqcd.einsum import _wick_tasklist as tasklist
    from pyqcd.einsum._dedicated_kernels import parse_equation

    terms, _ = parse_equation(target)
    table = dims("distill")
    shapes = None
    for seq in range(1, 40):
        task = tasklist.lookup(seq)
        if task is None:
            continue
        corpus_terms, _ = parse_equation(task.equation)
        sizes = _label_sizes(corpus_terms, task.roles, table)
        if all(ch in sizes for term in terms for ch in term):
            shapes = [tuple(sizes[ch] for ch in term) for term in terms]
            break
    assert shapes is not None, "找不到覆盖该方程的语料 roles"

    rng = np.random.default_rng(20261116)
    ops = [rng.standard_normal(s) + 1j * rng.standard_normal(s) for s in shapes]
    direct = pe.contract(target, *ops, backend="numpy", optimize="greedy")

    mapped = list(ops)
    mapped[0] = np.transpose(mapped[0], result.operand_permutations[0])
    via = pe.contract(result.equation, *mapped, backend="numpy", optimize="greedy")

    reference = np.einsum(target, *ops, optimize="greedy")
    scale = max(float(np.max(np.abs(reference))), 1e-300)
    assert float(np.max(np.abs(via - direct))) <= 1e-12 * scale, "映射后数值不一致"
    assert float(np.max(np.abs(direct - reference))) <= 1e-12 * scale, "直接算就不是真值"


def test_operand_order_mapping_is_numerically_equivalent():
    """操作数位置交换：改方程 + 同序重排张量 → 结果逐元素不变。"""
    target = "menf,cadb,ac,em,bd,fn->"
    result = _equivalence.normalize_equation(target)
    assert result is not None, "接口层没找到等价写法"
    assert result.equation == "cadb,menf,ac,em,bd,fn->", result.equation
    assert result.operand_order == (1, 0, 2, 3, 4, 5), result.operand_order

    rng = np.random.default_rng(7)
    sizes = {"c": 3, "a": 3, "d": 3, "b": 3, "m": 3, "e": 3, "n": 3, "f": 3}
    lhs = target.split("->")[0].split(",")
    ops = [rng.standard_normal(tuple(sizes[c] for c in t)) for t in lhs]
    direct = pe.contract(target, *ops, backend="numpy", optimize="greedy")

    reordered = [ops[i] for i in result.operand_order]
    via = pe.contract(result.equation, *reordered, backend="numpy", optimize="greedy")
    scale = max(abs(direct), 1e-300)
    assert float(np.max(np.abs(via - direct))) <= 1e-12 * scale, "映射后数值不一致"


def test_mapping_really_reaches_the_specialized_path():
    """映射的**目的**：等价写法能命中语料，直接写法不能。

    只断言 `_specialize.match` 的命中差异——这是本层存在的理由。
    """
    target = "acdb,menf,ac,em,bd,fn->"
    assert _specialize.match(target) is None, "该写法本不应命中"
    result = _equivalence.normalize_equation(target)
    assert result is not None
    assert _specialize.match(result.equation) is not None, "等价写法应当命中"


# ------------------------------------------------------------------ 2. 保守性

def test_invalid_or_unmappable_inputs_do_not_raise():
    """非法 / 无法映射的输入一律返回 None 或空列表，绝不抛异常。"""
    for bad in ["", "->", "abc", "a,->", ",", "a->b", None, 123,
                "a,b->c", "ij,jk", "ij->ijk", "a,,b->"]:
        _equivalence.normalize_equation(bad)          # 不抛即通过
        _equivalence.find_equivalents(bad, limit=3)


def test_already_matching_equation_is_not_rewritten():
    """本身已命中时必须返回 identity，不产生任何张量调整。"""
    hit = "cadb,menf,ac,em,bd,fn->"
    result = _equivalence.normalize_equation(hit)
    assert result is not None and result.kind == "identity", result
    assert not result.operand_permutations, result.operand_permutations
    assert result.operand_order is None, result.operand_order


def test_unmappable_equation_returns_none():
    """没有任何等价写法能命中时返回 None（不猜）。"""
    assert _equivalence.normalize_equation("qz,wx->qz") is None


def test_high_rank_enumeration_stays_bounded():
    """高秩方程不得因阶乘枚举而爆炸。"""
    import time
    start = time.perf_counter()
    candidates = _equivalence.find_equivalents("abcdefgh,ijklmnop->", limit=5)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"枚举耗时 {elapsed:.2f}s 过久"
    assert len(candidates) <= 5, len(candidates)


# ------------------------------------------------------------------ 3. 集成

def test_verify_specialized_reports_the_equivalence():
    """`verify_specialized` 的 MISS 报告里必须带出等价映射与张量调整。"""
    report = verify_specialized.query("acdb,menf,ac,em,bd,fn->")
    assert not report["matched"], "该写法本不应命中"
    equivalent = report.get("equivalent")
    assert equivalent is not None, report
    assert equivalent["rewritten"] is True, equivalent
    assert equivalent["equation"] == "cadb,menf,ac,em,bd,fn->", equivalent
    assert equivalent["kind"] == "operand-axes", equivalent
    assert equivalent["operand_permutations"][0] == [1, 0, 2, 3], equivalent

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        verify_specialized.main(["acdb,menf,ac,em,bd,fn->"])
    text = buffer.getvalue()
    for token in ("等价写法", "cadb,menf,ac,em,bd,fn->", "transpose(1, 0, 2, 3)"):
        assert token in text, f"输出缺少 {token!r}：{text}"


def test_verify_specialized_json_carries_the_equivalence():
    report = verify_specialized.query("acdb,menf,ac,em,bd,fn->")
    payload = json.dumps(report, ensure_ascii=False)
    assert "cadb,menf,ac,em,bd,fn->" in payload
    assert report["equivalent"]["operand_permutations"][0] == [1, 0, 2, 3]


TESTS = (
    test_axis_permutation_mapping_is_numerically_equivalent,
    test_operand_order_mapping_is_numerically_equivalent,
    test_mapping_really_reaches_the_specialized_path,
    test_invalid_or_unmappable_inputs_do_not_raise,
    test_already_matching_equation_is_not_rewritten,
    test_unmappable_equation_returns_none,
    test_high_rank_enumeration_stays_bounded,
    test_verify_specialized_reports_the_equivalence,
    test_verify_specialized_json_carries_the_equivalence,
)


def main() -> int:
    import unittest

    passed = skipped = failed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}", flush=True)
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}", flush=True)
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    print(f"{passed} passed, {skipped} skipped, {failed} failed", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
