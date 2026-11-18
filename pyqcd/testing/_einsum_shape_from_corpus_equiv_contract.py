"""契约：省略 ``--shape`` 时，语料匹配必须**把等价改写后的写法也算进去**。

缺口
----
``test.shapes_from_corpus`` 原先只做**字符串精确比较**
（``task.equation != equation``）。于是把语料的某条方程换个轴序——交换两个操作数、
转置某个操作数的轴、换输出轴序——就一律判成 ``error.corpus_miss``，尽管
``_equivalence.normalize_equation`` 明明能把它改写成语料里的那一条
（``cwdx,eufv,mknl,mo,aopb,…->GMHew`` 这类实测方程正是如此）。用户的原话是：
先确认能不能参考 ``wick_tasklist``（**等价转换后的结果也纳入**），没有才报错。

修法（三阶段，次序固定）
------------------------
1. **直接匹配**：逐字同一条（旧行为，一字不改）；
2. **等价转换后匹配**：``mapping = _equivalence.normalize_equation(equation)``，
   再拿 ``mapping.equation`` 查语料；命中后形状按
   ``operand_order`` / ``operand_permutations`` **反向还原**成用户原方程的轴序
   （``output_permutation`` 不参与：输出形状本就按用户方程的 ``->`` 段拼）；
3. 都没有（或**还原不可靠**）→ 仍是 ``error.corpus_miss``，文案不变。

为什么"还原"必须逐轴锁死
------------------------
等价写法的形状是配着**语料那条写法**的轴序算出来的。直接把它喂给用户的方程，
形状往往一样"合法"（同一批标签、同一套尺寸），只有数值悄悄错位——这类错误
数值测试抓不到（``contract`` 与 ``np.einsum`` 用同一批张量，当然一致）。所以下面
第 2 条按**逐轴**比对（不是只比总元素数），第 3 条再要求张量在**该方程下**自洽
（标签尺寸冲突会直接抛错，例如把 ``cdab`` 的形状按 ``cadb`` 的轴序给出来）。

本模块钉五条能各自独立失败的断言
--------------------------------
1. **直接命中**：语料内方程省略 ``--shape`` → 成功，``matched_via == "direct"``，
   CLI 退出码 0；
2. **等价命中**：语料方程的轴序变体 → 成功，``matched_via == "equivalence"``，
   ``target_equation`` 指向语料那条，且形状与"把语料操作数按 mapping 转回变体"
   得到的**逐轴**形状完全相同（同时确认它与未还原的轴序**不一样**，
   证明还原真的发生了，不是照抄）；
3. **数值正确**：该变体按还原后的形状跑 ``contract()``（``as_equivalence=False``，
   不经库内改写）与 ``np.einsum`` 的 ``rel <= 1e-12``；
4. **仍报错**：语料里根本没有的方程 → ``UsageError``（``code == 2``），
   正文与 :func:`pyqcd.einsum.test.translate` 的 ``error.corpus_miss`` **逐字相同**；
5. **不可还原时宁报错**：monkeypatch ``normalize_equation`` 让它返回一份
   **无法反向施加**的 mapping（``operand_order`` 不是排列 / ``perm`` 不是排列 /
   字段长度对不上）→ 必须报 ``corpus_miss``，**不许**给出错位形状。

口径边界
--------
形状口径（``_topology_gen_common.ROLE_SHAPES`` 与
``bench_specialized._label_sizes`` 的取舍）**不在本契约范围内**：本模块只钉
"用哪条 roles + 命中来源"，不改标签 → 尺寸的映射方式。故第 2/3 条用场景
``smoke``（毫秒级）里的语料方程，不与生产性能口径混淆——这里比的是**形状对齐**，
没有任何性能结论。

可直接运行::

    python -B -m pyqcd.testing._einsum_shape_from_corpus_equiv_contract
"""
from __future__ import annotations

import contextlib
import io
import unittest
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import numpy as np

from ..einsum import _equivalence, contract
from ..einsum import test as cli

#: 语料里的一条方程（seq 1，6 个操作数、输出标量）与它的**轴序变体**：
#: ① 交换前两个操作数位置；② ``cadb -> cdab``（``transpose(0,2,1,3)``）；
#: ③ ``bd -> db``（``transpose(1,0)``）。三者合起来覆盖 ``operand_order``、
#: ``operand_permutations`` 两类改写。
BASE = "cadb,menf,ac,em,bd,fn->"
VARIANT = "menf,cdab,ac,em,db,fn->"

#: 形状小、又能让"被转置的两根轴长度不同"（``Ns=4`` vs ``Mom=2``/``Nev=12``），
#: 逐轴比对才有区分度——两根等长轴互换是测不出方向错误的。
SCENE = "smoke"

#: 语料里根本不存在的方程（标签集合与任何语料条目都对不上）。
ABSENT = "qz,qz->q"


def _cli_args(equation: str, *extra: str) -> list[str]:
    """一条最小可跑的 CLI 命令（``--profile false``：不碰任何缓存/目录）。"""
    return ["--equation", equation, "--scene", SCENE,
            "--device", "cpu", "--backend", "numpy", "--dtype", "complex128",
            "--compare", "none", "--rounds", "1", "--warmup", "1",
            "--profile", "false", *extra]


def _main(equation: str, *extra: str) -> tuple[int, str, str]:
    """跑一次真实 CLI；返回 ``(退出码, stdout, stderr)``。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(_cli_args(equation, *extra))
    return code, out.getvalue(), err.getvalue()


def _caller_shapes(variant: str) -> list[tuple[int, ...]]:
    """把语料操作数按 mapping **转回**变体方程的轴序，给出逐轴形状（真值）。

    方向照 ``_api._contract_via_equivalence``：``rewritten[i] =
    transpose(caller[order[i]], perms[i])``，所以反过来是
    ``caller[order[i]] = transpose(rewritten[i], perms[i])``。
    """
    mapping = _equivalence.normalize_equation(variant)
    assert mapping is not None, f"{variant} 应有等价改写"
    base_shapes, matched, _sizes = cli.shapes_from_corpus(BASE, SCENE)
    assert matched, BASE
    order = mapping.operand_order or tuple(range(len(base_shapes)))
    perms = list(mapping.operand_permutations) or [None] * len(base_shapes)
    caller: list[tuple[int, ...] | None] = [None] * len(base_shapes)
    for index, shape in enumerate(base_shapes):
        tensor = np.zeros(shape, dtype=np.complex128)
        caller[order[index]] = (tuple(tensor.shape) if perms[index] is None
                                else tuple(np.transpose(tensor, perms[index]).shape))
    assert all(item is not None for item in caller), caller
    return [shape for shape in caller if shape is not None]


# ── 1. 直接命中 ─────────────────────────────────────────────────────────────
def test_direct_hit_reports_direct():
    """语料内方程省略 ``--shape`` → 成功，来源标成 ``direct``，CLI 退出码 0。"""
    detail: dict[str, Any] = {}
    shapes, matched, sizes = cli.shapes_from_corpus(BASE, SCENE, detail)
    assert matched, f"{BASE} 本就在语料里"
    assert detail["matched_via"] == "direct", detail
    assert detail["target_equation"] == BASE, detail
    assert len(shapes) == 6, shapes
    assert all(isinstance(size, int) for size in sizes.values()), sizes
    code, stdout, stderr = _main(BASE)
    assert code == 0, (code, stdout, stderr)
    assert stdout.count(f"PASS {BASE}") == 1, stdout
    # 直接命中没有"另一条方程"可报，明细行不该出现。
    assert "shape source:" not in stdout, stdout


# ── 2. 等价命中 + 逐轴还原 ───────────────────────────────────────────────────
def test_axis_order_variant_hits_via_equivalence():
    """轴序变体 → ``matched_via == "equivalence"``，形状按用户原方程的轴序给出。"""
    detail: dict[str, Any] = {}
    shapes, matched, sizes = cli.shapes_from_corpus(VARIANT, SCENE, detail)
    assert matched, f"{VARIANT} 是 {BASE} 的轴序变体，应命中"
    assert detail["matched_via"] == "equivalence", detail
    assert detail["target_equation"] == BASE, detail
    # ``sizes`` 的键仍是**用户的字母**（映射只在语料内部做字母双射）。
    assert {"c", "d", "a", "b"} <= set(sizes), sizes
    assert len(shapes) == 6, shapes
    # 逐轴比对（**不是**只比总元素数）：两根被转置的轴长度不同，方向错就一定不等。
    assert [tuple(shape) for shape in shapes] == _caller_shapes(VARIANT), (
        shapes, _caller_shapes(VARIANT))
    # 还原真的发生了：未还原的轴序（语料那条写法的形状）必须与给出的**不同**。
    base_shapes, hit, _ = cli.shapes_from_corpus(BASE, SCENE)
    assert hit and [tuple(shape) for shape in base_shapes] != [tuple(s) for s in shapes], (
        "变体与语料的形状完全相同，这条用例证明不了还原：", base_shapes, shapes)


# ── 3. 数值正确（形状没被错位） ─────────────────────────────────────────────
def test_variant_shapes_contract_matches_numpy():
    """按还原后的形状构造张量，``contract`` 与 ``np.einsum`` 的 ``rel <= 1e-12``。"""
    shapes, matched, _sizes = cli.shapes_from_corpus(VARIANT, SCENE)
    assert matched and shapes, shapes
    rng = np.random.default_rng(20261118)
    operands = [(rng.standard_normal(shape)
                 + 1j * rng.standard_normal(shape)).astype(np.complex128)
                for shape in shapes]
    # ``as_equivalence=False``：这里要验的是"这套形状在这条方程下自洽"，
    # 不借助库内改写替我们纠正轴序。
    actual = contract(VARIANT, *operands, as_equivalence=False)
    expected = np.einsum(VARIANT, *operands)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    scale = float(np.max(np.abs(expected))) or 1.0
    rel = float(np.max(np.abs(actual - expected))) / scale
    assert rel <= 1e-12, f"{VARIANT}: rel={rel:.3e} 超出 1e-12（形状可能错位）"


# ── 4. 语料里没有 → 仍报 corpus_miss（退出码 2） ────────────────────────────
def test_absent_equation_still_errors():
    """完全不在语料的方程 → ``corpus_miss``，退出码 2，文案与旧版逐字相同。"""
    detail: dict[str, Any] = {}
    shapes, matched, sizes = cli.shapes_from_corpus(ABSENT, SCENE, detail)
    assert not matched, (shapes, detail)
    assert shapes == [] and sizes == {}, (shapes, sizes)
    assert detail == {}, detail
    try:
        _main(ABSENT)
    except cli.UsageError as exc:
        assert exc.code == 2, exc.code
        expected = cli.translate("error.corpus_miss", "zh",
                                 equation=ABSENT, scene=SCENE)
        assert exc.message == expected, (exc.message, expected)
    else:
        raise AssertionError(f"{ABSENT} 不在语料，本该报 error.corpus_miss")


# ── 5. 还原不可靠 → 宁报错，不给错位形状 ────────────────────────────────────
def test_unrestorable_mapping_is_a_miss():
    """mapping 无法反向施加时按未命中处理（不许静默给出错位形状）。"""
    real = _equivalence.normalize_equation(VARIANT)
    assert real is not None and real.operand_permutations, real
    broken = {
        # 位置不是整个操作数集合的排列
        "operand_order": replace(real, operand_order=(0, 0, 1, 2, 3, 4)),
        # 轴置换不是排列（长度对、值重复）
        "perm-not-a-permutation": replace(
            real, operand_permutations=((0, 0, 1, 3),) + tuple(
                real.operand_permutations[1:])),
        # 轴置换长度与操作数秩不符
        "perm-rank": replace(
            real, operand_permutations=((0, 1),) + tuple(
                real.operand_permutations[1:])),
        # 字段长度与操作数个数不符
        "perm-count": replace(real, operand_permutations=((0, 2, 1, 3),)),
        # 字段看着合法、却与方程对不上（少了那两个转置）：还原会得到错位形状，
        # 逐轴交叉校验必须拦住它。
        "missing-perm": replace(real, operand_permutations=()),
    }
    for label, mapping in broken.items():
        detail: dict[str, Any] = {}
        with patch.object(_equivalence, "normalize_equation", return_value=mapping):
            shapes, matched, _sizes = cli.shapes_from_corpus(VARIANT, SCENE, detail)
            assert not matched, (label, shapes, detail)
            assert shapes == [], (label, shapes)
            try:
                _main(VARIANT)
            except cli.UsageError as exc:
                assert exc.code == 2, (label, exc.code)
                assert VARIANT in exc.message, (label, exc.message)
            else:
                raise AssertionError(f"{label}: 不可还原的 mapping 本该报 corpus_miss")


TESTS = (
    test_direct_hit_reports_direct,
    test_axis_order_variant_hits_via_equivalence,
    test_variant_shapes_contract_matches_numpy,
    test_absent_equation_still_errors,
    test_unrestorable_mapping_is_a_miss,
)


def test_einsum_shape_from_corpus_equiv_contracts():
    """注册入口：逐条运行 ``TESTS``；全跳过时也 SkipTest（不许静默通过）。"""
    passed = skipped = 0
    failures = []
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}", flush=True)
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    if failures:
        raise AssertionError(
            f"{len(failures)} 条语料等价匹配契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"shape-from-corpus equivalence contracts did not run ({skipped} skipped)")


def main() -> int:
    """直接运行本模块：返回失败条数（0 = 全通过）。"""
    passed = skipped = failed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}")
        except Exception:                                 # noqa: BLE001
            failed += 1
            print(f"FAIL {test.__name__}")
            import traceback

            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS {test.__name__}")
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
