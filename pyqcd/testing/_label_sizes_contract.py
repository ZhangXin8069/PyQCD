"""契约：``bench_specialized._label_sizes`` 的「角色 → 物理尺寸」映射。

这是本库**唯一**被授权用来构造生产尺寸的映射（``_topology_kernel_gen``、
``bench_topology_vs_dedicated``、``bench_dtype_matrix`` 等都消费它），错绑的后果
不是某条图算错，而是整条「生产口径」的结论失真。

语料里顶点张量的指标段有**两种形态**，此前被同一个 ``zip`` 处理，附录形态因此整体
错位一格：

==============================  ========================  ===========================
形态                             指标段                    物理形状
==============================  ========================  ===========================
基础表（seq 1..2582）            纯小写：``bd`` / ``fnp``  ``(Lt, Mom, Nev, ...)``
附录（动量指标段，seq ≥ 2583）   ``<大写动量><小写>...``   ``(Mom, Nev, ...)``
                                如 ``Nbd`` / ``Nfnp``
==============================  ========================  ===========================

依据：``refer/sush/wick_12cases.md:2784``（大写 ``M``/``N`` 是 ``Vindex`` 引入的
外部动量标签）与附录样例 ``:2820`` ``cadb,menf,ac,em,Mbd,Nfn->MN``；
``refer/sush/_distillation/corr_wick.py`` 的 ``creat_sink_source`` 把 ``Vindex[i]``
直接前缀到 VVV/VDV 的指标串（``:222-226``），gamma 同位置的 ``Gindex`` 前缀是
流 gamma 的 Lorentz 分量轴（``:509``），不是动量。

本模块钉死三件事：

1. **基础表逐位不变**：期望值是修复前实测快照，硬编码而不从实现反推；
2. **附录形态**：打头大写 → ``Mom``，其余收缩轴 → ``Nev``；
3. **形态不明即 fail closed**：既非纯小写、也非大写打头的指标段不许猜。

可直接运行::

    python -m pyqcd.testing._label_sizes_contract
"""
from __future__ import annotations

import unittest

from ..einsum import _kernel_kupl as kupl
from ..einsum import bench_specialized
from ..einsum import _wick_tasklist as tasklist
from ..einsum._bench_scene import dims as scene_dims
from ._topology_kernel_gen import plan_steps

#: 顶点张量角色前缀（``VVV`` 三腿、``VDV`` 两腿）。
VERTEX_ROLES = ("VVV", "VDV")
#: 基础表代表 seq（混合元数 6/7/13）。
BASE_SEQS = (1, 3, 9)
#: 附录代表 seq（13 元数五夸克 2pt、10 元数重子、14 元数六传播子）。
APPENDIX_SEQS = (2591, 3437, 2711)

#: 基础表逐操作数的硬编码期望形状（**修复前**的实测值，逐位不许漂移）。
BASE_EXPECTED = {
    1: {
        "scene": [(4, 4, 5, 40), (4, 4, 5, 40), (4, 4), (4, 4), (40, 5), (40, 5)],
        "distill": [(4, 4, 1, 1), (4, 4, 1, 1), (4, 4), (4, 4), (1, 1), (1, 1)],
        "distill_curr": [(4, 4, 9, 1), (4, 4, 9, 1), (4, 4), (4, 4), (1, 9), (1, 9)],
    },
    3: {
        "scene": [(50, 4, 40, 40), (4, 4, 5, 5), (4, 50, 50, 50), (4, 4), (4, 4),
                  (40, 5, 50), (40, 5, 50)],
        "distill": [(100, 4, 1, 1), (4, 4, 1, 1), (4, 100, 100, 100), (4, 4), (4, 4),
                    (1, 1, 100), (1, 1, 100)],
        "distill_curr": [(400, 4, 1, 1), (4, 4, 9, 9), (4, 400, 400, 400), (4, 4),
                         (4, 4), (1, 9, 400), (1, 9, 400)],
    },
    9: {
        "scene": [(4, 4, 5, 40), (50, 4, 40, 40), (4, 4, 5, 5), (4, 50, 50, 50),
                  (4, 4, 5, 40), (4, 4), (4, 4), (4, 4), (4, 4), (40, 5),
                  (40, 5, 50), (40, 5, 50), (40, 5)],
        "distill": [(4, 4, 1, 1), (100, 4, 1, 1), (4, 4, 1, 1), (4, 100, 100, 100),
                    (4, 4, 1, 1), (4, 4), (4, 4), (4, 4), (4, 4), (1, 1),
                    (1, 1, 100), (1, 1, 100), (1, 1)],
        "distill_curr": [(4, 4, 9, 1), (400, 4, 1, 1), (4, 4, 9, 9), (4, 400, 400, 400),
                         (4, 4, 9, 1), (4, 4), (4, 4), (4, 4), (4, 4), (1, 9),
                         (1, 9, 400), (1, 9, 400), (1, 9)],
    },
}

#: ``seq=1`` 在 ``distill_curr`` 档的完整绑定（逐 label，修复前快照）。
BASE_LABELS_EXPECTED = {
    1: {"distill_curr": {"a": 4, "b": 1, "c": 4, "d": 9, "e": 4,
                         "f": 1, "m": 4, "n": 9}},
}


def _bind(equation: str, roles, table):
    """(terms, sizes, 逐操作数形状)。"""
    terms = equation.split("->")[0].split(",")
    sizes = bench_specialized._label_sizes(terms, roles, table)
    return terms, sizes, [tuple(sizes[label] for label in term) for term in terms]


def _batch_profile(equation: str, sizes) -> list[tuple[str, int, int]]:
    """逐步 ``(式子, count_batch, batch_rank)``，与 KUPL 执行器同一口径。

    ``count_batch`` 是各 batch 轴长之积（``_kernel_kupl._dedicated_step`` 的
    ``"batch"``，等于 ``count_batch``）；``batch_rank`` 是 batch 轴个数。
    """
    steps, _terms, _out = plan_steps(equation, sizes)
    profile = []
    for step in steps:
        spec = kupl._dedicated_step(
            step["left"], step["right"], step["out"],
            tuple(sizes[label] for label in step["left"]),
            tuple(sizes[label] for label in step["right"]))
        assert spec is not None, step
        profile.append((f'{step["left"]},{step["right"]}->{step["out"]}',
                        spec["batch"], spec["batch_rank"]))
    return profile


def test_base_table_bindings_are_unchanged():
    """基础表代表 seq 的绑定必须与修复前**逐轴相同**（硬编码期望值）。

    基础表里 ``VDV``/``VVV`` 首轴是 ``Lt``、次轴是 ``Mom``，所以 ``distill`` 档
    （``Lt=1``、``Mom_src=1``）的 ``bd`` 是 ``(1, 1)``、``fnp`` 是 ``(1, 1, 100)``。
    这条断言与附录形态无关：修附录映射不得动基础表一个数。
    """
    for seq, scenes in BASE_EXPECTED.items():
        task = tasklist.by_seq()[seq]
        for scene, expected in scenes.items():
            table = scene_dims(scene)
            terms, _sizes, shapes = _bind(task.equation, task.roles, table)
            assert shapes == [tuple(shape) for shape in expected], (
                seq, scene, shapes, expected)
            assert len(terms) == len(expected), (seq, scene)
    for seq, scenes in BASE_LABELS_EXPECTED.items():
        task = tasklist.by_seq()[seq]
        for scene, expected in scenes.items():
            _terms, sizes, _shapes = _bind(task.equation, task.roles,
                                           scene_dims(scene))
            assert sizes == expected, (seq, scene, sizes, expected)


def test_appendix_momentum_label_binds_to_mom():
    """附录形态：打头大写标签 → ``Mom``，其余收缩轴 → ``Nev``。

    修复前 ``Nbd`` 被算成 ``(Lt, Mom, Nev)``（``distill`` 档即 ``(1, 9, 400)``），
    物理上是 ``(Mom, Nev, Nev)``。
    """
    for seq in APPENDIX_SEQS:
        task = tasklist.by_seq()[seq]
        for scene in ("distill", "distill_curr"):
            table = scene_dims(scene)
            terms, sizes, shapes = _bind(task.equation, task.roles, table)
            found = 0
            for term, role, shape in zip(terms, task.roles, shapes):
                if not role.startswith(VERTEX_ROLES) or not term[:1].isupper():
                    continue
                found += 1
                expected = (table["Mom"],) + (table["Nev"],) * (len(term) - 1)
                assert shape == expected, (seq, scene, term, shape, expected)
                assert sizes[term[0]] == table["Mom"], (seq, scene, term, sizes)
                assert all(sizes[label] == table["Nev"] for label in term[1:]), (
                    seq, scene, term, sizes)
            assert found >= 2, (seq, scene, found)

    # 报告里引用的定点核对：seq=2591 @ distill_curr（Mom=9、Nev=400）。
    task = tasklist.by_seq()[2591]
    terms, sizes, shapes = _bind(task.equation, task.roles,
                                 scene_dims("distill_curr"))
    shape_of = dict(zip(terms, shapes))
    assert shape_of["Nbd"] == (9, 400, 400), shape_of["Nbd"]
    assert shape_of["Nfnp"] == (9, 400, 400, 400), shape_of["Nfnp"]
    assert shape_of["Mhjl"] == (9, 400, 400, 400), shape_of["Mhjl"]
    assert shape_of["Mrt"] == (9, 400, 400), shape_of["Mrt"]
    # 大写标签是动量轴、紧随的小写才是 Nev 轴。
    assert sizes["N"] == 9 and sizes["M"] == 9, sizes
    assert sizes["b"] == 400 and sizes["d"] == 400, sizes


def test_appendix_forms_expose_batch_capable_steps():
    """改正后附录形态确实出现 ``count_batch > 1`` 的步，重数恰为 ``Mom`` 的幂。

    修复前动量标签被绑成 ``Lt=1``，于是 ``distill`` 与 ``distill_curr`` 两档的
    ``count_batch`` 恒为 1——``pyqcd/einsum/AGENTS.md`` 里「distill 口径没有
    ``count_batch > 1`` 的步」正是这个错绑造成的**口径产物**，不是生产事实。

    每个 batch 轴都是动量标签，所以 ``count_batch == Mom ** batch_rank`` 必须逐步
    成立；``Mom_src=1`` 的 ``distill`` 档自然退回 1（数量随 ``Mom`` 而非实现）。
    """
    table = scene_dims("distill_curr")
    for seq, rank_expected in ((2591, 1), (3437, 1), (2711, 2)):
        task = tasklist.by_seq()[seq]
        _terms, sizes, _shapes = _bind(task.equation, task.roles, table)
        profile = _batch_profile(task.equation, sizes)
        for text, count, rank in profile:
            assert count == table["Mom"] ** rank, (seq, text, count, rank)
        widest = max(profile, key=lambda row: row[1])
        assert widest[2] == rank_expected, (seq, widest)
        assert widest[1] > 1, (seq, widest)
        assert widest[1] == table["Mom"] ** rank_expected, (seq, widest)

    # seq=2591 的双现值：``distill_curr`` 的 Mom=9 → 9；``distill`` 的 Mom=1 → 1。
    task = tasklist.by_seq()[2591]
    observed = {}
    for scene in ("distill", "distill_curr"):
        table = scene_dims(scene)
        _terms, sizes, _shapes = _bind(task.equation, task.roles, table)
        observed[scene] = max(count for _text, count, _rank
                              in _batch_profile(task.equation, sizes))
    assert observed == {"distill": 1, "distill_curr": 9}, observed


def test_base_table_has_no_batch_step():
    """基础表的每一步 ``batch_rank`` 都是 0，因此 ``count_batch`` 恒为 1。

    这与上一条配对：``count_batch > 1`` 只能出现在附录形态（动量轴进 batch 组），
    基础表没有可比照的动量轴——两档结论的差异来自语料形态，不是实现分支。
    """
    for seq in BASE_SEQS + (129, 855, 1599, 2582):
        task = tasklist.by_seq()[seq]
        for scene in ("scene", "distill", "distill_curr"):
            _terms, sizes, _shapes = _bind(task.equation, task.roles,
                                           scene_dims(scene))
            profile = _batch_profile(task.equation, sizes)
            ranks = {rank for _text, _count, rank in profile}
            assert ranks == {0}, (seq, scene, ranks)
            assert all(count == 1 for _text, count, _rank in profile), (seq, scene)


def test_vertex_form_is_fail_closed():
    """形态不明的指标段必须报错，而不是猜一个绑法。

    语料形态若变化（大写跑进非首轴、腿数与指标数不符），错的绑法会静默污染收缩序，
    比抛异常危险得多。
    """
    table = scene_dims("distill_curr")
    # 大写不在首轴：既非纯小写、也不是动量打头。
    for term in ("bNd", "bndM", "N"):
        for role in ("VDV_0", "VVV_0"):
            try:
                bench_specialized._vertex_axes(term, role, table)
            except ValueError:
                continue
            raise AssertionError(f"{role} {term!r} should be rejected")
    # 腿数不符：VDV 带动量必须是 3 轴、VVV 带动量必须是 4 轴。
    for term, role in (("Nbd", "VVV_0"), ("Nfnp", "VDV_0"), ("bd", "VVV_0"),
                       ("fnp", "VDV_0")):
        try:
            bench_specialized._vertex_axes(term, role, table)
        except ValueError:
            continue
        raise AssertionError(f"{role} {term!r} should be rejected")
    # 未知角色照旧报错（回归保护）。
    try:
        bench_specialized._label_sizes(["ab"], ["mystery_0"], table)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown role should raise")


def test_whole_corpus_binds_without_guessing():
    """全语料 × 全场景都必须能绑定（形态漂移的第一道闸）。

    语料 8638 条里约 2/3 是附录形态（带动量指标段）。任何一条落到"既非纯小写也
    非大写打头"都会在这里暴露，而不是等到某个 bench 出数时才炸。
    """
    prefixed = plain = 0
    for task in tasklist.TASKS:
        terms = task.equation.split("->")[0].split(",")
        for scene in ("scene", "distill", "distill_curr"):
            table = scene_dims(scene)
            sizes = bench_specialized._label_sizes(terms, task.roles, table)
            assert sizes, (task.seq, scene)
            for term, role in zip(terms, task.roles):
                if not role.startswith(VERTEX_ROLES):
                    continue
                if term[:1].isupper():
                    prefixed += 1
                    assert sizes[term[0]] == table["Mom"], (task.seq, scene, term)
                else:
                    plain += 1
    # 两种形态都必须真实存在，否则这条"全语料"断言是空的。
    assert prefixed >= 20000, prefixed
    assert plain >= 5000, plain


TESTS = (
    test_base_table_bindings_are_unchanged,
    test_appendix_momentum_label_binds_to_mom,
    test_appendix_forms_expose_batch_capable_steps,
    test_base_table_has_no_batch_step,
    test_vertex_form_is_fail_closed,
    test_whole_corpus_binds_without_guessing,
)


def main() -> int:
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
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    print(f"{passed} passed, {skipped} skipped, {failed} failed", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
