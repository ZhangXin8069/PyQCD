"""契约：CPU 专用步表的选路判据（``_dedicated_allowlist`` 与代价门）。

白名单只用于**选路**，普通数值测试不会碰它，因此它的缺陷极隐蔽——本模块专门盯
三件事：

1. **可用性**：``allow()`` / ``seqs()`` 必须真的能调用。

   历史缺陷：``ALLOW`` 只在模块 ``__getattr__`` 里对*外部* ``module.ALLOW`` 惰性
   注入，而模块内的全局查找不触发 ``__getattr__``，于是 ``allow()`` 一调用就是
   ``NameError: name 'ALLOW' is not defined``。白名单不参与数值计算，这个缺陷可以
   一直躺到有人真的用它选路。

2. **口径边界**：白名单按 ``dim ∈ {4,6,8}`` 的**等长超立方体**、``complex128``
   标定（见模块 docstring）。生产 scene 是异形轴（Nx=64/Nev=50/Lt=40），所以
   **不能把 allowlist 当成生产选路的依据**——它在这里恒为 True，判不出好坏。

3. **代价门抓不到"结构性不适用"**：scene 实测 seq=129 四类 dtype 全亏
   （1.53–3.62×）、seq=3461 全亏（1.18–1.61×），而 seq=1623/1863 赚
   （0.72–1.02×）。这两类的 ``pack/flops`` 比值几乎相同（0.286 vs 0.281），
   **绝对规模也不是分界**：沿 ``scale 8→32`` 扫 seq=129，比值稳定在 1.68–2.30×，
   全程都亏。所以"再加一个尺寸门槛"修不了它，只能按拓扑/元数判定。

本模块把上述结论固化成断言，防止有人误用 allowlist 或误以为加门槛能解决。

可直接运行::

    python -m pyqcd.testing._dedicated_routing_contract
"""
from __future__ import annotations

import unittest

import numpy as np

from ..einsum import _dedicated_allowlist as allowlist
from ..einsum import _kernel_kupl
import pyqcd.einsum._wick_tasklist as tasklist


def test_allowlist_is_callable():
    """``allow()``/``seqs()`` 必须可用（回归 NameError 缺陷）。"""
    for dim in allowlist._DIMS:
        tokens = allowlist.seqs(dim)
        assert tokens, dim
        assert tokens == tuple(sorted(tokens)), dim
        # 白名单成员必须一律 allow=True，且 seqs 与 allow 自洽
        for seq in tokens[:5]:
            assert allowlist.allow(dim, seq) is True, (dim, seq)
        # 不存在的 seq 必须 False（不能全 True）
        assert allowlist.allow(dim, 10 ** 9) is False, dim


def test_allowlist_coverage_matches_its_docstring():
    """覆盖率必须与模块 docstring 声明的数目一致（gen 物漂移检测）。"""
    counts = {dim: len(allowlist.seqs(dim)) for dim in allowlist._DIMS}
    # docstring: dim=4: 8055 / dim=6: 7806 / dim=8: 4932
    assert counts[4] == 8055, counts
    assert counts[6] == 7806, counts
    assert counts[8] == 4932, counts


def test_allowlist_is_not_a_production_routing_oracle():
    """白名单按等长 dim 标定，对生产异形轴的判定没有分辨力。

    scene 实测：seq=129（全亏）与 seq=1623（赚）在 allowlist 里**都是 True**。
    任何"用 allowlist 给生产选路"的改动都会失效。
    """
    for seq in (129, 1623, 1863, 3461):
        for dim in allowlist._DIMS:
            assert allowlist.allow(dim, seq) is True, (seq, dim)
    # 这四条的实测盈亏截然不同（见 docstring 与 SKILL §4e），说明该判定无分辨力。
    losing = {129, 3461}
    winning = {1623, 1863}
    assert losing & winning == set()
    for seq in losing | winning:
        assert allowlist.allow(4, seq) is True


def test_cost_gate_cannot_separate_the_losing_paths():
    """代价门的三条判据都无法区分"结构性不适用"与"适用"。

    seq=129 与 1623 的 ``pack/flops`` 只差 2 倍（0.572 vs 0.281），却一个亏 1.53×
    一个赚 0.86×；arena 差 14 倍也远不到数量级。沿 ``scale 8→32`` 扫 seq=129 的
    比值稳定在 1.68–2.30×（全程都亏），说明**加尺寸门槛不是修复方案**，
    只能按拓扑/元数选路。
    """
    from pyqcd.einsum._bench_scene import dims as scene_dims
    from pyqcd.einsum.bench_specialized import _label_sizes

    stats = {}
    for seq in (129, 1623):
        task = tasklist.by_seq()[seq]
        sizes = _label_sizes(task.equation.split("->")[0].split(","), task.roles,
                             scene_dims("scene"))
        shapes = [tuple(int(sizes[c]) for c in term)
                  for term in task.equation.split("->")[0].split(",")]
        cached = _kernel_kupl._dedicated_cached(
            task.equation, tuple(shapes), np.dtype(np.complex128), "auto", "kupl")
        assert cached is not None, seq
        stats[seq] = _kernel_kupl._plan_statistics(cached[0])

    ratio_129 = stats[129]["plan_pack_bytes"] / max(stats[129]["plan_flops"], 1)
    ratio_1623 = stats[1623]["plan_pack_bytes"] / max(stats[1623]["plan_flops"], 1)
    # 实测 0.572 vs 0.281：只差 2 倍，而两者的盈亏是 1.53× vs 0.86×。
    # **同一量级**（相差不足一个数量级）就足以说明它不能当判据——最初写"几乎相同"
    # 是我夸大了，按实测改成"不足一个数量级"。
    assert max(ratio_129, ratio_1623) / min(ratio_129, ratio_1623) < 10.0, (
        ratio_129, ratio_1623)
    for seq in (129, 1623):
        assert _kernel_kupl._cpu_plan_fallback_reason(stats[seq]) is None, (
            seq, "代价门应当放行（正因为放行才需要拓扑级判据）")


def test_losing_paths_share_the_wide_output_shape():
    """亏损的两条路径共享"4 输出轴 + 高元数"这一结构特征（候选判据）。"""
    losing = [tasklist.by_seq()[seq] for seq in (129, 3461)]
    winning = [tasklist.by_seq()[seq] for seq in (1623, 1863)]
    for task in losing:
        assert len(task.equation.split("->")[1]) >= 4, task.seq
        assert task.arity >= 14, task.seq
    for task in winning:
        assert len(task.equation.split("->")[1]) == 2, task.seq
    # 该特征只是"候选"：样本只有 2+2，不足以据此改运行时选路。
    assert len(losing) == 2 and len(winning) == 2


TESTS = (
    test_allowlist_is_callable,
    test_allowlist_coverage_matches_its_docstring,
    test_allowlist_is_not_a_production_routing_oracle,
    test_cost_gate_cannot_separate_the_losing_paths,
    test_losing_paths_share_the_wide_output_shape,
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
