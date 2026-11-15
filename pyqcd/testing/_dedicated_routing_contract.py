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


def test_cpu_plan_gate_rejects_plans_with_excess_pack_traffic_per_flop():
    """``pack/flops`` 比值判据：病态计划被拒、GEMM 主导的计划放行。

    实测（2026-11-15，本机 CPU complex128）：seq=856 的 dedicated 计划
    ``pack_bytes=8,281,198,592`` / ``flops=2,096,413,216`` = **3.95 B/FLOP**，
    只有 step 0 就把 ``ekfl``(3.9 GiB) 整块重排成扁平 GEMM 布局。绝对判据
    （``pack >= 1 GiB`` 且 ``flops <= 1e9``）对它是**放行**的，于是
    ``try_dedicated`` 实测 8662.7 ms，而 `specialize=False` 的通用路径 2330 ms、
    改后实际回落的 `kupl_topology` 2675 ms（3.24×）。
    比值判据要挡的正是这种"活干得也多、但搬得更多"的计划。

    对照组是真正的 GEMM 主导计划（``ij,jk->ik`` 4000x4000）：比值 0.012，
    必须仍然放行——否则新判据会把正常收缩也打回通用路径。
    """
    equation = "ambn,cgdh,ekfl,oipj,ce,mo,gi,bdf,np,hjl->ak"
    shapes = ((400, 4, 1, 1), (4, 4, 9, 1), (4, 400, 400, 400), (4, 4, 9, 9),
              (4, 4), (4, 4), (4, 4), (1, 9, 400), (1, 9), (1, 9, 400))
    cached = _kernel_kupl._dedicated_cached(
        equation, shapes, np.dtype(np.complex128), "auto", "kupl")
    assert cached is not None, equation
    stats = _kernel_kupl._plan_statistics(cached[0])
    ratio = stats["plan_pack_bytes"] / stats["plan_flops"]

    assert ratio >= _kernel_kupl._HEAVY_PACK_RATIO_LIMIT, stats
    # 绝对判据单独看会放行它：这条用例之所以存在，就是因为旧判据漏掉了它。
    absolute_only = (
        stats["plan_arena_bytes"] >= _kernel_kupl._CPU_PLAN_ARENA_LIMIT
        or stats["plan_max_step_macs"] >= _kernel_kupl._CPU_PLAN_MAX_STEP_MACS_LIMIT
        or (stats["plan_pack_bytes"] >= _kernel_kupl._HEAVY_PACK_MIN_BYTES
            and stats["plan_flops"] <= _kernel_kupl._HEAVY_PACK_MAX_FLOPS))
    assert absolute_only is False, stats

    reason = _kernel_kupl._cpu_plan_fallback_reason(stats)
    assert reason is not None, ("比值超限的计划必须被拒", stats)
    for token in ("pack/flops", str(stats["plan_pack_bytes"]),
                  str(stats["plan_flops"])):
        assert token in reason, (token, reason)

    # 对照组：GEMM 主导（比值 ≪ 1）必须放行。
    gemm = _kernel_kupl._dedicated_cached(
        "ij,jk->ik", ((4000, 4000), (4000, 4000)),
        np.dtype(np.complex128), "auto", "kupl")
    assert gemm is not None
    gemm_stats = _kernel_kupl._plan_statistics(gemm[0])
    gemm_ratio = gemm_stats["plan_pack_bytes"] / gemm_stats["plan_flops"]
    assert gemm_ratio < 0.1, gemm_stats
    assert _kernel_kupl._cpu_plan_fallback_reason(gemm_stats) is None, gemm_stats

    # ``flops == 0`` 时比值无意义：只走绝对判据，且不得除零。
    zero_work = {"plan_flops": 0, "plan_pack_bytes": 0,
                 "plan_arena_bytes": 0, "plan_max_step_macs": 0}
    assert _kernel_kupl._cpu_plan_fallback_reason(zero_work) is None
    zero_work["plan_pack_bytes"] = _kernel_kupl._HEAVY_PACK_MIN_BYTES
    reason = _kernel_kupl._cpu_plan_fallback_reason(zero_work)
    assert reason is not None and "pack-bound" in reason, reason
    assert "pack/flops" not in reason, reason


TESTS = (
    test_allowlist_is_callable,
    test_allowlist_coverage_matches_its_docstring,
    test_allowlist_is_not_a_production_routing_oracle,
    test_cost_gate_cannot_separate_the_losing_paths,
    test_losing_paths_share_the_wide_output_shape,
    test_cpu_plan_gate_rejects_plans_with_excess_pack_traffic_per_flop,
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
