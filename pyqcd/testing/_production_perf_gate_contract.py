"""闸门：性能依据只认生产尺寸（强制要求，2026-11-14 依 `sush-einsum.py` 重写）。

这条约束写在 `AGENTS.md`、`pyqcd/einsum/AGENTS.md` 与 `skills/pyqcd-einsum/SKILL.md`
里，但只写文档拦不住——历史上最典型的三次误判都是"小尺寸测出的倍率被当成生产结论"：

* `pack/flops` 比值被当作盈亏判据（seq=129 vs 1623 只差 2×，盈亏相反）；
* `dim ≤ 8` 的 ded/gen 比值（0.19–0.53×）被当成"特化普遍更快"，而 scene 档
  CPU 侧四类 dtype 其实是 **0.80–1.67×（普遍略慢）**；
* `--scale 16` 的抽样倍率被写进文档当生产数字。

2026-11-14 的**第四次**误判更隐蔽：把 `scene`/`full` 档（`Nev*Nx = 3200²` 级矩阵乘）
的倍率当成**蒸馏收缩**的性能结论。核对 `refer/sush/sush-einsum.py` 后发现该脚本的
张量根本没有空间轴（`Nx`/`Lt`/`lattice` 全文匹配数为 0），规模只由
`Nev`(100/400)、`Mom`(1/3)、`Ns=4` 决定。因此本模块现在区分**两条互不替代的口径**：

* ``distill`` —— `distill`/`distill_curr` 档，主口径，尺寸逐字取自该脚本；
* ``spatial`` —— `scene`/`full` 档，只支撑"空间×本征模展平后的矩阵乘"。

跨口径引用（把 `spatial` 的数字当蒸馏结论，或反之）由 `claim_basis()` 判为非法。

本模块把约束变成可执行断言：

1. **生产场景必须存在且不退化成小尺寸**，且 `distill` 档逐维度等于参考脚本字面值
   （`REFERENCE_DIMENSIONS` 做来源锁定）。
2. **基准入口必须区分档位**：`bench_dtype_matrix` 的 `--scene` 与 `--dims` 互斥可辨，
   且报告头必须打印场景说明（否则读日志的人会把 `--scale` 的数字贴到生产维度上）。
3. **性能结论必须带证据等级**：`PRODUCTION_SCENES`、`BASES`、`classify_measurement()`
   与 `claim_basis()`，供报告/技能复核"这条数字算不算结论、算哪条口径的结论"。
4. **已知的"不可当结论"清单**：把上面那些历史误判固化成样例，断言分类器把它们
   判为 `unverified`/`path-only`，并断言跨口径引用被拒。

可直接运行::

    python -m pyqcd.testing._production_perf_gate_contract
"""
from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from ..einsum import _bench_scene

# 两条互不替代的生产口径（2026-11-14 依 `refer/sush/sush-einsum.py` 重写）：
#
#   * ``distill``：基准脚本的真实蒸馏张量，规模只由 ``Nev``(100/400)、``Mom``(1/3)、
#     ``Ns`` 决定，**没有空间轴**（该脚本里 ``Nx``/``Lt``/``lattice`` 匹配数为 0）。
#     这是本库的**主口径**。
#   * ``spatial``：把空间×本征模展平后的 ``Nev*Nx``(3200²) 级矩阵乘，来自
#     ``_distillation/contraction.py`` 的 peram 展平描述。它**不是**蒸馏收缩的
#     生产尺寸，只能支撑"展平矩阵乘"这一类结论。
BASES = {
    "distill": ("distill", "distill_curr"),
    "spatial": ("scene", "full"),
}
# 唯一可以产出性能结论的场景档；`smoke` 与任何 `--scale` 都不在内。
PRODUCTION_SCENES = tuple(scene for scenes in BASES.values() for scene in scenes)
# 每个口径的生产参数下限：低于它就只能算路径证据。逐项取自对应来源的实际值，
# 这样一旦有人把生产参数改小，本闸门会直接失败而不是静默放宽。
MIN_PRODUCTION = {
    "distill": {"Nev_src": 100, "Nev_curr": 400, "Mom_src": 1,
                "Mom_curr": 3, "Ns": 4, "Nc": 3},
    "spatial": {"Nx": 32, "Nev": 100, "Lt": 64, "Ns": 4, "Mom": 8},
}
# 基准脚本 `refer/sush/sush-einsum.py` 里的字面值（来源锁定：脚本改了这里才会失败）。
REFERENCE_DIMENSIONS = {
    "Nev_src": 100,      # :170  nev_src = 100
    "Nev_curr": 400,     # :171  nev_curr = 400
    "Mom_src": 1,        # :172  n_mom_src = 1
    "Mom_curr": 3,       # :173  n_mom_curr = 3
    "Ns": 4,             # :200  peram (4,4,nev,nev)
    "Nc": 3,             # _distillation/constant.py:1
}


def basis_of(scene: str | None) -> str | None:
    """场景档属于哪条生产口径；不在生产集合里返回 ``None``。"""
    for basis, scenes in BASES.items():
        if scene in scenes:
            return basis
    return None


def classify_measurement(scene: str | None, scale: int | None,
                         repeat: int) -> str:
    """把一次测量归类为 ``production`` / ``path-only`` / ``unverified``。

    判据与 `AGENTS.md` 一致：

    * ``production``：档位在生产集合里，未被 ``--scale`` 覆盖，且中位次数 ≥10；
    * ``path-only``：能跑通但尺寸不够（``smoke``、任意 ``scale``、等长 dim 扫描）；
    * ``unverified``：档位未知或重复次数不足。
    """
    if scene is None or scene not in PRODUCTION_SCENES:
        return "unverified"
    if scale is not None:
        return "path-only"
    if repeat < 10:
        return "unverified"
    return "production"


def claim_basis(scene: str | None, claimed: str) -> bool:
    """``scene`` 档的数字能否用来支撑 ``claimed`` 口径的结论。

    两条口径的尺度由不同的量决定（蒸馏看 ``Nev``，展平看 ``Nev*Nx``），所以
    **跨口径引用一律非法**：``scene``/``full`` 的倍率不能当蒸馏结论，反之亦然。
    """
    actual = basis_of(scene)
    if actual is None or claimed is None:
        return False
    return actual == claimed


# ── 1. 生产场景本身不得退化 ─────────────────────────────────────────────────

def test_production_scene_is_actually_production_sized():
    """`full` 档必须等于生产参数；参数被别人改小时本闸门失败。"""
    production = dict(_bench_scene.PRODUCTION)
    for key, floor in MIN_PRODUCTION["spatial"].items():
        assert production.get(key, 0) >= floor, (key, production.get(key), floor)
    full = _bench_scene.dims("full")
    for key, value in production.items():
        assert full[key] == value, (key, full[key], value)


def test_distill_scene_matches_reference_script_literals():
    """蒸馏口径的每个维度必须与 `sush-einsum.py` 的字面值逐项相等。

    这是**来源锁定**：参考脚本改了（或有人"顺手调小"了场景），本条立刻失败。
    """
    for scene in BASES["distill"]:
        table = _bench_scene.dims(scene)
        for key, expected in REFERENCE_DIMENSIONS.items():
            assert table.get(key) == expected, (scene, key, table.get(key), expected)
        for key, floor in MIN_PRODUCTION["distill"].items():
            assert table.get(key, 0) >= floor, (scene, key, table.get(key), floor)
    # 两档的差别只有 `Nev`/`Mom` 这一组：源端 100/1，current 端 400/3。
    src = _bench_scene.dims("distill")
    curr = _bench_scene.dims("distill_curr")
    assert (src["Nev"], src["Mom"]) == (100, 1), (src["Nev"], src["Mom"])
    assert (curr["Nev"], curr["Mom"]) == (400, 3), (curr["Nev"], curr["Mom"])


def test_distill_scene_has_no_spatial_axis():
    """蒸馏张量没有空间轴：`distill` 档的 `Nx`/`Lt` 只能是 1 的占位。"""
    for scene in BASES["distill"]:
        table = _bench_scene.dims(scene)
        assert table["Nx"] == 1 and table["Lt"] == 1, (scene, table["Nx"], table["Lt"])


def test_distill_tensor_shapes_are_heterogeneous():
    """真实蒸馏形状必须是异形轴，且出现 `(4,4,100,400)` 这类跨端点组合。"""
    shapes = _bench_scene.distill_tensor_shapes()
    flat = [tuple(shape) for shape in shapes.values()]
    assert flat, shapes
    seen = {shape for shape in flat}
    # 参考脚本 `nev_by_time` 让 peram 出现四种端点组合。
    for want in ((4, 4, 100, 100), (4, 4, 100, 400),
                 (4, 4, 400, 100), (4, 4, 400, 400)):
        assert want in seen, (want, sorted(seen)[:8])
    # 判定"异形"：至少有一个张量的各轴长度不全相等。
    assert any(len(set(shape)) > 1 for shape in flat), flat


def test_scene_tier_is_not_a_smoke_size():
    """`scene` 档必须是"接近生产、但仍装得下"的档，不能退化成 smoke。"""
    scene = _bench_scene.dims("scene")
    smoke = _bench_scene.dims("smoke")
    for key in ("Nev", "Nx"):
        assert scene[key] > smoke[key] * 2, (key, scene[key], smoke[key])
    # 单张 peram 的元素数：形状是 **(Lt, Nx, Ns, Ns, Nev, Nev)**。
    # 我第一版漏了 ``Nx``，把 scene 算成 1.6e6 元素（实际 1.024e8 = 1.53 GiB），
    # 于是断言把正确实现判成失败——这类"足迹算错"正是本闸门要防的。
    def peram_elements(table):
        return (table["Lt"] * table["Nx"] * table["Ns"] ** 2
                * table["Nev"] ** 2)

    scene_elements = peram_elements(scene)
    full_elements = peram_elements(_bench_scene.dims("full"))
    assert scene_elements == 102_400_000, scene_elements        # docstring: 1.53 GiB
    assert full_elements == 327_680_000, full_elements          # docstring: 4.88 GiB
    assert full_elements > scene_elements, (full_elements, scene_elements)


def test_general_shapes_derive_from_production_parameters():
    """通用基准的形状必须由生产参数推出（`Nev*Nx` / `Lt*Nx`），不能是手写常数。"""
    table = _bench_scene.dims("full")
    shapes = _bench_scene.generic_shapes("full")
    assert shapes["side"] == table["Nev"] * table["Nx"], shapes["side"]
    assert shapes["matrix"] == (shapes["side"], shapes["side"])
    assert shapes["spatial_matrix"] == (table["Lt"] * table["Nx"],) * 2


# ── 2. 基准入口必须能区分档位 ───────────────────────────────────────────────

def test_bench_entry_separates_scene_from_scaled():
    """`bench_dtype_matrix` 必须把 `--scene` 与 `--scale` 记录进输出。"""
    from ..testing.einsum import bench_dtype_matrix

    describe = _bench_scene.describe("scene", scale=None)
    assert "scene=scene" in describe, describe
    assert "Nev=" in describe and "Nx=" in describe, describe
    scaled = _bench_scene.describe("scene", scale=16)
    assert "Nx=16" in scaled, scaled

    source = open(bench_dtype_matrix.__file__, encoding="utf-8").read()
    for needle in ("--scene", "--scale", "describe(", "mem-budget"):
        assert needle in source, needle


# ── 3. 分类器本身的行为 ─────────────────────────────────────────────────────

def test_classifier_marks_small_runs_as_non_conclusions():
    """历史误判样例：这些测量都不能产出性能结论。"""
    cases = (
        # (scene, scale, repeat, 期望分类)
        (None, None, 100, "unverified"),        # 等长 dim 扫描：没有生产档
        ("smoke", None, 100, "unverified"),     # 冒烟档
        ("scene", 16, 100, "path-only"),        # --scale 缩小
        ("scene", 32, 20, "path-only"),         # 即便 scale=32 也不是生产
        ("distill", 8, 100, "path-only"),       # 蒸馏口径同样不许 --scale
        ("scene", None, 3, "unverified"),       # 中位次数不足
        ("scene", None, 10, "production"),
        ("full", None, 10, "production"),
        ("full", None, 20, "production"),
        ("distill", None, 10, "production"),
        ("distill_curr", None, 10, "production"),
    )
    for scene, scale, repeat, expected in cases:
        actual = classify_measurement(scene, scale, repeat)
        assert actual == expected, (scene, scale, repeat, actual, expected)


def test_cross_basis_claims_are_rejected():
    """跨口径引用必须被判非法：`spatial` 的数字不能当蒸馏结论，反之亦然。"""
    # 同口径：允许。
    assert claim_basis("distill", "distill")
    assert claim_basis("distill_curr", "distill")
    assert claim_basis("scene", "spatial")
    assert claim_basis("full", "spatial")
    # 跨口径：一律拒绝——这正是 2026-11-14 重写要钉死的那条。
    assert not claim_basis("scene", "distill")
    assert not claim_basis("full", "distill")
    assert not claim_basis("distill", "spatial")
    assert not claim_basis("distill_curr", "spatial")
    # 非生产档：没有任何口径。
    assert not claim_basis("smoke", "distill")
    assert not claim_basis(None, "spatial")
    assert basis_of("smoke") is None and basis_of(None) is None


def test_known_wrong_claims_are_classified_as_non_production():
    """把真实历史误判固化成样例，防止被重新当作证据。"""
    wrong_claims: tuple[tuple[str, Any], ...] = (
        ("dim=4 的 ded/gen 0.19×（等长超立方体）", classify_measurement(None, None, 100)),
        ("--scale 16 抽样的 0.33×", classify_measurement("scene", 16, 50)),
        ("smoke 档的中位倍率", classify_measurement("smoke", None, 100)),
    )
    for claim, verdict in wrong_claims:
        assert verdict != "production", (claim, verdict)
    # 2026-11-14 新增的两类：跨口径引用与"用 _label_sizes 反推蒸馏形状"。
    assert not claim_basis("scene", "distill"), "scene 的倍率被当成蒸馏结论"
    assert not claim_basis("full", "distill"), "full 的倍率被当成蒸馏结论"


# ── 4. 生产档的真实规模可达性 ───────────────────────────────────────────────

def test_production_scene_footprint_is_documented():
    """两档的内存足迹都必须能被算出来，且与 `_bench_scene` 的注释一致。"""
    def peram_bytes(table):
        return (table["Lt"] * table["Nx"] * table["Ns"] ** 2
                * table["Nev"] ** 2) * 16                       # complex128

    scene_bytes = peram_bytes(_bench_scene.dims("scene"))
    full_bytes = peram_bytes(_bench_scene.dims("full"))
    # 与 `_bench_scene` docstring 的两个数字逐一对齐（差 1% 以内）：
    #   scene 1.53 GiB、full 4.88 GiB。
    assert abs(scene_bytes / 2 ** 30 - 1.53) < 0.01, scene_bytes / 2 ** 30
    assert abs(full_bytes / 2 ** 30 - 4.88) < 0.01, full_bytes / 2 ** 30
    # 六张传播子的光操作数就 ~9 GiB（scene）：15 GB 机器跑 scene 已接近上限，
    # 这正是"生产档跑不动时必须降 --scale 并标 path-only"的依据。
    assert scene_bytes * 6 >= (8 << 30), scene_bytes

    documented = open(_bench_scene.__file__, encoding="utf-8").read()
    for needle in ("1.53 GiB", "4.88 GiB"):
        assert needle in documented, (
            f"{needle} 必须写在 _bench_scene 注释里，否则报告无法核对")


TESTS = (
    test_production_scene_is_actually_production_sized,
    test_distill_scene_matches_reference_script_literals,
    test_distill_scene_has_no_spatial_axis,
    test_distill_tensor_shapes_are_heterogeneous,
    test_scene_tier_is_not_a_smoke_size,
    test_general_shapes_derive_from_production_parameters,
    test_bench_entry_separates_scene_from_scaled,
    test_classifier_marks_small_runs_as_non_conclusions,
    test_cross_basis_claims_are_rejected,
    test_known_wrong_claims_are_classified_as_non_production,
    test_production_scene_footprint_is_documented,
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
