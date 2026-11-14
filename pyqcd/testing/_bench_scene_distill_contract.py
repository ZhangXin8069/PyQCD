"""契约：蒸馏生产场景（``distill`` / ``distill_curr``）的来源锁定与向后兼容。

背景：``_bench_scene.PRODUCTION`` 的 ``Nx=32`` / ``Lt=64`` / ``Mom=8`` 来自
``refer/sush/_distillation/contraction.py`` 的「空间×本征模展平」描述；而基准脚本
``refer/sush/sush-einsum.py`` 里实际进 einsum 的张量**没有任何空间轴**（该文件里
``Nx`` / ``Lt`` / ``lattice`` 的 grep 匹配数为 0），规模只由 ``Nev``（100/400）、
``Mom``（1/3）、``Ns=4`` 决定。本模块把这条差异固化成可执行断言，防止有人
「按看起来合理」把 ``Nx`` 加回蒸馏场景，或把两个 ``Nev`` 合并成一个。

四组断言：

1. **来源锁定**：``PRODUCTION_DISTILL`` / ``SCENES["distill*"]`` 的每个数字都等于
   参考脚本里的字面值，且注释里带 ``文件:行``。这里是**硬编码副本**——参考脚本改了
   而本库没跟，就失败。
2. **向后兼容**：``dims("full"/"scene"/"smoke")`` 与 ``scale=8/16/32/48`` 的返回值
   与「引入蒸馏场景之前」的基线逐项相同（基线见 :data:`LEGACY_DIMS` 等）。
3. **形状是异形的**：``distill_tensor_shapes`` 给出参考脚本的真实轴长组合
   （``4``/``100``/``400``/``3`` 混排），并保留 ``(3, 400, 400)`` 的动量轴。
4. **内存预算**：两段的操作数总量都远小于 :data:`_bench_scene.
   DISTILL_MEMORY_LIMIT_GIB`（15 GiB），所以 ``Nev_curr = 400`` 不需要缩小档。

可直接运行::

    python -m pyqcd.testing._bench_scene_distill_contract
"""
from __future__ import annotations

import unittest

from ..einsum import _bench_scene

# ── 1. 参考脚本的字面值（来源锁定；左边是键，右边是 (值, 出处)） ───────────────
REF_LITERALS = {
    "Nev_src": (100, "refer/sush/sush-einsum.py:170"),
    "Nev_curr": (400, "refer/sush/sush-einsum.py:171"),
    "Mom_src": (1, "refer/sush/sush-einsum.py:172"),
    "Mom_curr": (3, "refer/sush/sush-einsum.py:173"),
    "Ns": (4, "refer/sush/sush-einsum.py:200"),
    "Nc": (3, "refer/sush/_distillation/constant.py:1"),
    "Nt_labels": (4, "refer/sush/sush-einsum.py:179"),
}

# dims() 返回的两个蒸馏档（Nx/Lt = 1 表示「该轴不存在」的占位，不是缩小值）。
DISTILL_DIMS = {
    "distill": {"Nx": 1, "Lt": 1, "Nev": 100, "Ns": 4, "Mom": 1},
    "distill_curr": {"Nx": 1, "Lt": 1, "Nev": 400, "Ns": 4, "Mom": 3},
}

# ── 2. 改动前的基线（2026-11-14 逐项 dump，先记录后改动） ──────────────────────
LEGACY_DIMS = {
    "full": {"Nx": 32, "Lt": 64, "Nev": 100, "Ns": 4, "Mom": 8},
    "scene": {"Nx": 64, "Lt": 40, "Nev": 50, "Ns": 4, "Mom": 5},
    "smoke": {"Nx": 20, "Lt": 12, "Nev": 12, "Ns": 4, "Mom": 2},
}
#: 键顺序也锁定：``bench_specialized`` / ``_production_perf_gate_contract`` 的存在
#: 依赖这五个键，顺序变了说明有人换过结构。
LEGACY_KEYS = ("Nx", "Lt", "Nev", "Ns", "Mom")
LEGACY_SCALE = {
    8: {"Nx": 8, "Lt": 16, "Nev": 25, "Ns": 4, "Mom": 2},
    16: {"Nx": 16, "Lt": 32, "Nev": 50, "Ns": 4, "Mom": 4},
    32: {"Nx": 32, "Lt": 64, "Nev": 100, "Ns": 4, "Mom": 8},
    48: {"Nx": 48, "Lt": 96, "Nev": 150, "Ns": 4, "Mom": 12},
}
LEGACY_DESCRIBE = {
    "full": "scene=full  Nx=32 Nev=100 Lt=64 Mom=8 Ns=4  "
            "(production: Nx=32 Nev=100 Lt=64 Mom=8 Ns=4)",
    "scene": "scene=scene  Nx=64 Nev=50 Lt=40 Mom=5 Ns=4  "
             "(production: Nx=32 Nev=100 Lt=64 Mom=8 Ns=4)",
    "smoke": "scene=smoke  Nx=20 Nev=12 Lt=12 Mom=2 Ns=4  "
             "(production: Nx=32 Nev=100 Lt=64 Mom=8 Ns=4)",
}
LEGACY_DESCRIBE_SCALE = {
    8: "scene=scene  Nx=8 Nev=25 Lt=16 Mom=2 Ns=4  "
       "(production: Nx=32 Nev=100 Lt=64 Mom=8 Ns=4)",
    48: "scene=scene  Nx=48 Nev=150 Lt=96 Mom=12 Ns=4  "
        "(production: Nx=32 Nev=100 Lt=64 Mom=8 Ns=4)",
}
LEGACY_PRODUCTION = {"Nx": 32, "Lt": 64, "Nev": 100, "Ns": 4, "Mom": 8}

# ── 3. 参考脚本的真实张量形状（硬编码副本，出处见注释） ───────────────────────
REF_TENSOR_SHAPES_3PT = {
    "peram[...]": {(4, 4, 100, 100), (4, 4, 100, 400), (4, 4, 400, 100),
                   (4, 4, 400, 400)},                       # :200 + :184-189
    "VVV_0@tsink": {(100, 100, 100)},                       # :209 分配 → :241 切动量轴
    "VVV_0@tsrc": {(100, 100, 100)},                        # :214 → :242
    "VDV_0@tsrc": {(1, 100, 100)},                          # :219 → :243
    "VDV_0@tcur0": {(3, 400, 400)},                         # :224 → :244
    "VDV_0@tcur1": {(3, 400, 400)},                         # :245
    "gamma_4": {(4, 4)},                                    # :248
    "gamma_7": {(4, 4)},                                    # :249
    "gamma_curr": {(4, 4, 4)},                              # :250
}


def test_distill_literals_match_reference():
    """``PRODUCTION_DISTILL`` 每个数字都必须等于参考脚本字面值。"""
    for key, (value, source) in REF_LITERALS.items():
        actual = _bench_scene.PRODUCTION_DISTILL[key]
        assert actual == value, (key, actual, value, source)
    assert tuple(_bench_scene.DISTILL_TIMES) == ("tsink", "tsrc", "tcur0", "tcur1")
    assert dict(_bench_scene.DISTILL_NEV_BY_TIME) == {
        "tsink": 100, "tsrc": 100, "tcur0": 400, "tcur1": 400}
    assert _bench_scene.DISTILL_SETTINGS["optimize"] == "dp"       # :176
    assert _bench_scene.DISTILL_SETTINGS["dtype"] == "complex128"  # :177
    assert _bench_scene.DISTILL_SETTINGS["seed"] == 20261114       # :175


def test_distill_scene_dims_lock_the_two_nev():
    """两个 Nev 必须分开存在；``Nx``/``Lt`` 只能是「无此轴」的占位 1。"""
    for scene, expected in DISTILL_DIMS.items():
        table = _bench_scene.dims(scene)
        for key, value in expected.items():
            assert table[key] == value, (scene, key, table[key], value)
        # 消费方（_label_sizes / bench_dtype_matrix）需要的键必须齐。
        for key in LEGACY_KEYS:
            assert key in table, (scene, key)
        # 两个 Nev 不能合并成一个：distill 用 100、distill_curr 用 400。
        assert table["Nev_src"] == 100 and table["Nev_curr"] == 400, scene
        assert table["Mom_src"] == 1 and table["Mom_curr"] == 3, scene
    assert _bench_scene.dims("distill")["Nev"] == 100
    assert _bench_scene.dims("distill_curr")["Nev"] == 400


def test_scenes_registry_contains_distill():
    """新档必须进 ``SCENES``，且既有档位的对象没被换掉。"""
    assert set(_bench_scene.SCENES) >= {
        "full", "scene", "smoke", "distill", "distill_curr"}
    assert _bench_scene.SCENES["full"] is _bench_scene.PRODUCTION
    assert _bench_scene.SCENES["scene"] is _bench_scene.DEFAULT_SCENE
    assert _bench_scene.SCENES["smoke"] is _bench_scene.SMOKE_SCENE
    try:
        _bench_scene.dims("not-a-scene")
    except SystemExit:
        pass
    else:
        raise AssertionError("未知场景必须 SystemExit，不能静默回落到 scene")


def test_legacy_scene_dims_are_unchanged():
    """``full``/``scene``/``smoke`` 与 ``PRODUCTION`` 逐项等同改动前。"""
    assert dict(_bench_scene.PRODUCTION) == LEGACY_PRODUCTION
    for scene, expected in LEGACY_DIMS.items():
        actual = _bench_scene.dims(scene)
        assert actual == expected, (scene, actual, expected)
        assert tuple(actual) == LEGACY_KEYS, (scene, tuple(actual))
    for scene, expected in LEGACY_DESCRIBE.items():
        actual = _bench_scene.describe(scene)
        assert actual == expected, (scene, actual, expected)


def test_scale_path_is_unchanged():
    """``scale=`` 路径行为不变：与改动前逐项相同，且压过 scene 选择。"""
    for scale, expected in LEGACY_SCALE.items():
        actual = _bench_scene.dims("scene", scale=scale)
        assert actual == expected, (scale, actual, expected)
        assert tuple(actual) == LEGACY_KEYS, (scale, tuple(actual))
        # scale 给定时 scene 参数被忽略（含新档），行为与改动前一致。
        for scene in ("full", "scene", "smoke", "distill", "distill_curr"):
            assert _bench_scene.dims(scene, scale=scale) == expected, (scene, scale)
    for scale, expected in LEGACY_DESCRIBE_SCALE.items():
        actual = _bench_scene.describe("scene", scale=scale)
        assert actual == expected, (scale, actual, expected)


def test_describe_keeps_legacy_prefix_for_distill():
    """蒸馏档的说明必须仍以旧格式开头，并额外印出两个 Nev / 两个 Mom。"""
    text = _bench_scene.describe("distill")
    assert text.startswith("scene=distill  Nx=1 Nev=100 Lt=1 Mom=1 Ns=4  "), text
    assert "(production: Nx=32 Nev=100 Lt=64 Mom=8 Ns=4)" in text, text
    for needle in ("Nev_src=100", "Nev_curr=400", "Mom_src=1", "Mom_curr=3",
                   "Nc=3", "无 Nx/Lt 轴"):
        assert needle in text, (needle, text)


def test_distill_tensor_shapes_are_literal_and_anisotropic():
    """形状构造器给出参考脚本的真实轴长，且**不是** ``dim`` 等长。"""
    shapes = _bench_scene.distill_tensor_shapes("3pt")
    perams = {shape for name, shape in shapes.items() if name.startswith("peram[")}
    assert perams == REF_TENSOR_SHAPES_3PT["peram[...]"], perams
    for name, expected in REF_TENSOR_SHAPES_3PT.items():
        if name == "peram[...]":
            continue
        assert shapes[name] in expected, (name, shapes[name], expected)
    # 异形：至少一张张量内部轴长不相等（混合 4 / 100 / 400）。
    assert any(len(set(shape)) > 1 and max(shape) > 4 for shape in shapes.values())
    assert (4, 4, 100, 400) in perams
    assert (3, 400, 400) in {shapes["VDV_0@tcur0"], shapes["VDV_0@tcur1"]}

    legacy = _bench_scene.distill_tensor_shapes("6peram")
    assert sum(name.startswith("peram") for name in legacy) == 6      # :377-382
    assert sum(name.startswith("gamma") for name in legacy) == 4      # :385-389
    assert {legacy[f"VVV{i}"] for i in range(1, 5)} == {(3, 100, 100, 100)}   # :390
    assert {legacy[f"VDV{i}"] for i in range(1, 3)} == {(3, 100, 100)}        # :391


def test_distill_footprint_fits_memory_budget():
    """两段的操作数总量必须远低于 15 GiB；dtype 是 complex128 = 16 B。"""
    limit = _bench_scene.DISTILL_MEMORY_LIMIT_GIB
    assert limit == 15.0, limit
    three = _bench_scene.distill_memory("3pt")
    six = _bench_scene.distill_memory("6peram")
    assert three["itemsize"] == 16 and six["itemsize"] == 16     # :177 complex128
    assert three["total_elements"] == 21_940_096, three["total_elements"]
    assert six["total_elements"] == 13_020_064, six["total_elements"]
    for memory in (three, six):
        assert memory["total_gib"] < limit, memory["total_gib"]
    assert abs(three["total_gib"] - 0.327) < 0.001, three["total_gib"]
    assert abs(six["total_gib"] - 0.194) < 0.001, six["total_gib"]
    # 单张最大张量也远小于上限（不存在"单张就爆内存"的档）。
    for memory in (three, six):
        assert memory["largest"][2] < 1.0, memory["largest"]


def test_consumers_can_consume_distill_dims():
    """``_label_sizes`` 拿到新档的维度表不能抛 KeyError（消费方契约）。"""
    from ..einsum.bench_specialized import _label_sizes
    from ..einsum import _wick_tasklist

    task = _wick_tasklist.by_seq()[3]           # 3pt 7 op（VVV + gamma + 3 peram）
    terms = task.equation.split("->")[0].split(",")
    for scene in ("distill", "distill_curr"):
        sizes = _label_sizes(terms, task.roles, _bench_scene.dims(scene))
        assert sizes, scene
        for label, extent in sizes.items():
            assert extent >= 1, (scene, label, extent)
    # peram 的酉空间标签必须落在该档的 Nev 上。
    sizes = _label_sizes(terms, task.roles, _bench_scene.dims("distill_curr"))
    assert 400 in set(sizes.values()), sizes
    _bench_scene.generic_shapes("distill")      # 不得抛（形状仍是旧口径）


def test_label_sizes_cannot_express_distill_axes():
    """已知限制（锁定）：``_label_sizes`` 的等长口径**表达不出**蒸馏轴长。

    它把 ``VVV`` 当 ``(Lt, Mom, Nev, Nev, Nev)``、``VDV`` 当 ``(Lt, Mom, Nev, Nev)``，
    且全方程只有一个 ``Nev``；``dims("distill*")`` 里 ``Nx=Lt=1``（无此轴占位）、
    ``Mom`` 是 1 或 3，于是这些占位会作为**真实轴长**漏进 peram 的轴里，
    与参考脚本的 ``(4,4,100,400)`` 之类形状无关。
    结论：``bench_specialized --scene distill`` 的形状只能当 path-only 证据；
    真实蒸馏形状必须用 :func:`_bench_scene.distill_tensor_shapes`。
    """
    from ..einsum.bench_specialized import _label_sizes
    from ..einsum import _wick_tasklist

    task = _wick_tasklist.by_seq()[3]
    terms = task.equation.split("->")[0].split(",")
    sizes = _label_sizes(terms, task.roles, _bench_scene.dims("distill"))
    shapes = [tuple(sizes[c] for c in term) for term in terms]
    # 占位/动量值 1 与 3 漏进了操作数轴（真实蒸馏形状里没有 1 或 3 的酉空间轴）。
    assert any(1 in shape for shape in shapes), shapes
    real = _bench_scene.distill_tensor_shapes("3pt")
    eigen_shapes = [shape for name, shape in real.items() if name.startswith("peram[")]
    assert all(1 not in shape[2:] for shape in eigen_shapes), eigen_shapes
    assert any(400 in shape[2:] for shape in eigen_shapes), eigen_shapes


TESTS = (
    test_distill_literals_match_reference,
    test_distill_scene_dims_lock_the_two_nev,
    test_scenes_registry_contains_distill,
    test_legacy_scene_dims_are_unchanged,
    test_scale_path_is_unchanged,
    test_describe_keeps_legacy_prefix_for_distill,
    test_distill_tensor_shapes_are_literal_and_anisotropic,
    test_distill_footprint_fits_memory_budget,
    test_consumers_can_consume_distill_dims,
    test_label_sizes_cannot_express_distill_axes,
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
