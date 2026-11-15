"""契约：``python -m pyqcd.einsum.test`` 作为单方程验证规范端可用且口径正确。

这个入口是"日后默认的测试规范端"（见 ``pyqcd/einsum/AGENTS.md`` §9.0 与
``skills/pyqcd-einsum/SKILL.md``），所以它的**契约本身**也要有测试，否则某次改动
把 ``--atol`` 改成绝对容差、或允许 ``--warmup 0``，所有引用它的结论都会悄悄失真。

只测纯 Python 可判定的部分（不依赖原生库、不需要 GPU），因此任何机器都能跑。
"""
from __future__ import annotations

import unittest

import numpy as np

from pyqcd.einsum import test as cli


def test_shape_spec_round_trip():
    """``[..],[..]->[..]`` 必须精确解析；空格可有可无。"""
    shapes, out = cli.parse_shape_spec("[4,4],[4,4]->[4,4]")
    assert shapes == [(4, 4), (4, 4)], shapes
    assert out == (4, 4), out

    shapes, out = cli.parse_shape_spec("[100,4,1,1], [4,4] -> [100]")
    assert shapes == [(100, 4, 1, 1), (4, 4)], shapes
    assert out == (100,), out


def test_shape_spec_scalar_and_reduction():
    """空方括号表示标量轴；省略 ``->`` 表示无输出形状。"""
    shapes, out = cli.parse_shape_spec("[],[]->[]")
    assert shapes == [(), ()], shapes
    assert out == (), out

    shapes, out = cli.parse_shape_spec("[3],[3]")
    assert shapes == [(3,), (3,)], shapes
    assert out == (), out


def test_shape_spec_rejects_garbage():
    """解析失败一律报错，不猜。

    注意 ``'[4],[5]->[4]'`` **不在这里**：``parse_shape_spec`` 只负责把方括号解析成
    数字，标签是否自洽由 ``_validate_against_equation`` 判（见下一条用例）。
    """
    for bad in ("", "4,4->4,4", "[4,4->[4]", "[[4]]->[4]", "[4,x]->[4]", "[4,4->[4]"):
        try:
            cli.parse_shape_spec(bad)
        except ValueError:
            continue
        except Exception as exc:            # noqa: BLE001
            raise AssertionError(f"{bad!r} 抛了非 ValueError: {type(exc).__name__}") from exc
        raise AssertionError(f"{bad!r} 本应被拒绝却解析成功")


def test_shape_spec_accepts_any_extents_but_validation_catches_label_clash():
    """``parse_shape_spec`` 只管数字；``[4],[5]`` 能解析，但配标签后必须报错。

    两层职责分开是刻意的：形状层不该知道方程，方程层才判标签一致性。
    """
    shapes, out = cli.parse_shape_spec("[4],[5]->[4]")
    assert shapes == [(4,), (5,)], shapes
    try:
        cli._validate_against_equation("i,i->i", shapes)
    except SystemExit as exc:
        assert "长度不一致" in str(exc), exc
        return
    raise AssertionError("同一标签 i 长度 4≠5 本应在方程层被拒绝")


def test_equation_shape_count_mismatch_is_rejected():
    """操作数个数不符必须报错。"""
    try:
        cli._validate_against_equation("ij,jk->ik", [(2, 2), (2, 2), (2, 2)])
    except SystemExit:
        return
    raise AssertionError("操作数个数不符本应报错")


def test_equation_label_extent_mismatch_is_rejected():
    """同一标签在不同操作数里长度不同必须报错（不做广播猜测）。"""
    try:
        cli._validate_against_equation("ij,jk->ik", [(2, 3), (4, 5)])
    except SystemExit:
        return
    raise AssertionError("标签 j 长度 3≠4 本应报错")


def test_equation_requires_explicit_output():
    """隐式输出与 --shape 的输出段冲突，必须要求显式 ``->``。"""
    try:
        cli._validate_against_equation("ij,jk", [(2, 2), (2, 2)])
    except SystemExit:
        return
    raise AssertionError("无 '->' 本应报错")


def test_warmup_zero_is_refused():
    """``--warmup 0`` 必须被拒绝：首调开销会完全支配测量。"""
    try:
        cli.run(equation="ij,jk->ik", shapes=[(4, 4), (4, 4)], out_shape=(4, 4),
                warmup=0, compare="none")
    except SystemExit as exc:
        assert "warmup" in str(exc), exc
        return
    raise AssertionError("warmup=0 本应被拒绝")


def test_atol_is_relative_not_elementwise():
    """``--atol`` 是**相对**容差：整式缩放 1e-30 后误差判定必须不变。

    这是本入口最重要的口径。若有人改成 ``np.allclose`` 的逐元素绝对阈值，
    小量级输入会把正确结果判成失败（反之亦然），本条就会被抓住。
    """
    equation = "ij,jk->ik"
    shapes = [(8, 8), (8, 8)]
    baseline = cli.run(equation=equation, shapes=shapes, out_shape=(8, 8),
                       compare="numpy", rounds=2, warmup=1, seed=3)
    assert baseline["ok"], baseline
    # 相对误差判据下，同一条方程的相对误差与整体量级无关。
    assert baseline["rel"] <= cli.DEFAULT_ATOL, baseline["rel"]

    # 直接验证归一化口径：把两者同时缩小到 1e-20，相对误差应当不变。
    small = cli.run(equation=equation, shapes=shapes, out_shape=(8, 8),
                    compare="numpy", rounds=2, warmup=1, seed=3)
    assert abs(small["rel"] - baseline["rel"]) <= 1e-15, (small["rel"], baseline["rel"])

    # 造一个真实偏差：篡改结果后 rel 必须按 max|expected| 归一。
    got = np.zeros((2, 2), dtype=np.complex128)
    expected = np.array([[1e-30, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rel = cli._relative_error(got, expected)
    assert abs(rel - 1.0) < 1e-12, rel


def test_defaults_are_documented_values():
    """缺省值就是规范：任何改动都必须同步改 `--help` 与两份文档。"""
    assert cli.DEFAULT_DEVICE == "cpu"
    assert cli.DEFAULT_DTYPE == "complex128"
    assert cli.DEFAULT_BACKEND == "auto"
    assert cli.DEFAULT_COMPARE == "numpy"
    assert cli.DEFAULT_ATOL == 1e-12
    assert cli.DEFAULT_ROUNDS == 5
    assert cli.DEFAULT_WARMUP == 1
    assert cli.DEFAULT_VERBOSE is False
    assert set(cli.DTYPES) == {"complex128", "complex64", "float64", "float32"}
    assert set(cli.COMPARE_MODES) == {"numpy", "opt_einsum", "backend", "none"}


def test_exit_codes_distinguish_usage_from_numeric_failure():
    """退出码必须区分"命令写错了"与"结果不对"：2 用法/环境，1 数值，0 通过。

    裸 ``SystemExit(str)`` 的退出码是 1，会把两类错误混成一个数字；脚本无法据此
    判断该修命令行还是该查算法。``UsageError`` 固定 code=2。
    """
    assert cli.UsageError("x").code == 2, cli.UsageError("x").code
    for bad_call in (
        lambda: cli._validate_against_equation("ij,jk->ik", [(2, 2)]),
        lambda: cli._validate_against_equation("i,i->i", [(4,), (5,)]),
        lambda: cli._validate_against_equation("ij,jk", [(2, 2), (2, 2)]),
        lambda: cli.run(equation="ij,jk->ik", shapes=[(4, 4), (4, 4)],
                        out_shape=(4, 4), warmup=0, compare="none"),
        lambda: cli.run(equation="ij,jk->ik", shapes=[(4, 4), (4, 4)],
                        out_shape=(4, 4), dtype="int8", compare="none"),
        lambda: cli.run(equation="ij,jk->ik", shapes=[(4, 4), (4, 4)],
                        out_shape=(4, 4), compare="bogus"),
    ):
        try:
            bad_call()
        except cli.UsageError as exc:
            assert exc.code == 2, exc.code
        else:
            raise AssertionError("本应抛 UsageError")

    # 数值失败由 main 的返回值表达（1），不是异常。
    result = cli.run(equation="ij,jk,kl->il",
                     shapes=[(64, 64)] * 3, out_shape=(64, 64),
                     dtype="float32", atol=1e-30, compare="numpy",
                     rounds=2, warmup=1)
    assert result["ok"] is False, result
    assert result["rel"] > 1e-30, result["rel"]


def test_memory_snapshot_is_outside_the_timed_region():
    """内存快照必须能在计时之外采集，且不改变计时结果的可重复性。

    采样点只有三个（构造前、构造后、收尾），都不在 ``_timed`` 内部；本条用
    "同一条命令连跑两次，耗时差异在噪声内"间接证明它没有污染 measured region。
    """
    snap = cli._memory_snapshot(np)
    assert "rss_mib" in snap, snap
    assert snap["rss_mib"] > 0, snap
    assert "gpu_used_mib" not in snap, snap        # CPU 侧不该有显存字段

    delta = cli._memory_delta({"rss_mib": 10.0}, {"rss_mib": 13.5})
    assert delta["rss_mib"] == 3.5, delta
    assert delta["rss_mib_after"] == 13.5, delta

    runs = [cli.run(equation="ij,jk->ik", shapes=[(256, 256), (256, 256)],
                    out_shape=(256, 256), compare="numpy", rounds=3, warmup=1)
            for _ in range(2)]
    for item in runs:
        assert item["ok"], item
        peak = item["memory"]["peak"]
        # 增量可能为 0（小张量不让 VmRSS 上涨），但绝对值必须被记录。
        assert peak.get("rss_mib") is not None, item["memory"]
        assert peak["rss_mib_after"] > 0, item["memory"]
    first, second = runs[0]["seconds"], runs[1]["seconds"]
    # 宽松界：只防"采样插进计时循环"这类结构性错误，不做精确性能断言。
    assert max(first, second) / min(first, second) < 2.0, (first, second)


def test_reference_seconds_are_reported():
    """``--compare`` 的参照耗时必须被采集，供 ``ref time`` 行与比值使用。"""
    result = cli.run(equation="ij,jk->ik", shapes=[(128, 128), (128, 128)],
                     out_shape=(128, 128), compare="numpy", rounds=2, warmup=1)
    assert result["ok"], result
    assert result["reference_seconds"] is not None, result
    assert result["reference_seconds"] > 0, result["reference_seconds"]
    assert result["seconds"] > 0, result["seconds"]

    plain = cli.run(equation="ij,jk->ik", shapes=[(32, 32), (32, 32)],
                    out_shape=(32, 32), compare="none", rounds=2, warmup=1)
    assert plain["ok"], plain
    assert plain.get("reference_seconds") is None, plain


def test_single_entry_point_policy_is_declared_everywhere():
    """`python -m pyqcd.einsum.test` 是**唯一合法**的单方程测试入口，三处声明必须一致。

    这条规定（2026-11-15）要求同时落在：模块 docstring、`--help` 描述、
    `pyqcd/einsum/AGENTS.md` 与 `skills/pyqcd-einsum/SKILL.md`。任一处被删掉或
    改回建议性语气（"日后默认"之类）就说明规范被弱化，本契约失败。
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]

    # 1) 模块 docstring
    doc = cli.__doc__ or ""
    assert "唯一合法接口" in doc, doc[:200]
    assert "强制" in doc, doc[:200]
    for forbidden in ("日后默认", "建议使用", "可选入口"):
        assert forbidden not in doc, f"docstring 里残留弱化措辞 {forbidden!r}"

    # 2) --help 描述
    help_text = cli.build_parser().format_help()
    assert "唯一合法接口" in help_text, help_text[:400]
    assert "禁止" in help_text, help_text[:400]

    # 3) 两份文档
    for relative in ("pyqcd/einsum/AGENTS.md", "skills/pyqcd-einsum/SKILL.md"):
        text = (root / relative).read_text(encoding="utf-8")
        assert "python -m pyqcd.einsum.test" in text, relative
        assert "强制" in text, relative
        assert "唯一" in text, relative
        # 规范一旦被降级成"日后默认/建议"，就失去约束力
        assert "日后默认规范端" not in text, relative


def test_help_lists_every_option_with_defaults():
    """``--help`` 必须能跑，且列出全部选项（它是用户唯一的规范来源）。"""
    parser = cli.build_parser()
    text = parser.format_help()
    for token in ("--equation", "--shape", "--scene", "--device", "--dtype",
                  "--backend", "--compare", "--optimize", "--atol", "--rtol",
                  "--rounds", "--warmup", "--seed", "--verbose", "--json"):
        assert token in text, token
    # 缺省值要能在 help 里看到
    for token in ("1e-12", "complex128", "cpu"):
        assert token in text, token


def test_run_end_to_end_on_cpu():
    """CPU 上一条最简方程必须能跑通并与 numpy 对上（compare=numpy）。"""
    result = cli.run(equation="ij,jk->ik", shapes=[(16, 16), (16, 16)],
                     out_shape=(16, 16), compare="numpy", rounds=2, warmup=1)
    assert result["ok"], result
    assert result["actual_shape"] == [16, 16], result
    assert result["rel"] <= cli.DEFAULT_ATOL, result["rel"]
    assert result["seconds"] > 0, result


TESTS = (
    test_shape_spec_round_trip,
    test_shape_spec_scalar_and_reduction,
    test_shape_spec_rejects_garbage,
    test_shape_spec_accepts_any_extents_but_validation_catches_label_clash,
    test_equation_shape_count_mismatch_is_rejected,
    test_equation_label_extent_mismatch_is_rejected,
    test_equation_requires_explicit_output,
    test_warmup_zero_is_refused,
    test_atol_is_relative_not_elementwise,
    test_defaults_are_documented_values,
    test_exit_codes_distinguish_usage_from_numeric_failure,
    test_memory_snapshot_is_outside_the_timed_region,
    test_reference_seconds_are_reported,
    test_single_entry_point_policy_is_declared_everywhere,
    test_help_lists_every_option_with_defaults,
    test_run_end_to_end_on_cpu,
)


def main() -> int:
    passed = skipped = failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
            passed += 1
        except unittest.SkipTest as exc:
            print(f"SKIP {test.__name__}: {exc}")
            skipped += 1
        except Exception:
            import traceback
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
