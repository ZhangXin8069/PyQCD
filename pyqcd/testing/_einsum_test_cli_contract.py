"""契约：``python -m pyqcd.einsum.test`` 作为单方程验证规范端可用且口径正确。

这个入口是"日后默认的测试规范端"（见 ``pyqcd/einsum/AGENTS.md`` §9.0 与
``skills/pyqcd-einsum/SKILL.md``），所以它的**契约本身**也要有测试，否则某次改动
把 ``--atol`` 改成绝对容差、或在**不报告实际值**的情况下改 warmup/rounds 缺省，
所有引用它的结论都会悄悄失真（缺省 ``--warmup 1`` / ``--rounds 3`` 是项目契约，
报告里的 ``warmup_effective`` / ``wall_ms`` 必须如实反映它们）。

只测纯 Python 可判定的部分（不依赖原生库、不需要 GPU），因此任何机器都能跑。
"""
from __future__ import annotations

import os
import re
import unittest
from unittest.mock import patch

import numpy as np

from pyqcd.einsum import test as cli

# 本模块验证的是 CLI 的**纯 Python 口径**（形状解析、参数校验、容差、报告字段）。
# 配置表与自动 tune 都会让 `cli.run` 走"实测过的最快路线"，于是本模块的断言会随
# **用户家目录里那张真实表**变化（实测出现过命中的提示与 ProfileConflict）。
# 因此在模块导入时显式关掉两者；`conftest.main()` 对整套测试做同一件事。
os.environ.setdefault("PYQCD_EINSUM_PROFILE", "0")
os.environ.setdefault("PYQCD_EINSUM_AUTOTUNE", "0")


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


def test_warmup_zero_is_allowed_but_negative_is_refused():
    """``--warmup 0`` 仍合法，缺省为 1，负数才是用法错误。"""
    report = cli.run(equation="ij,jk->ik", shapes=[(4, 4), (4, 4)], out_shape=(4, 4),
                     warmup=0, rounds=1, compare="none")
    assert report["ok"], report
    assert report["warmup"] == 0, report["warmup"]
    assert report["warmup_effective"] == 0, report["warmup_effective"]

    try:
        cli.run(equation="ij,jk->ik", shapes=[(4, 4), (4, 4)], out_shape=(4, 4),
                warmup=-1, compare="none")
    except SystemExit as exc:
        assert "warmup" in str(exc), exc
        return
    raise AssertionError("warmup=-1 本应被拒绝")


def test_atol_is_relative_not_elementwise():
    """``--atol`` 是**相对**容差：整式缩放 1e-30 后误差判定必须不变。

    这是本入口最重要的口径。若有人改成 ``np.allclose`` 的逐元素绝对阈值，
    小量级输入会把正确结果判成失败（反之亦然），本条就会被抓住。
    """
    equation = "ij,jk->ik"
    shapes = [(8, 8), (8, 8)]
    baseline = cli.run(equation=equation, shapes=shapes, out_shape=(8, 8),
                       compare="opt_einsum", rounds=2, warmup=1, seed=3)
    assert baseline["ok"], baseline
    # 相对误差判据下，同一条方程的相对误差与整体量级无关。
    assert baseline["rel"] <= cli.DEFAULT_ATOL, baseline["rel"]

    # 直接验证归一化口径：把两者同时缩小到 1e-20，相对误差应当不变。
    small = cli.run(equation=equation, shapes=shapes, out_shape=(8, 8),
                    compare="opt_einsum", rounds=2, warmup=1, seed=3)
    assert abs(small["rel"] - baseline["rel"]) <= 1e-15, (small["rel"], baseline["rel"])

    # 造一个真实偏差：篡改结果后 rel 必须按 max|expected| 归一。
    got = np.zeros((2, 2), dtype=np.complex128)
    expected = np.array([[1e-30, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rel = cli._relative_error(got, expected)
    assert abs(rel - 1.0) < 1e-12, rel


def test_timed_reports_the_minimum_sample():
    """``_timed`` 报告三个样本的最小值，而不是中位数。"""
    ticks = iter((0.010, 0.030, 0.040, 0.070, 0.080, 0.090))
    with patch.object(cli.time, "perf_counter", side_effect=lambda: next(ticks)):
        seconds = cli._timed(lambda: None, warmup=1, rounds=3)
    assert abs(seconds - 0.010) < 1e-15, seconds


def test_defaults_are_documented_values():
    """缺省值就是规范：任何改动都必须同步改 `--help` 与两份文档。"""
    assert cli.DEFAULT_DEVICE == "cpu"
    assert cli.DEFAULT_DTYPE == "complex128"
    assert cli.DEFAULT_BACKEND == "auto"
    # ``--compare`` 缺省**关闭**（``none``）：不给就不跑参考基准。``numpy`` 档已删除，
    # 它原来的等价写法是 ``--backend numpy --compare backend``。
    assert cli.DEFAULT_COMPARE == "none"
    assert cli.DEFAULT_ATOL == 1e-12
    assert cli.DEFAULT_ROUNDS == 3, cli.DEFAULT_ROUNDS
    assert cli.DEFAULT_WARMUP == 1, cli.DEFAULT_WARMUP
    assert cli.DEFAULT_TUNE_TIMEOUT == 15.0, cli.DEFAULT_TUNE_TIMEOUT
    assert cli.DEFAULT_TUNE_BUDGET == 1000.0, cli.DEFAULT_TUNE_BUDGET
    parser = cli.build_parser(language="en")
    args = parser.parse_args(["--equation", "ij,jk->ik",
                              "--shape", "[2,2],[2,2]"])
    assert args.warmup == 1, args.warmup
    assert args.rounds == 3, args.rounds
    assert args.tune_timeout == 15.0, args.tune_timeout
    assert args.tune_budget == 1000.0, args.tune_budget
    assert cli.DEFAULT_VERBOSE is False
    assert set(cli.DTYPES) == {"complex128", "complex64", "float64", "float32"}
    assert set(cli.COMPARE_MODES) == {"opt_einsum", "backend", "none"}


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
                        out_shape=(4, 4), warmup=-1, compare="none"),
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
                     dtype="float32", atol=1e-30, compare="opt_einsum",
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

    # **结构性**判据，不用"两次耗时比值"：256² 的收缩只有几毫秒，比值在负载下
    # 天然抖动（实测 8.2 ms vs 28.0 ms，比值 3.4），那条断言与"采样是否插进计时
    # 循环"毫无关系，只会变成偶发红。这里直接在 `_timed` 打开期间盯着采样器：
    # 任何一次 `_memory_snapshot` 落在计时区内就立即失败。
    real_timed = cli._timed
    real_snapshot = cli._memory_snapshot
    inside = {"active": False, "violations": 0, "snapshots": 0}

    def guarded_timed(call, warmup, rounds, sync=None):
        inside["active"] = True
        try:
            return real_timed(call, warmup, rounds, sync)
        finally:
            inside["active"] = False

    def guarded_snapshot(xp):
        inside["snapshots"] += 1
        if inside["active"]:
            inside["violations"] += 1
        return real_snapshot(xp)

    cli._timed = guarded_timed
    cli._memory_snapshot = guarded_snapshot
    try:
        item = cli.run(equation="ij,jk->ik", shapes=[(256, 256), (256, 256)],
                       out_shape=(256, 256), compare="opt_einsum",
                       rounds=3, warmup=1)
    finally:
        cli._timed = real_timed
        cli._memory_snapshot = real_snapshot
    assert item["ok"], item
    peak = item["memory"]["peak"]
    # 增量可能为 0（小张量不让 VmRSS 上涨），但绝对值必须被记录。
    assert peak.get("rss_mib") is not None, item["memory"]
    assert peak["rss_mib_after"] > 0, item["memory"]
    assert inside["snapshots"] >= 2, (
        "采样点必须真的被调用（构造前 / 构造后 / 收尾）", inside)
    assert inside["violations"] == 0, (
        "内存采样不许落在 _timed 的计时区内", inside)


def test_reference_seconds_are_reported():
    """``--compare`` 的参照耗时必须被采集，供 ``ref time`` 行与比值使用。"""
    result = cli.run(equation="ij,jk->ik", shapes=[(128, 128), (128, 128)],
                     out_shape=(128, 128), compare="opt_einsum", rounds=2, warmup=1)
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
                  "--backend", "--compare", "--specialize", "--equivalence",
                  "--optimize", "--atol",
                  "--rtol", "--rounds", "--warmup", "--seed", "--verbose", "--json"):
        assert token in text, token
    # 缺省值要能在 help 里看到
    for token in ("1e-12", "complex128", "cpu"):
        assert token in text, token


def test_run_end_to_end_on_cpu():
    """CPU 上一条最简方程必须能跑通并与 numpy 对上（compare=numpy）。"""
    result = cli.run(equation="ij,jk->ik", shapes=[(16, 16), (16, 16)],
                     out_shape=(16, 16), compare="opt_einsum", rounds=2, warmup=1)
    assert result["ok"], result
    assert result["actual_shape"] == [16, 16], result
    assert result["rel"] <= cli.DEFAULT_ATOL, result["rel"]
    assert result["seconds"] > 0, result


# ── --specialize 三态 ───────────────────────────────────────────────────────
#: 语料 seq=1（distill 尺寸）：在 ``_specialize`` 索引里，专供"专用路径开/关"对照。
SPECIALIZE_EQUATION = "cadb,menf,ac,em,bd,fn->"
SPECIALIZE_SHAPES = [(4, 4, 1, 1), (4, 4, 1, 1), (4, 4), (4, 4), (1, 1), (1, 1)]


def _specialize_env():
    import os

    from pyqcd.einsum import _specialize

    return os, _specialize


def test_specialize_tri_state_maps_to_contract_argument():
    """``--specialize auto/true/false`` 必须映射成 ``None/True/False`` 三态。

    ``auto`` 是 ``None`` 的别名（跟随 ``PYQCD_EINSUM_SPECIALIZE``），不是 ``True``：
    写成 ``True`` 会把环境变量关掉的专用路径强行打开，等于换了缺省行为。
    """
    assert cli.DEFAULT_SPECIALIZE == "auto", cli.DEFAULT_SPECIALIZE
    assert set(cli.SPECIALIZE_MODES) == {"auto", "true", "false"}, cli.SPECIALIZE_MODES

    assert cli.parse_specialize("auto") is None
    assert cli.parse_specialize(None) is None
    assert cli.parse_specialize("true") is True
    assert cli.parse_specialize(True) is True
    assert cli.parse_specialize("false") is False
    assert cli.parse_specialize(False) is False
    # 大小写与常见真/假写法（与 --verbose 同口径）
    assert cli.parse_specialize("AUTO") is None
    assert cli.parse_specialize("TRUE") is True
    assert cli.parse_specialize("off") is False

    for bad in ("bogus", "specialize", 1.5):
        try:
            cli.parse_specialize(bad)
        except cli.UsageError as exc:
            assert exc.code == 2, exc.code
        else:
            raise AssertionError(f"{bad!r} 本应被拒绝")


def test_specialize_false_really_disables_the_native_path():
    """**核心保险丝**：``--specialize false`` 必须真的不碰任何原生入口。

    用一条**语料内**方程（seq=1，拓扑命中原生命中）。若 ``specialize`` 没被传到
    ``contract()`` 底下，``LAST_INFO`` 会留下原生命中，本契约立刻失败。

    反方向（auto/true 命中）只在原生库可用的机器上断言——没有 ``.so`` 的机器本来
    就没有可命中的路径，那不算本契约的失败面。
    """
    os, _specialize = _specialize_env()
    saved = os.environ.pop(_specialize.FUSED_DISABLE_ENV, None)
    try:
        auto = cli.run(equation=SPECIALIZE_EQUATION, shapes=SPECIALIZE_SHAPES,
                       out_shape=(), compare="none", specialize=None,
                       rounds=2, warmup=1)
        assert auto["ok"], auto
        assert auto["specialize"] == "auto", auto
        # auto 的"实际是否启用"必须等于库自己的判定（即真的跟随了环境变量）
        assert auto["specialize_effective"] is _specialize.enabled(None), auto

        forced_off = cli.run(equation=SPECIALIZE_EQUATION, shapes=SPECIALIZE_SHAPES,
                             out_shape=(), compare="none", specialize=False,
                             rounds=2, warmup=1)
        assert forced_off["ok"], forced_off
        assert forced_off["specialize"] == "false", forced_off
        assert forced_off["specialize_effective"] is False, forced_off
        # 关掉专用路径后，两个原生入口都不该留下任何记录
        assert forced_off["kernel_info"] == {}, forced_off["kernel_info"]
        assert "specialize_warning" not in forced_off, forced_off["specialize_warning"]

        if auto["kernel_info"]:
            # 本机有原生库：auto 必须留下 path，强制 true 必须与 auto 一致
            for info in auto["kernel_info"].values():
                assert "path" in info, auto["kernel_info"]
            forced_on = cli.run(equation=SPECIALIZE_EQUATION, shapes=SPECIALIZE_SHAPES,
                                out_shape=(), compare="none", specialize=True,
                                rounds=2, warmup=1)
            assert forced_on["ok"], forced_on
            assert forced_on["specialize"] == "true", forced_on
            assert forced_on["specialize_effective"] is True, forced_on
            assert forced_on["kernel_info"] == auto["kernel_info"], (
                forced_on["kernel_info"], auto["kernel_info"])
    finally:
        os.environ.pop(_specialize.FUSED_DISABLE_ENV, None)
        if saved is not None:
            os.environ[_specialize.FUSED_DISABLE_ENV] = saved


def test_specialize_is_forwarded_to_the_compare_backend_reference():
    """``--compare backend`` 的参照路径必须拿到**同一个** ``specialize``。

    否则跨后端互证会把"专用 vs 通用"当成后端差异来比较，结论失真。用同一条语料
    方程（它会命中原生命中）双向验证：``None`` 留下 ``path``，``False`` 什么都不留。
    """
    import numpy as np

    from pyqcd.einsum import _kernel_cuda, _kernel_kupl

    operands = [np.ones(shape, dtype=np.complex128) for shape in SPECIALIZE_SHAPES]

    cli._clear_kernel_info()
    auto = cli.pe_contract(SPECIALIZE_EQUATION, operands, "numpy", None)
    assert np.shape(auto) == (), np.shape(auto)
    hit_kernel_info = dict(_kernel_kupl.LAST_INFO)
    hit_cuda_info = dict(_kernel_cuda.LAST_INFO)

    cli._clear_kernel_info()
    off = cli.pe_contract(SPECIALIZE_EQUATION, operands, "numpy", False)
    assert np.shape(off) == (), np.shape(off)
    # 关掉专用路径：两个原生入口都不该留下记录（不论本机是否装了原生库）
    assert _kernel_kupl.LAST_INFO == {}, _kernel_kupl.LAST_INFO
    assert _kernel_cuda.LAST_INFO == {}, _kernel_cuda.LAST_INFO

    if hit_kernel_info or hit_cuda_info:
        # 本机有原生库：auto 必须留下 path，即上面那次断言不是空转
        assert "path" in hit_kernel_info or "path" in hit_cuda_info, (
            hit_kernel_info, hit_cuda_info)


def test_corpus_shape_inference_does_not_import_the_testing_package():
    """省略 ``--shape`` 时的形状反推**不得**依赖 ``pyqcd.testing``（连带 scipy）。

    回归点（2026-11-15 用户实测）：``shapes_from_corpus`` 曾
    ``from pyqcd.testing._topology_kernel_gen import parse``，而
    ``pyqcd/testing/__init__.py`` 顶层 ``from scipy.linalg import expm``。于是
    在没装 scipy 的环境里，一个**根本不需要 scipy** 的纯 einsum 形状反推直接
    ``ModuleNotFoundError: No module named 'scipy'``。这同时违反本库反模式
    "生产代码不 import ``pyqcd.testing``"。

    本条锁两件事：①源码里不存在该 import；②输出形状按方程输出段**真的推对**
    （原实现把它兜成 ``()``，导致非标量输出误报 ``actual shape=[100,100]
    expected=[]``）。
    """
    import inspect

    from pyqcd.einsum import test as test_module

    # 只查**代码行**，跳过 docstring 与注释（说明文字里本该能提到它）。
    import ast

    tree = ast.parse(inspect.getsource(test_module.shapes_from_corpus))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    offenders = [name for name in imported if name.startswith("pyqcd.testing")]
    assert not offenders, (
        f"shapes_from_corpus 不得 import pyqcd.testing（会连带 scipy）：{offenders}")

    # 语料内的 16 操作数方程，输出是 [e,s] 而非标量。
    equation = "cqdr,eafb,mknl,oupv,igjh,wsxt,ac,mo,gi,kq,uw,bd,fnp,hj,lrt,vx->es"
    shapes, matched, label_sizes = test_module.shapes_from_corpus(equation, "distill")
    assert matched, f"{equation} 应在语料里"
    assert len(shapes) == 16
    assert tuple(label_sizes[ch] for ch in "es") == (100, 100), (
        f"输出 [e,s] 应为 (100, 100)，得到 {label_sizes}")
    assert shapes[1] == (100, 4, 1, 1), f"第二个操作数形状不对：{shapes[1]}"


def test_corpus_shape_inference_reports_non_scalar_output_shape():
    """省略 ``--shape`` 时，非标量输出的形状必须被正确报告（不能兜成 ``[]``）。

    回归点同上：``main`` 曾把 ``out_shape`` 设成 ``None`` 再兜成 ``()``，
    于是 ``rel`` 达标却因形状不符判 FAIL。
    """
    import contextlib
    import io

    from pyqcd.einsum import test as test_module

    equation = "cqdr,eafb,mknl,oupv,igjh,wsxt,ac,mo,gi,kq,uw,bd,fnp,hj,lrt,vx->es"
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
        code = test_module.main([
            "--equation", equation, "--scene", "distill",
            "--device", "cpu", "--backend", "numpy", "--dtype", "complex128",
            "--compare", "none", "--rounds", "1", "--warmup", "1",
        ])
    output = buffer.getvalue()
    assert code == 0, f"应当通过，得到退出码 {code}：{output}"
    assert "expected=[]" not in output, f"输出形状被兜成空元组：{output}"


def test_verbose_lists_every_operand_shape():
    """``--verbose true`` 必须逐个列出操作数形状，且格式可直接回填 ``--shape``。

    回归点（2026-11-15 用户要求）：明细原先只有一行 ``operands``（总数 + 输出形状），
    排查"哪一路轴长不对"时看不到逐个形状。现在多一行 ``shape``。

    断言：①行存在；②每个操作数的形状都出现；③用 ", " 分隔且每项形如 ``[d1,d2]``，
    于是可以原样粘回 ``--shape``。
    """
    import contextlib
    import io
    import re

    from pyqcd.einsum import test as test_module

    equation = "cqdr,eafb,mknl,oupv,igjh,wsxt,ac,mo,gi,kq,uw,bd,fnp,hj,lrt,vx->es"
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        code = test_module.main([
            "--equation", equation, "--scene", "distill",
            "--device", "cpu", "--backend", "numpy", "--dtype", "complex128",
            "--compare", "none", "--rounds", "1", "--warmup", "1",
            "--verbose", "true",
        ])
    text = err.getvalue()
    assert code == 0, f"应当通过，得到 {code}：{text}"
    lines = [ln for ln in text.splitlines() if ln.startswith("shape    :")]
    assert lines, f"缺少 shape 行：{text}"

    # 形状明细在紧随其后的续行里（以空白缩进）。
    match = re.search(r"^shape    :.*\n\s+(\[.*)$", text, re.MULTILINE)
    assert match, f"缺少形状明细续行：{text}"
    rendered = match.group(1).strip()
    tokens = rendered.split(", ")
    assert len(tokens) == 16, f"应列出 16 个操作数形状，得到 {len(tokens)}：{rendered}"
    for token in tokens:
        assert re.fullmatch(r"\[\d+(,\d+)*\]", token), f"形状格式不对：{token}"
    # 第一个与第二个操作数的形状必须真的出现在里面（语料 distill 口径）。
    assert "[4,4,1,1]" in tokens and "[100,4,1,1]" in tokens, (
        f"形状内容不对：{tokens}")

    # 该明细本身必须能当 --shape 的输入段用（输出段另加）。
    parsed, _out = test_module.parse_shape_spec(rendered + "->[1]", language="en")
    assert len(parsed) == 16, f"明细无法回填 --shape：{parsed}"


def test_shape_spec_without_arrow_derives_the_output_from_the_equation():
    """``--shape`` 省略 ``->…`` 时，输出形状必须按**方程输出段**推导，不是当标量。

    回归点（2026-11-15 用户实测）：用户给了 16 个输入形状、没写输出段，方程是
    ``...->es``。旧实现一律把输出当 ``()``，于是 ``operands`` 行显示
    ``输出 [] (1 元素)``、最终报 ``actual shape=[100, 100] expected=[]``，
    而 ``rel`` 其实早已达标。

    这条同时锁"真的整式归约仍然成立"：方程 ``ij,jk->`` 配 ``[4,4],[4,4]``
    必须仍得 ``()``。
    """
    from pyqcd.einsum import test as cli

    # 1) 有输出段：从输入形状里查出 e=100、s=7。
    _shapes, out = cli.parse_shape_spec(
        "[100,4,3,5], [7,5]", equation="eabc,sc->es")
    assert out == (100, 7), f"输出应为 (100, 7)，得到 {out}"

    # 2) 真整式归约：输出段为空 → ()。
    _shapes, out = cli.parse_shape_spec("[4,4],[4,4]", equation="ij,jk->")
    assert out == (), f"整式归约的输出应为 ()，得到 {out}"

    # 3) 显式给了输出段时以它为准（不被方程覆盖）。
    _shapes, out = cli.parse_shape_spec("[4,4],[4,4]->[4,4]",
                                        equation="ij,jk->ik")
    assert out == (4, 4), out


def test_shape_spec_mismatches_are_usage_errors_not_crashes():
    """``--shape`` 与方程不符时必须是 ``UsageError``（退出码 2），不是裸 ValueError。

    ``--help`` 承诺"0 通过 / 1 数值失败 / 2 参数错误"。旧实现让
    ``parse_shape_spec`` 的 ValueError 直接冒泡，退出码变成 1，脚本无法区分
    "改命令行"与"查算法"。
    """
    from pyqcd.einsum import test as cli

    cases = [
        ("[4,4]", "ij,jk->ik"),                    # 操作数个数不符
        ("[4,4],[4,4,4]", "ij,jk->ik"),            # 秩不符
        ("[4,4],[5,4]", "ij,jk->ik"),              # 同一标签长度冲突
    ]
    for spec, equation in cases:
        try:
            cli.parse_shape_spec(spec, equation=equation)
        except ValueError as exc:
            assert not isinstance(exc, SystemExit), exc
            # main() 负责把它转成 UsageError；这里直接验 main 的退出码。
        else:
            raise AssertionError(f"{spec} vs {equation} 应当报错")
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                code = cli.main([
                    "--equation", equation, "--shape", spec,
                    "--device", "cpu", "--compare", "none",
                ])
            except SystemExit as exc:
                code = exc.code
        assert code == 2, f"{spec} vs {equation} 的退出码应为 2，得到 {code}"


def test_verbose_names_the_route_that_actually_ran():
    """``--verbose`` 必须给出**实际执行入口**，且形态提示不得与它矛盾。

    回归点（2026-11-15 用户实测）：提示行写
    ``specialized path: topology:3fd782ac [kernel: kupl_dedicated] 真实语料拓扑
    （整条计划交给原生拓扑入口；不支持时回落计划缓存）``——括号里是实际内核
    ``kupl_dedicated``，文案却说"交给拓扑入口"。原因是
    ``_specialized_kernel_attempts`` 先试 ``*_dedicated``、失败才试 ``*_topology``，
    而形态条目的 ``note`` 是按形态写死的、不承诺本次走哪条。

    本条锁两件事：①``route`` 行列出了实际 path；②形态提示不再声称"整条计划
    交给原生拓扑入口"这种与 dedicated 命中相矛盾的说法。
    """
    import contextlib
    import io

    from pyqcd.einsum import _specialize
    from pyqcd.einsum import test as cli

    # 1) 形态条目文案必须与"先 dedicated 后 topology"的实际顺序一致。
    entry = _specialize._topology_entry("0" * 32)
    note = entry.get("note", "")
    assert "拓扑入口" not in note or "dedicated" in note, (
        f"形态文案仍宣称走拓扑入口、与实际顺序矛盾：{note!r}")
    assert entry.get("name", "").startswith("topology:"), entry

    # 2) route 行必须报出实际 path。
    # ``--kernel-force dedicated`` 是必须的：``kernel_force`` 的缺省值已是 ``generic``
    # （``auto`` 已删除），那条路不匹配、不下发内核，route 只会写"通用"——这条断言
    # 就变成空断言。
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        code = cli.main([
            "--equation", "cqdr,eafb,mknl,oupv,igjh,wsxt,ac,mo,gi,kq,uw,bd,fnp,hj,lrt,vx->es",
            "--shape", "[4,4,9,9], [100,4,9,9], [4,4,9,9], [4,4,100,9], [4,4,9,9], "
                       "[4,100,9,100], [4,4], [4,4], [4,4], [4,4], [4,4], [9,9], "
                       "[9,9,100], [9,9], [9,9,100], [9,9]",
            "--device", "cpu", "--backend", "auto", "--dtype", "complex128",
            "--compare", "none", "--rounds", "1", "--warmup", "1",
            "--verbose", "true", "--specialize", "true",
            "--kernel-force", "dedicated",
        ])
    text = err.getvalue()
    assert code == 0, f"应当通过，得到 {code}：{text}"
    routes = [ln for ln in text.splitlines() if ln.startswith("route    :")]
    assert routes, f"缺少 route 行：{text}"
    # 必须点名**特化 + 原生符号**，让人一眼看出不是通用内核。
    assert "qcd_kupl_dedicated_apply" in routes[0] or \
        "qcd_kupl_topology_apply" in routes[0], f"route 应给出原生符号：{routes[0]}"
    assert "特化" in routes[0], f"route 应标明是特化内核：{routes[0]}"
    # 通用路径必须明确说"通用"，不能留空或含糊。
    from pyqcd.einsum.test import _route_of
    assert "通用" in _route_of({}), _route_of({})
    assert "generic" in _route_of({}, "en"), _route_of({}, "en")


def test_tolerance_is_atol_plus_rtol_and_rtol_is_reported():
    """判定式必须是 ``rel <= atol + rtol``，且 atol/rtol 都要进报告。

    回归点（2026-11-15 第三轮巡查实证）：原式是 ``rel <= atol + rtol * rel``，
    等价于 ``rel * (1 - rtol) <= atol``。后果有二：

    1. **``rtol >= 1`` 时左边恒 <= 0 → 判定无条件通过**（假 PASS）。实测
       complex64 语料方程 rel=1.82e-07：``--rtol 0.999`` FAIL、``--rtol 1.0`` PASS。
    2. ``rtol < 1`` 时又被无端收紧：``--rtol 0.999`` 把有效阈值压成
       ``atol / 0.001``，用户要的"99.9% 容差"根本没生效。

    本条用固定 rel 验证语义，并检查报告字段。
    """
    # **必须真正驱动 run() 的判定分支**，否则契约是空壳：第一版只在这里验证
    # 自己写的算术（`1.8e-7 <= 1e-12 + 0.5`），把公式改回 `atol + rtol*rel`
    # 契约照样通过（2026-11-15 第 2 轮复查的变异测试抓出来的）。
    #
    # 做法：monkeypatch `_relative_error` 注入一个**受控的 rel**，选在两式结论
    # 相反的区间——
    #   新式 `rel <= atol + rtol`；旧式 `rel <= atol + rtol*rel` == `rel*(1-rtol) <= atol`
    # 取 rel = 0.05、atol = 1e-9、rtol = 0.25：
    #   新式 0.05 <= 1e-9 + 0.25 = 0.25            → PASS
    #   旧式 0.05*(1-0.25)=0.0375 <= 1e-9          → FAIL
    # 两式结论必须相反，因此这条断言能钉死公式。
    import contextlib
    import io

    from pyqcd.einsum import test as cli

    injected = 0.05
    original = cli._relative_error
    cli._relative_error = lambda actual, expected: injected
    try:
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(stderr):
            code = cli.main([
                "--equation", "ij,jk->ik", "--shape", "[8,8],[8,8]->[8,8]",
                "--device", "cpu", "--dtype", "complex128",
                "--compare", "opt_einsum", "--rounds", "1", "--warmup", "1",
                "--atol", "1e-9", "--rtol", "0.25",
            ])
    finally:
        cli._relative_error = original
    assert code == 0, (
        f"rel={injected} 应满足 rel <= atol + rtol（=0.25），得到退出码 {code}；"
        f"若为 1 说明判定式退化成了 atol + rtol*rel（旧式的 bug）。"
        f"stderr={stderr.getvalue()[:200]}")

    # 第二个方向：``rtol > 1`` 时旧式**恒真**（``rel*(1-rtol)`` 是负数，恒 <= atol）。
    # 选 rel = 3.0、atol = 1e-12、rtol = 2.0：
    #   新式 3.0 <= 1e-12 + 2.0 = 2.0   → **FAIL**（正确：超出 2.0 的容差）
    #   旧式 3.0*(1-2.0) = -3.0 <= 1e-12 → PASS（假 PASS，正是被修的坑）
    # 所以"必须 FAIL"这条断言能钉死旧式。
    injected = 3.0
    cli._relative_error = lambda actual, expected: injected
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = cli.main([
                "--equation", "ij,jk->ik", "--shape", "[8,8],[8,8]->[8,8]",
                "--device", "cpu", "--dtype", "complex128",
                "--compare", "opt_einsum", "--rounds", "1", "--warmup", "1",
                "--atol", "1e-12", "--rtol", "2.0",
            ])
    finally:
        cli._relative_error = original
    assert code == 1, (
        f"rel={injected} 超过 atol+rtol=2.0，必须 FAIL；得到 {code} 说明判定式退化成"
        f" atol+rtol*rel（rtol>1 时恒真）。")

    # 报告字段：atol/rtol/tolerance 必须都在。
    import contextlib
    import io
    import json
    import tempfile

    from pyqcd.einsum import test as cli

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/r.json"
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = cli.main([
                "--equation", "ij,jk->ik", "--shape", "[8,8],[8,8]->[8,8]",
                "--device", "cpu", "--dtype", "complex128",
                "--compare", "opt_einsum", "--rounds", "1", "--warmup", "1",
                "--atol", "1e-12", "--rtol", "0.25", "--json", path,
            ])
        assert code == 0, code
        report = json.load(open(path, encoding="utf-8"))
    for key in ("atol", "rtol", "tolerance"):
        assert key in report, f"报告缺 {key}：{sorted(report)}"
    assert report["atol"] == 1e-12, report["atol"]
    assert report["rtol"] == 0.25, report["rtol"]
    assert abs(report["tolerance"] - (1e-12 + 0.25)) < 1e-18, report["tolerance"]


def test_equations_the_library_accepts_are_also_accepted_by_the_cli():
    """测试入口不得比被测库更严：对角 / 带空格 / ellipsis 三种写法都必须能跑。

    回归点（2026-11-15 第三轮巡查 A-1/A-2/C-1）：CLI 曾在三道校验上用
    ``_dedicated_kernels.parse_equation``（给原生下发的保守校验：拒绝操作数内
    重复标签、拒绝省略号、不去空格）与裸 ``split(",")``，于是

    * ``ii->i``（对角）→ rc=2，而 ``ii->i --shape "[4,4]->[4]"`` PASS；
    * ``"ij, jk -> ik"``（逗号后有空格）→ 报"秩是 4"；
    * ``...ij,...jk->...ik``（ellipsis）→ 所有写法都 rc=2。

    三者 ``contract()`` 本身全都支持（``parser.py`` 去空格 + 展开 ellipsis）。
    本条锁住"测试入口与库同口径"。
    """
    import contextlib
    import io

    from pyqcd.einsum import test as cli

    cases = [
        ("ii->i", "[4,4]", None),                       # 对角，省略输出段
        ("ii->i", "[4,4]->[4]", None),                  # 对角，显式输出段
        ("ij, jk -> ik", "[4,4],[4,4]", None),          # 逗号后空格
        ("...ij,...jk->...ik", "[2,4,4],[2,4,4]", None),  # ellipsis
    ]
    for equation, shape, _unused in cases:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                code = cli.main([
                    "--equation", equation, "--shape", shape,
                    "--device", "cpu", "--compare", "none", "--rounds", "1",
                ])
            except SystemExit as exc:
                code = exc.code
        assert code == 0, f"{equation!r} {shape!r} 应当通过，得到 {code}"

    # 反例仍必须报错（别把校验改松到什么都放行）。
    for equation, shape in [("ii->i", "[4,5]"),        # 对角要求 4==5
                            ("ij,jk->ik", "[4,4]")]:   # 操作数个数不符
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                code = cli.main([
                    "--equation", equation, "--shape", shape,
                    "--device", "cpu", "--compare", "none", "--rounds", "1",
                ])
            except SystemExit as exc:
                code = exc.code
        assert code == 2, f"{equation!r} {shape!r} 应当报错，得到 {code}"


def test_unwritable_json_path_is_a_usage_error():
    """``--json`` 写不进去必须是退出码 2（参数/环境错误），不是 1。

    回归点（2026-11-15 第六轮复查实证）：裸 ``open()`` 让
    ``FileNotFoundError`` 冒泡成 traceback + rc=1，脚本会误判成"结果不对"。
    ``--help`` 明确承诺 2 = 参数或环境错误。
    """
    import contextlib
    import io

    from pyqcd.einsum import test as cli

    stderr = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(stderr):
        try:
            code = cli.main([
                "--equation", "ij,jk->ik", "--shape", "[4,4],[4,4]->[4,4]",
                "--device", "cpu", "--compare", "none", "--rounds", "1",
                "--json", "/nonexistent_dir_xyz/x.json",
            ])
        except SystemExit as exc:
            code = exc.code
    assert code == 2, f"应为 2，得到 {code}"


def test_help_lists_specialize_and_verbose_reports_it():
    """``--help`` 是用户唯一权威来源：缺省值表必须有 ``--specialize``。

    同时 verbose 必须打印"我要什么 / 实际是什么"，否则用户无法判断选项是否生效。
    """
    text = cli.build_parser().format_help()
    assert "--specialize" in text, text
    for token in ("auto", "true", "false", "第一手段"):
        assert token in text, token

    import contextlib
    import io

    for label, expected_effective in (("false", "False"), ("true", "True")):
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            result = cli.run(equation="ij,jk->ik", shapes=[(8, 8), (8, 8)],
                             out_shape=(8, 8), compare="none", rounds=2, warmup=1,
                             specialize=(label == "true"), verbose=True)
        assert result["ok"], result
        printed = buffer.getvalue()
        assert f"specialize: {label}（实际 {expected_effective}）" in printed, printed
        assert "kernel   :" in printed, printed


# ── --compare-optimize：只作用于参照，缺省不得改既有结论 ─────────────────────
#: 三条链的方程：足够多步，``greedy`` 与 ``dp`` 会选出不同的收缩序。
COMPARE_OPT_EQUATION = "ij,jk,kl,lm,mn->in"
COMPARE_OPT_SHAPES = [(24, 24)] * 5


def _compare_opt_run(**kwargs):
    """跑一条小方程并返回报告（rounds 压到 2，本组不验性能）。"""
    return cli.run(equation=COMPARE_OPT_EQUATION, shapes=COMPARE_OPT_SHAPES,
                   out_shape=(24, 24), rounds=2, warmup=1, **kwargs)


def test_compare_optimize_default_reproduces_the_old_reference_behaviour():
    """**最重要的一条**：缺省 ``auto`` 下参照的实际优化器与改动前逐字一致。

    改动前的参照是：``numpy`` 恒 ``greedy``；``opt_einsum`` 在
    ``--optimize`` 为 auto/None 时 ``greedy``、否则跟随 ``--optimize``。
    这两条断言就是那个旧行为的拷贝——只要有人"顺手改成 dp"或让参照跟随新选项，
    本条立刻红，避免悄悄改掉所有历史对照结论。

    注意：``--compare numpy`` 这一档已从 CLI 删除（numpy 参照的等价写法是
    ``--backend numpy --compare backend``，那条路**没有**优化器可设）。旧规则仍以
    纯函数形式保留在 :func:`cli.resolve_compare_optimize` 里，供直接调用方使用，
    所以下面按"端到端只能走 opt_einsum、解析层两条都钉"来分组。
    """
    assert cli.DEFAULT_COMPARE_OPTIMIZE == "auto", cli.DEFAULT_COMPARE_OPTIMIZE
    assert set(cli.COMPARE_OPTIMIZERS) == {
        "auto", "greedy", "dp", "optimal", "branch-all", "branch-2", "branch-1",
    }, cli.COMPARE_OPTIMIZERS

    # opt_einsum 参照：auto → greedy；显式 --optimize 则跟随（旧行为逐字保留）
    report = _compare_opt_run(compare="opt_einsum", optimize="auto")
    assert report["ok"], report
    assert report["compare_optimize"] == "auto", report
    assert report["compare_optimize_effective"] == "greedy", report
    assert "optimize='greedy'" in report["reference"], report["reference"]

    report = _compare_opt_run(compare="opt_einsum", optimize="optimal")
    assert report["ok"], report
    assert report["compare_optimize_effective"] == "optimal", report
    assert "optimize='optimal'" in report["reference"], report["reference"]

    # 解析层同样逐字对照（不需要真的跑收缩）
    assert cli.resolve_compare_optimize("numpy", "auto", "auto") == "greedy"
    assert cli.resolve_compare_optimize("numpy", None, None) == "greedy"
    assert cli.resolve_compare_optimize("numpy", "auto", "optimal") == "optimal"
    assert cli.resolve_compare_optimize("opt_einsum", "auto", "auto") == "greedy"
    assert cli.resolve_compare_optimize("opt_einsum", "dp", "auto") == "dp"
    assert cli.resolve_compare_optimize("backend", "auto", "auto") is None
    assert cli.resolve_compare_optimize("none", "auto", "auto") is None


def test_compare_optimize_dp_reaches_the_opt_einsum_reference():
    """显式 ``dp`` 必须真的到达 ``oe.contract``，并写进报告与 verbose。

    ``dp`` 是 opt_einsum 的路径名、numpy 没有——所以这条只能走 opt_einsum。
    同时确认它**不再**跟随主路径的 ``--optimize auto``（那正是缺口本身）。
    """
    report = _compare_opt_run(compare="opt_einsum", optimize="auto",
                              compare_optimize="dp")
    assert report["ok"], report
    assert report["compare_optimize"] == "dp", report
    assert report["compare_optimize_effective"] == "dp", report
    assert "optimize='dp'" in report["reference"], report["reference"]
    # 参照行**必须同时报出后端**：只写 ``optimize=`` 时，GPU 上拿 numpy 参照
    # （`oe.contract` 的 backend 缺省 'auto' 按数组类型选）与主路径的 GPU 数字
    # 并列展示，读者无从发现"这两个数不同机器"（2026-11-19 用户实测）。
    assert "backend='numpy'" in report["reference"], report["reference"]
    # 主路径的字段没被这个选项改写
    assert report["optimize"] == "auto", report

    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        cli.run(equation=COMPARE_OPT_EQUATION, shapes=COMPARE_OPT_SHAPES,
                out_shape=(24, 24), rounds=2, warmup=1, verbose=True,
                compare="opt_einsum", optimize="auto", compare_optimize="dp")
    printed = buffer.getvalue()
    # verbose 的 reference 行打的是**参照实际用的**优化器
    assert "reference: opt_einsum.contract(backend='numpy', optimize='dp')" in printed, printed
    assert "ref opt  : 要 dp" in printed, printed


def test_compare_optimize_rejects_names_the_reference_cannot_honour():
    """不支持的名字必须报 ``UsageError``（退出码 2），**不做静默回退**。

    numpy 侧实测只认 ``greedy``/``optimal``：源码里就这两个分支，其余抛
    ``KeyError('Path name %s not found', ...)``。把 ``dp`` 硬映射成 ``greedy``
    会让"对照实验"给出一个完全不相关的数字，比报错危险得多。

    ``--compare numpy`` 已从 CLI 删除，所以"numpy 拒绝 dp"只能在**解析层**钉
    （:func:`cli.resolve_compare_optimize` 仍保留那条旧规则）；端到端的用法错误
    走 ``opt_einsum``。
    """
    # 解析层：numpy 参照的旧规则照旧拒绝 dp/branch-*（纯函数，无 CLI 档位也成立）
    for bad in ("dp", "branch-all", "branch-2", "branch-1"):
        try:
            cli.resolve_compare_optimize("numpy", "auto", bad)
        except cli.UsageError as exc:
            assert exc.code == 2, exc.code
            assert "numpy" in str(exc), exc
        else:
            raise AssertionError(f"numpy 参照本应拒绝 {bad!r}")

    # 完全不在枚举里的值，端到端与解析层都要拒绝
    for bad in ("bogus", "auto-hq", ""):
        try:
            _compare_opt_run(compare="opt_einsum", compare_optimize=bad)
        except cli.UsageError as exc:
            assert exc.code == 2, exc.code
        else:
            raise AssertionError(f"opt_einsum 参照本应拒绝 {bad!r}")
        try:
            cli.resolve_compare_optimize("numpy", "auto", bad)
        except cli.UsageError as exc:
            assert exc.code == 2, exc.code
        else:
            raise AssertionError(f"解析层的 numpy 参照本应拒绝 {bad!r}")

    # numpy 支持的两个名字在解析层放行；端到端用 opt_einsum 验证同名取值真的到位
    assert set(cli.NUMPY_COMPARE_OPTIMIZERS) == {"greedy", "optimal"}
    for good in cli.NUMPY_COMPARE_OPTIMIZERS:
        assert cli.resolve_compare_optimize("numpy", "auto", good) == good
        report = _compare_opt_run(compare="opt_einsum", compare_optimize=good)
        assert report["ok"], report
        assert report["compare_optimize_effective"] == good, report
        assert f"optimize='{good}'" in report["reference"], report["reference"]


def test_compare_optimize_does_not_touch_the_main_path():
    """该选项**只影响参照**：同一方程在 greedy 与 dp 下主路径的计划不变。

    断言主路径的 ``optimize`` 字段与 ``kernel_info``/``actual_shape`` 完全一致——
    主路径的收缩序由 ``--optimize`` 管，参照优化器改了也不该动它。**不做任何耗时
    断言**：``--rounds 2`` 的噪声（实测同一对调用的比值可到 4.01）远大于本改动能的
    差异，性能数字不是正确性判据。
    """
    runs = [_compare_opt_run(compare="opt_einsum", optimize="optimal",
                             compare_optimize=value)
            for value in ("greedy", "dp")]
    first, second = runs
    for item in runs:
        assert item["ok"], item

    assert first["optimize"] == second["optimize"] == "optimal", runs
    assert first["actual_shape"] == second["actual_shape"] == [24, 24], runs
    assert first["kernel_info"] == second["kernel_info"], runs
    assert first["specialize_effective"] == second["specialize_effective"], runs
    # 参照侧确实换了：这条与上一条一起说明"变的是参照，不是主路径"
    assert first["compare_optimize_effective"] == "greedy", first
    assert second["compare_optimize_effective"] == "dp", second
    # 数值必须一致（换的只是收缩序）
    assert abs(first["rel"] - second["rel"]) < 1e-10, (first["rel"], second["rel"])

    # 刻意**不**断言两次的耗时比值：参照侧的优化器只改收缩序，主路径逐字相同，
    # 而 ``--rounds 2`` 的计时噪声（实测同一对调用比值在 1.0–4.01 之间抖动）远大于
    # 本改动能带来的差异。性能数字不做 CI 判据——上面五条相等性断言才是"只动参照"
    # 的证据。


def test_compare_optimize_is_inert_for_none():
    """``none`` 没有可设的优化器：非 auto 值不报错，只标无效。

    报告里 ``compare_optimize_effective`` 为 ``None``，且不产生 reference。
    """
    report = _compare_opt_run(compare="none", compare_optimize="dp")
    assert report["ok"], report
    assert report["compare_optimize"] == "dp", report
    assert report["compare_optimize_effective"] is None, report
    assert "reference" not in report, report

    auto = _compare_opt_run(compare="none", compare_optimize="auto")
    assert auto["ok"], auto
    assert auto["compare_optimize_effective"] is None, auto

    for value in cli.COMPARE_OPTIMIZERS:
        assert cli.resolve_compare_optimize("none", "auto", value) is None, value


def test_compare_optimize_is_inert_for_backend():
    """``backend`` needs a usable second backend; the optimizer stays inert."""
    try:
        import cupy
        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            raise unittest.SkipTest("no CUDA device is visible")
    except ImportError as error:
        raise unittest.SkipTest(f"cupy unavailable: {error}") from error
    except unittest.SkipTest:
        raise
    except Exception as error:  # noqa: BLE001 - only known CUDA runtime error skips
        if type(error).__name__ == "CUDARuntimeError":
            raise unittest.SkipTest(f"CUDA runtime unavailable: {error}") from error
        raise

    for value in ("dp", "auto"):
        report = _compare_opt_run(compare="backend", compare_optimize=value)
        assert report["ok"], report
        assert report["compare_optimize"] == value, report
        assert report["compare_optimize_effective"] is None, report
        assert "optimize" not in report["reference"], report["reference"]

    for value in cli.COMPARE_OPTIMIZERS:
        assert cli.resolve_compare_optimize("backend", "auto", value) is None, value


def test_help_lists_compare_optimize():
    """``--help`` 是唯一权威来源：选项名、全部取值与缺省值都必须出现。"""
    text = cli.build_parser().format_help()
    assert "--compare-optimize" in text, text
    for value in cli.COMPARE_OPTIMIZERS:
        assert value in text, value
    # 缺省值要与 DEFAULT_COMPARE_OPTIMIZE 一致地出现在 help 文本里
    assert f"缺省 {cli.DEFAULT_COMPARE_OPTIMIZE}" in text, text[-1500:]
    # 必须写明它只管参照，否则用户会以为它改主路径
    assert "只影响参照" in text, text
    # 必须写明不跨库重映射（这条是本选项最容易被误解的地方）
    assert "重映射" in text, text


# ── --language：只翻译、不改行为 ────────────────────────────────────────────
#: 中日韩统一表意文字：``--language en`` 的 help 里一个都不许出现。
CJK = re.compile(r"[\u4e00-\u9fff]")


def _help_text(language: str) -> str:
    return cli.build_parser(language=language).format_help()


def test_language_modes_and_default_are_documented():
    """``--language`` 的取值与缺省必须是规范端的一部分。

    缺省 ``auto`` **不等于** ``notice_language()``：它只在 ``PYQCD_EINSUM_LANG``
    显式设了时才跟随，未设时保持本入口的历史行为（中文）。直接跟随 locale 会让
    ``LANG=C.UTF-8`` 的机器上不传选项的输出从中文变英文——那是无声的破坏。
    """
    assert cli.DEFAULT_LANGUAGE == "auto", cli.DEFAULT_LANGUAGE
    assert set(cli.LANGUAGE_MODES) == {"auto", "en", "zh-cn"}, cli.LANGUAGE_MODES
    assert cli.LANGUAGE_ENV == "PYQCD_EINSUM_LANG", cli.LANGUAGE_ENV

    text = _help_text("zh")
    assert "--language" in text, text
    for value in cli.LANGUAGE_MODES:
        assert value in text, value
    assert cli.LANGUAGE_ENV in text, text
    # 必须写明只改输出、不写环境变量
    assert "环境变量" in text, text


def test_language_help_is_localised():
    """``--language en`` 的 help 里不许有中文；``zh-cn`` 必须有。"""
    english = _help_text("en")
    assert not CJK.search(english), CJK.findall(english)[:20]
    # 英文 help 仍要保留规范端身份与全部选项（翻译不是删条款）
    assert "only sanctioned entry point" in english, english[:400]
    for token in ("--equation", "--shape", "--scene", "--device", "--dtype",
                  "--backend", "--compare", "--compare-optimize", "--specialize",
                  "--optimize", "--atol", "--rtol", "--rounds", "--warmup",
                  "--seed", "--verbose", "--json", "--language"):
        assert token in english, token

    chinese = _help_text("zh")
    assert CJK.search(chinese), chinese[:400]
    assert "唯一合法接口" in chinese, chinese[:400]
    # 不传语言时仍是中文（缺省行为零变化，见上一条的 docstring）
    assert cli.build_parser().format_help() == chinese


def test_language_resolution_follows_the_env_var_only_when_auto():
    """``resolve_language``：显式值确定；``auto`` 只看 ``PYQCD_EINSUM_LANG``。

    刻意**不**跟随 locale：本机的 ``LANG`` 是 ``C.UTF-8``（``notice_language()``
    返回 ``en``），若 ``auto`` 跟随 locale，缺省输出就会从中文变英文。
    """
    import os

    saved = os.environ.pop(cli.LANGUAGE_ENV, None)
    try:
        assert cli.resolve_language(None) == "zh"
        assert cli.resolve_language("auto") == "zh"
        assert cli.resolve_language("en") == "en"
        assert cli.resolve_language("zh-cn") == "zh"
        assert cli.resolve_language("zh") == "zh"     # 与 notice_language 同口径
        assert cli.resolve_language("EN") == "en"

        os.environ[cli.LANGUAGE_ENV] = "en"
        assert cli.resolve_language("auto") == "en"
        os.environ[cli.LANGUAGE_ENV] = "zh"
        assert cli.resolve_language("auto") == "zh"
        # 显式选项**压过**环境变量
        os.environ[cli.LANGUAGE_ENV] = "zh"
        assert cli.resolve_language("en") == "en"
        os.environ[cli.LANGUAGE_ENV] = "en"
        assert cli.resolve_language("zh-cn") == "zh"
    finally:
        os.environ.pop(cli.LANGUAGE_ENV, None)
        if saved is not None:
            os.environ[cli.LANGUAGE_ENV] = saved

    for bad in ("fr", "zh-tw", "jp", "zh_cn"):
        try:
            cli.resolve_language(bad)
        except cli.UsageError as exc:
            assert exc.code == 2, exc.code
        else:
            raise AssertionError(f"{bad!r} 本应被拒绝")


def test_language_rejects_unknown_values_with_exit_code_two():
    """非法 ``--language`` 必须退出码 2（用法错误），不是数值失败。

    走 argparse 的 ``choices``（``--help`` 里列的就是这三个），错误信息可见；
    同时 ``--language fr --help`` 仍要打印 help 并以 2 退出，不能被提前吞掉。
    """
    import contextlib
    import io

    for argv in (["--equation", "ij,jk->ik", "--shape", "[4,4],[4,4]->[4,4]",
                  "--language", "fr"],
                 ["--equation", "ij,jk->ik", "--shape", "[4,4],[4,4]->[4,4]",
                  "--language=zh-tw"]):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            try:
                cli.main(argv)
            except SystemExit as exc:
                assert exc.code == 2, exc.code
            else:
                raise AssertionError(f"{argv[-1]!r} 本应被拒绝")
        assert "--language" in errors.getvalue(), errors.getvalue()

    # ``--language fr --help`` 必须仍然失败（2），不能被 --help 短路成 0
    errors = io.StringIO()
    with contextlib.redirect_stderr(errors), contextlib.redirect_stdout(io.StringIO()):
        try:
            cli.main(["--language", "fr", "--help"])
        except SystemExit as exc:
            assert exc.code == 2, exc.code
        else:
            raise AssertionError("--language fr --help 本应以 2 退出")
    assert "--language" in errors.getvalue(), errors.getvalue()

    # 直接调 run 也必须按语言报错（run 是外部脚本的入口，绕过 argparse）
    try:
        cli.run(equation="ij,jk->ik", shapes=[(2, 2), (2, 2)], out_shape=(2, 2),
                compare="none", language="fr")
    except cli.UsageError as exc:
        assert exc.code == 2, exc.code
    else:
        raise AssertionError("run(language='fr') 本应被拒绝")


def test_language_does_not_change_results():
    """**核心保险丝**：同一个方程在 en 与 zh-cn 下的数值结果必须完全一致。

    翻译只该改文案。若有人顺手在某一分支里改了容差、参照或收缩序，本条立刻红。
    """
    equation = "ij,jk,kl->il"
    shapes = [(32, 32), (32, 32), (32, 32)]
    runs = [cli.run(equation=equation, shapes=shapes, out_shape=(32, 32),
                    compare="opt_einsum", rounds=2, warmup=1, seed=7, language=lang)
            for lang in ("en", "zh-cn")]
    english, chinese = runs
    for item in runs:
        assert item["ok"], item
    assert english["rel"] == chinese["rel"], (english["rel"], chinese["rel"])
    assert english["ok"] == chinese["ok"], runs
    assert english["actual_shape"] == chinese["actual_shape"], runs
    assert english["ok"] is True, english

    # 报告里的机器可读字段完全一致；只有文案字段允许不同
    for field in ("equation", "shapes", "out_shape", "device", "dtype", "backend",
                  "compare", "atol", "rounds", "warmup", "optimize",
                  "compare_optimize", "compare_optimize_effective", "specialize",
                  "specialize_effective", "n_operands", "input_elements",
                  "output_elements", "kernel_info", "actual_shape"):
        assert english[field] == chinese[field], (field, english[field],
                                                  chinese[field])


def test_language_only_changes_the_verbose_wording():
    """``--verbose true`` 的键名不变，值按语言翻译；数值不变。"""
    import contextlib
    import io

    reports = {}
    for lang in ("en", "zh-cn"):
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            reports[lang] = cli.run(equation="ij,jk->ik",
                                    shapes=[(8, 8), (8, 8)], out_shape=(8, 8),
                                    compare="opt_einsum", rounds=2, warmup=1,
                                    verbose=True, language=lang)
        reports[lang]["_printed"] = buffer.getvalue()

    english, chinese = reports["en"], reports["zh-cn"]
    for lang, report in reports.items():
        text = report["_printed"]
        # 键名是稳定的机器可读标签，两种语言都必须保留
        for key in ("equation", "device", "operands", "specialize", "time",
                    "ref opt", "reference", "ref time", "rel", "memory", "peak",
                    "pyqcd", "kernel"):
            assert f"{key}" in text, (lang, key, text)
        assert "PASS" not in text, text          # 结果行在 stdout，不在明细里

    en_text = english["_printed"]
    zh_text = chinese["_printed"]
    assert not CJK.search(en_text), CJK.findall(en_text)[:20]
    assert CJK.search(zh_text), zh_text[:400]
    # 值是翻译过的，不是同一串
    assert "operand(s)" in en_text, en_text
    assert "个，共" in zh_text, zh_text
    assert "no native hit (generic path)" in en_text or "kernel   : {" in en_text, \
        en_text
    # 数值一致
    assert english["rel"] == chinese["rel"], (english["rel"], chinese["rel"])
    assert english["n_operands"] == chinese["n_operands"] == 2, reports


def test_language_does_not_pollute_the_environment():
    """``--language`` 只影响本次输出，**不写** ``PYQCD_EINSUM_LANG``。

    若它去写环境变量，同进程后续调用（包括库里的 ``notice_language()``）会被
    悄悄改掉语言，而调用方根本无法预期。
    """
    import os

    saved = os.environ.pop(cli.LANGUAGE_ENV, None)
    try:
        before = dict(os.environ)
        for argv in (["--equation", "ij,jk->ik", "--shape", "[4,4],[4,4]->[4,4]",
                      "--language", "en", "--compare", "none"],
                     ["--equation", "ij,jk->ik", "--shape", "[4,4],[4,4]->[4,4]",
                      "--compare", "none"]):
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                assert cli.main(argv) == 0, argv
            assert cli.LANGUAGE_ENV not in os.environ, sorted(os.environ)
            assert dict(os.environ) == before, "main() 改了环境变量"
    finally:
        os.environ.pop(cli.LANGUAGE_ENV, None)
        if saved is not None:
            os.environ[cli.LANGUAGE_ENV] = saved


def test_translate_fails_loudly_on_missing_keys_and_variants():
    """``translate`` 缺 key / 缺语言必须抛 KeyError，**不静默返回 key 本身**。

    静默兜底会让漏翻的字符串以"看似正常的英文"混进输出里，比直接崩更难发现。
    同时钉住"每条文案都有 en/zh 两份"，否则某一语言下必然漏字。
    """
    for key, variants in cli.MESSAGES.items():
        assert set(variants) == {"en", "zh"}, (key, sorted(variants))
        for lang in ("en", "zh"):
            assert variants[lang].strip(), (key, lang)

    try:
        cli.translate("no.such.key", "en")
    except KeyError as exc:
        assert "no.such.key" in str(exc), exc
    else:
        raise AssertionError("缺 key 本应抛 KeyError")

    try:
        cli.translate("help.equation", "fr")
    except KeyError as exc:
        assert "fr" in str(exc), exc
    else:
        raise AssertionError("缺语言变体本应抛 KeyError")

    # ``zh-cn``/``zh`` 与 ``en`` 都是接受的别名，不该被当成未知语言
    assert cli.translate("detail.kernel_none", "zh-cn") == \
        cli.translate("detail.kernel_none", "zh")
    assert cli.translate("detail.kernel_none", "EN") == \
        cli.translate("detail.kernel_none", "en")


# ── --equivalence：等价写法报告（只影响报告，不改执行） ──────────────────────
#: 未命中语料、但只差第 0 个操作数的轴序即可落到 seq=1 上的方程。
#: 语料里是 ``cadb,menf,ac,em,bd,fn->``；这里把前两位 ``c``/``a`` 对调。
EQUIV_AXES_EQUATION = "adcb,menf,ac,em,bd,fn->"
EQUIV_AXES_SHAPES = [(4, 1, 4, 1), (4, 4, 1, 1), (4, 4), (4, 4), (1, 1), (1, 1)]

#: 未命中语料、只差**输出段**轴序的方程（对应 ``->MNag``，本处写 ``->aMNg``）。
#: 这一族必须验证**输出侧**回填，否则用户改完输入会拿到轴序错了的答案。
EQUIV_OUTPUT_EQUATION = "ambn,codp,egfh,ce,mo,Mbdf,Nnph->aMNg"
EQUIV_OUTPUT_SHAPES = [(2, 2, 2, 2), (2, 2, 2, 2), (2, 2, 2, 2), (2, 2), (2, 2),
                       (2, 2, 2, 2), (2, 2, 2, 2)]
EQUIV_OUTPUT_SHAPE = (2, 2, 2, 2)

#: 语料命中的方程（seq=1）。它**不该**出现等价写法那一段。
EQUIV_HIT_EQUATION = "cadb,menf,ac,em,bd,fn->"
EQUIV_HIT_SHAPES = [(4, 4, 1, 1), (4, 4, 1, 1), (4, 4), (4, 4), (1, 1), (1, 1)]


def _equivalence_env():
    import os

    return os, cli.EQUIVALENCE_ENV


def _verbose_text(equation, shapes, out_shape, **kwargs):
    """跑一次 ``main``，返回 (退出码, 明细文本)。明细在 stderr。"""
    import contextlib
    import io

    argv = ["--equation", equation,
            "--shape", ",".join("[" + ",".join(str(d) for d in shape) + "]"
                                for shape in shapes)
                       + "->[" + ",".join(str(d) for d in out_shape) + "]",
            "--device", "cpu", "--backend", "numpy",
            "--compare", "none", "--rounds", "1", "--warmup", "1",
            "--verbose", "true"]
    argv += kwargs.pop("extra", [])
    buffer = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(buffer):
        code = cli.main(argv)
    return code, buffer.getvalue()


def test_equivalence_reports_both_the_input_and_the_output_side():
    """``--equivalence true``：明细必须给出等价写法，且**输入与输出两侧都在**。

    用户的原话是"转换输入的同时也有对应变换输出，正确性必须确保"——只报"怎么改
    张量"而不报"结果怎么回填"，照着做会得到一个轴序错了的答案。所以两侧都要检查：

    * ``operand-axes`` 族：``张量：transpose(...), -, ...`` + ``结果：-``（输出不变）
    * ``output-axes`` 族：``张量：-`` + ``结果：transpose(...)``（输入不变）
    """
    code, text = _verbose_text(EQUIV_AXES_EQUATION, EQUIV_AXES_SHAPES, (),
                               extra=["--equivalence", "true"])
    assert code == 0, f"应当通过，得到 {code}：{text}"
    equiv_lines = [line for line in text.splitlines() if line.startswith("equiv    :")]
    assert equiv_lines, f"缺少 equiv 行：{text}"
    assert "存在等价写法" in equiv_lines[0], equiv_lines
    assert "operand-axes" in equiv_lines[0], equiv_lines
    # 改写后的方程必须点名语料里的写法
    assert "cadb,menf,ac,em,bd,fn->" in text, text
    # 输入侧：第 0 个操作数转置，其余用 '-' 占位
    assert "transpose(2, 0, 1, 3)" in text, text
    assert "张量：" in text and "-" in text, text
    # 输出侧必须在场（该族为空时明确写 '-'，不能整行省略）
    assert "结果：" in text, text
    assert "逐元素相同" in text, text

    code, text = _verbose_text(EQUIV_OUTPUT_EQUATION, EQUIV_OUTPUT_SHAPES,
                               EQUIV_OUTPUT_SHAPE,
                               extra=["--equivalence", "true"])
    assert code == 0, f"应当通过，得到 {code}：{text}"
    assert "output-axes" in text, text
    assert "张量：-" in text, text
    # 输出侧的真实置换必须出现——这是"正确性"的一半
    assert "结果：transpose(2, 0, 1, 3)" in text, text
    # 该置换必须能用 numpy 语义复现原方程的结果（不只是"看起来有"）
    operands = [np.ones(shape, dtype=np.complex128)
                for shape in EQUIV_OUTPUT_SHAPES]
    direct = np.einsum("ambn,codp,egfh,ce,mo,Mbdf,Nnph->aMNg", *operands)
    rewritten = np.einsum("ambn,codp,egfh,ce,mo,Mbdf,Nnph->MNag", *operands)
    assert np.array_equal(direct, rewritten.transpose((2, 0, 1, 3))), (
        "报出来的输出置换与本模块的语义不一致")


def test_equivalence_false_removes_the_section_and_the_json_field():
    """``--equivalence false``：明细里没有该段，``--json`` 里没有该字段。"""
    import json
    import tempfile

    code, text = _verbose_text(EQUIV_AXES_EQUATION, EQUIV_AXES_SHAPES, (),
                               extra=["--equivalence", "false"])
    assert code == 0, f"应当通过，得到 {code}：{text}"
    assert not [line for line in text.splitlines()
                if line.startswith("equiv    :")], text
    assert "存在等价写法" not in text, text

    with tempfile.TemporaryDirectory() as folder:
        for label, flag, present in (("off", "false", False), ("on", "true", True)):
            path = f"{folder}/{label}.json"
            code, text = _verbose_text(EQUIV_AXES_EQUATION, EQUIV_AXES_SHAPES, (),
                                       extra=["--equivalence", flag,
                                              "--json", path])
            assert code == 0, f"应当通过，得到 {code}：{text}"
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
            assert ("equivalence" in report) is present, (label, sorted(report))
            if present:
                payload = report["equivalence"]
                # 结构化字段是脚本化流程要的东西，名字必须稳定
                for field in ("equation", "rewritten", "kind", "operand_order",
                              "operand_permutations", "output_permutation",
                              "description"):
                    assert field in payload, (field, sorted(payload))
                assert payload["kind"] == "operand-axes", payload
                assert payload["operand_permutations"][0] == [2, 0, 1, 3], payload


def test_equivalence_auto_follows_the_environment_variable_only():
    """缺省 ``auto`` 只跟随 ``PYQCD_EINSUM_EQUIVALENCE``；``0`` 等同 ``false``。

    与 ``--specialize``/``--language`` 同口径：``auto`` 是 ``None``（跟随环境变量），
    不是 ``True``——写成 ``True`` 会把用户用环境变量关掉的报告强行打开。
    """
    os, env_name = _equivalence_env()
    assert cli.DEFAULT_EQUIVALENCE == "auto", cli.DEFAULT_EQUIVALENCE
    assert set(cli.EQUIVALENCE_MODES) == {"auto", "true", "false"}, cli.EQUIVALENCE_MODES
    assert env_name == "PYQCD_EINSUM_EQUIVALENCE", env_name

    assert cli.parse_equivalence("auto") is None
    assert cli.parse_equivalence(None) is None
    assert cli.parse_equivalence("true") is True
    assert cli.parse_equivalence("TRUE") is True
    assert cli.parse_equivalence("off") is False
    for bad in ("bogus", "equivalence", 1.5):
        try:
            cli.parse_equivalence(bad)
        except cli.UsageError as exc:
            assert exc.code == 2, exc.code
        else:
            raise AssertionError(f"{bad!r} 本应被拒绝")

    saved = os.environ.pop(env_name, None)
    try:
        # 未设 → 开启（缺省行为）
        assert cli.equivalence_enabled(None) is True
        # 显式选项压过环境变量
        os.environ[env_name] = "0"
        assert cli.equivalence_enabled(None) is False
        assert cli.equivalence_enabled(True) is True
        os.environ[env_name] = "1"
        assert cli.equivalence_enabled(None) is True
        assert cli.equivalence_enabled(False) is False

        # 端到端：环境变量 0 时明细里没有该段（缺省 auto）
        os.environ[env_name] = "0"
        code, text = _verbose_text(EQUIV_AXES_EQUATION, EQUIV_AXES_SHAPES, ())
        assert code == 0, text
        assert "存在等价写法" not in text, text
        # 显式 true 仍然压过环境变量
        code, text = _verbose_text(EQUIV_AXES_EQUATION, EQUIV_AXES_SHAPES, (),
                                   extra=["--equivalence", "true"])
        assert code == 0, text
        assert "存在等价写法" in text, text
    finally:
        os.environ.pop(env_name, None)
        if saved is not None:
            os.environ[env_name] = saved


def test_equivalence_never_changes_numbers_shapes_or_route():
    """**核心保险丝**：该选项只加报告，``rel``/``out_shape``/``ok`` 必须完全相同。

    它若被顺手接进 ``contract()``（例如"自动改用等价方程再算"），耗时会变、路径
    会变，而且一旦改写有误数值也会变——本条只断言**数值与形状**，不断言耗时具体值
    （耗时本来就只要求"可差"，不承诺相等）。
    """
    shapes = [(8, 8), (8, 8)]
    equation = "ij,jk->ik"
    for equation_, shapes_, out_shape in (
            ("ij,jk->ik", shapes, (8, 8)),
            (EQUIV_AXES_EQUATION, EQUIV_AXES_SHAPES, ()),
            (EQUIV_OUTPUT_EQUATION, EQUIV_OUTPUT_SHAPES, EQUIV_OUTPUT_SHAPE)):
        runs = {flag: cli.run(equation=equation_, shapes=shapes_, out_shape=out_shape,
                             compare="opt_einsum", rounds=2, warmup=1, seed=5,
                             equivalence=flag)
                for flag in (True, False)}
        on, off = runs[True], runs[False]
        for report in runs.values():
            assert report["ok"], report
        for field in ("rel", "out_shape", "ok", "actual_shape", "max_abs_diff",
                      "kernel_info", "specialize_effective", "n_operands"):
            assert on[field] == off[field], (equation_, field, on[field], off[field])
        # 只有报告字段不同
        assert "equivalence" in on, (equation_, sorted(on))
        assert "equivalence" not in off, (equation_, sorted(off))


def test_equivalence_section_is_absent_when_the_equation_already_hits():
    """已命中特化路径时**不出**该段：不干扰正常输出。"""
    code, text = _verbose_text(EQUIV_HIT_EQUATION, EQUIV_HIT_SHAPES, (),
                               extra=["--equivalence", "true"])
    assert code == 0, f"应当通过，得到 {code}：{text}"
    assert not [line for line in text.splitlines()
                if line.startswith("equiv    :")], text
    assert "存在等价写法" not in text, text

    report = cli.run(equation=EQUIV_HIT_EQUATION, shapes=EQUIV_HIT_SHAPES,
                     out_shape=(), compare="none", rounds=2, warmup=1,
                     equivalence=True)
    assert report["ok"], report
    assert "equivalence" not in report, sorted(report)


def test_equivalence_says_none_when_no_equivalence_exists():
    """没有任何等价写法时，明细必须**明说无**，而不是悄悄不打印。"""
    code, text = _verbose_text("ij,jk->ik", [(4, 4), (4, 4)], (4, 4),
                               extra=["--equivalence", "true"])
    assert code == 0, f"应当通过，得到 {code}：{text}"
    lines = [line for line in text.splitlines() if line.startswith("equiv    :")]
    assert lines, f"缺少 equiv 行：{text}"
    assert "无" in lines[0] and "等价写法" in lines[0], lines
    assert "transpose" not in lines[0], lines


def test_help_lists_equivalence():
    """``--help`` 是唯一权威来源：选项名、取值、环境变量与缺省值都要出现。"""
    for language in ("zh", "en"):
        text = _help_text(language)
        assert "--equivalence" in text, language
        for value in cli.EQUIVALENCE_MODES:
            assert value in text, (language, value)
        assert cli.EQUIVALENCE_ENV in text, language
        # 必须写明"只做报告"
        assert "只做报告" in text or "Report only" in text, language
    # 中文 epilog 也要提到它
    assert cli.EQUIVALENCE_ENV in _help_text("zh"), _help_text("zh")


TESTS = (
    test_shape_spec_round_trip,
    test_shape_spec_scalar_and_reduction,
    test_shape_spec_rejects_garbage,
    test_shape_spec_accepts_any_extents_but_validation_catches_label_clash,
    test_equation_shape_count_mismatch_is_rejected,
    test_equation_label_extent_mismatch_is_rejected,
    test_equation_requires_explicit_output,
    test_warmup_zero_is_allowed_but_negative_is_refused,
    test_atol_is_relative_not_elementwise,
    test_timed_reports_the_minimum_sample,
    test_defaults_are_documented_values,
    test_exit_codes_distinguish_usage_from_numeric_failure,
    test_memory_snapshot_is_outside_the_timed_region,
    test_reference_seconds_are_reported,
    test_single_entry_point_policy_is_declared_everywhere,
    test_help_lists_every_option_with_defaults,
    test_run_end_to_end_on_cpu,
    test_specialize_tri_state_maps_to_contract_argument,
    test_specialize_false_really_disables_the_native_path,
    test_specialize_is_forwarded_to_the_compare_backend_reference,
    test_corpus_shape_inference_does_not_import_the_testing_package,
    test_corpus_shape_inference_reports_non_scalar_output_shape,
    test_verbose_lists_every_operand_shape,
    test_shape_spec_without_arrow_derives_the_output_from_the_equation,
    test_shape_spec_mismatches_are_usage_errors_not_crashes,
    test_verbose_names_the_route_that_actually_ran,
    test_tolerance_is_atol_plus_rtol_and_rtol_is_reported,
    test_equations_the_library_accepts_are_also_accepted_by_the_cli,
    test_unwritable_json_path_is_a_usage_error,
    test_help_lists_specialize_and_verbose_reports_it,
    test_compare_optimize_default_reproduces_the_old_reference_behaviour,
    test_compare_optimize_dp_reaches_the_opt_einsum_reference,
    test_compare_optimize_rejects_names_the_reference_cannot_honour,
    test_compare_optimize_does_not_touch_the_main_path,
    test_compare_optimize_is_inert_for_none,
    test_compare_optimize_is_inert_for_backend,
    test_help_lists_compare_optimize,
    test_language_modes_and_default_are_documented,
    test_language_help_is_localised,
    test_language_resolution_follows_the_env_var_only_when_auto,
    test_language_rejects_unknown_values_with_exit_code_two,
    test_language_does_not_change_results,
    test_language_only_changes_the_verbose_wording,
    test_language_does_not_pollute_the_environment,
    test_translate_fails_loudly_on_missing_keys_and_variants,
    test_equivalence_reports_both_the_input_and_the_output_side,
    test_equivalence_false_removes_the_section_and_the_json_field,
    test_equivalence_auto_follows_the_environment_variable_only,
    test_equivalence_never_changes_numbers_shapes_or_route,
    test_equivalence_section_is_absent_when_the_equation_already_hits,
    test_equivalence_says_none_when_no_equivalence_exists,
    test_help_lists_equivalence,
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
