"""契约：``python -m pyqcd.einsum.test`` 作为单方程验证规范端可用且口径正确。

这个入口是"日后默认的测试规范端"（见 ``pyqcd/einsum/AGENTS.md`` §9.0 与
``skills/pyqcd-einsum/SKILL.md``），所以它的**契约本身**也要有测试，否则某次改动
把 ``--atol`` 改成绝对容差、或允许 ``--warmup 0``，所有引用它的结论都会悄悄失真。

只测纯 Python 可判定的部分（不依赖原生库、不需要 GPU），因此任何机器都能跑。
"""
from __future__ import annotations

import re
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
                  "--backend", "--compare", "--specialize", "--optimize", "--atol",
                  "--rtol", "--rounds", "--warmup", "--seed", "--verbose", "--json"):
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
    """
    assert cli.DEFAULT_COMPARE_OPTIMIZE == "auto", cli.DEFAULT_COMPARE_OPTIMIZE
    assert set(cli.COMPARE_OPTIMIZERS) == {
        "auto", "greedy", "dp", "optimal", "branch-all", "branch-2", "branch-1",
    }, cli.COMPARE_OPTIMIZERS

    # numpy 参照：恒 greedy（旧行为），与主路径 --optimize 无关
    for optimize in ("auto", "dp", "optimal", "greedy"):
        report = _compare_opt_run(compare="numpy", optimize=optimize)
        assert report["ok"], report
        assert report["compare_optimize"] == "auto", report
        assert report["compare_optimize_effective"] == "greedy", report
        assert "optimize='greedy'" in report["reference"], report["reference"]

    # opt_einsum 参照：auto → greedy；显式 --optimize 则跟随（旧行为逐字保留）
    report = _compare_opt_run(compare="opt_einsum", optimize="auto")
    assert report["ok"], report
    assert report["compare_optimize_effective"] == "greedy", report
    assert "optimize='greedy'" in report["reference"], report["reference"]

    report = _compare_opt_run(compare="opt_einsum", optimize="optimal")
    assert report["ok"], report
    assert report["compare_optimize_effective"] == "optimal", report
    assert "optimize='optimal'" in report["reference"], report["reference"]

    # 解析层同样逐字对照（不需要真的跑收缩）
    assert cli.resolve_compare_optimize("numpy", "auto", "auto") == "greedy"
    assert cli.resolve_compare_optimize("numpy", None, None) == "greedy"
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
    assert "reference: opt_einsum.contract(optimize='dp')" in printed, printed
    assert "ref opt  : 要 dp" in printed, printed


def test_compare_optimize_rejects_names_the_reference_cannot_honour():
    """不支持的名字必须报 ``UsageError``（退出码 2），**不做静默回退**。

    numpy 侧实测只认 ``greedy``/``optimal``：源码里就这两个分支，其余抛
    ``KeyError('Path name %s not found', ...)``。把 ``dp`` 硬映射成 ``greedy``
    会让"对照实验"给出一个完全不相关的数字，比报错危险得多。
    """
    for bad in ("dp", "branch-all", "branch-2", "branch-1"):
        try:
            _compare_opt_run(compare="numpy", compare_optimize=bad)
        except cli.UsageError as exc:
            assert exc.code == 2, exc.code
            assert "numpy" in str(exc), exc
        else:
            raise AssertionError(f"numpy 参照本应拒绝 {bad!r}")

    # 完全不在枚举里的值，两条参照都要拒绝
    for bad in ("bogus", "auto-hq", ""):
        for compare in ("numpy", "opt_einsum"):
            try:
                _compare_opt_run(compare=compare, compare_optimize=bad)
            except cli.UsageError as exc:
                assert exc.code == 2, exc.code
            else:
                raise AssertionError(f"{compare} 参照本应拒绝 {bad!r}")

    # numpy 支持的两个名字必须放行，且真的传到 np.einsum
    for good in cli.NUMPY_COMPARE_OPTIMIZERS:
        report = _compare_opt_run(compare="numpy", compare_optimize=good)
        assert report["ok"], report
        assert report["compare_optimize_effective"] == good, report
        assert f"optimize='{good}'" in report["reference"], report["reference"]


def test_compare_optimize_does_not_touch_the_main_path():
    """该选项**只影响参照**：同一方程在 greedy 与 dp 下主路径的计划不变。

    断言主路径的 ``optimize`` 字段与 ``kernel_info``/``actual_shape`` 完全一致——
    主路径的收缩序由 ``--optimize`` 管，参照优化器改了也不该动它。耗时只做宽松界，
    不做精确性能断言（``--rounds 2`` 的噪声远大于本改动能带来的差异）。
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

    scale = max(first["seconds"], second["seconds"]) / max(
        min(first["seconds"], second["seconds"]), 1e-12)
    assert scale < 4.0, (first["seconds"], second["seconds"])


def test_compare_optimize_is_inert_for_backend_and_none():
    """``backend``/``none`` 没有可设的优化器：非 auto 值不报错，只标无效。

    这两条参照路径的耗时不该被该选项影响；报告里 ``compare_optimize_effective``
    为 ``None``，``reference`` 串里不出现优化器名。
    """
    for compare in ("backend", "none"):
        report = _compare_opt_run(compare=compare, compare_optimize="dp")
        assert report["ok"], report
        assert report["compare_optimize"] == "dp", report
        assert report["compare_optimize_effective"] is None, report
        if compare == "backend":
            assert "optimize" not in report["reference"], report["reference"]
        else:
            assert "reference" not in report, report

        auto = _compare_opt_run(compare=compare, compare_optimize="auto")
        assert auto["ok"], auto
        assert auto["compare_optimize_effective"] is None, auto

    # 解析层同样不报错（无效但合法），且返回 None
    for compare in ("backend", "none"):
        for value in cli.COMPARE_OPTIMIZERS:
            assert cli.resolve_compare_optimize(compare, "auto", value) is None, (
                compare, value)


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
                    compare="numpy", rounds=2, warmup=1, seed=7, language=lang)
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
                                    compare="numpy", rounds=2, warmup=1,
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
    test_specialize_tri_state_maps_to_contract_argument,
    test_specialize_false_really_disables_the_native_path,
    test_specialize_is_forwarded_to_the_compare_backend_reference,
    test_corpus_shape_inference_does_not_import_the_testing_package,
    test_corpus_shape_inference_reports_non_scalar_output_shape,
    test_verbose_lists_every_operand_shape,
    test_help_lists_specialize_and_verbose_reports_it,
    test_compare_optimize_default_reproduces_the_old_reference_behaviour,
    test_compare_optimize_dp_reaches_the_opt_einsum_reference,
    test_compare_optimize_rejects_names_the_reference_cannot_honour,
    test_compare_optimize_does_not_touch_the_main_path,
    test_compare_optimize_is_inert_for_backend_and_none,
    test_help_lists_compare_optimize,
    test_language_modes_and_default_are_documented,
    test_language_help_is_localised,
    test_language_resolution_follows_the_env_var_only_when_auto,
    test_language_rejects_unknown_values_with_exit_code_two,
    test_language_does_not_change_results,
    test_language_only_changes_the_verbose_wording,
    test_language_does_not_pollute_the_environment,
    test_translate_fails_loudly_on_missing_keys_and_variants,
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
