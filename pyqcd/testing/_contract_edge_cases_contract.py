"""Edge cases of the public contraction API.

These are the cases that a "drop-in `opt_einsum` replacement" claim lives or dies
on: every NumPy dtype, ellipsis and repeated labels, scalars, zero-extent
dimensions, non-contiguous operands, every ``optimize`` spelling, ``out=``,
``contract_expression``, ``shared_intermediates`` and the error types for invalid
input. Everything is compared against ``numpy.einsum`` (or ``opt_einsum`` where
the reference is about planning rather than numbers).

``dtype=``/``order=`` are deliberately *not* required to work: ``opt_einsum``
3.4.0 rejects them as well, so matching that behaviour is the compatible one.
"""

from __future__ import annotations

import numpy as np

import pyqcd.einsum as pe

RNG = np.random.default_rng(20261108)


def _rel_error(got, ref) -> float:
    got = np.asarray(got)
    ref = np.asarray(ref)
    assert got.shape == ref.shape, f"shape {got.shape} != {ref.shape}"
    if got.dtype == bool or ref.dtype == bool:
        assert np.array_equal(got.astype(bool), ref.astype(bool))
        return 0.0
    scale = max(float(np.max(np.abs(ref))) if ref.size else 1.0, 1e-30)
    return float(np.max(np.abs(got - ref)) / scale) if ref.size else 0.0


def _check(subscripts: str, operands, **kwargs) -> None:
    got = pe.contract(subscripts, *operands, **kwargs)
    ref = np.einsum(subscripts, *operands, optimize="optimal")
    assert _rel_error(got, ref) <= 1e-10, f"{subscripts}: value mismatch"


def test_all_dtypes() -> None:
    for dtype in (np.float32, np.float64, np.complex64, np.complex128):
        a = RNG.standard_normal((3, 4)).astype(dtype)
        b = RNG.standard_normal((4, 5)).astype(dtype)
        _check("ij,jk->ik", [a, b])
        _check("ij,jk", [a, b])            # implicit output
    _check("ij,jk->ik", [np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4)])
    _check("ij,jk->ik", [np.eye(3, dtype=bool), np.eye(3, dtype=bool)])


def test_syntax_edges() -> None:
    _check("...ij,...jk->...ik",
           [RNG.standard_normal((2, 3, 4)), RNG.standard_normal((2, 4, 5))])
    _check("ii->i", [RNG.standard_normal((4, 4))])       # diagonal
    _check("ii->", [RNG.standard_normal((4, 4))])        # trace
    _check("i,i->", [RNG.standard_normal(4), RNG.standard_normal(4)])
    _check(",ij->ij", [np.array(2.0), RNG.standard_normal((2, 3))])
    _check("ij->ij", [RNG.standard_normal((3, 4))])      # identity
    _check("ij->ji", [RNG.standard_normal((3, 4))])
    _check("ij,ij->ij", [RNG.standard_normal((3, 4)), RNG.standard_normal((3, 4))])
    _check("ij,jk,kl->il",
           [RNG.standard_normal((2, 3)), RNG.standard_normal((3, 4)),
            RNG.standard_normal((4, 5))])


def test_noncontiguous_and_empty() -> None:
    base = RNG.standard_normal((4, 6))
    _check("ij,jk->ik", [base[::2, ::2], RNG.standard_normal((3, 5))])
    _check("ij,jk->ik", [np.zeros((0, 4)), np.zeros((4, 5))])


def test_optimize_spellings() -> None:
    a = RNG.standard_normal((2, 3))
    b = RNG.standard_normal((3, 4))
    c = RNG.standard_normal((4, 5))
    for optimize in (False, True, "greedy", "optimal", "dp", "auto"):
        _check("ij,jk,kl->il", [a, b, c], optimize=optimize)


def test_out_argument() -> None:
    a = RNG.standard_normal((2, 3))
    b = RNG.standard_normal((3, 4))
    out = np.zeros((2, 4))
    result = pe.contract("ij,jk->ik", a, b, out=out)
    assert np.allclose(np.asarray(result), a @ b)


def test_expression_and_cache() -> None:
    a = RNG.standard_normal((2, 3))
    b = RNG.standard_normal((3, 4))
    expression = pe.contract_expression("ij,jk->ik", (2, 3), (3, 4))
    assert np.allclose(expression(a, b), a @ b)
    with pe.shared_intermediates() as cache:
        pe.contract("ij,jk->ik", a, b)
        pe.contract("ij,jk->ik", a, b)
    assert isinstance(cache, dict)


def test_opt_cost_matches_reference() -> None:
    import opt_einsum

    a = RNG.standard_normal((2, 3))
    b = RNG.standard_normal((3, 4))
    c = RNG.standard_normal((4, 5))
    _, ours = pe.contract_path("ij,jk,kl->il", a, b, c, optimize="optimal")
    _, theirs = opt_einsum.contract_path("ij,jk,kl->il", a, b, c, optimize="optimal")
    assert int(ours.opt_cost) == int(theirs.opt_cost)


def test_invalid_input_errors() -> None:
    a = RNG.standard_normal((2, 3))
    for subscripts, operands in (
        ("ij,jk->ik", [a]),                                   # operand count
        ("ij,jk->ik", [a, RNG.standard_normal((5, 4))]),      # label size
        ("ij->ik", [a]),                                      # unknown label
    ):
        try:
            pe.contract(subscripts, *operands)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"{subscripts}: expected ValueError/TypeError")


def test_native_failure_falls_back_without_raising() -> None:
    """回落是硬不变量：原生路径里任何异常都必须降级，而不是抛给调用方。

    这里连"规划阶段"抛异常也一并模拟——历史上只有"执行阶段"被包住，
    `_try_native_contract` 自身抛错会直接冒到用户代码里。
    """
    import warnings

    import pyqcd.einsum._api as api

    a = np.ones((16, 16))
    b = np.ones((16, 16))
    original = api._try_native_contract

    def explode(*args, **kwargs):
        raise RuntimeError("simulated native planning failure")

    api._try_native_contract = explode
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = [pe.contract("ij,jk->ik", a, b) for _ in range(4)]
    finally:
        api._try_native_contract = original

    for result in results:
        assert np.allclose(np.asarray(result), a @ b)
    assert len(caught) <= 1, f"expected at most one warning, got {len(caught)}"
    # 恢复后必须重新用回原生路径（没有留下永久的禁用状态）
    assert np.allclose(np.asarray(pe.contract("ij,jk->ik", a, b)), a @ b)


def test_kernel_skip_warns_immediately() -> None:
    """注册了专用 kernel 却没跑成时，必须立刻告警并给出原因。

    静默回落会让"PyQCD 很慢"变成一个无法诊断的现象；这条契约保证用户第一次
    遇到降级就能看到提示（同一形态+原因只报一次）。
    """
    import os
    import warnings

    import pyqcd.einsum._api as api

    saved = {name: os.environ.get(name) for name in
             ("PYQCD_EINSUM_KERNEL_KUPL", "PYQCD_EINSUM_KERNEL_CUDA")}
    api._KERNEL_SKIP_WARNED.clear()
    os.environ["PYQCD_EINSUM_KERNEL_KUPL"] = "0"
    os.environ["PYQCD_EINSUM_KERNEL_CUDA"] = "0"
    try:
        rng = np.random.default_rng(0)

        def draw(*shape):
            return (rng.standard_normal(shape)
                    + 1j * rng.standard_normal(shape)).astype(np.complex128)

        n = 8
        operands = [draw(4, 4, n, n), draw(4, 4, n, n), draw(4, 4, n, n),
                    draw(4, 4), draw(4, 4), draw(4, n, n, n), draw(4, n, n, n),
                    draw(4, 4)]
        equation = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                result = pe.contract(equation, *operands)
        reference = np.einsum(equation, *operands, optimize="optimal")
        assert np.allclose(np.asarray(result), reference)
        messages = [str(item.message) for item in caught]
        assert any("did not run" in message for message in messages), messages
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        api._KERNEL_SKIP_WARNED.clear()


def test_gpu_probe_survives_missing_device() -> None:
    """装了 cupy 却没有可见 GPU 时，`--gpu-probe` 必须给结论而不是 traceback。"""
    import os

    from pyqcd.einsum import _probe

    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        report = _probe.probe()
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
    assert isinstance(report, dict) and report.get("verdict"), report
    assert report["kernel_status"]["state"] in {"no-device", "ok", "probe-failed",
                                                "cupy-missing", "disabled"}


def test_probe_reports_without_raising() -> None:
    from pyqcd.einsum import _probe

    report = _probe.probe()
    assert "verdict" in report and "kernel_status" in report
    text = _probe.format_report(report)
    assert "verdict" in text and "kernel status" in text


def test_elementwise_equations_bypass_the_planner() -> None:
    """无求和标签的方程必须走 backend 原生 einsum（分步执行慢 2–3 倍）。

    Hadamard 4000² 实测：分步 89 ms vs 原生 37 ms；trace 0.53 ms vs 0.05 ms。
    """
    import pyqcd.einsum._api as api

    delegable = ["ij,ij->ij", "ij->ji", "ii->i", "ii->", "i,j->ij", "ij,ij->ji"]
    planned = ["ij,jk->ik", "ij,jk,kl->il", "...ij,...jk->...ik", "ij,jk->ik"]
    for equation in delegable:
        assert api._is_elementwise_equation(equation), equation
    for equation in planned:
        assert not api._is_elementwise_equation(equation), equation

    rng = np.random.default_rng(11)
    a = rng.standard_normal((16, 16))
    b = rng.standard_normal((16, 16))
    for equation in ("ij,ij->ij", "ij,ij->ji", "ij->ji", "ii->i", "ii->"):
        operands = [a] if equation.count(",") == 0 else [a, b]
        assert np.allclose(np.asarray(pe.contract(equation, *operands)),
                           np.einsum(equation, *operands, optimize="optimal")), equation
    assert np.allclose(np.asarray(pe.contract("i,j->ij", a[0], b[0])),
                       np.outer(a[0], b[0]))


def test_kupl_plan_gate_by_batch_structure() -> None:
    """KUPL 计划执行器只在带 batch 维时启用（2D 链上它反而慢 6–10%）。"""
    import pyqcd.einsum._api as api

    assert api._batched_operands([np.zeros((2, 3, 4))]) is True
    assert api._batched_operands([np.zeros((4, 4)), np.zeros((4, 4))]) is False
    assert api._batched_operands([np.zeros((4, 4)), np.zeros((3, 4, 4))]) is True

    rng = np.random.default_rng(12)
    batched = [rng.standard_normal((4, 16, 16)), rng.standard_normal((4, 16, 16))]
    assert np.allclose(np.asarray(pe.contract("bij,bjk->bik", *batched)),
                       np.einsum("bij,bjk->bik", *batched, optimize="optimal"))
    chain = [rng.standard_normal((32, 32)) for _ in range(3)]
    assert np.allclose(np.asarray(pe.contract("ij,jk,kl->il", *chain)),
                       np.einsum("ij,jk,kl->il", *chain, optimize="optimal"))
