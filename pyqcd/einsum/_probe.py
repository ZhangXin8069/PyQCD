"""Diagnose whether the CUDA fast path is actually active.

``pyqcd.einsum`` falls back silently (by contract: a native failure only warns
once and the numbers stay correct), which makes a *slow* GPU run hard to explain:
the timings look like "PyQCD is slow" when in fact the kernel never ran.

``probe()`` answers the question directly: does the build see a device, does the
CUDA archive expose the topology entry, and does one tiny contraction through it
agree with the reference result.

The topology dtype contract has **four** equal dtypes (``complex128`` /
``complex64`` / ``float64`` / ``float32``), each enabled only by its own native
2x2 measurement. The report therefore carries a per-dtype section
(``report["dtypes"]``) taken straight from ``_kernel_cuda.probe_dtypes()``: one
whole-site "fallback" says nothing about *which* dtypes need a rebuild, and a
single failing dtype must not be allowed to condemn the other three.

    python -m pyqcd --gpu-probe
"""

from __future__ import annotations

from typing import Any

_PROBE_EQUATION = "ij,jk->ik"


def _cupy():
    try:
        import cupy  # noqa: F401
    except ImportError:
        return None
    return cupy


def _dtype_line(dtypes: Any) -> str:
    """Render ``complex128=ok complex64=fail(stage=native: ...) ...``.

    The per-dtype facts come from the kernel probe; this only renders them, in
    contract order, with the real stage and detail text for the failing ones.
    """
    if not isinstance(dtypes, dict) or not dtypes:
        return "n/a — no dtype probe ran"
    parts = []
    for name, record in dtypes.items():
        record = record if isinstance(record, dict) else {}
        if record.get("ok"):
            parts.append(f"{name}=ok")
        else:
            stage = record.get("stage") or "unknown"
            detail = record.get("detail") or "no detail recorded"
            parts.append(f"{name}=fail(stage={stage}: {detail})")
    return " ".join(parts)


def _kernel_status() -> dict[str, Any]:
    """Probe the four contract dtypes, then run one end-to-end contraction.

    Returns ``{"state", "detail", "dtypes", "verified"}``:

    * ``state`` stays inside the set the existing diagnostics contract allows
      (``ok`` / ``fallback`` / ``mismatch`` / ``call-failed`` / ``disabled`` /
      ``cupy-missing``);
    * ``detail`` always carries the per-dtype verdict (``complex128=ok
      complex64=fail(stage=native: ...) ...``);
    * ``dtypes`` is the per-dtype fact table from ``_kernel_cuda.probe_dtypes()``;
    * ``verified`` is the dtype the 2x2 end-to-end comparison actually agreed on
      (``None`` if none did) — ``state`` alone cannot tell "one dtype verified,
      the rest fall back" from "nothing ran".

    The end-to-end dtype is the **first available contract dtype**, not a
    hard-wired ``complex64``: one failing dtype must not condemn the other three.
    """
    from . import _kernel_cuda as kernel

    def result(state: str, detail: str, dtypes: Any = None,
               verified: str | None = None) -> dict[str, Any]:
        return {"state": state, "detail": detail, "dtypes": dtypes or {},
                "verified": verified}

    cupy = _cupy()
    if cupy is None:
        return result("cupy-missing", "install cupy: pip install \"pyqcd[cuda]\"")
    if not kernel.enabled():
        return result("disabled",
                      "PYQCD_EINSUM_KERNEL_CUDA=0 is set, or the library is absent")

    import numpy as np

    dtypes = kernel.probe_dtypes()
    summary = _dtype_line(dtypes)
    available = [name for name, record in dtypes.items()
                 if isinstance(record, dict) and record.get("ok")]
    if not available:
        # 四类全失败：逐类原因必须都在 detail 里，不许只说一句"回落"。
        return result("fallback",
                      f"the topology entry did not run for any contract dtype: "
                      f"{summary}", dtypes)

    dtype_name = available[0]                      # 契约内第一个可用（顺序即契约序）
    dtype = np.dtype(dtype_name)
    left = cupy.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    right = cupy.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=dtype)
    try:
        actual = kernel.try_topology_apply(_PROBE_EQUATION, (left, right))
    except Exception as error:  # noqa: BLE001 - diagnostics only
        return result("call-failed",
                      f"{type(error).__name__}: {error}; dtypes: {summary}", dtypes)
    if actual is None:
        reason = kernel.LAST_FAILURE.get("reason") or "unknown reason"
        return result("fallback",
                      f"topology entry did not run: {reason}; dtypes: {summary}",
                      dtypes)

    expected = np.einsum(
        _PROBE_EQUATION, cupy.asnumpy(left), cupy.asnumpy(right))
    got = cupy.asnumpy(actual)
    # 容差与 ``_kernel_cuda._probe_dtype`` **同一份映射**（相对口径：单精度 1e-4、
    # 双精度 1e-9）。写死一个绝对值会让 complex128/float64 的对照失去意义，也会和
    # 原生探测用两把尺子。
    tolerance = float(kernel._PROBE_TOLERANCE[dtype_name])
    scale = float(np.max(np.abs(expected))) or 1.0
    agrees = (got.shape == expected.shape and got.dtype == dtype
              and bool(np.allclose(got, expected, rtol=tolerance,
                                   atol=tolerance * scale)))
    if not agrees:
        observed = (f"shape {tuple(got.shape)} dtype {got.dtype} "
                    f"max|diff|={float(np.max(np.abs(got - expected))):.3e}"
                    if got.shape == expected.shape else
                    f"shape {tuple(got.shape)} != {tuple(expected.shape)}")
        return result("mismatch",
                      f"topology entry disagreed with numpy on {dtype_name} "
                      f"({observed}, rtol={tolerance:g}); dtypes: {summary}",
                      dtypes)
    if len(available) == len(dtypes):
        return result("ok",
                      f"topology entry returned the reference result for "
                      f"{dtype_name}; dtypes: {summary}", dtypes, dtype_name)
    return result("fallback",
                  f"only some contract dtypes can run the topology entry "
                  f"(verified {dtype_name} end to end); dtypes: {summary}",
                  dtypes, dtype_name)


def probe() -> dict[str, Any]:
    """Return a dict describing the GPU fast path in this installation."""
    from . import build_info

    info = build_info()
    cupy = _cupy()
    report: dict[str, Any] = {
        "build_info": info,
        "cupy": None if cupy is None else cupy.__version__,
        "device": None,
        "kernel_enabled": None,
        "kernel_status": None,
        "dtypes": {},
        "verdict": "",
        "hints": [],
    }
    if cupy is None:
        report["verdict"] = "cupy missing — GPU kernels cannot receive numpy data"
        report["kernel_status"] = {"state": "cupy-missing",
                                   "detail": "cupy is not importable"}
        report["hints"].append('pip install "pyqcd[cuda]"')
        return report

    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as error:  # noqa: BLE001
        device_count = 0
        report["device"] = f"unavailable ({type(error).__name__}: {error})"
    if device_count == 0:
        report.setdefault("device", "no CUDA device visible")
        report["kernel_enabled"] = None
        report["kernel_status"] = {"state": "no-device",
                                   "detail": "cupy is installed but no GPU is visible "
                                             "to this process (check CUDA_VISIBLE_DEVICES)"}
        report["verdict"] = ("no GPU available — pyqcd.einsum runs the CPU/native "
                             "path; GPU kernels cannot be checked")
        report["hints"].append("unset CUDA_VISIBLE_DEVICES or run on a GPU node")
        return report
    try:
        properties = cupy.cuda.runtime.getDeviceProperties(0)
        report["device"] = properties["name"].decode()
        report["compute_capability"] = (
            f"{properties['major']}.{properties['minor']}")
    except Exception as error:  # noqa: BLE001
        report["device"] = f"unavailable ({type(error).__name__}: {error})"

    from . import _kernel_cuda as kernel

    try:
        report["kernel_enabled"] = bool(kernel.enabled())
        outcome = _kernel_status()
        report["kernel_status"] = {"state": outcome["state"],
                                   "detail": outcome["detail"]}
        report["dtypes"] = outcome["dtypes"]
        status = outcome["state"]
        detail = outcome["detail"]
        verified = outcome["verified"]
    except Exception as error:  # noqa: BLE001 - 诊断必须给出结论而不是抛异常
        report["kernel_status"] = {"state": "probe-failed",
                                   "detail": f"{type(error).__name__}: {error}"}
        report["verdict"] = "GPU probe failed — see kernel_status"
        report["hints"].append(str(error))
        return report

    available = [name for name, record in report["dtypes"].items()
                 if isinstance(record, dict) and record.get("ok")]
    missing = [name for name, record in report["dtypes"].items()
               if not (isinstance(record, dict) and record.get("ok"))]
    if status == "ok":
        report["verdict"] = "CUDA fast path is active"
    elif status == "mismatch":
        # 原生真的跑了但数字不对：这既不是"回落"也不是"没问题"，必须如实说。
        report["verdict"] = ("CUDA fast path ran but disagreed with numpy — "
                             "results through the topology entry are not trustworthy")
        report["hints"].append(detail)
    elif available and missing:
        # 部分可用：既不能说 active（有 dtype 在回落），也不能说 NOT active
        # （其余 dtype 真的在跑原生）。逐类点名，方便对照 dtypes 那一行。
        # "部分可用"的判据是**真的有一类端到端对上**（verified 非空）：四类探测都
        # 声称通过却仍未下发，那就不是"部分可用"，而是回落。
        if verified is None:
            report["verdict"] = "CUDA fast path is NOT active — results fall back"
            report["hints"].append(detail)
            if info.get("backend") != "cuda":
                report["hints"].append(
                    f"build_info() backend is {info.get('backend')!r}, not 'cuda'")
            return report
        report["verdict"] = (
            "CUDA fast path is partially active — dtype(s) "
            f"{', '.join(missing)} fall back")
        report["hints"].append(detail)
        for name in missing:
            record = report["dtypes"].get(name) or {}
            report["hints"].append(
                f"{name}: {record.get('stage') or 'unknown'} — "
                f"{record.get('detail') or 'no detail recorded'}")
    else:
        report["verdict"] = "CUDA fast path is NOT active — results fall back"
        report["hints"].append(detail)
        if info.get("backend") != "cuda":
            report["hints"].append(
                f"build_info() backend is {info.get('backend')!r}, not 'cuda'")
    return report


def _format_kernel_status(status: Any) -> str:
    """Render the kernel status, which is ``None`` when no CUDA kernel exists.

    Without ``cupy`` (CPU-only install) the probe has nothing to report here; a
    ``None`` status used to raise ``TypeError`` instead of printing a line.
    """
    if not status:
        return "n/a — no CUDA kernel in this installation"
    state = status.get("state", "unknown")
    detail = status.get("detail", "")
    return f"{state} — {detail}" if detail else str(state)


def format_report(report: dict[str, Any]) -> str:
    info = report["build_info"]
    lines = [
        "GPU probe",
        f"  build_info     : backend={info.get('backend')} "
        f"native_available={info.get('native_available')} "
        f"array_backend={info.get('array_backend')}",
        f"  cupy           : {report['cupy']}",
        f"  device         : {report['device']} "
        f"(compute {report.get('compute_capability', '?')})",
        f"  kernel enabled : {report['kernel_enabled']}",
        "  kernel status  : " + _format_kernel_status(report.get("kernel_status")),
        f"  dtypes         : {_dtype_line(report.get('dtypes'))}",
        f"  verdict        : {report['verdict']}",
    ]
    for hint in report["hints"]:
        lines.append(f"  hint           : {hint}")
    return "\n".join(lines)
