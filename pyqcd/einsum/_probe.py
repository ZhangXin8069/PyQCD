"""Diagnose whether the CUDA fast path is actually active.

``pyqcd.einsum`` falls back silently (by contract: a native failure only warns
once and the numbers stay correct), which makes a *slow* GPU run hard to explain:
the timings look like "PyQCD is slow" when in fact the kernel never ran.

``probe()`` answers the question directly: does the build see a device, does the
CUDA archive expose the topology entry, and does one tiny contraction through it
agree with the reference result.

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


def _kernel_status() -> tuple[str, str]:
    """Run one tiny topology contraction through the CUDA entry."""
    from . import _kernel_cuda as kernel

    cupy = _cupy()
    if cupy is None:
        return "cupy-missing", "install cupy: pip install \"pyqcd[cuda]\""
    if not kernel.enabled():
        return "disabled", "PYQCD_EINSUM_KERNEL_CUDA=0 is set, or the library is absent"

    import numpy as np

    left = cupy.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=cupy.complex64)
    right = cupy.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=cupy.complex64)
    try:
        actual = kernel.try_topology_apply(_PROBE_EQUATION, (left, right))
    except Exception as error:  # noqa: BLE001 - diagnostics only
        return "call-failed", f"{type(error).__name__}: {error}"
    if actual is None:
        reason = kernel.LAST_FAILURE.get("reason") or "unknown reason"
        return "fallback", f"topology entry did not run: {reason}"
    expected = np.einsum(
        _PROBE_EQUATION, cupy.asnumpy(left), cupy.asnumpy(right))
    got = cupy.asnumpy(actual)
    if got.shape != expected.shape or not np.allclose(got, expected, atol=1e-6):
        return "mismatch", "topology entry returned a different result"
    return "ok", "topology entry returned the reference result"


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
        status, detail = _kernel_status()
        report["kernel_status"] = {"state": status, "detail": detail}
    except Exception as error:  # noqa: BLE001 - 诊断必须给出结论而不是抛异常
        report["kernel_status"] = {"state": "probe-failed",
                                   "detail": f"{type(error).__name__}: {error}"}
        report["verdict"] = "GPU probe failed — see kernel_status"
        report["hints"].append(str(error))
        return report

    if status == "ok":
        report["verdict"] = "CUDA fast path is active"
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
        f"  verdict        : {report['verdict']}",
    ]
    for hint in report["hints"]:
        lines.append(f"  hint           : {hint}")
    return "\n".join(lines)
