"""Diagnose whether the CUDA fast path is actually active.

``pyqcd.einsum`` falls back silently (by contract: a native failure only warns
once and the numbers stay correct), which makes a *slow* GPU run hard to explain:
the timings look like "PyQCD is slow" when in fact the kernel never ran.

``probe()`` answers the question directly: does the build see a device, does the
CUDA library expose a usable kernel, does that kernel return OK for a real
operand set, and do the projected shapes hit the specialised path.

    python -m pyqcd --gpu-probe
"""

from __future__ import annotations

import contextlib
import io
from typing import Any

NUCLEON_2PT = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"
SANDWICH = "ab,...bce,cd->...dae"


def _cupy():
    try:
        import cupy  # noqa: F401
    except ImportError:
        return None
    return cupy


def _kernel_status() -> tuple[str, str]:
    """Call the CUDA 2pt kernel once and report its status code."""
    from . import _kernel_cuda as kernel

    cupy = _cupy()
    if cupy is None:
        return "cupy-missing", "install cupy: pip install \"pyqcd[cuda]\""
    if not kernel.enabled():
        return "disabled", "PYQCD_EINSUM_KERNEL_CUDA=0 is set, or the library is absent"

    def draw(*shape):
        generator = cupy.random.default_rng(0)
        return (generator.standard_normal(shape)
                + 1j * generator.standard_normal(shape)).astype(cupy.complex128)

    n = 8
    operands = [draw(4, 4, n, n), draw(4, 4, n, n), draw(4, 4, n, n),
                draw(4, 4), draw(4, 4), draw(4, n, n, n), draw(4, n, n, n),
                draw(4, 4)]
    try:
        entry = kernel._function_2pt()
    except Exception as error:  # noqa: BLE001 - diagnostics only
        return "entry-missing", f"{type(error).__name__}: {error}"
    if entry is None:
        return "entry-missing", "libpyqcdeinsum_cuda.so does not export the kernel"

    import ctypes

    ordered = [operands[0], operands[1], operands[2], operands[3],
               operands[4], operands[7], operands[5], operands[6]]
    dims = (ctypes.c_int64 * 10)(n, 4, 4, 0, 123, 123, 123, 123, 123, 0)
    out = cupy.zeros((4,), dtype=cupy.complex128)
    pointers = (ctypes.c_void_p * len(ordered))(
        *[int(operand.data.ptr) for operand in ordered])
    try:
        status = entry(kernel.QCD_DTYPE_COMPLEX128, pointers,
                       int(out.data.ptr), dims, None)
    except Exception as error:  # noqa: BLE001
        return "call-failed", f"{type(error).__name__}: {error}"
    if status == 0:
        return "ok", "kernel returned QCD_STATUS_OK"
    names = getattr(kernel, "STATUS_NAMES", ())
    label = names[status] if 0 <= status < len(names) else str(status)
    return f"status-{status}", (
        f"kernel returned {label}; the cubin/PTX was built for another "
        "architecture — rebuild with PYQCD_CUDA_ARCHITECTURES=<your compute cap>")


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
        "kernel_hint": None,
        "specialised": None,
        "verdict": "",
        "hints": [],
    }
    if cupy is None:
        report["verdict"] = "cupy missing — GPU kernels cannot receive numpy data"
        report["hints"].append('pip install "pyqcd[cuda]"')
        return report

    try:
        properties = cupy.cuda.runtime.getDeviceProperties(0)
        report["device"] = properties["name"].decode()
        report["compute_capability"] = (
            f"{properties['major']}.{properties['minor']}")
    except Exception as error:  # noqa: BLE001
        report["device"] = f"unavailable ({type(error).__name__})"

    from . import _kernel_cuda as kernel

    report["kernel_enabled"] = bool(kernel.enabled())
    status, detail = _kernel_status()
    report["kernel_status"] = {"state": status, "detail": detail}

    # does the specialised path announce a kernel (rather than a plan cache)?
    generator = cupy.random.default_rng(0)

    def draw(*shape):
        return (generator.standard_normal(shape)
                + 1j * generator.standard_normal(shape)).astype(cupy.complex128)

    n = 8
    operands = [draw(4, 4, n, n), draw(4, 4, n, n), draw(4, 4, n, n),
                draw(4, 4), draw(4, 4), draw(4, n, n, n), draw(4, n, n, n),
                draw(4, 4)]
    buffer = io.StringIO()
    try:
        from . import _specialize

        _specialize._announced.clear()
        with contextlib.redirect_stdout(buffer):
            from . import contract

            contract(NUCLEON_2PT, *operands)
        cupy.cuda.Device(0).synchronize()
    except Exception as error:  # noqa: BLE001
        report["kernel_hint"] = f"call failed: {type(error).__name__}: {error}"
    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    hint = next((line for line in lines if "专用路径" in line), "")
    report["kernel_hint"] = hint or (lines[0] if lines else "(no hint printed)")
    report["specialised"] = "kernel:" in hint

    if status == "ok" and report["specialised"]:
        report["verdict"] = "CUDA fast path is active"
    elif status == "ok":
        report["verdict"] = ("kernel works, but this operand set did not hit it "
                             "(shape contract mismatch)")
    else:
        report["verdict"] = "CUDA fast path is NOT active — results fall back"
        report["hints"].append(detail)
        if info.get("backend") != "cuda":
            report["hints"].append(
                f"build_info() backend is {info.get('backend')!r}, not 'cuda'")
    return report


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
        f"  kernel status  : {report['kernel_status']['state']} — "
        f"{report['kernel_status']['detail']}",
        f"  hit hint       : {report['kernel_hint']}",
        f"  verdict        : {report['verdict']}",
    ]
    for hint in report["hints"]:
        lines.append(f"  hint           : {hint}")
    return "\n".join(lines)
