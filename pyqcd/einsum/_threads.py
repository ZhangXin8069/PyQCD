"""Default thread configuration for the native backends.

The KUPL backend runs two parallelism layers: OpenMP loops inside the library
(``libgomp``) and a dynamically loaded BLAS (``libopenblas``). Left alone, both
default to "all logical cores", which on a hyper-threaded laptop means 8 OpenMP
threads *and* 8 BLAS threads — measured 3-6x slower than the tuned setting.

This module picks a balanced default, applies it to the already-loaded runtimes,
and steps aside as soon as the user asks for something else.

Two profiles are used, because the kernel path and the BLAS path want opposite
settings on a small machine (measured, 4 physical cores):

====================  =========  =========  =============
profile               OMP        BLAS       tuned for
====================  =========  =========  =============
``kernel``            cores      2          topology / step-table kernels
``generic``           cores//2   4          plain GEMM chains and packing
====================  =========  =========  =============

``contract()`` selects the profile per call (a matched specialised shape uses
``kernel``, everything else uses ``generic``) and only touches the runtimes when
the profile actually changes.

Priority (highest wins)
-----------------------
1. ``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` (the standard variables)
2. ``PYQCD_EINSUM_OMP_THREADS`` / ``PYQCD_EINSUM_BLAS_THREADS``
3. auto: ``OMP = min(physical_cores, 8)``, ``BLAS = clamp(physical_cores // 2, 1, 4)``
4. ``PYQCD_EINSUM_THREADS=off`` disables the auto configuration entirely

Measured on a 4-physical-core / 8-thread laptop (KUPL backend, ``pyqcd 0.1.14``):

===========  ===========  ==========  =============
OMP threads  BLAS threads  GEMM 700²   topology
===========  ===========  ==========  =============
unset        unset        27.3 ms     2.1 ms
1            4            4.3 ms      3.7 ms
4            2            **5.8 ms**  **1.1 ms**
4            8            27.5 ms     2.7 ms
===========  ===========  ==========  =============
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

DISABLE_ENV = "PYQCD_EINSUM_THREADS"
OMP_ENV = "PYQCD_EINSUM_OMP_THREADS"
BLAS_ENV = "PYQCD_EINSUM_BLAS_THREADS"
STANDARD_OMP_ENV = "OMP_NUM_THREADS"
STANDARD_BLAS_ENV = "OPENBLAS_NUM_THREADS"

_MAX_OMP = 8
_MAX_BLAS = 4

_state: dict[str, object] = {"applied": False, "omp": None, "blas": None,
                             "source": "not applied",
                             "runtime_applied": [], "exported_env": {}}
_runtime_complete = False


def _disabled() -> bool:
    return os.environ.get(DISABLE_ENV, "").strip().lower() in {"0", "off", "no",
                                                              "false", "none"}


def physical_cores() -> int:
    """Physical (not hyper-thread) core count, with a portable fallback."""
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        text = ""
    cores_per_socket = 0
    sockets = set()
    for line in text.splitlines():
        if line.startswith("cpu cores") and not cores_per_socket:
            cores_per_socket = int(line.split(":", 1)[1])
        elif line.startswith("physical id"):
            sockets.add(line.split(":", 1)[1].strip())
    if cores_per_socket and sockets:
        return max(cores_per_socket * len(sockets), 1)
    if cores_per_socket:
        return cores_per_socket
    detected = os.cpu_count() or 1
    return max(detected // 2, 1) if detected > 1 else 1


def _explicit(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    exported = (_state.get("exported_env") or {}).get(name)
    if exported is not None and value.strip() == exported:
        return None                      # 我们自己写的，不算用户设置
    try:
        threads = int(value.strip())
    except ValueError:
        return None
    return threads if threads > 0 else None


def profiles() -> dict[str, tuple[int, int]]:
    """Return ``{name: (omp, blas)}`` scaled to this machine."""
    cores = max(physical_cores(), 1)
    kernel_omp = min(cores, _MAX_OMP)
    kernel_blas = min(max(cores // 2, 1), _MAX_BLAS)
    generic_omp = min(max(cores // 2, 1), _MAX_OMP)
    generic_blas = min(max(cores, 2), _MAX_BLAS)
    return {"kernel": (kernel_omp, kernel_blas),
            "generic": (generic_omp, generic_blas)}


def _pinned() -> tuple[int | None, int | None] | None:
    """Return the user's explicit thread request, or ``None`` when unpinned."""
    omp = _explicit(STANDARD_OMP_ENV)
    blas = _explicit(STANDARD_BLAS_ENV)
    source = "standard environment" if (omp is not None or blas is not None) else ""
    if omp is None:
        omp = _explicit(OMP_ENV)
        if omp is not None:
            source = source or OMP_ENV
    if blas is None:
        blas = _explicit(BLAS_ENV)
        if blas is not None:
            source = source or BLAS_ENV
    if omp is None and blas is None:
        return None
    return omp, blas


def resolve(profile: str = "kernel") -> tuple[int | None, int | None, str]:
    """Return ``(omp_threads, blas_threads, source)`` for ``profile``."""
    if _disabled():
        return None, None, f"disabled by {DISABLE_ENV}"
    pinned = _pinned()
    table = profiles()
    omp, blas = table.get(profile, table["kernel"])
    if pinned is None:
        cores = max(physical_cores(), 1)
        return omp, blas, f"auto {profile} ({cores} physical cores)"
    pinned_omp, pinned_blas = pinned
    if (_explicit(STANDARD_OMP_ENV) is not None
            or _explicit(STANDARD_BLAS_ENV) is not None):
        source = "standard environment"
    elif _explicit(OMP_ENV) is not None:
        source = OMP_ENV
    else:
        source = BLAS_ENV
    return (pinned_omp if pinned_omp is not None else omp,
            pinned_blas if pinned_blas is not None else blas,
            source)


def _loaded_libraries() -> dict[str, str]:
    found: dict[str, str] = {}
    try:
        lines = Path("/proc/self/maps").read_text().splitlines()
    except OSError:
        return found
    for line in lines:
        if ".so" not in line:
            continue
        path = line.split()[-1]
        lowered = path.lower()
        if "blas" in lowered and "blas" not in found:
            found["blas"] = path
        elif "gomp" in lowered and "gomp" not in found:
            found["gomp"] = path
    return found


def _set_runtime_threads(omp: int | None, blas: int | None,
                         respect_standard: bool) -> list[str]:
    """Push the values into the already-loaded runtimes; return what worked."""
    applied: list[str] = []
    libraries = _loaded_libraries()
    if omp is not None and not respect_standard and "gomp" in libraries:
        try:
            gomp = ctypes.CDLL(libraries["gomp"])
            gomp.omp_set_num_threads(ctypes.c_int(int(omp)))
            applied.append("gomp")
        except (OSError, AttributeError):
            pass
    if blas is not None and "blas" in libraries:
        try:
            handle = ctypes.CDLL(libraries["blas"])
        except OSError:
            handle = None
        if handle is not None:
            # 不同发行版的 OpenBLAS 导出名不同：conda/系统库用 openblas_*，
            # numpy/scipy 的 wheel 用 scipy_openblas*（见 numpy.libs/*.so）。
            for symbol in ("openblas_set_num_threads",
                           "openblas_set_num_threads64_",
                           "scipy_openblas_set_num_threads",
                           "scipy_openblas_set_num_threads64_",
                           "scipy_openblas_set_num_threads_64_",
                           "scipy_goto_set_num_threads64_"):
                function = getattr(handle, symbol, None)
                if function is None:
                    continue
                try:
                    function(ctypes.c_int(int(blas)))
                except (TypeError, ValueError):
                    continue
                applied.append("blas")
                _state["blas_symbol"] = symbol
                break
    return applied


def apply(profile: str = "kernel", force: bool = False) -> dict[str, object]:
    """Apply one profile; safe to call repeatedly (cheap when unchanged)."""
    if _state["applied"] and not force and _state.get("profile") == profile:
        return dict(_state)
    omp, blas, source = resolve(profile)
    if omp is None and blas is None:
        _state.update({"applied": True, "profile": profile, "omp": None,
                       "blas": None, "source": source,
                       "runtime_applied": []})
        return dict(_state)

    explicit_omp = _explicit(STANDARD_OMP_ENV) is not None
    explicit_blas = _explicit(STANDARD_BLAS_ENV) is not None
    applied = _set_runtime_threads(omp, blas, respect_standard=explicit_omp)

    # 同时导出到标准变量：GOMP 在**加载时**读 OMP_NUM_THREADS，而我们的原生库
    # 是延迟 dlopen 的，导出后它初始化时就能拿到正确值（运行时 API 只对已加载的库有效）。
    # 记录在 _state["exported_env"]，_pinned() 会忽略这些值，档位切换不会失效。
    exported = dict(_state.get("exported_env") or {})
    if not explicit_omp and omp is not None:
        os.environ[STANDARD_OMP_ENV] = str(omp)
        exported[STANDARD_OMP_ENV] = str(omp)
    if not explicit_blas and blas is not None:
        os.environ[STANDARD_BLAS_ENV] = str(blas)
        exported[STANDARD_BLAS_ENV] = str(blas)
    _state["exported_env"] = exported

    _state.update({"applied": True, "profile": profile, "omp": omp, "blas": blas,
                   "source": source,
                   "runtime_applied": applied})
    return dict(_state)


def apply_profile(profile: str) -> None:
    """Switch to ``profile`` when nothing is pinned and the profile changed."""
    if _disabled() or _pinned() is not None:
        return
    if _state.get("applied") and _state.get("profile") == profile:
        return
    apply(profile)


def current() -> dict[str, object]:
    """Return the state without changing the active profile."""
    if not _state.get("applied"):
        return apply()
    return dict(_state)


def describe() -> str:
    state = current()
    if state["omp"] is None:
        return f"threads: untouched ({state['source']})"
    runtimes = state.get("runtime_applied") or []
    extra = f"; runtime: {', '.join(runtimes)}" if runtimes else ""
    return (f"threads: omp={state['omp']} blas={state['blas']} "
            f"[{state['source']}{extra}]")


def reset() -> None:  # pragma: no cover - test helper
    _state.update({"applied": False, "omp": None, "blas": None,
                   "source": "not applied"})


def ensure_runtime() -> None:
    """Re-apply the thread counts once the native runtimes are actually loaded.

    ``apply()`` runs at import time, when ``libgomp``/``libopenblas`` may not be
    mapped yet. The native backends call this before their first use so the
    OpenMP default is not missed when another library loaded GOMP first.
    """
    global _runtime_complete
    if _runtime_complete or not _state.get("applied") or _state.get("omp") is None:
        return
    attempts = int(_state.get("runtime_attempts", 0)) + 1
    _state["runtime_attempts"] = attempts
    if attempts > 3:                     # 三次仍没成（例如库还没加载）就放弃
        _runtime_complete = True
        return
    respect = _explicit(STANDARD_OMP_ENV) is not None
    applied = _set_runtime_threads(_state["omp"], _state["blas"],
                                   respect_standard=respect)
    if applied:
        _state["runtime_applied"] = sorted(set(list(_state.get("runtime_applied") or []) + applied))
    libraries = _loaded_libraries()
    need = {"gomp", "blas"} - set(_state.get("runtime_applied") or [])
    if not need or not (need & set(libraries)):
        _runtime_complete = True
    _state["runtime_applied"] = sorted(set(_state.get("runtime_applied") or []))
