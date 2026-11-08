"""``python -m pyqcd`` — report version, native backend and subpackages.

Usage::

    python -m pyqcd            # version + backend + subpackage list
    python -m pyqcd --version  # version only
    python -m pyqcd --json     # machine-readable
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys

from . import __all__ as SUBPACKAGES
from . import __version__

_OPTIONAL = {
    "numpy": "required",
    "h5py": "required",
    "scipy": "required",
    "torch": 'pyqcd[torch]',
    "matplotlib": 'pyqcd[plot]',
    "cupy": 'pyqcd[cuda]',
    "mpi4py": 'pyqcd[mpi]',
}


def _present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _native() -> dict[str, object]:
    try:
        from . import einsum as pe

        return pe.build_info()
    except Exception as error:  # pragma: no cover - diagnostics only
        return {"error": f"{type(error).__name__}: {error}"}


def _report() -> dict[str, object]:
    return {
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "subpackages": list(SUBPACKAGES),
        "optional_dependencies": {
            name: {"installed": _present(name), "extra": extra}
            for name, extra in _OPTIONAL.items()
        },
        "native": _native(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pyqcd",
        description="Report the installed PyQCD version, native backend and "
                    "available subpackages.")
    parser.add_argument("--version", action="store_true",
                        help="print the version only")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output")
    parser.add_argument("--gpu-probe", action="store_true",
                        help="check whether the CUDA fast path is actually active")
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.gpu_probe:
        from .einsum import _probe

        report = _probe.probe()
        if args.json:
            import json as _json
            print(_json.dumps(report, indent=2, ensure_ascii=False, default=str))
        else:
            print(_probe.format_report(report))
        return 0

    report = _report()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    native = report["native"]
    print(f"pyqcd {report['version']}  (Python {report['python']})")
    if isinstance(native, dict) and "backend" in native:
        print(f"  native backend : {native['backend']}")
        print(f"  mpi            : {native['mpi']} ({native['mpi_label']})")
        print(f"  library        : {native['native_library']}")
    else:
        print(f"  native backend : unknown ({native})")
    print("  subpackages    : " + ", ".join(report["subpackages"]))
    try:
        from .einsum import _threads

        print("  threads        : " + _threads.describe().replace("threads: ", ""))
    except Exception:  # noqa: BLE001 - 诊断信息不应影响入口
        pass
    missing = [
        f"{name} ({info['extra']})"
        for name, info in report["optional_dependencies"].items()
        if not info["installed"] and info["extra"] != "required"
    ]
    if missing:
        print("  optional not installed: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
