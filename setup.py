"""Build entry for the PyQCD einsum native stack.

Mirrors ``PyQCU/setup.py``: a custom ``build_ext`` owns the native library
lifecycle (shelling out to ``build-nv.sh`` / ``build-kupl.sh``), then Cython
compiles the thin bridge against the selected backend.

Environment:
    PYQCD_EINSUM_BACKEND       kupl (default) | cuda
    PYQCD_SKIP_NATIVE_BUILD    set to 1 to reuse an existing libpyqcdeinsum.so
    PYQCD_EINSUM_SKIP_CYTHON   set to 1 to build no Python extension at all
"""

import os
import subprocess

from setuptools import setup
from setuptools.command.build_ext import build_ext

HOME = os.path.dirname(os.path.abspath(__file__))

BUILD_BACKEND = os.environ.get("PYQCD_EINSUM_BACKEND", "kupl").strip().lower()
if BUILD_BACKEND not in ("kupl", "cuda"):
    raise RuntimeError(
        f"invalid PYQCD_EINSUM_BACKEND={BUILD_BACKEND!r}; expected kupl or cuda")

BACKEND_DIR = os.path.join(HOME, "cpp", BUILD_BACKEND, "einsum")
BUILD_SCRIPT = os.path.join(
    HOME, "build-nv.sh" if BUILD_BACKEND == "cuda" else "build-kupl.sh")

print(f"PyQCD einsum build backend: {BUILD_BACKEND}")
print(f"native library directory: {BACKEND_DIR}")


class NativeBuildExt(build_ext):
    def run(self):
        if os.environ.get("PYQCD_SKIP_NATIVE_BUILD") != "1":
            subprocess.check_call(["bash", BUILD_SCRIPT])
        super().run()


def _extensions():
    if os.environ.get("PYQCD_EINSUM_SKIP_CYTHON") == "1":
        return []
    try:
        from Cython.Build import cythonize
    except ImportError:
        return []
    from distutils.core import Extension

    extension = Extension(
        "pyqcd.einsum.cuda.einsum",
        [os.path.join(HOME, "pyqcd", "einsum", "cuda", "einsum.pyx")],
        include_dirs=[
            os.path.join(HOME, "pyqcd", "einsum", "cuda"),
            os.path.join(BACKEND_DIR, "python"),
            os.path.join(BACKEND_DIR, "include"),
        ],
        library_dirs=[BACKEND_DIR],
        runtime_library_dirs=[BACKEND_DIR],
        libraries=["pyqcdeinsum"],
        language="c++",
    )
    return cythonize([extension], language_level="3")


setup(
    name="pyqcd-einsum",
    version="0.1.0",
    description="Native contraction engine for PyQCD with an opt_einsum-compatible API.",
    packages=[
        "pyqcd.einsum",
        "pyqcd.einsum.backends",
        "pyqcd.einsum.cuda",
        "pyqcd.einsum.kupl",
    ],
    package_data={"pyqcd.einsum.cuda": ["*.pyi", "*.pxd"]},
    ext_modules=_extensions(),
    cmdclass={"build_ext": NativeBuildExt},
    python_requires=">=3.10",
    zip_safe=False,
)
