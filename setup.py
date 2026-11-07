"""Build entry for the PyQCD einsum native and CUDA Cython layers."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

HOME = Path(__file__).resolve().parent
RAW_BACKEND = os.environ.get("PYQCD_EINSUM_BACKEND", "auto").strip().lower()


def _cuda_sources() -> tuple[Path, ...]:
    source_dir = HOME / "cpp" / "cuda" / "einsum" / "src"
    return tuple(sorted((*source_dir.glob("*.cu"), *source_dir.glob("*.cpp"))))


def _resolve_backend(value: str) -> str:
    if value in {"kupl", "cuda"}:
        return value
    if value != "auto":
        raise RuntimeError(
            f"invalid PYQCD_EINSUM_BACKEND={value!r}; expected kupl, cuda, or auto")
    if shutil.which(os.environ.get("NVCC", "nvcc")) and _cuda_sources():
        return "cuda"
    return "kupl"


BUILD_BACKEND = _resolve_backend(RAW_BACKEND)
BACKEND_DIR = HOME / "cpp" / BUILD_BACKEND / "einsum"
BUILD_SCRIPT = HOME / ("build-nv.sh" if BUILD_BACKEND == "cuda" else "build-kupl.sh")

print(f"PyQCD einsum build backend: {BUILD_BACKEND}")
print(f"native library directory: {BACKEND_DIR}")


def _stage_native_library() -> None:
    library_name = (
        "libpyqcdeinsum_cuda.so"
        if BUILD_BACKEND == "cuda"
        else "libpyqcdeinsum_kupl.so"
    )
    source = BACKEND_DIR / library_name
    if not source.is_file():
        raise RuntimeError(f"native library was not built: {source}")
    destination = HOME / "pyqcd" / "einsum" / BUILD_BACKEND / library_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 必须原子落位：就地 O_TRUNC 覆盖正在被其它进程 mmap 的 .so 会让其在退出期 SIGSEGV。
    staged = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    shutil.copy2(source, staged)
    os.replace(staged, destination)


class NativeBuildExt(build_ext):
    """Own native compilation before setuptools builds Python extensions."""

    def run(self):
        if os.environ.get("PYQCD_SKIP_NATIVE_BUILD") != "1":
            environment = os.environ.copy()
            environment["PYQCD_EINSUM_BACKEND"] = BUILD_BACKEND
            subprocess.check_call(["bash", str(BUILD_SCRIPT)], cwd=HOME, env=environment)
        super().run()
        _stage_native_library()


class NativeBuildPy(build_py):
    """Build native output before collecting package data into a wheel."""

    def run(self):
        self.run_command("build_ext")
        _stage_native_library()
        super().run()


def _extensions():
    if BUILD_BACKEND != "cuda":
        return []
    if os.environ.get("PYQCD_EINSUM_SKIP_CYTHON") == "1":
        return []

    try:
        from Cython.Build import cythonize
    except ImportError as error:
        raise RuntimeError(
            "CUDA backend requires Cython; install Cython or use `bash install.sh kupl`"
        ) from error

    cuda_dir = HOME / "pyqcd" / "einsum" / "cuda"
    native_dir = HOME / "cpp" / "cuda" / "einsum"
    shared_include = HOME / "cpp" / "kupl" / "einsum" / "include"
    source = cuda_dir / "einsum.pyx"
    header = native_dir / "python" / "pyqcdeinsum.h"
    api_declaration = cuda_dir / "einsum_api.pxd"
    generated_dir = Path(
        os.environ.get(
            "PYQCD_EINSUM_CYTHON_BUILD_DIR",
            HOME / "build" / "cython",
        )
    ).expanduser()

    extension = Extension(
        "pyqcd.einsum.cuda.einsum",
        [str(source)],
        depends=[
            str(header),
            str(shared_include / "einsum_define.h"),
            str(api_declaration),
        ],
        include_dirs=[
            str(cuda_dir),
            str(native_dir / "python"),
            str(native_dir / "include"),
            str(shared_include),
        ],
        library_dirs=[str(native_dir)],
        runtime_library_dirs=["$ORIGIN", str(native_dir)],
        libraries=["pyqcdeinsum_cuda"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
        extra_link_args=["-Wl,-rpath,$ORIGIN", f"-Wl,-rpath,{native_dir}"],
    )
    return cythonize(
        [extension],
        build_dir=str(generated_dir),
        compiler_directives={"language_level": 3},
        force=True,
    )


setup(
    name="pyqcd-einsum",
    version="0.1.0",
    description="Native contraction engine for PyQCD with an opt_einsum-compatible API.",
    packages=[
        "pyqcd",
        "pyqcd.einsum",
        "pyqcd.einsum.backends",
        "pyqcd.einsum.cuda",
        "pyqcd.einsum.kupl",
    ],
    package_data={
        "pyqcd.einsum.cuda": ["*.pyi", "*.pxd", "libpyqcdeinsum_cuda.so"],
        "pyqcd.einsum.kupl": ["libpyqcdeinsum_kupl.so"],
    },
    ext_modules=_extensions(),
    cmdclass={"build_ext": NativeBuildExt, "build_py": NativeBuildPy},
    python_requires=">=3.10",
    zip_safe=False,
)
