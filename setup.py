"""Build entry for the PyQCD einsum native and CUDA Cython layers.

Project metadata (name, version, dependencies, ...) lives in ``pyproject.toml``;
this file only supplies the build mechanics that PEP 621 cannot express:
compiling/staging the native library and the optional Cython CUDA extension.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings
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
        try:
            import Cython  # noqa: F401
        except ImportError:
            # 隔离构建环境里没有 Cython 时，CUDA 扩展无法编译；
            # 明确退到纯 CPU 的 KUPL 后端，而不是让 pip install 直接失败。
            print("PyQCD einsum: Cython is unavailable, using the kupl backend")
            return "kupl"
        return "cuda"
    return "kupl"


BUILD_BACKEND = _resolve_backend(RAW_BACKEND)
BACKEND_DIR = HOME / "cpp" / BUILD_BACKEND / "einsum"
BUILD_SCRIPT = HOME / ("build-nv.sh" if BUILD_BACKEND == "cuda" else "build-kupl.sh")

print(f"PyQCD einsum build backend: {BUILD_BACKEND}")
print(f"native library directory: {BACKEND_DIR}")

_SKIP_WARNED = False


def _native_library_path() -> Path:
    """Archive ``build-*.sh`` produces for the selected backend."""
    return BACKEND_DIR / f"libpyqcdeinsum_{BUILD_BACKEND}.so"


def _stage_native_library() -> None:
    source = _native_library_path()
    if not source.is_file():
        raise RuntimeError(f"native library was not built: {source}")
    destination = HOME / "pyqcd" / "einsum" / BUILD_BACKEND / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 必须原子落位：就地 O_TRUNC 覆盖正在被其它进程 mmap 的 .so 会让其在退出期 SIGSEGV。
    staged = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    shutil.copy2(source, staged)
    os.replace(staged, destination)


def _toolchain_problem() -> str | None:
    """Return why the native build cannot run, or None when it can."""
    if not BUILD_SCRIPT.is_file():
        return f"build script {BUILD_SCRIPT.name} is missing"
    if shutil.which("bash") is None:
        return "bash is not available on PATH"
    # CUDA 后端需要两套工具链：nvcc 编译 .cu，主机 C++ 编译器编译 Cython 扩展。
    compilers = [("CXX", "c++", "host C++ compiler")]
    if BUILD_BACKEND == "cuda":
        compilers.insert(0, ("NVCC", "nvcc", "CUDA compiler"))
    for variable, default, description in compilers:
        compiler = os.environ.get(variable, default)
        if shutil.which(compiler) is None:
            return f"{description} {compiler!r} is not available on PATH"
    return None


def _skip_native(reason: str) -> None:
    """Warn once about a skipped native build, or fail when it was requested."""
    detail = (
        f"PyQCD einsum: {reason}. Installing without the native {BUILD_BACKEND} "
        "library; pyqcd.einsum then runs the pure-Python planner "
        "(pyqcd/einsum/_plan_py.py)."
    )
    if os.environ.get("PYQCD_EINSUM_REQUIRE_NATIVE") == "1":
        raise RuntimeError(
            f"{detail} PYQCD_EINSUM_REQUIRE_NATIVE=1 is set, so this is a hard error.")
    global _SKIP_WARNED
    if not _SKIP_WARNED:
        _SKIP_WARNED = True
        warnings.warn(
            f"{detail} Set PYQCD_EINSUM_REQUIRE_NATIVE=1 to make this a hard error.",
            RuntimeWarning, stacklevel=2)


def _stage_native_library_or_fallback() -> None:
    """Stage the native library, degrading to the pure-Python plan on failure."""
    try:
        _stage_native_library()
    except (RuntimeError, OSError) as error:
        _skip_native(str(error))


class NativeBuildExt(build_ext):
    """Own native compilation before setuptools builds Python extensions.

    A missing toolchain (or a failing compiler) degrades the install to the
    pure-Python planner instead of aborting it; the sdist is installable on
    machines without a C++ compiler or CUDA.
    """

    def run(self):
        native_ready = True
        if os.environ.get("PYQCD_SKIP_NATIVE_BUILD") != "1":
            problem = _toolchain_problem()
            if problem is not None:
                _skip_native(problem)
                native_ready = False
            else:
                environment = os.environ.copy()
                environment["PYQCD_EINSUM_BACKEND"] = BUILD_BACKEND
                try:
                    subprocess.check_call(
                        ["bash", str(BUILD_SCRIPT)], cwd=HOME, env=environment)
                except subprocess.CalledProcessError as error:
                    _skip_native(
                        f"native build failed ({BUILD_SCRIPT.name} exited "
                        f"with {error.returncode})")
                    native_ready = False
        elif not _native_library_path().is_file():
            native_ready = False
        if not native_ready:
            # 没有原生库时，链接它的 Cython 扩展必然失败；纯 Python 规划等价。
            self.extensions = []
        super().run()
        _stage_native_library_or_fallback()


class NativeBuildPy(build_py):
    """Build native output before collecting package data into a wheel."""

    def run(self):
        self.run_command("build_ext")
        _stage_native_library_or_fallback()
        super().run()


def _extensions():
    if BUILD_BACKEND != "cuda":
        return []
    if os.environ.get("PYQCD_EINSUM_SKIP_CYTHON") == "1":
        return []
    if (os.environ.get("PYQCD_SKIP_NATIVE_BUILD") == "1"
            and not _native_library_path().is_file()):
        # 扩展要链接原生 CUDA 库；跳过原生构建且库里不存在时，链接注定失败。
        _skip_native(f"native library is missing ({_native_library_path()})")
        return []

    try:
        from Cython.Build import cythonize
    except ImportError:
        _skip_native("Cython is unavailable, so the CUDA Cython extension "
                     "cannot be built")
        return []

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
    # name/version/description/requires-python and the rest of the metadata are
    # declared once in pyproject.toml ([project]); declaring them here too makes
    # setuptools refuse to build.
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
    zip_safe=False,
)
