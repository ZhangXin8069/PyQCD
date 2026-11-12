"""Build entry for the PyQCD native layers (KUPL and CUDA).

Project metadata (name, version, dependencies, ...) lives in ``pyproject.toml``;
this file only supplies the build mechanics that PEP 621 cannot express:
compiling/staging the native libraries and the optional Cython CUDA extension.

Backend policy
--------------
* ``PYQCD_EINSUM_BACKEND=kupl`` -> KUPL only.
* ``PYQCD_EINSUM_BACKEND=cuda`` -> KUPL, then CUDA. KUPL is built first so a
  failing CUDA build still leaves a working native library.
* ``PYQCD_EINSUM_BACKEND=auto`` (default) -> KUPL always, plus CUDA when
  ``nvcc``, the CUDA sources and Cython are all available.

KUPL is the baseline native backend, so a failed KUPL build aborts the install:
the alternative is a wheel that ships no native kernel at all and falls back to
the pure-Python planner without telling the user. ``PYQCD_SKIP_NATIVE_BUILD=1``
(and ``PYQCD_EINSUM_REQUIRE_NATIVE=0``) opt into that pure-Python install on
purpose. A CUDA failure is *always* only a warning: the package stays installed
with the KUPL library.
"""

from __future__ import annotations

import os
import shutil
import sys
import subprocess
import time
import warnings
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

HOME = Path(__file__).resolve().parent
RAW_BACKEND = os.environ.get("PYQCD_EINSUM_BACKEND", "auto").strip().lower()
_WARNED_BACKENDS: set[str] = set()

# make.sh 用该退出码表示"产物校验失败"（与编译失败区分）：构建命令本身成功，
# 但产出的 .so 不满足运行期前提（例如缺 OpenBLAS 运行库，装上去 dlopen 会失败）。
# 这种库不能被当作可复用产物，否则源码树里的旧库会掩盖真实缺陷。
VALIDATION_EXIT_CODE = 3


def _notify_progress(message: str) -> None:
    """Report build progress on the terminal.

    pip captures the build backend's output and only replays it on failure, so a
    two-minute compile looks like a hang. Writing to ``/dev/tty`` reaches the user
    anyway; without a terminal it degrades to stderr.
    """
    line = f"[pyqcd build] {message}\n"
    try:
        with open("/dev/tty", "w") as terminal:
            terminal.write(line)
            terminal.flush()
    except OSError:
        sys.stderr.write(line)
        sys.stderr.flush()


def _cuda_sources() -> tuple[Path, ...]:
    source_dir = HOME / "cpp" / "cuda" / "einsum" / "src"
    return tuple(sorted((*source_dir.glob("*.cu"), *source_dir.glob("*.cpp"))))


def _cython_available() -> bool:
    if os.environ.get("PYQCD_EINSUM_SKIP_CYTHON") == "1":
        return False
    try:
        import Cython  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_backends(value: str) -> tuple[str, ...]:
    """Backends to build, in build order; KUPL is always the baseline."""
    if value == "kupl":
        return ("kupl",)
    if value == "cuda":
        return ("kupl", "cuda")
    if value != "auto":
        raise RuntimeError(
            f"invalid PYQCD_EINSUM_BACKEND={value!r}; expected kupl, cuda, or auto")
    if shutil.which(os.environ.get("NVCC", "nvcc")) and _cuda_sources():
        if _cython_available():
            return ("kupl", "cuda")
        print("PyQCD einsum: Cython is unavailable, building the kupl backend only")
    return ("kupl",)


BACKENDS = _resolve_backends(RAW_BACKEND)
BUILD_SCRIPTS = {
    backend: HOME / ("build-nv.sh" if backend == "cuda" else "build-kupl.sh")
    for backend in BACKENDS
}
print(f"PyQCD einsum build backends: {', '.join(BACKENDS)}")


def _native_library_path(backend: str) -> Path:
    """Archive ``build-*.sh`` produces for ``backend``."""
    return HOME / "cpp" / backend / "einsum" / f"libpyqcdeinsum_{backend}.so"


def _build_log(backend: str) -> Path:
    """Compiler log ``make.sh`` leaves behind for ``backend``."""
    return HOME / "cpp" / backend / "einsum" / "build" / "build.log"


def _stage_native_library(backend: str) -> None:
    source = _native_library_path(backend)
    if not source.is_file():
        raise RuntimeError(f"native library was not built: {source}")
    destination = HOME / "pyqcd" / "einsum" / backend / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 必须原子落位：就地 O_TRUNC 覆盖正在被其它进程 mmap 的 .so 会让其在退出期 SIGSEGV。
    staged = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    shutil.copy2(source, staged)
    os.replace(staged, destination)


def _host_compiler_works(compiler: str | None = None) -> bool:
    """Compile a trivial program with the host compiler; False when it cannot."""
    compiler = compiler or os.environ.get("CXX", "c++")
    if shutil.which(compiler) is None:
        return False
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "probe.cpp"
        source.write_text("int main() { return 0; }\n")
        try:
            subprocess.run([compiler, str(source), "-o", str(Path(tmp) / "probe")],
                           check=True, capture_output=True, timeout=180)
        except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
            return False
    return True


def _host_compiler() -> str | None:
    """Compiler ``make.sh`` resolves: ``CXX`` if set, else g++ -> c++.

    Probing ``c++`` alone reports a missing toolchain on clusters whose module
    file only exposes ``g++``, even though ``make.sh`` would have compiled fine.
    """
    explicit = os.environ.get("CXX")
    if explicit:
        return explicit if shutil.which(explicit) else None
    for candidate in ("g++", "c++"):
        if shutil.which(candidate):
            return candidate
    return None


def _toolchain_problem(backend: str) -> str | None:
    """Return why the native build for ``backend`` cannot run, or None."""
    script = BUILD_SCRIPTS[backend]
    if not script.is_file():
        return (f"build script {script.name} is missing; fix: the source tree is "
                "incomplete — re-extract the sdist or re-clone the repository")
    if shutil.which("bash") is None:
        return ("bash is not available on PATH; fix: apt install bash | "
                "dnf install bash | conda install -c conda-forge bash")
    if backend == "cuda":
        # CUDA 需要两套工具链：nvcc 编译 .cu，主机 C++ 编译器编译 Cython 扩展。
        nvcc = os.environ.get("NVCC", "nvcc")
        if shutil.which(nvcc) is None:
            return (f"CUDA compiler {nvcc!r} is not available on PATH; fix: install "
                    "the CUDA toolkit and put nvcc on PATH, or build the CPU backend "
                    "only with PYQCD_EINSUM_BACKEND=kupl (CUDA is opportunistic)")
        host = _host_compiler()
        if host is None:
            return ("host C++ compiler ('CXX', 'g++' or 'c++') is not "
                    "available on PATH; fix: apt install build-essential | "
                    "dnf install gcc-c++ | conda install -c conda-forge gxx_linux-64")
        if not _host_compiler_works(host):
            return (f"host C++ compiler {host!r} cannot compile a trivial program; "
                    "fix: repair the toolchain (apt install build-essential) or set CXX")
        return None
    if shutil.which("nm") is None:
        # make.sh 用 nm 校验 applyPlanQcd 导出符号；缺 nm 时它只会报“符号缺失”。
        return ("binutils ('nm') is not available on PATH; fix: apt install binutils | "
                "dnf install binutils | conda install -c conda-forge binutils")
    if _host_compiler() is None:
        return ("host C++ compiler ('CXX', 'g++' or 'c++') is not "
                "available on PATH; fix: apt install build-essential | "
                "dnf install gcc-c++ | conda install -c conda-forge gxx_linux-64")
    return None


def _native_required(backend: str) -> bool:
    """Whether a missing ``backend`` library must abort the install.

    KUPL is the baseline: without it the wheel carries no native kernel, so the
    build fails loudly unless the user explicitly asked for the pure-Python
    planner (``PYQCD_SKIP_NATIVE_BUILD=1`` / ``PYQCD_EINSUM_REQUIRE_NATIVE=0``).
    CUDA stays opportunistic because KUPL is still installed.
    """
    if backend == "cuda":
        return False
    if os.environ.get("PYQCD_SKIP_NATIVE_BUILD") == "1":
        return False
    return os.environ.get("PYQCD_EINSUM_REQUIRE_NATIVE", "1") != "0"


def _skip_native(reason: str, backend: str) -> None:
    """Warn about a skipped/failed native build, or abort for the KUPL baseline."""
    detail = f"PyQCD einsum: {reason}. The native {backend} library is not installed."
    if backend != "cuda" and _native_required(backend):
        raise RuntimeError(
            f"{detail} KUPL is the baseline native backend, so this install would "
            "ship no native kernel and silently fall back to the pure-Python "
            "planner (pyqcd/einsum/_plan_py.py) at runtime. Install a C++ "
            "toolchain (CXX/g++/c++, bash and binutils) and reinstall, or ask for "
            "the pure-Python planner on purpose with PYQCD_SKIP_NATIVE_BUILD=1 "
            "(or PYQCD_EINSUM_REQUIRE_NATIVE=0).")
    if backend != "cuda":
        detail += (" pyqcd.einsum then runs the pure-Python planner "
                   "(pyqcd/einsum/_plan_py.py).")
    # CUDA 是机会性构建：失败只警告，KUPL 仍是可用的原生后端。
    if backend in _WARNED_BACKENDS:
        return
    _WARNED_BACKENDS.add(backend)
    suffix = ("pyqcd.einsum keeps using the kupl backend."
              if backend == "cuda"
              else "Continuing with the pure-Python planner because "
                   "PYQCD_SKIP_NATIVE_BUILD=1 / PYQCD_EINSUM_REQUIRE_NATIVE=0.")
    warnings.warn(f"{detail} {suffix}", RuntimeWarning, stacklevel=2)


def _stage_built_libraries() -> None:
    """Stage every backend whose library exists; never abort the install."""
    for backend in BACKENDS:
        try:
            _stage_native_library(backend)
        except (RuntimeError, OSError) as error:
            _skip_native(str(error), backend)


class NativeBuildExt(build_ext):
    """Own native compilation before setuptools builds Python extensions.

    A failing CUDA build degrades the install to the KUPL library, which stays
    opportunistic. A failing KUPL build aborts the install (see
    :func:`_native_required`) unless the user opted into the pure-Python planner.
    """

    def _build_backend(self, backend: str) -> bool:
        problem = _toolchain_problem(backend)
        if problem is not None:
            _skip_native(problem, backend)
            return False
        environment = os.environ.copy()
        environment["PYQCD_EINSUM_BACKEND"] = backend
        if backend != "cuda" and "CXX" not in os.environ:
            # make.sh 默认用 g++；只有它缺席时才显式透传探测到的编译器（如仅有 c++），
            # 不影响 CXX 已设置或 g++ 存在（含 MPI 自动探测）的既有成功路径。
            resolved = _host_compiler()
            if resolved is not None and resolved != "g++":
                environment["CXX"] = resolved
        started = time.time()
        _notify_progress(f"compiling the native {backend} library (this takes a "
                         f"while; pip will stay silent until it finishes)")
        try:
            subprocess.check_call(
                ["bash", str(BUILD_SCRIPTS[backend])], cwd=HOME, env=environment)
        except subprocess.CalledProcessError as error:
            if (error.returncode != VALIDATION_EXIT_CODE
                    and _native_library_path(backend).is_file()):
                # 源码树里已有可复用的库（重复安装、增量开发）：不必因本次失败中止。
                # 产物校验失败（退出码 3）不走这条路：那个旧库同样不可信。
                _notify_progress(
                    f"reusing the existing native {backend} library "
                    f"({BUILD_SCRIPTS[backend].name} exited with "
                    f"{error.returncode})")
                return True
            log = _build_log(backend)
            detail = (f"native build failed ({BUILD_SCRIPTS[backend].name} exited "
                      f"with {error.returncode}")
            if log.is_file():
                detail += f"; compiler log: {log}"
            _skip_native(f"{detail})", backend)
            return False
        _notify_progress(f"native {backend} library built in "
                         f"{time.time() - started:.1f}s")
        return True

    def run(self):
        built: dict[str, bool] = {}
        skip = os.environ.get("PYQCD_SKIP_NATIVE_BUILD") == "1"
        for backend in BACKENDS:
            built[backend] = (
                _native_library_path(backend).is_file() if skip
                else self._build_backend(backend)
            )
        if not built.get("cuda", False):
            # CUDA 扩展链接 libpyqcdeinsum_cuda.so；没有它时编译必然失败。
            self.extensions = [
                extension for extension in self.extensions
                if not extension.name.startswith("pyqcd.einsum.cuda")
            ]
        super().run()
        _stage_built_libraries()


class NativeBuildPy(build_py):
    """Build native output before collecting package data into a wheel."""

    def run(self):
        self.run_command("build_ext")
        _stage_built_libraries()
        super().run()


def _extensions():
    if "cuda" not in BACKENDS:
        return []
    if os.environ.get("PYQCD_EINSUM_SKIP_CYTHON") == "1":
        return []
    if (os.environ.get("PYQCD_SKIP_NATIVE_BUILD") == "1"
            and not _native_library_path("cuda").is_file()):
        # 扩展要链接原生 CUDA 库；跳过原生构建且库里不存在时，链接注定失败。
        _skip_native(
            f"native library is missing ({_native_library_path('cuda')})", "cuda")
        return []

    try:
        from Cython.Build import cythonize
    except ImportError:
        _skip_native("Cython is unavailable, so the CUDA Cython extension "
                     "cannot be built; fix: pip install cython (or keep the CPU "
                     "backend with PYQCD_EINSUM_SKIP_CYTHON=1)", "cuda")
        return []

    def _rel(path: Path) -> str:
        """Prefer a HOME-relative path so SOURCES.txt never leaks the user's home."""
        try:
            return str(Path(path).relative_to(HOME))
        except ValueError:
            return str(path)

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
        [_rel(source)],
        depends=[
            _rel(header),
            _rel(shared_include / "einsum_define.h"),
            _rel(api_declaration),
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
    extensions = cythonize(
        [extension],
        build_dir=_rel(generated_dir),
        compiler_directives={"language_level": 3},
        force=True,
    )
    # cythonize() rewrites `depends` to absolute paths, which makes setuptools warn
    # ("the path must be relative") when it computes the manifest. Re-relativize
    # them so the sdist step stays warning-free and SOURCES.txt never leaks $HOME.
    for ext in extensions:
        ext.depends = [_rel(Path(dep)) for dep in (ext.depends or [])]
    return extensions


setup(
    # name/version/description/requires-python and the rest of the metadata are
    # declared once in pyproject.toml ([project]); declaring them here too makes
    # setuptools refuse to build.
    packages=find_packages(
        include=["pyqcd", "pyqcd.*"],
        exclude=["pyqcd.testing", "pyqcd.testing.*"],
    ),
    package_data={
        "pyqcd": ["py.typed"],
        "pyqcd.einsum.cuda": ["*.pyi", "*.pxd", "libpyqcdeinsum_cuda.so"],
        "pyqcd.einsum.kupl": ["libpyqcdeinsum_kupl.so"],
    },
    ext_modules=_extensions(),
    cmdclass={"build_ext": NativeBuildExt, "build_py": NativeBuildPy},
    zip_safe=False,
)
