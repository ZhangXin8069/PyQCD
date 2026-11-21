"""Metadata consistency guard for the published distribution.

A release drifts when the version in ``pyproject.toml`` and the version the code
reports disagree, or when the code ships a subpackage the metadata forgets. Both
have happened in this repository, so they are asserted here.

Checks
------
* ``pyproject.toml`` version == ``pyqcd.__version__`` == ``pyqcd.einsum._version``
* ``pyqcd.__all__`` matches the subpackage directories on disk
* every subpackage is importable from a source checkout
* the sdist manifest excludes ``pyqcd.testing`` and includes ``pyqcd/py.typed``
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch

import pyqcd
import pyqcd.einsum._version as _version
from pyqcd.testing._toml_compat import load as load_toml

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "pyqcd"


def _pyproject() -> dict:
    return load_toml(ROOT / "pyproject.toml")


def test_versions_agree() -> None:
    project = _pyproject()
    declared = project["project"]["version"]
    assert declared == pyqcd.__version__, (
        f"pyproject.toml says {declared}, pyqcd.__version__ says {pyqcd.__version__}")
    assert declared == _version.__version__, (
        f"pyproject.toml says {declared}, einsum._version says {_version.__version__}")
    assert project["project"]["name"] == "pyqcd"


def test_python39_toml_fallback_reads_project_metadata() -> None:
    """The 3.9 compatibility loader must use a strict TOML parser."""
    project = load_toml(ROOT / "pyproject.toml")["project"]
    assert project["name"] == "pyqcd"
    assert project["version"] == _version.__version__
    assert isinstance(project["optional-dependencies"], dict)


def test_public_subpackages_exist() -> None:
    on_disk = {
        path.name for path in PACKAGE_DIR.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
        and path.name != "testing"
    }
    declared = set(pyqcd.__all__)
    assert declared == on_disk, (
        f"pyqcd.__all__ and the package directories disagree: "
        f"only in __all__={sorted(declared - on_disk)}, "
        f"only on disk={sorted(on_disk - declared)}")


def test_subpackages_import() -> None:
    import importlib

    for name in sorted(pyqcd.__all__):
        importlib.import_module(f"pyqcd.{name}")


def test_manifest_matches_policy() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "prune pyqcd/testing" in manifest, (
        "pyqcd/testing must never be shipped in the sdist")
    assert "include pyqcd/py.typed" in manifest, (
        "PEP 561 marker must be shipped with the sdist")
    assert "global-exclude AGENTS.md" in manifest, (
        "internal contributor notes must not ship")
    assert "prune data/tmp" in manifest, (
        "temporary agent/build data must not ship")
    assert "prune cpp/kupl/einsum/tests" in manifest, (
        "compiled C++ test sources must not ship")


def test_missing_nvcc_keeps_kupl_install_and_explains_cuda_skip() -> None:
    """CUDA 工具链缺席时应只跳过 CUDA，明确提示，并保留 KUPL 安装路径。"""
    environment = os.environ.copy()
    environment["NVCC"] = "/__pyqcd_missing_nvcc__/nvcc"
    environment["PYQCD_EINSUM_BACKEND"] = "auto"
    process = subprocess.run(
        [sys.executable, "setup.py", "--name"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = process.stdout + process.stderr
    assert process.returncode == 0, output
    assert "CUDA backend not built because nvcc is not on PATH" in output, output
    assert "locally compiled KUPL backend will still be installed" in output, output
    assert "PyQCD einsum build backends: kupl" in output, output


def _rebind_scope_function(scope: dict, name: str):
    """Bind a function loaded by ``runpy`` to the returned scope dictionary."""
    original = scope[name]
    rebound = types.FunctionType(
        original.__code__, scope, original.__name__, original.__defaults__,
        original.__closure__)
    rebound.__kwdefaults__ = original.__kwdefaults__
    return rebound


def _setup_scope() -> tuple[dict, dict]:
    captured: dict = {}
    environment = {
        "PYQCD_EINSUM_BACKEND": "kupl",
        "PYQCD_EINSUM_SKIP_CYTHON": "1",
    }
    with patch.dict(os.environ, environment, clear=False), \
            patch("setuptools.setup", side_effect=lambda **kwargs: captured.update(kwargs)):
        scope = runpy.run_path(str(ROOT / "setup.py"), run_name="pyqcd_setup_contract")
    return scope, captured


def test_setup_never_packages_archives_from_stale_backend_directories() -> None:
    """KUPL-only setup metadata must not name a stale CUDA archive."""
    scope, captured = _setup_scope()
    package_data = captured["package_data"]
    flattened = [
        pattern
        for patterns in package_data.values()
        for pattern in patterns
    ]
    assert not any(pattern.endswith(".so") for pattern in flattened), package_data

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        kupl = root / "kupl.so"
        cuda = root / "cuda.so"
        kupl.write_bytes(b"kupl")
        cuda.write_bytes(b"stale cuda")
        paths = {"kupl": kupl, "cuda": cuda}
        scope["BACKENDS"] = ("kupl",)
        scope["_native_library_path"] = lambda backend: paths[backend]
        scope["_missing_native_exports"] = lambda path, backend: ()
        verified = _rebind_scope_function(scope, "_native_package_data")()

    assert "pyqcd.einsum.kupl" in verified, verified
    assert "libpyqcdeinsum_cuda.so" not in verified.get(
        "pyqcd.einsum.cuda", []), verified


def test_staging_rejects_archive_with_missing_required_exports() -> None:
    """The staging path itself, not only a source-text test, must fail closed."""
    scope, _captured = _setup_scope()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = root / "source.so"
        source.write_bytes(b"synthetic")
        scope["HOME"] = root
        scope["_native_library_path"] = lambda backend: source
        scope["_missing_native_exports"] = (
            lambda path, backend: ("qcd_cuda_topology_apply",))
        try:
            _rebind_scope_function(scope, "_stage_native_library")("cuda")
        except RuntimeError as error:
            assert "missing required exports" in str(error), error
        else:
            raise AssertionError("a native archive with missing exports was staged")
        assert not (
            root / "pyqcd" / "einsum" / "cuda" / "libpyqcdeinsum_cuda.so"
        ).exists()


def test_build_config_records_the_installed_backend_set() -> None:
    """The installed package must retain why each backend is present or absent."""
    scope, _captured = _setup_scope()
    scope["BACKENDS"] = ("kupl",)
    scope["RAW_BACKEND"] = "auto"
    scope["_BUILD_NOTES"] = ["CUDA backend not built because nvcc is not on PATH"]
    scope["_native_package_data"] = lambda: {
        "pyqcd": ["py.typed"],
        "pyqcd.einsum.kupl": ["libpyqcdeinsum_kupl.so"],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        _rebind_scope_function(scope, "_write_build_config")(tmpdir)
        payload = json.loads(
            (Path(tmpdir) / "pyqcd" / "einsum" / "_build_config.json")
            .read_text(encoding="utf-8"))

    assert payload["requested_backend"] == "auto", payload
    assert payload["configured_backends"] == ["kupl"], payload
    assert payload["installed_backends"] == ["kupl"], payload
    assert "nvcc is not on PATH" in payload["notes"][0], payload


def test_build_info_shape() -> None:
    import pyqcd.einsum as pe

    info = pe.build_info()
    for key in ("backend", "native_available", "array_backend",
                "native_library"):
        assert key in info, f"build_info() is missing {key!r}"
    assert info["backend"] in {"cuda", "kupl", "python"}
    assert info["native_available"] is (info["backend"] != "python")


def test_pypi_description_probes_match_the_published_readme() -> None:
    """Keep release_check's PyPI page probes tied to the actual README headings."""
    from pyqcd.testing.release_check import PYPI_DESCRIPTION_PROBES

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = [label for label, probe in PYPI_DESCRIPTION_PROBES.items()
               if probe not in readme]
    assert not missing, f"README is missing release_check probes: {missing}"
