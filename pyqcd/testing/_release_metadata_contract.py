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

import tomllib
from pathlib import Path

import pyqcd
import pyqcd.einsum._version as _version

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "pyqcd"


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_versions_agree() -> None:
    project = _pyproject()
    declared = project["project"]["version"]
    assert declared == pyqcd.__version__, (
        f"pyproject.toml says {declared}, pyqcd.__version__ says {pyqcd.__version__}")
    assert declared == _version.__version__, (
        f"pyproject.toml says {declared}, einsum._version says {_version.__version__}")
    assert project["project"]["name"] == "pyqcd"


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


def test_build_info_shape() -> None:
    import pyqcd.einsum as pe

    info = pe.build_info()
    for key in ("backend", "native_available", "array_backend",
                "native_library"):
        assert key in info, f"build_info() is missing {key!r}"
    assert info["backend"] in {"cuda", "kupl", "python"}
    assert info["native_available"] is (info["backend"] != "python")
