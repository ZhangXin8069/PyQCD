"""Python 3.9-compatible TOML loading for release metadata tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _parser() -> Any:
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            try:
                from setuptools._vendor import tomli as tomllib  # type: ignore[no-redef]
            except (ImportError, ModuleNotFoundError):
                try:
                    from pip._vendor import tomli as tomllib  # type: ignore[no-redef]
                except (ImportError, ModuleNotFoundError) as error:
                    raise ImportError(
                        "Python < 3.11 requires `tomli` (or setuptools/pip "
                        "vendored TOML support) for release metadata checks") from error
    return tomllib


def load(path: str | Path) -> dict[str, Any]:
    """Strictly load TOML; malformed metadata never gets a string fallback."""
    with Path(path).open("rb") as handle:
        return _parser().load(handle)
