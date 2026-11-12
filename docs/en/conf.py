"""Sphinx configuration for the English PyQCD documentation site.

Sources live in ``docs/en/``; the Chinese translation lives in ``docs/zh/`` and
is built by its own Read the Docs project (``docs/zh/.readthedocs.yaml``).

``pyqcd`` itself is not imported here, so the documentation build never needs
the native einsum libraries (or CUDA) to be present.

Scope: only the hand-written pages of this directory are indexed. The research
notes and LaTeX reports kept directly under ``docs/`` are *not* part of this
site.
"""

project = "PyQCD"
author = "PyQCD contributors"
copyright = "PyQCD contributors"

# 版本从 pyproject.toml 读，避免每次发版手改这里（曾经漏改导致站点停旧版）。
def _project_version() -> str:
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    found = re.search(r'(?m)^version = "([^"]+)"', text)
    return found.group(1) if found else "0.0.0"


version = ".".join(_project_version().split(".")[:2])
release = _project_version()

extensions = ["myst_parser"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"

# The source directory is docs/en/, so the research reports in docs/ and the
# Chinese subtree in docs/zh/ are outside of it and never indexed.
exclude_patterns = ["_build"]
