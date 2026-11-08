"""Sphinx configuration for the PyQCD documentation site.

The English sources live in ``docs/`` (``index.rst`` plus ``guide/*.md``); the
Chinese translation lives in ``docs/zh/`` and is excluded from this build.

``pyqcd`` itself is not imported here, so the documentation build never needs
the native einsum libraries (or CUDA) to be present.

Note: the research notes and LaTeX reports stored in ``docs/`` are *not* part
of this site — only the hand-written ``guide/`` pages are indexed.
"""

project = "PyQCD"
author = "PyQCD contributors"
copyright = "PyQCD contributors"

version = "0.1"
release = "0.1.0"

extensions = ["myst_parser"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"

# Only the hand-written guide pages are part of this site: the Chinese subtree
# is built by its own project, and benchmark artifacts must never be indexed.
exclude_patterns = ["_build", "zh", "**/artifacts/**"]
