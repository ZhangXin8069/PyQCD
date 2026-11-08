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

version = "0.1"
release = "0.1.11"

extensions = ["myst_parser"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"

# The source directory is docs/en/, so the research reports in docs/ and the
# Chinese subtree in docs/zh/ are outside of it and never indexed.
exclude_patterns = ["_build"]
