"""Sphinx configuration for the PyQCD documentation.

Built on Read the Docs with the settings from ``.readthedocs.yaml``; see
``docs/requirements.txt`` for the pinned set of required packages.

Note: ``pyqcd`` itself is intentionally *not* imported here, so the docs build
does not need the (heavy, CUDA-dependent) package to be installed.
"""

project = "PyQCD"
author = "PyQCD contributors"
copyright = "PyQCD contributors"

# Version shown in the docs; kept in sync with setup.py.
version = "0.1"
release = "0.1.0"

# myst_parser renders the Markdown reports that already live in docs/.
extensions = ["myst_parser"]

# Both the Sphinx-native .rst sources and the existing Markdown reports.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"

# Do not pull benchmark artifacts or previous build output into the docs.
exclude_patterns = ["**/artifacts/**", "_build"]
