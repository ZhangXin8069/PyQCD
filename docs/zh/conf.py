"""PyQCD 中文文档（Read the Docs 项目 ``pyqcd-zh``）的 Sphinx 配置。

源文件位于 ``docs/zh/``；英文站点使用同一 ``docs/`` 下的 ``conf.py``，
其 ``exclude_patterns`` 已把本目录排除，两者互不干扰。

本文件**不 import pyqcd**，因此文档构建不依赖原生库（也不需要 CUDA）。
"""

project = "PyQCD（中文文档）"
author = "PyQCD contributors"
copyright = "PyQCD contributors"

version = "0.1"
release = "0.3.4"

language = "zh_CN"
html_search_language = "zh"

extensions = ["myst_parser"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"

exclude_patterns = ["_build", "**/artifacts/**"]
