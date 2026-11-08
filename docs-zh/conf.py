"""PyQCD 中文文档（Read the Docs 项目 ``pyqcd-zh``）的 Sphinx 配置。

构建入口为仓库根的 ``.readthedocs.yaml`` 之外的另一份配置
``docs-zh/readthedocs.yaml``，依赖见 ``docs-zh/requirements.txt``。

注意：这里**不 import pyqcd**，让文档构建不依赖（需要 CUDA 与原生库的）包安装。
"""

project = "PyQCD（中文文档）"
author = "PyQCD contributors"
copyright = "PyQCD contributors"

# 与 setup.py 中的版本保持同步。
version = "0.1"
release = "0.1.0"

# 全站简体中文：interface 文案与搜索分词都用中文。
language = "zh_CN"
html_search_language = "zh"

# myst_parser 负责渲染 docs-zh/ 下已有的中文 Markdown 页面。
extensions = ["myst_parser"]

# 同时支持 Sphinx 原生的 .rst 源与 Markdown 源。
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"

# 不把基准产物与上一次的构建输出带进文档。
exclude_patterns = ["**/artifacts/**", "_build"]
