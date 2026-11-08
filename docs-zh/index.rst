PyQCD 中文文档
==============

PyQCD（PyQCD By ZhangXin）是面向格点 QCD 的 Python / CUDA 计算仓库，核心目标是
**使用梯度流重整化方案计算核子中的胶子 TMD-PDF**。仓库提供格点常数与 γ 矩阵、
规范场与 Wilson 线、夸克传播子、Wick 收缩与重子算符、物理关联函数、einsum 收缩
内核、梯度流、重整化与 PDF/TMD 分析链路等组件。

本页是中文文档（Read the Docs 项目 ``pyqcd-zh``）的索引。这里的页面是 ``docs-zh/``
目录下的 Markdown 源文件，由 MyST 解析器（``myst-parser``）渲染。仓库 ``docs/``
目录下另有大量中文 Markdown 报告与 LaTeX/PDF 报告，由主项目（仓库根的
``.readthedocs.yaml`` 搭配 ``docs/conf.py``）的 Sphinx 工程收录，本工程不重复收录。

.. toctree::
   :maxdepth: 2
   :caption: 中文文档

   overview
   install

阅读顺序
--------

建议先读「项目总览」了解仓库定位、主要模块与 einsum 子系统的现状，再读
「安装与构建」完成本地构建、确认后端与 MPI 标识，并找到测试入口。

说明
----

- 站点语言由 ``docs-zh/conf.py`` 中的 ``language = "zh_CN"`` 与
  ``html_search_language = "zh"`` 设定，全站为简体中文。
- 构建配置是 ``docs-zh/readthedocs.yaml``；Read the Docs 项目默认只读仓库根的
  ``.readthedocs.yaml``，所以需要在项目 Settings 的 "Configuration file" 中把路径
  指到 ``docs-zh/readthedocs.yaml``。
- 文档构建只安装 ``docs-zh/requirements.txt`` 中的依赖，不安装 ``pyqcd`` 包本身。
