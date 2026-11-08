PyQCD 中文文档
==============

PyQCD 是一个用 Python 编写的格点 QCD 工具库，可选配原生加速内核。它提供传播子与
算符收缩、梯度流、重整化、胶子 TMD-PDF 提取所需的基础组件，并自带一个与
``opt_einsum`` 兼容的收缩引擎。

本站只讲两件事：**这个库能做什么**，以及**怎么用**。

.. toctree::
   :maxdepth: 2
   :caption: 目录

   overview
   usage
   install

从哪里开始
----------

* :doc:`overview` —— 各子包提供什么，收缩引擎分几层。
* :doc:`usage` —— 每个主要模块的可运行示例。
* :doc:`install` —— 如何构建原生后端、选择后端、启用 MPI，以及测试入口。

其他语言
--------

* English — https://py-qcd.readthedocs.io/en/latest/usage.html

范围说明
--------

本目录下的页面是为文档站专门编写的功能与用法说明。仓库 ``docs/`` 下另有研究报告
与 LaTeX/PDF 报告，属于独立材料，**不纳入本站**。
