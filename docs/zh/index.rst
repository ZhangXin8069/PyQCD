PyQCD 中文文档
==============

``pyqcd`` 是发布在 PyPI 上的格点 QCD 工具库。它提供传播子与算符收缩、梯度流、
重整化、胶子 TMD-PDF 提取所需的基础组件，并自带一个与 ``opt_einsum`` 兼容的收缩引擎。

.. code-block:: bash

   pip install pyqcd

本站以**已安装的包**为出发点：所有示例都假定 ``import pyqcd`` 可用，不需要仓库源码。

.. toctree::
   :maxdepth: 2
   :caption: 目录

   overview
   usage
   performance
   install

从哪里开始
----------

* :doc:`overview` —— 安装之后这个包能提供什么。
* :doc:`usage` —— 每个主要模块的可运行示例。
* :doc:`performance` —— 相对 ``opt_einsum`` 的实测加速，以及不适用的情况。
* :doc:`install` —— 安装、可选 extras、原生后端与可调开关。

代码托管
--------

三个平台同步同一份历史：

* 国内开发（主开发，GitLab）— https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD
* 国内发布（国内镜像，Gitee）— https://gitee.com/zhangxin8069/PyQCD
* 国外发布（海外镜像，GitHub）— https://github.com/zhangxin8069/PyQCD

其他语言
--------

* English — https://py-qcd.readthedocs.io/en/latest/usage.html
