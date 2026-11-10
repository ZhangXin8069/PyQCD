PyQCD
=====

``pyqcd`` is a lattice-QCD toolkit distributed on PyPI. It ships the building
blocks for propagator and operator contractions, gradient flow, renormalisation
and gluon TMD-PDF extraction, together with a self-contained contraction engine
that exposes an ``opt_einsum``-compatible API.

.. code-block:: bash

   pip install pyqcd

This site is written from the point of view of the installed package: every
example assumes ``import pyqcd`` works and needs nothing else from the
repository.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   usage
   performance
   install
   acknowledgements
   appendix

Where to start
--------------

* :doc:`overview` — what the package provides once installed.
* :doc:`usage` — runnable examples for every major module.
* :doc:`performance` — measured speed-ups over ``opt_einsum``, and where they
  do not apply.
* :doc:`install` — installation, optional extras, native backends and knobs.

Repositories
------------

All three mirrors carry the same history:

* 国内开发 (primary, GitLab) — https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD
* 国内发布 (China mirror, Gitee) — https://gitee.com/zhangxin8069/PyQCD
* 国外发布 (global mirror, GitHub) — https://github.com/zhangxin8069/PyQCD

Other languages
---------------

* 中文 — https://py-qcd.readthedocs.io/zh-cn/latest/usage.html
