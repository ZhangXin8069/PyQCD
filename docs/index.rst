PyQCD 文档
===========

PyQCD 是一个面向格点 QCD 计算的 Python / CUDA 框架，涵盖规范场与 Wilson 线、
夸克传播子、物理关联函数、einsum 内核、梯度流以及 PDF/TMD 分析链路等组件。

本页是文档索引。下方目录收录 ``docs/`` 下已有的 Markdown 报告，由 MyST 解析器
（``myst-parser``）渲染；中文 LaTeX 报告（``.tex`` 及其编译后的 ``.pdf``）不在本
Sphinx 工程中构建，请直接阅读源文件。

.. toctree::
   :maxdepth: 1
   :caption: 项目说明与计划

   AGENTS
   EXAMPLES_MIGRATION_20260927
   plan_tmd_full_production_20261101

.. toctree::
   :maxdepth: 1
   :caption: Einsum 内核开发报告

   pyqcd_einsum_overall_20261108
   pyqcd_einsum_agents_20261108
   pyqcd_einsum_bugs_20261108
   pyqcd_einsum_bugfree_20261108
   pyqcd_einsum_build_matrix_20261108
   pyqcd_einsum_build_test_20261108
   pyqcd_einsum_compat_layer_20261108
   pyqcd_einsum_gpu_l24_20261108
   pyqcd_einsum_kernels_20261108
   pyqcd_einsum_performance_status_20261108
   pyqcd_einsum_progress_20261108
   pyqcd_einsum_progress_20261108_0610
   pyqcd_einsum_progress_20261108_0635
   pyqcd_einsum_progress_20261108_0700
   pyqcd_einsum_progress_20261108_0720
   pyqcd_einsum_progress_20261108_0740
   pyqcd_einsum_progress_20261108_0752
   pyqcd_einsum_progress_20261108_0800
   pyqcd_einsum_regression_20261108
   pyqcd_einsum_segfault_analysis_20261108
   pyqcd_einsum_specialization_20261108

.. toctree::
   :maxdepth: 1
   :caption: TMD、胶子 PDF 与结果汇总

   gradient_flow
   gluon_tmd_analysis_notes_20261029
   absorb_dev20_gluon_tmd_audit_20261022
   form_pyqcd_20260928
   form_pyqcd_20261020
   form_pyqcd_20261021
   form_pyqcd_20261027
   form_pyqcd_20261101
   summary_tmd_full_production_20261101
   summary_tmd_ext_production_20261103
