# 致谢

PyQCD 的对比测试工作与本套文档建立在以下开源项目之上。它们被用作独立实现、
测试对象或参考资料；PyQCD 不把其源码内嵌进运行时。

* NumPy、SciPy、opt_einsum —— 数组语义、参考收缩与收缩规划。
* CuPy —— CUDA 对比所使用的 GPU 数组后端。
* MatX（NVIDIA）—— CUDA 张量库；其 `cutensor::einsum` 是 `cpp/cuda/matx`
  桥的测试对象。其 32 位计算精度被登记为上游限制，不是 PyQCD 的选择。
* Einsums —— C++20 张量代数库；其 `compute_graph::einsum` 是
  `cpp/cpu/cpu-einsums` 桥的测试对象。
* OpenBLAS 与 LAPACK —— 被测引擎使用的 CPU BLAS/LAPACK 实现。
* Sphinx、MyST-Parser 与 Read the Docs —— 文档构建与托管。
* 源码仓库 `refer/` 下归档的格点 QCD 包（PyQUDA、EasyDistillation、
  LQCD_Master、lamet-agent 等）—— API 与工作流参考。

参考仓库的固定提交与许可证记录在源码仓库的 `refer/git-rep/README.md`。
