# 致谢

PyQCD 的推导、实现与验证，建立在他人首先公开发表的算法和工作流之上，也依赖于
一批开源数值与文档工具。谨向下列所有作者与维护者致谢。

本页是本站的最后一页：PyQCD 所参考的代码与 Git 仓库统一在此致谢，而不散落在
正文各章之中。

## 开源软件

* NumPy、SciPy、opt_einsum —— 数组语义、参考收缩与收缩规划。
* CuPy —— CUDA 对比所使用的 GPU 数组后端。
* MatX（NVIDIA）—— CUDA 张量库；其 `cutensor::einsum` 曾是（现已移除的）
  `cpp/cuda/matx` 桥的测试对象。其 32 位计算精度被登记为上游限制，不是 PyQCD 的选择。
* Einsums —— C++20 张量代数库；其 `compute_graph::einsum` 曾是（现已移除的）
  `cpp/cpu/cpu-einsums` 桥的测试对象。
* OpenBLAS 与 LAPACK —— 被测引擎使用的 CPU BLAS/LAPACK 实现。
* Sphinx、MyST-Parser 与 Read the Docs —— 文档构建与托管。

## 参考代码

源码仓库的 `refer/` 目录按贡献者归档了 PyQCD 算法、约定与重整化逻辑的参考来源。
PyQCD 重新实现其逻辑，不 import、也不把参考实现内嵌进运行时。

| 目录 | 贡献者 | 参考内容 |
|---|---|---|
| `refer/donghx/` | 董 HX | 质子 2pt 蒸馏与 OPE 算符 |
| `refer/zhangxin/` | 张 X | 完整胶子 PDF 工作流与数据分析框架 |
| `refer/huangcl/` | 黄 CL | 多步 Chroma 流水线（contract、ratio、energy） |
| `refer/sush/` | 苏 SH | 蒸馏收缩框架与 `lqcddb` 包 |
| `refer/zengch/` | 曾 C | 重整化、混合方案与拟合逻辑（`pyqcd/renorm` 的移植来源） |
| `refer/papers/`、`refer/books/` | — | 物理推导与验证所用的文献与教材 |

## 参考 Git 仓库

下列外部仓库以 Git submodule 固定在 `refer/git-rep/<name>`，工作树保持上游 commit
的只读状态。它们被用作软件参考、对比对象与架构范例，均未内嵌进发布包。

| 项目 | 来源 | 参考内容 |
|---|---|---|
| PyQUDA | https://github.com/CLQCD/PyQUDA | QUDA 的 Python/Cython 封装，面向 GPU/MPI 格点 QCD |
| LQCD_Skills | https://github.com/AI4LGT/LQCD_Skills | 从物理观测量到数值结果的格点 QCD agent 技能集合 |
| lamet-agent | https://github.com/Greyyy-HJC/lamet-agent | LaMET/LQCD 分析 agent 的 Python 脚手架 |
| EasyDistillation | https://github.com/IHEP-LQCD/EasyDistillation | 蒸馏方法（本征矢、perambulator、elemental 自动缩并）的多后端实现 |
| LQCD_Master | https://github.com/sjtu-sai-agents/LQCD_Master | 自然语言到 PyQUDA 程序与 Slurm 脚本的 agentic 流水线 |
| PyQCU | https://gitlab.itp.ac.cn/zhangxin8069/PyQCU | torch、mpi4py 与 h5py 架构范例 |
| opt_einsum | https://github.com/dgasmith/opt_einsum | 收缩路径规划与成本诊断 |
| Einsums | https://github.com/Einsums/Einsums | C++20 张量代数、PackedGemm 与延迟执行 |
| MatX | https://github.com/NVIDIA/MatX | CUDA C++17 张量库与 GPU `einsum` 后端 |

`PyQUDA` 内部以 `pycparser` 作为嵌套 submodule，需
`git submodule update --init --recursive` 初始化。各库的固定 commit、许可证与用途
清单记录在 `refer/git-rep/README.md`。

## 谨致谢意

感谢所有把代码、工作流与笔记归档到 `refer/` 的作者，感谢上表每个开源项目的
作者与维护者。正是这些公开的软件与文档，使本工具库得以基于独立实现重新推导、
交叉检查并修正自身；若仍有错误，责任完全在 PyQCD 一方。

这些是格点 QCD 领域优秀的软件工具包。
