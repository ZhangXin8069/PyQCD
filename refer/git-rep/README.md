# refer/git-rep 参考库清单

## 致谢

谨向下列每个项目的作者与维护者致谢：感谢你们把软件与文档公开发布，使 PyQCD 的
算法实现、接口设计与物理工作流得以基于独立实现重新推导、交叉检查并修正。

### PyQUDA

感谢 CLQCD 公开 QUDA 的 Python/Cython 封装，为 GPU/MPI 格点计算的组织方式提供了范例。

### LQCD_Skills

感谢 AI4LGT 公开从物理观测量到数值结果的格点 QCD agent 技能集合。

### lamet-agent

感谢 Greyyy-HJC 公开 LaMET/LQCD 分析 agent 的 Python 脚手架。

### EasyDistillation

感谢 IHEP-LQCD 公开多后端蒸馏实现：本征矢、perambulator 与 elemental 自动缩并。

### LQCD_Master

感谢 sjtu-sai-agents 公开自然语言到 PyQUDA 程序与 Slurm 脚本的 agentic 流水线。

### PyQCU

感谢维护者公开 torch、mpi4py 与 h5py 的架构范例。

### opt_einsum

感谢 dgasmith 公开收缩路径规划与成本诊断工具。

### Einsums

感谢 Einsums 项目公开 C++20 张量代数、PackedGemm 与延迟执行实现。

### MatX

感谢 NVIDIA 公开 CUDA C++17 张量库与 GPU `einsum` 后端。

### pycparser

感谢 Eliben 等项目作者公开 C 解析器；它作为 PyQUDA 的嵌套 submodule 被使用。

## 参考库清单

本目录只读收录 lattice QCD 参考软件、技能库与数值基础设施库，全部以 Git submodule 固定在上游
commit。子库工作树保持上游状态，不修改、不把上游文件混入主库源码。本库对这些库的
分析文档统一存放于 `refer/git-rep/docs/`。

| 路径 | 来源 | 用途 | 固定 commit |
|---|---|---|---|
| `refer/git-rep/PyQUDA` | https://github.com/CLQCD/PyQUDA | QUDA 的 Python/Cython 封装，面向 GPU/MPI 格点 QCD 计算 | `138d8ad6e4d6e0805ecc7ec0a59a45febe0482ce` |
| `refer/git-rep/LQCD_Skills` | https://github.com/AI4LGT/LQCD_Skills | 从物理观测量到数值结果的格点 QCD agent 技能集合 | `90bcbb8739c80b41a3965f41dab50bdfc9841b16` |
| `refer/git-rep/lamet-agent` | https://github.com/Greyyy-HJC/lamet-agent | LaMET/LQCD 分析 agent 的 Python 脚手架 | `9355c52ab7f5061fcc1befacfb7e1b65a0b32db1` |
| `refer/git-rep/EasyDistillation` | https://github.com/IHEP-LQCD/EasyDistillation | 蒸馏方法（本征矢、perambulator、elemental 与自动缩并）的多后端实现 | `f74bf7de5ebb3ae94107c147339f66c5552cf240` |
| `refer/git-rep/LQCD_Master` | https://github.com/sjtu-sai-agents/LQCD_Master | 自然语言到 PyQUDA 程序与 Slurm 脚本的 agentic 计算流水线 | `e4a443e08d5904371fc280ce93c00ff84df21893` |
| `refer/git-rep/PyQCU` | https://gitlab.itp.ac.cn/zhangxin8069/PyQCU.git | torch、mpi4py 与 h5py 架构范例及 PyQCD 实现参考 | `8d7002f908aa2c162705f840b604814971753171` |
| `refer/git-rep/opt_einsum` | https://github.com/dgasmith/opt_einsum | 张量缩并路径优化库（`einsum_path`/`contract`），供缩并成本诊断与路径加速参考 | `fccdd5aa2a933c8bd1452f44689d15e737982cc0` |
| `refer/git-rep/Einsums` | https://github.com/Einsums/Einsums | C++20 张量代数库：编译期模板与运行时字符串两套 `einsum`、PackedGemm 后端、ComputeGraph 延迟执行与 NumPy 风格 Python 绑定 | `628971cd886cd5bb9ecb71d87289522d6625910d` |
| `refer/git-rep/MatX` | https://github.com/NVIDIA/MatX | CUDA C++17 张量计算库：GPU 端 `einsum`（cuTENSOR/cuTensorNet 后端）、GEMM 与算子融合 | `0ac86644a82d63e578f695e335fb2d01e1cb06ae` |

`PyQUDA` 内部的 `pyquda_core/pycparser` 是其上游嵌套 submodule
（https://github.com/eliben/pycparser.git），需以 `git submodule update --init
--recursive` 初始化。
