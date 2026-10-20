# refer/git-rep 参考库清单

本目录只读收录 lattice QCD 参考软件与技能库，全部以 Git submodule 固定在上游
commit。子库工作树保持上游状态，不修改、不把上游文件混入主库源码。本库对这些库的
分析文档统一存放于 `refer/git-rep/docs/`。

| 路径 | 来源 | 用途 | 固定 commit |
|---|---|---|---|
| `refer/git-rep/PyQUDA` | https://github.com/CLQCD/PyQUDA | QUDA 的 Python/Cython 封装，面向 GPU/MPI 格点 QCD 计算 | `138d8ad6e4d6e0805ecc7ec0a59a45febe0482ce` |
| `refer/git-rep/LQCD_Skills` | https://github.com/AI4LGT/LQCD_Skills | 从物理观测量到数值结果的格点 QCD agent 技能集合 | `90bcbb8739c80b41a3965f41dab50bdfc9841b16` |
| `refer/git-rep/lamet-agent` | https://github.com/Greyyy-HJC/lamet-agent | LaMET/LQCD 分析 agent 的 Python 脚手架 | `9355c52ab7f5061fcc1befacfb7e1b65a0b32db1` |
| `refer/git-rep/EasyDistillation` | https://github.com/IHEP-LQCD/EasyDistillation | 蒸馏方法（本征矢、perambulator、elemental 与自动缩并）的多后端实现 | `f74bf7de5ebb3ae94107c147339f66c5552cf240` |
| `refer/git-rep/LQCD_Master` | https://github.com/sjtu-sai-agents/LQCD_Master | 自然语言到 PyQUDA 程序与 Slurm 脚本的 agentic 计算流水线 | `e4a443e08d5904371fc280ce93c00ff84df21893` |

`PyQUDA` 内部的 `pyquda_core/pycparser` 是其上游嵌套 submodule
（https://github.com/eliben/pycparser.git），需以 `git submodule update --init
--recursive` 初始化。

> 致谢：感谢上述项目及其作者公开软件与文档，为 PyQCD 的算法实现、接口设计和物理
> 工作流提供了重要参考。These are excellent software toolkits for lattice QCD.
