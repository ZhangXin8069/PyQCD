# refer/git-rep 参考库清单

本目录只读收录 lattice QCD 参考软件与技能库。除 `LQCD_Skills` 以 submodule 引入外，
其余四个为已验证的普通参考拷贝（非独立仓库，由主仓库跟踪）；其 `docs/` 中的
`analy_*`、`pure_*`、`report_*` TeX/PDF 是本库产出的分析文档，按本地 form 例外原地保留。

| 路径 | 来源 | 用途 | 引入方式 |
|---|---|---|---|
| `refer/git-rep/PyQUDA` | https://github.com/CLQCD/PyQUDA | QUDA 的 Python/Cython 封装，面向 GPU/MPI 格点 QCD 计算 | 普通参考拷贝 |
| `refer/git-rep/LQCD_Skills` | https://github.com/AI4LGT/LQCD_Skills | 从物理观测量到数值结果的格点 QCD agent 技能集合 | Git submodule |
| `refer/git-rep/lamet-agent` | https://github.com/Greyyy-HJC/lamet-agent | LaMET/LQCD 分析 agent 的 Python 脚手架 | 普通参考拷贝 |
| `refer/git-rep/EasyDistillation` | https://github.com/IHEP-LQCD/EasyDistillation | 蒸馏方法（本征矢、perambulator、elemental 与自动缩并）的多后端实现 | 普通参考拷贝 |
| `refer/git-rep/LQCD_Master` | https://github.com/sjtu-sai-agents/LQCD_Master | 自然语言到 PyQUDA 程序与 Slurm 脚本的 agentic 计算流水线 | 普通参考拷贝 |

> 致谢：感谢上述项目及其作者公开软件与文档，为 PyQCD 的算法实现、接口设计和物理
> 工作流提供了重要参考。These are excellent software toolkits for lattice QCD.
