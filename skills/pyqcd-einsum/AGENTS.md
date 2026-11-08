# AGENTS.md — skills/pyqcd-einsum

本目录维护 `pyqcd-einsum` 技能，覆盖 opt_einsum 兼容 API、原生收缩计划、CUDA/KUPL
执行、并行与性能验收。

- `SKILL.md` 是权威入口；详细 ABI、计划和性能证据在 `references/`。
- 委派/门限、线程档与测量协议的实测判据见 `SKILL.md`「性能调试与路径选择」。
- 物理缩并语义交给 `pyqcd-distillation`，平台与计时纪律交给 `pyqcd-infra`。
- 只记录当前可复现证据；实现存在、数值正确和 3x 性能必须分别标注。
- 不修改 `refer/git-rep/opt_einsum`，生产代码不得 import 参考区或测试包。
