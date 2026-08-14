# AGENTS.md — agent/docker-v20260805/lib

`docker-v20260805` 内自包含蒸馏/收缩框架——`examples/sush/lqcddb` 的扁平快照（逐字复制，**不 import**），使流水线无需依赖 examples 树的包。生成此库的流水线技能为 `docker-v20260805`（扁平副本：`agent/docker-v20260805.skill.md`）。

## 模块

`backend.py`（后端切换）、`base_functions.py`（Levi-Civita、动量表、缓存 einsum）、`gamma_matrix.py`（DR γ 18 种）、`sigma_matrix.py`、`constants.py`、`baroperator.py`、`seqperam.py`、`autowick.py`、`dynamic.py`、`vertex.py`（VdV/VVV）、`analyse.py`、`io_readers.py`、`__init__.py`。

## 约定

- **禁止从 `examples/sush/` import**——本目录是自包含副本，与上游 lqcddb 手工同步
- 新模块/图/输出属于流水线自身 `output/`，不放这里
