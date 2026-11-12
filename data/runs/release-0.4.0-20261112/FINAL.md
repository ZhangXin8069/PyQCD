# PyQCD 0.4.0 发布证据

日期：2026-11-12

## 发布候选

| 项 | 值 |
|---|---|
| 版本 | `0.4.0` |
| 原生 ABI 版本 | `0.1.0`（未随包版本修改） |
| sdist | `pyqcd-0.4.0.tar.gz` |
| 大小 | `1,056,739` bytes |
| SHA-256 | `032a56302b50c8b1a500a0976a66615296bec6e34cbe2fec5d00ce7da29fa25e` |

## 关键门禁

| 门禁 | 作业 | 结果 |
|---|---|---|
| CPU 离线发布检查 | `1648768` | `11/11 passed` |
| GPU 统一套件 | `1648767` | `120 passed / 1 skipped / 0 failed` |
| KUPL 原生契约 | `1648678` | `16/16 passed` |
| CUDA 原生契约 | `1648678` | `16/16 passed` |
| 计划资源门 | `1648678` | `12/12 passed` |
| Python 热路径 | `1648678` | `4/4 passed` |
| form 审计 | `1648778` | registered exceptions 19, unexpected 0 |

GPU 统一套件的唯一跳过项是 `xelatex is unavailable`；该依赖与 einsum 发布路径无关。

## sdist 二次构建

作业 `1648752`：

* setuptools 隔离构建生成 sdist；
* `twine check` PASSED；
* 内容审计通过，无 `pyqcd.testing`、无 `.so`、无 `AGENTS.md`、无 `/home/` 绝对路径；
* 新建 venv 从 sdist 成功构建并安装 KUPL，`backend=kupl`；
* 新建 venv 使用 `PYQCD_SKIP_NATIVE_BUILD=1` 成功安装，`backend=python`；
* 两种安装均在源码树外导入 `site-packages` 中的 `pyqcd`，版本为 `0.4.0`。

## 生产性能边界

* 43 条 full-scene：CPU 对 PyQCD generic `42/43` 不慢，中位 `0.8612x`；
  GPU `42/43` 不慢，中位 `0.2760x`。
* 对 `opt_einsum 3.4.0 dp`：CPU `36/43` 不慢，GPU `42/43` 不慢。
* CPU `seq=6585`：`22.445 s -> 10.310 s`。
* 8638 条计划审计：CPU 放行 3833、GPU 放行 2864；计划不可用 0。
* 43 条 full-scene 最大相对误差：CPU `6.95e-15`，GPU `6.68e-15`。

详细数据和规律见
`data/runs/einsum-opt-20261112/expanded-paths-20261112/FINAL.md` 与
`EXPERIENCE.md`。
