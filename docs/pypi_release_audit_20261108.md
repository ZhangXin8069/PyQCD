# PyPI 发布硬化台账（2026-11-08 起）

> 判定口径：**一律以 `pip install pyqcd==<ver>` 从 PyPI 远程安装到全新 venv 的结果为准**，
> 本地 sdist、源码树里的 `import pyqcd` 都不算证据。
> 约束：不修改核心代码（接口级 debug 允许）。经验实时回写 `skills/pypi-release/` 与
> `skills/readthedocs/`。

## 验收清单

### A. 规范性（Packaging compliance）
| # | 检查项 | 判定方法 | 当前状态 |
|---|---|---|---|
| A1 | 包名/版本/摘要/许可证/Python 版本元数据正确 | PyPI JSON API 字段核对 | 0.1.4 ✅ |
| A2 | 只发布 sdist（wheel 标签对 `.so` 不成立） | PyPI JSON 的 `urls` 只有 `.tar.gz` | 0.1.4 ✅ |
| A3 | sdist 无 `pyqcd/testing`、无 `.so`、无绝对路径条目 | 解包 `tar -tzf` 审计 | 0.1.4 ✅ |
| A4 | 所有声明的子包都在 sdist 内 | `find_packages` 列表 vs tar 列表 | 0.1.4 ✅（12 个） |
| A5 | 依赖声明完整（无隐性 import 失败） | 全新 venv 里逐个 `import pyqcd.<sub>` | 0.1.4 ✅（12/12） |
| A6 | 无可选重依赖被硬编入（torch/matplotlib 按需） | `find_spec` 断言 + 子包 import | 0.1.4 ✅ |
| A7 | `twine check` 全绿；README 渲染为 Markdown | twine + PyPI `description_content_type` | 0.1.4 ✅ |
| A8 | PEP 561 类型信息（如有 `.pyi`/`py.typed` 需与声明一致） | 待核 | 待查 |
| A9 | 项目 URL 指向真实可达平台 | 三个托管地址 200 | 0.1.4 ✅ |
| A10 | 无 setuptools 弃用告警（license 表等） | 构建日志 | 待查（D5） |

### B. 易用性（First-run usability）
| # | 检查项 | 判定方法 | 当前状态 |
|---|---|---|---|
| B1 | 安装即用：一条命令、无额外手动步骤 | 全新 venv `pip install pyqcd` | ✅ |
| B2 | `import pyqcd` 后的可发现性 | `dir(pyqcd)`、`__version__` | 待评 |
| B3 | 文档示例在**已安装包**上可直接跑通 | 逐段执行 docs 示例（不含源码仓库专属） | 待查 |
| B4 | 错误信息可行动（缺 extra 时提示装哪个 extra） | 触发 ImportError 检查文案 | 待查 |
| B5 | `pyqcd.einsum` 与 `opt_einsum` API 对等 | 38 项 API 差分 | 待复核 |
| B6 | 基准/测试命令明确标注"需源码仓库" | 文档措辞 | ✅ |

### C. 生产力价值（Productivity）
| # | 检查项 | 判定方法 | 当前状态 |
|---|---|---|---|
| C1 | 无需源码即可完成一次真实收缩（2pt 形态） | 纯安装环境跑 E1 方程 + 正确性 | 待查 |
| C2 | 与 `opt_einsum` 可互换（同一脚本只改 import） | 双实现同脚本对照 | 待查 |
| C3 | 性能优势可在安装环境复现（A/B 开关可见） | `specialize=False` 对照 + 计时 | 待查 |
| C4 | 物理组件可组合（gamma + 规范 + 收缩） | 端到端小例子 | 待查 |

## 迭代记录

| 轮次 | 时间 | 版本 | 发现 | 处置 |
|---|---|---|---|---|
| R0 | 11:45 | 0.1.4 | 建台账；A5/A6 已由 0.1.4 修复（h5py/scipy 声明、torch/matplotlib 延迟） | — |
| R1 | 12:05 | 0.1.4（待 0.1.5） | B2 可发现性差（`__init__` 无 docstring/`__all__`，注释里子包列表过期）；A8 `Typing :: Typed` 但无 `py.typed` | 已加模块 docstring + `__all__`（12 子包）+ `pyqcd/py.typed` + `package_data`；新增 `python -m pyqcd` 自检入口并写入中英文 install 文档 |
| R2 | 12:40 | 0.1.5 | A组 9 条（SOURCES 泄漏家目录、py.typed 漏打包、license 弃用、classifier 缺 3.13/3.14、plot 报错不可行动、AGENTS.md 进 sdist…）；B组 9 条（安装不 hermetic、usage 两处 API 签名错、torch/matplotlib 裸 ModuleNotFoundError、顶层零可发现性、混合精度无处方、native_library 指向不存在文件）；C组终止前发现 `cached_contract` 依赖未声明的 opt_einsum | 0.1.5 已修：cythonize 相对路径；`py.typed`+package_data+MANIFEST；PEP 639 `license`+`license-files`+setuptools>=77；3.13/3.14 classifier；`pyqcd[plot]`/`[torch]` 可行动报错；PEP 562 `__getattr__/__dir__`；`cached_contract` 回退自带引擎；`build_info().native_available`；主机编译器试编译失败只跳过 CUDA；wickplot 必需键校验；MANIFEST `prune pyqcd/testing`；usage.md 三处代码块改为真实签名并通过顺序执行 |
| R3 | 13:10 | 0.1.5 | 0.1.5 从 PyPI 远程复验通过：12/12 子包 import（无 torch/matplotlib）、`pyqcd.einsum` 懒加载、`cached_contract` 无 opt_einsum、`py.typed` 就位、torch/plot 报错可行动；Python 3.14 用 `setup.py build` + `PYTHONPATH` 实测 `import pyqcd` 与 `python -m pyqcd` 正常（该环境无 numpy，后端如实报 unknown） | 3.13/3.14 classifier 保留（3.14 已验证构建与导入）；README 补 `python -m pyqcd` |
| R4 | 13:25 | 0.1.5 | S1 复测通过：`CC=/bin/false CXX=/bin/false NVCC=/bin/false` 下 `pip install pyqcd==0.1.5` 退出码 0，`backend='python'`、`native_available=False`、12/12 子包与 contract 正常；A10 复检：license 弃用告警已消失、SOURCES 绝对路径 0；新增文档示例门 | 新增 `pyqcd/testing/docs_examples.py`（en/zh 各 16 块、0 失败）；MANIFEST 精简（`global-exclude` 的 4 条无害告警保留，已在技能中说明不要为消警告删安全排除） |
| R5 | 14:20 | 0.1.6 | C 组三条：①「专用路径小规模慢 10.9×」经复核为**测量假象**（`specialize=False` 连计划缓存一起关；且合成形状从不命中 kernel）；②E4/E5 在 REGISTRY 里标 `fused` 并打印不存在的 CUDA kernel → 改为 `plan-cache` 并据实打印 `[kernel: …]`/`[plan-cache]`；③`build_info()` 只报 `backend` 误导（numpy 输入其实走 KUPL） → 新增 `array_backend` | 先加再撤规模门（实测默认从不变差：nvec 8/16/24 上默认比 `SPECIALIZE=0` 快 1.3×/7.5×/6.3×）；0.1.6 已发布并从 PyPI 复验：12/12 子包、`[kernel: kupl_nucleon_2pt]` 标签正确、rel 3.5e-15/1.7e-14 |
| R6 | 15:05 | 0.1.7 | C 组遗留接口缺陷：`compute_dual_field_strength` 缺分量静默返回 `None`、传 ndarray 抛无信息量 `AttributeError`；缺元数据一致性回归测试 | 0.1.7：`TypeError`/`ValueError` 带处方；新增 `pyqcd/testing/_release_metadata_contract.py`（版本三处一致、`__all__`↔磁盘、子包可导入、MANIFEST 策略、`build_info` 字段）并接入 `conftest`，`python -m pyqcd.testing` 115→**116 passed**；PyPI 远程复验 12/12 子包、两条错误文案正确 |
| R7 | 16:05 | 0.1.8 | ①文档性能表不可复现（E4/E5 的 8.33×/3.371× 依赖发行版里没有的融合 kernel）；②`pyqcd[cuda]` 指向源码包 `cupy`，安装必然失败；③`/tmp` tmpfs 被历史 venv 撑满导致解析失败被误判 | 0.1.8：性能表改为"发布即可复现"三行（E1 6.02×/5.83×、E5 2.91×）+ 历史数字单独标注不可复现；`cuda` extra → `cupy-cuda12x`；清理 /tmp（100%→49%）并把 `df -h /tmp` 作为失败排查第一步；PyPI 复验三个 extras 全部 `Would install` 通过 |
| R8 | 16:55 | 0.1.9 | ①`pyqcd[plot]` 只验证过"能解析"，未验证可用；②文档未说明"为何无 wheel / 首次安装要编译"；③A9 五个 URL 复检 | 0.1.9：干净 venv 装 `pyqcd[plot]` 端到端跑通 `wick_contraction → plot_figure_wick`（107 KB PNG，12 键）；install 页（中英）补"无平台 wheel 的理由 + 首次安装约 1 分钟 + pip 缓存"；五个 URL 全部 200；`pyqcd[torch]` 因 539 MB 轮子仅做 dry-run，标注未验证 |
| R9 | 17:40 | 0.1.9（无新发行） | ①文档 API 引用此前只靠"代码块能跑"间接保障，伪代码块不被执行；②文档里的 bash 命令未被任何门禁覆盖；③RTD 站点是否已重建到 0.1.9 未知 | 新增 `pyqcd/testing/_docs_api_contract.py`（AST 解析，35 个引用 0 失败，含"至少 20 引用"防空转）并接入 `conftest`，`python -m pyqcd.testing` 116→**117 passed**；三条文档命令 `--help` + bench `--smoke` 实跑通过；RTD en/zh 已确认重建（含 6.02×、auditwheel、cupy-cuda12x、plan-cache）。本包内容未变，**不发新版本** |
| R10 | 18:35 | 0.1.9（无新发行） | ①`pyqcd[torch]` 仅做过 dry-run，未端到端；②PyPI 项目页无法用 HTML 抓取核对（bot 挑战） | torch extra 端到端通过：torch 2.15.0+cu132、`set_backend('torch')`、torch 张量 contract 与 `a @ b` 一致、numpy 输入仍可用、`gamma(0)` 返回 torch 复数张量、`python -m pyqcd` 正确列出缺失 extras；PyPI 页改用 JSON `info.description` 探针核对（Repositories/三平台/文档/python -m pyqcd/6.02 全部命中）。包内容未变，不发新版本 |
