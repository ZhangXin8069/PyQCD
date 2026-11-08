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
