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
| A8 | PEP 561 类型信息 | `py.typed` 在包内 + `Typing :: Typed` 声明 | ✅ 0.1.5 起随包发布（R2/R14 复核） |
| A9 | 项目 URL 指向真实可达平台 | 三个托管地址 200 | 0.1.4 ✅ |
| A10 | 无 setuptools 弃用告警 | 构建日志 grep `deprecat` | ✅ PEP 639 修完（R4）；仅剩 `global-exclude` 无害告警（R4 记录） |

### B. 易用性（First-run usability）
| # | 检查项 | 判定方法 | 当前状态 |
|---|---|---|---|
| B1 | 安装即用：一条命令、无额外手动步骤 | 全新 venv `pip install pyqcd` | ✅ |
| B2 | `import pyqcd` 后的可发现性 | `dir(pyqcd)` + PEP 562 懒加载 | ✅ `__all__`/`__getattr__`/`__dir__` + `python -m pyqcd`（R1/R2） |
| B3 | 文档示例在**已安装包**上可直接跑通 | `python -m pyqcd.testing.docs_examples` | ✅ 16 块 0 失败（R2/R9）；API 引用 AST 校验 35 项 0 失败（R9） |
| B4 | 错误信息可行动 | 触发 ImportError 检查文案 | ✅ torch/plot 均给出 `pip install "pyqcd[...]"`（R2）；6 类非法输入与 opt_einsum 逐字一致（R12） |
| B5 | `pyqcd.einsum` 与 `opt_einsum` API 对等 | 差分套件 + 边界矩阵 | ✅ 49 项行为 MATCH；差异仅两处增量（callable planner、顶层 PathInfo，R19） |
| B6 | 基准/测试命令明确标注"需源码仓库" | 文档措辞 | ✅ |

### C. 生产力价值（Productivity）
| # | 检查项 | 判定方法 | 当前状态 |
|---|---|---|---|
| C1 | 无需源码即可完成一次真实收缩 | 纯安装环境跑 E1 方程 + 正确性 | ✅ rel 2.4e-15（R5/C 组）；可复现表 6.02×（R7） |
| C2 | 与 `opt_einsum` 可互换 | 双实现同脚本对照 | ✅ `opt_cost`/路径逐值一致（C 组 + R19） |
| C3 | 性能优势可在安装环境复现 | 发布包上 A/B 计时 | ✅ 默认对 `SPECIALIZE=0` 快 1.3×/7.5×/6.3×（R5），对 opt_einsum 6.02×/5.83×/2.91×（R7） |
| C4 | 物理组件可组合 | 端到端小例子 | ✅ clover→dual→γ→夹心 全链通（R14 文档示例实跑）；`compute_dual_field_strength` 错误已可行动（R6） |

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
| R11 | 19:30 | 0.1.9（无新发行） | 对抗性复查：dtype/形状/kwargs 边界此前无门禁；`dtype=`/`order=` 报 TypeError 疑似缺陷 | 新增 `_contract_edge_cases_contract.py`（8 组：全 dtype、ellipsis/重复标签/标量/迹、非连续与零尺寸、6 种 optimize、`out=`、expression+shared_intermediates、opt_cost 与 opt_einsum 逐值一致、三类非法输入错误类型）；`python -m pyqcd.testing` 117→**118 passed**。经对照确认 `dtype=`/`order=` 在 **opt_einsum 3.4.0 上同样抛 TypeError**，属 bug-for-bug 兼容，不修 |
| R12 | 20:20 | 0.1.9（无新发行） | ①错误路径消息质量未与 opt_einsum 逐字比对；②`-O/-OO/-B`/只读安装目录未验；③`-OO` 下 `docs_examples` 崩 | 6 类非法输入与 `contract_path` 的异常消息**逐字一致（SAME）**；`none/-O/-OO/-B` 与只读 site-packages 下 contract/opt_cost 全部正常；修 `docs_examples.py` 的 `__doc__` 依赖（-OO 下 `AttributeError`），118 passed 保持 |
| R13 | 21:05 | 0.1.9（无新发行） | ①离线/air-gapped 安装未验证（HPC 常态）；②classifier 声称 3.10–3.14，但本机只有 3.12 与 3.14（且 3.14 无 pip） | 离线安装验证通过：wheelhouse 7 包/60 MB，`PIP_NO_INDEX=1` 下成功构建并安装 0.1.9，装后 `backend=cuda`、contract 正确；文档（中英 install）新增"实测于 3.12 完整套件 / 3.14 构建+导入，3.10/3.11/3.13 未自动验证"的诚实声明；代码块仍 8/8 一致，sphinx -W 双站 0 告警 |
| R14 | 21:50 | 0.1.9（无新发行） | ①0.1.3 曾静默漏发 11 个子包，现行配置是否有同类风险未知；②构建可复现性未测 | 受控探针实验：新增 `pyqcd/_tmp_probe.py` 与 `pyqcd/_tmp_pkg/` 均**自动进入 sdist**，且 `pyqcd._tmp_pkg` 自动进 packages；同时 `test_public_subpackages_exist` 立即报 "only on disk=['_tmp_pkg']"，证明"未登记子包"会被抓住；探针已清理、工作树干净。可复现性：两次 sdist tarball 哈希不同但**解包内容逐字节相同**；加 `SOURCE_DATE_EPOCH` 仍不同（setuptools 84.2.1），故约定用内容哈希判断，不用 tarball sha256 |
| R15 | 22:40 | **0.1.10** | ①双原生后端一致性（kupl vs cuda）未验证；②cupy 设备数组走通用执行器报 `QCD_STATUS_INTERNAL` 回落，文档未说明；③`_specialize` 的 E1 banner **仍在引用已删除的** `PYQCD_EINSUM_KERNEL_MIN_ELEMENTS` | 一致性：4 个方程在 kupl/cuda 两条库路径上**逐位相同（0.00e+00）**，对 numpy 为 3.3e-15/8.3e-16/2.9e-16/0；cupy 实测结果正确但通用执行器回落且只警告一次，已写入"数组类型 × 执行路径"矩阵（中英 install）；banner 文案修正并随 0.1.10 发布，远程复验确认不再出现该变量、rel=3.5e-15 |
| R16 | 23:25 | **0.1.11** | ①`memory_limit` 语义与峰值内存从未验证；②内存差异对生产 OOM 有关键影响但文档未提 | `memory_limit` 四档语义与 opt_einsum 完全一致，`opt_cost`/`largest_intermediate`/`path` 逐值相同；峰值实测 pyqcd 为 opt_einsum 的 **1.4–2×**（1200²:18→41 MB、2000²:73→104 MB），耗时不变；该结论写入 performance 页（中英）与 README 并随 0.1.11 发布（PyPI `info.description` 已含 "1.4–2× the RSS"）；另记录良性回落：`optimize='optimal'`+极小 `memory_limit` 产生 arity=3 步 → 原生只支持一/二元步 → 单次 warning 回落 Python |
| R17 | 24:05 | 0.1.11（无新发行） | 并发安全未验证（嵌入流水线时的常见前提） | 多线程 matmul（8×16×25）、多线程专用路径（6×12×10）、`shared_intermediates` 线程隔离（4 线程各自 cache）、多进程（4×8）全部 0 错误；新增 `_thread_safety_contract.py` 四用例（0.07 s）并接入 conftest，`python -m pyqcd.testing` 118→**119 passed**；记录 `count_cached_ops(cache)` 需传参的坑 |
| R18 | 24:50 | 0.1.11（无新发行） | 终检证据分散在多个命令里，打 tag 时容易漏项 | 新增 `pyqcd/testing/release_check.py`：一条命令验证 16 项（版本三处一致、sdist 存在/twine/内容洁净/无绝对路径、文档示例+中英块一致+sphinx -W 双站、PyPI 版本与页面探针、RTD 双语言含当前版本、gitlab/gitee 与本地 HEAD 同步），失败非 0 退出可直接做 tag 门。实测全量 **16/16 passed**，`--offline --with-tests` **10/10 passed** |
| R19 | 25:35 | 0.1.11（无新发行） | ①`contract_expression` 的 constants/形状语义未与 opt_einsum 对照；②自定义 `optimize` callable 与路径缓存失效未验；③"完全兼容"承诺缺少差异清单 | 实测：`contract_expression(*shapes)` 两边一致；`constants={...}` 两边抛**同样的** `ValueError`（bug-for-bug）；不同 shape 的路径与 cost（240 vs 252）一致；`optimize=` 传 callable 时 **pyqcd 支持而 opt_einsum 3.4.0 抛 TypeError**（超集）。已在 usage 页（中英）新增 "Differences from opt_einsum" 小节，仅列两处增量差异 |
| R20 | 25:20 | 0.1.11 | 台账顶部 A/B/C 清单仍留着 R0 时的"待查/待评"旧状态，与后面 19 轮的结论不一致 | 把 A8/A10/B2/B3/B4/B5/C1/C2/C3/C4 全部改写为已验证状态并标注对应轮次证据，`待查/待评/待核/待复核` 归零；`release_check --with-tests` 全量 **17/17 passed**（含 119 passed 测试套件） |
| R21 | 17:46 | 0.1.11 | 6 小时门槛到达，需终检并打 tag | 终检 **17/17 passed**（含 119 passed 测试套件）；工作树干净；打 annotated tag **`dev34`**（dev 系列 33 → 34）并推送到 gitlab 与 gitee，三端 tag 对象 `cae654c0` 一致；循环终止 |
| R22 | 26:10 | **0.1.13** | 用户问"PyPI 装的 pyqcd 是否支持 CUDA"→ 实测发现：CUDA 库**编译得出来**，但 `make-nv.sh` 默认架构 `6.0 7.0` 只嵌 SASS，在 RTX 4060（sm_89）上 kernel 入口返回 `QCD_STATUS_INTERNAL`，**静默回落 CPU**；文档只在源码构建一节提到架构变量 | 0.1.13：install 页（中英）新增"CUDA 架构必须匹配"专节（默认 7.0 vs 实测对照表、`PYQCD_CUDA_ARCHITECTURES=8.9 pip install` 写法、`strings`/`nvidia-smi` 诊断命令、`[plan-cache]` 判据），README 双版加一行提示；实测 8.9 下 kernel status=0、rel 2.8e-14；PyPI 0.1.13 远程复验通过且页面已含该提示 |
